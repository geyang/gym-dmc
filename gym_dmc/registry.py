from __future__ import annotations

import copy
import importlib
import importlib.util
import re
import sys
from typing import (
    Callable,
    Type,
    Optional,
    Union,
    Tuple,
    Any,
)

from gym_dmc.wrappers.order_enforcing import OrderEnforcing
from gym_dmc.wrappers.time_limit import TimeLimit

if sys.version_info < (3, 10):
    pass  # type: ignore
else:
    pass

from dataclasses import dataclass, field

from .gym.core import Env


if sys.version_info >= (3, 8):
    pass
else:

    class Literal(str):
        def __class_getitem__(cls, item):
            return Any


ENV_ID_RE: re.Pattern = re.compile(r"^(?:(?P<namespace>[\w:-]+)\/)?(?:(?P<name>[\w:.-]+?))(?:-v(?P<version>\d+))?$")


def load(name: str) -> Type:
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def parse_env_id(id: str) -> Tuple[Optional[str], str, Optional[int]]:
    """Parse environment ID string format.

    This format is true today, but it's *not* an official spec.
    [username/](env-name)-v(version)    env-name is group 1, version is group 2

    2016-10-31: We're experimentally expanding the environment ID format
    to include an optional username.
    """
    match = ENV_ID_RE.fullmatch(id)
    if not match:
        raise RuntimeError(f"Malformed environment ID: {id}." f"(Currently all IDs must be of the form {ENV_ID_RE}.)")
    namespace, name, version = match.group("namespace", "name", "version")
    if version is not None:
        version = int(version)

    return namespace, name, version


@dataclass
class EnvSpec:
    """A specification for a particular instance of the environment. Used
    to register the parameters for official evaluations.

    Args:
        id_requested: The official environment ID
        entry_point: The Python entrypoint of the environment class (e.g. module.name:Class)
        reward_threshold: The reward threshold before the task is considered solved
        nondeterministic: Whether this environment is non-deterministic even after seeding
        max_episode_steps: The maximum number of steps that an episode can consist of
        order_enforce: Whether to wrap the environment in an orderEnforcing wrapper
        kwargs: The kwargs to pass to the environment class

    """

    id_requested: str
    entry_point: Optional[Union[Callable, str]] = field(default=None)

    reward_threshold: Optional[int] = field(default=None)
    nondeterministic: bool = field(default=False)
    max_episode_steps: Optional[int] = field(default=None)

    order_enforce: bool = field(default=True)
    kwargs: dict = field(default_factory=dict)

    namespace: Optional[str] = field(init=False)
    name: str = field(init=False)
    version: Optional[int] = field(init=False)

    def __post_init__(self):
        self.namespace, self.name, self.version = parse_env_id(self.id_requested)

    @property
    def eid(self) -> str:
        """
        `id_requested` is an InitVar meaning it's only used at initialization to parse
        the namespace, name, and version. This means we can define the dynamic
        property `id` to construct the `id` from the parsed fields. This has the
        benefit that we update the fields and obtain a dynamic id.
        """
        # namespace = "" if self.namespace is None else f"{self.namespace}/"
        name = self.name
        version = "" if self.version is None else f"-v{self.version}"
        return f"{name}{version}"
        # return f"{namespace}{name}{version}"

    def make(self, **kwargs) -> Env:
        """Instantiates an instance of the environment with appropriate kwargs"""
        _kwargs = self.kwargs.copy()
        _kwargs.update(kwargs)

        if callable(self.entry_point):
            env = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            env = cls(**_kwargs)

        # Make the environment aware of which spec it came from.
        spec = copy.deepcopy(self)
        spec.kwargs = _kwargs
        env.unwrapped.spec = spec
        if self.order_enforce:
            env = OrderEnforcing(env)

        assert env.spec is not None, "expected spec to be set to the unwrapped env."

        if env.spec.max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=env.spec.max_episode_steps)

        return env

    def __repr__(self):
        return f"""
        EnvSpec( 
            id_requested={self.id_requested},
            entry_point={self.entry_point},
            reward_threshold={self.reward_threshold},
            nondeterministic={self.nondeterministic},
            max_episode_steps={self.max_episode_steps},
            order_enforce={self.order_enforce},
            kwargs={self.kwargs}
        )"""


class EnvRegistry:
    """Register an env by ID. IDs remain stable over time and are
    guaranteed to resolve to the same environment dynamics (or be
    desupported). The goal is that results on a particular environment
    should always be comparable, and not depend on the version of the
    code that was running.
    """

    def __init__(self):
        self.specs = {}

    def make(self, eid: str, **kwargs) -> Env:
        if ":" in eid:
            module_name, eid = eid.split(":")
        else:
            module_name = None

        spec = self.specs[eid]

        spec_options = {**spec.kwargs, **kwargs}

        entry_point = spec.entry_point
        module_name, entrypoint_name = entry_point.split(":")

        module = importlib.import_module(module_name)
        entry_point = getattr(module, entrypoint_name)

        env = entry_point(eid, **spec_options)

        return env

    def register(self, eid: str, **kwargs) -> None:
        spec = EnvSpec(id_requested=eid, **kwargs)
        self.specs[spec.eid] = spec

    def __repr__(self):
        return repr(self.specs)


# Have a global registry
registry = EnvRegistry()


def register(eid: str, **kwargs) -> None:
    return registry.register(eid, **kwargs)
