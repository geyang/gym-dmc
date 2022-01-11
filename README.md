# `gym-dmc`, OpenAI Gym Plugin for DeepMind Control Suite

Link to other OpenAI Gym Plugins:

- `gym-sawyer`
- `gym-toy-nav`



## Update Log

- **2022-01-11**: Added a `env._get_obs()` method to allow one to obtain the observation after resetting the environment. **Version: `v0.2.1`**

## How To Use

Usage pattern:

```python
import gym

env = gym.make("dmc:Pendulum-swingup-v1")
```

For the full list of environments, you can print:
```python
from dm_control.suite import ALL_TASKS

print(*ALL_TASKS, sep="\n")

# Out[2]: ('acrobot', 'swingup')
#         ('acrobot', 'swingup_sparse')
...
```
We register all of these environments using the following
pattern:

> acrobot task "swingup_sparse" becomes `dmc:Acrobot-swingup_sparse-v1`

You can see the usage patten in [./specs/test_gym_dmc.py](./specs/test_gym_dmc.py):

```python
env = gym.make('dmc:Walker-walk-v1', frame_skip=4)
assert env._max_episode_steps == 250
assert env.reset().shape == (24,)

env = gym.make('dmc:Walker-walk-v1', from_pixels=True, frame_skip=4)
assert env._max_episode_steps == 250

env = gym.make('dmc:Cartpole-balance-v1', from_pixels=True, frame_skip=8)
assert env._max_episode_steps == 125
assert env.reset().shape == (3, 84, 84)

env = gym.make('dmc:Cartpole-balance-v1', from_pixels=True, frame_skip=8, channels_first=False)
assert env._max_episode_steps == 125
assert env.reset().shape == (84, 84, 3)

env = gym.make('dmc:Cartpole-balance-v1', from_pixels=True, frame_skip=8, channels_first=False, gray_scale=True)
assert env._max_episode_steps == 125
assert env.reset().shape == (84, 84, 1)
```

**Note, the `max_episode_steps` is calculated based on the `frame_skip`.** All DeepMind control domains terminate after 1000 simulation steps. So for `frame_skip=4`, the `max_episode_steps` should be 250.

Built with :heart: by Ge Yang