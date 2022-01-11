import gym.spaces as spaces
from gym import ObservationWrapper


class FlattenObservation(ObservationWrapper):
    """Observation wrapper that flattens the observation.

    Note: Different from the default gym wrapper by adding _get_obs function
    """

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)
        self.observation_space = spaces.flatten_space(env.observation_space)

    def observation(self, observation):
        return spaces.flatten(self.env.observation_space, observation)

    def _get_obs(self):
        obs = self.env.unwrapped._get_obs()
        return self.observation(obs)


class ObservationByKey(ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""

    def __init__(self, env, obs_key):
        super().__init__(env)
        self.obs_key = obs_key
        self.observation_space = env.observation_space[obs_key]

    def observation(self, observation):
        return observation[self.obs_key]
