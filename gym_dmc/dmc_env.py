import gym
from gym import spaces
import numpy as np

from dm_control import suite
from dm_env import specs


def convert_dm_control_to_gym_space(dm_control_space):
    r"""Convert dm_control space to gym space. """
    if isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(low=dm_control_space.minimum,
                           high=dm_control_space.maximum,
                           dtype=dm_control_space.dtype)
        assert space.shape == dm_control_space.shape
        return space
    elif isinstance(dm_control_space, specs.Array) and not isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(low=-float('inf'),
                           high=float('inf'),
                           shape=dm_control_space.shape,
                           dtype=dm_control_space.dtype)
        return space
    elif isinstance(dm_control_space, dict):
        space = spaces.Dict({key: convert_dm_control_to_gym_space(value)
                             for key, value in dm_control_space.items()})
        return space


class DMCEnv(gym.Env):
    def __init__(self, domain_name, task_name,
                 task_kwargs=None,
                 environment_kwargs=None,
                 visualize_reward=False,
                 height=84,
                 width=84,
                 camera_id=0,
                 frame_skip=1,
                 channels_first=True,
                 img_obs=False,
                 gray_scale=True
                 ):
        self.env = suite.load(domain_name,
                              task_name,
                              task_kwargs=task_kwargs,
                              environment_kwargs=environment_kwargs,
                              visualize_reward=visualize_reward)
        self.metadata = {'render.modes': ['human', 'rgb_array'],
                         'video.frames_per_second': round(1.0 / self.env.control_timestep())}

        self.observation_space = convert_dm_control_to_gym_space(self.env.observation_spec())
        self.action_space = convert_dm_control_to_gym_space(self.env.action_spec())
        self.viewer = None

        self.render_kwargs = dict(
            height=height,
            width=width,
            camera_id=camera_id,
        )
        self.frame_skip = frame_skip
        self.img_obs = img_obs
        self.gray_scale = gray_scale
        self.channels_first = channels_first

    def seed(self, seed):
        return self.env.task.random.seed(seed)

    def step(self, action):
        reward = 0
        extra = {'sim_state': self.env.physics.get_state().copy()}

        for i in range(self.frame_skip):
            ts = self.env.step(action)
            reward += ts.reward or 0
            done = ts.last()
            if done:
                break
        obs = ts.observation

        if self.img_obs:
            obs['img'] = self._get_obs_img()

        return obs, reward, done, extra

    def _get_obs_img(self):
        img = self.render("gray" if self.gray_scale else "rgb", **self.render_kwargs)
        return img.transpose([2, 0, 1]) if self.channels_first else img

    def reset(self):
        obs = self.env.reset().observation

        if self.img_obs:
            obs['img'] = self._get_obs_img()

        return obs

    def render(self, mode='human', height=None, width=None, camera_id=0, **kwargs):
        img = self.env.physics.render(
            width=self.render_kwargs['width'] if width is None else width,
            height=self.render_kwargs['height'] if height is None else height,
            camera_id=self.render_kwargs['camera_id'] if camera_id is None else camera_id,
            **kwargs).astype(np.uint8)
        if mode in ['rgb', 'rgb_array']:
            return img
        elif mode in ['gray', 'grey']:
            return img.mean(axis=-1, keepdims=True)
        elif mode == 'notebook':
            from IPython.display import display
            from PIL import Image
            img = Image.fromarray(img, "RGB")
            display(img)
            return img
        elif mode == 'human':
            from PIL import Image
            return Image.fromarray(img)
        else:
            raise NotImplementedError(f"`{mode}` mode is not implemented")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self.env.close()
