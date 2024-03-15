import gym
import numpy as np


def test_max_episode_steps():
    env = gym.make('dmc:Walker-walk-v1', frame_skip=4)
    assert env.spec.max_episode_steps == 250


def test_frame_skip():
    env = gym.make('dmc:Walker-walk-v1', frame_skip=8)
    assert env.spec.max_episode_steps == 125


def test_pixel_output():
    env = gym.make('dmc:Walker-walk-v1', from_pixels=True, frame_skip=8)
    obs, info = env.reset()
    assert obs.shape == (3, 84, 84)


def test_flat_obs():
    env = gym.make('dmc:Walker-walk-v1', frame_skip=4)
    obs, info = env.reset()
    assert obs.shape == (24,)


def test_flat_space_dtype():
    env = gym.make('dmc:Walker-walk-v1', frame_skip=4, space_dtype=np.float32)
    assert env.action_space.dtype == np.float32
    assert env.observation_space.dtype == np.float32


def test_channel_first():
    env = gym.make('dmc:Walker-walk-v1', from_pixels=True, frame_skip=8, channels_first=False)
    assert env.spec.max_episode_steps == 125
    obs, info = env.reset()
    assert obs.shape == (84, 84, 3)


def test_gray_scale():
    env = gym.make('dmc:Walker-walk-v1', from_pixels=True, frame_skip=8, channels_first=False,
                   gray_scale=True)
    assert env.spec.max_episode_steps == 125
    obs, info = env.reset()
    assert obs.shape == (84, 84, 1)
