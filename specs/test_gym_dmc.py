import numpy as np

import gym_dmc


def test_max_episode_steps():
    env = gym_dmc.make("dmc:Walker-walk-v1", frame_skip=4)
    assert env.spec.max_episode_steps == 250


def test_frame_skip():
    env = gym_dmc.make("dmc:Walker-walk-v1", frame_skip=8)
    assert env.spec.max_episode_steps == 125


def test_pixel_output():
    env = gym_dmc.make("dmc:Walker-walk-v1", from_pixels=True, frame_skip=8)
    assert env.reset().shape == (3, 84, 84)


def test_flat_obs():
    env = gym_dmc.make("dmc:Walker-walk-v1", frame_skip=4)
    assert env.reset().shape == (24,)


def test_flat_space_dtype():
    env = gym_dmc.make("dmc:Walker-walk-v1", frame_skip=4, space_dtype=np.float32)
    assert env.action_space.dtype == np.float32
    assert env.observation_space.dtype == np.float32


def test_channel_first():
    env = gym_dmc.make(
        "dmc:Walker-walk-v1",
        from_pixels=True,
        frame_skip=8,
        channels_first=False,
    )
    assert env.spec.max_episode_steps == 125
    assert env.reset().shape == (84, 84, 3)


def test_gray_scale():
    env = gym_dmc.make(
        "dmc:Walker-walk-v1",
        from_pixels=True,
        frame_skip=8,
        channels_first=False,
        gray_scale=True,
    )
    assert env.spec.max_episode_steps == 125
    assert env.reset().shape == (84, 84, 1)
