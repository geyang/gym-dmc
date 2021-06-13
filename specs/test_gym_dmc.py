import gym


def test_max_episode_steps():
    env = gym.make('dmc:Walker-walk-v1', from_pixels=True, frame_skip=4)
    assert env._max_episode_steps == 250


def test_flat_obs():
    env = gym.make('dmc:Walker-walk-v1', frame_skip=4)
    assert env.reset().shape == (24,)


def test_frame_skip():
    env = gym.make('dmc:Walker-walk-v1', from_pixels=True, frame_skip=8)
    assert env._max_episode_steps == 125
    assert env.reset().shape == (3, 84, 84)


def test_channel_first():
    env = gym.make('dmc:Walker-walk-v1', from_pixels=True, frame_skip=8, channels_first=False)
    assert env._max_episode_steps == 125
    assert env.reset().shape == (84, 84, 3)


def test_gray_scale():
    env = gym.make('dmc:Walker-walk-v1', from_pixels=True, frame_skip=8, channels_first=False,
                   gray_scale=True)
    assert env._max_episode_steps == 125
    assert env.reset().shape == (84, 84, 1)
