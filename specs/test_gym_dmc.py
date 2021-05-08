import gym


def test_make():
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
