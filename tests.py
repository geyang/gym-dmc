def test_gym_dmc():
    import gym
    env = gym.make("gym_dmc:Cartpole_swingup-v1")
    img = env.render('gray')
    assert img.shape == (84, 84, 1)