from gym.envs import register
from dm_control import suite

for domain_name, task_name in suite.ALL_TASKS:
    ID = f'{domain_name.capitalize()}-{task_name}-v1'
    register(id=ID,
             entry_point='gym_dmc.dmc_env:DMCEnv',
             kwargs=dict(domain_name=domain_name,
                         task_name=task_name,
                         channels_first=True,
                         width=84,
                         height=84,
                         frame_skip=4),
             )
    # does this interfere with frame_skip?
    # max_episode_steps=1000)
