from dm_control import suite
from gym.envs import register

for domain_name, task_name in suite.ALL_TASKS:
    ID = f'{domain_name.capitalize()}-{task_name}-v1'
    register(id=ID,
             entry_point='gym_dmc:make_env',
             kwargs=dict(
                 domain_name=domain_name,
                 task_name=task_name,
                 channels_first=True,
                 width=84,
                 height=84,
                 frame_skip=4),
             )
