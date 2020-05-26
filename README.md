# `gym-dmc`, OpenAI Gym Plugin for DeepMind Control Suite

Link to other OpenAI Gym Plugins:

- `gym-sawyer`
- `gym-toy-nav`

Usage pattern:
```python
import gym

env = gym.make("gym_dmc:Pendulum-swingup-v1")
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
> acrobot task "swingup_sparse" becomes "gym_dmc:Acrobot-swingup_sparse-v1"
