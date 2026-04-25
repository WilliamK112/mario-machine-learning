import sys, pathlib
sys.path.insert(0, str(pathlib.Path('.').resolve()))
from envs.mario import Mario
import numpy as np

env = Mario(task='1-1', action_repeat=4, death_penalty=200.0, time_limit=12000)
obs = env.reset()
print('reset OK')

# Aggressive random actions to try to trigger a death quickly
np.random.seed(0)
total = 0.0
for i in range(3000):
    a = np.random.randint(env.action_space.n)
    obs, r, done, info = env.step(a)
    total += r
    if done:
        print(f'EPISODE ENDED at step {i+1}')
        print(f'  total_reward = {total:.2f}  (death_penalty should show -200 if died)')
        print(f'  max_x = {info.get("max_x")}  x_pos = {info.get("x_pos")}')
        print(f'  life_lost logic: total negative means death_penalty fired')
        break
else:
    print(f'ran 3000 steps without dying, total={total:.2f}, x={info.get("x_pos")}')
env.close()
