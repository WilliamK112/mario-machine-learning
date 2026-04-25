import sys, pathlib
sys.path.insert(0, str(pathlib.Path('.').resolve()))
from envs.mario import Mario
import numpy as np

env = Mario(
    task='1-1',
    action_repeat=4,
    flag_base_bonus=300.0,
    time_bonus_scale=1.0,
    score_scale=0.0,
    death_penalty=0.0,
)
obs = env.reset()
print('obs:', list(obs.keys()), 'image:', obs['image'].shape)
print('action_space n =', env.action_space.n)

total = 0.0
for i in range(60):
    a = np.random.randint(env.action_space.n)
    obs, r, done, info = env.step(a)
    total += r
    if done:
        print(f'ep ended at step {i+1}, total={total:.2f}, info={info}')
        break
else:
    print(f'60 steps ok, total={total:.2f}, info={info}')

# Verify backward-compat (mario_run1 defaults)
print('\n--- backward-compat: run1 defaults ---')
env2 = Mario(task='1-1', action_repeat=4)
env2.reset()
_, r, _, _ = env2.step(0)
print(f'default env step reward = {r:.3f} (should be small, not involve time bonus)')
env.close(); env2.close()
print('\nv3 env: OK')
