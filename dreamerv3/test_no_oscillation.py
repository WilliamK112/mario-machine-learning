"""Verify new reward only pays out on NEW max x, not on oscillation."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('.').resolve()))
from envs.mario import Mario
import numpy as np

env = Mario(task='1-1', action_repeat=4, death_penalty=200.0, time_limit=12000)
obs = env.reset()

# Simulate: walk forward for 30 steps (action=1 = right)
print('=== Phase 1: walk forward for 30 steps ===')
r_total = 0
for _ in range(30):
    obs, r, done, info = env.step(1)
    r_total += r
    if done: break
print(f'  forward phase reward: {r_total:.2f}  max_x={info.get("max_x")}')

# Phase 2: walk backward (action=6 = left) for 20 steps
print('=== Phase 2: walk backward 20 steps (should get 0 reward) ===')
r_total = 0
for _ in range(20):
    obs, r, done, info = env.step(6)
    r_total += r
    if done: break
print(f'  backward phase reward: {r_total:.2f}  max_x={info.get("max_x")} x={info.get("x_pos")}')

# Phase 3: walk forward again, still behind old max (should be 0)
print('=== Phase 3: walk forward 15 steps BUT not past old max (should be 0) ===')
r_total = 0
for _ in range(15):
    obs, r, done, info = env.step(1)
    r_total += r
    if done: break
print(f'  re-forward phase reward: {r_total:.2f}  max_x={info.get("max_x")} x={info.get("x_pos")}')

# Phase 4: walk forward past old max (should give reward again)
print('=== Phase 4: walk forward 30 more to surpass old max ===')
r_total = 0
for _ in range(30):
    obs, r, done, info = env.step(1)
    r_total += r
    if done: break
print(f'  new-max phase reward: {r_total:.2f}  max_x={info.get("max_x")} x={info.get("x_pos")}')

env.close()
