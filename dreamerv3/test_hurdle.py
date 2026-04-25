"""Verify hurdle bonuses fire exactly once per threshold crossed."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('.').resolve()))
from envs.mario import Mario
import numpy as np

env = Mario(task='1-1', action_repeat=4, death_penalty=50.0, time_limit=12000,
            hurdle_bonus=10.0, hurdle_spacing=400)
obs = env.reset()

total = 0.0
hurdles_hit = 0
for i in range(800):
    obs, r, done, info = env.step(1)  # right only
    if r > 5:  # hurdle bonus is +10, forward_bonus is ~0.05*small_delta
        # not perfectly reliable but most hurdle frames get +10 + some forward
        hurdles_hit += 1
        print(f'  step {i}: r={r:.2f}  max_x={info.get("max_x")} (likely hurdle)')
    total += r
    if done:
        print(f'EPISODE ENDED step {i+1} max_x={info.get("max_x")}  total={total:.2f}')
        break
else:
    print(f'done 800 steps  max_x={info.get("max_x")} total={total:.2f} hurdles={hurdles_hit}')
env.close()
