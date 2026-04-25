import numpy as np, glob, os
import imageio.v2 as imageio
os.makedirs('viz', exist_ok=True)
LOGDIR = 'logs/mario_run4_plan2explore'
files = sorted(glob.glob(f'{LOGDIR}/eval_eps/*.npz'), key=os.path.getmtime)
rewards = [(f, float(np.load(f)['reward'].sum())) for f in files]
rewards.sort(key=lambda x: -x[1])
print('top 3 episodes by reward:')
for f, r in rewards[:3]:
    print(f'  r={r:.1f}  {os.path.basename(f)}')
best_f, best_r = rewards[0]
d = np.load(best_f)
frames = d['image']
big = np.repeat(np.repeat(frames, 4, axis=1), 4, axis=2)
# slow it down to 15 fps so you can see details
out = f'viz/BEST_r{best_r:.1f}.gif'
imageio.mimsave(out, big, duration=1/15, loop=0)
print('wrote', out, f'({len(frames)} frames at 15fps slow-mo)')
