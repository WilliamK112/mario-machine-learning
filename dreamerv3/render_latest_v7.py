import numpy as np, glob, os, imageio.v2 as imageio
from datetime import datetime
os.makedirs('viz', exist_ok=True)

LOGDIR = 'logs/mario_run3_rescue'
files = glob.glob(f'{LOGDIR}/eval_eps/*.npz') + glob.glob(f'{LOGDIR}/train_eps/*.npz')

# Only v7 episodes (after the wipe ~03:11)
V7_CUTOFF = datetime(2026, 4, 24, 3, 11, 0).timestamp()
v7_files = [f for f in files if os.path.getmtime(f) > V7_CUTOFF]
print(f'v7 episodes: {len(v7_files)}')

# score by real max_x
def max_x(f):
    d = np.load(f)
    return sum(x for x in d['reward'] if 0 < x < 50) / 0.05

scored = [(f, max_x(f)) for f in v7_files]
scored.sort(key=lambda x: -x[1])
print('Top 3 by real max_x:')
for f, mx in scored[:3]:
    print(f'  max_x={mx:.0f}  {os.path.basename(f)}')

for i, (f, mx) in enumerate(scored[:3]):
    d = np.load(f)
    frames = d['image']
    big = np.repeat(np.repeat(frames, 4, axis=1), 4, axis=2)
    src = 'eval' if 'eval_eps' in f else 'train'
    out = f'viz/V7_TOP{i+1}_{src}_x{mx:.0f}.gif'
    # slow playback
    imageio.mimsave(out, big, duration=1/15, loop=0)
    print(f'wrote {out}')
