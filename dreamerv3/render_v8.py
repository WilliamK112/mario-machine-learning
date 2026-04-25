import numpy as np, glob, os, imageio.v2 as imageio
from datetime import datetime
os.makedirs('viz', exist_ok=True)

LOGDIR = 'logs/mario_run3_rescue'
V8_CUTOFF = datetime(2026, 4, 24, 8, 30, 0).timestamp()
files = [f for f in glob.glob(f'{LOGDIR}/eval_eps/*.npz') + glob.glob(f'{LOGDIR}/train_eps/*.npz')
         if os.path.getmtime(f) > V8_CUTOFF]

def mx(f):
    d = np.load(f)
    return sum(x for x in d['reward'] if 0 < x < 50) / 0.05

scored = sorted([(f, mx(f)) for f in files], key=lambda t: -t[1])[:3]
print('v8 top 3 by real max_x:')
for f, x in scored:
    print(f'  x={x:.0f}  {os.path.basename(f)}')

for i, (f, x) in enumerate(scored[:2]):
    d = np.load(f)
    frames = d['image']
    big = np.repeat(np.repeat(frames, 4, axis=1), 4, axis=2)
    src = 'eval' if 'eval_eps' in f else 'train'
    out = f'viz/V8_TOP{i+1}_{src}_x{x:.0f}.gif'
    imageio.mimsave(out, big, duration=1/15, loop=0)
    print(f'wrote {out}')
