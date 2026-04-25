import numpy as np, glob, os
import imageio.v2 as imageio
os.makedirs('viz', exist_ok=True)
LOGDIR = 'logs/mario_run4_plan2explore'
files = sorted(glob.glob(f'{LOGDIR}/eval_eps/*.npz'), key=os.path.getmtime)
for f in files[-2:]:
    d = np.load(f)
    frames = d['image']
    r = float(d['reward'].sum())
    idx = files.index(f)
    big = np.repeat(np.repeat(frames, 3, axis=1), 3, axis=2)
    out = f'viz/latest_eval{idx:02d}_r{r:.1f}.gif'
    imageio.mimsave(out, big, duration=1/30, loop=0)
    print(out, f'({len(frames)} frames, r={r:.1f})')
