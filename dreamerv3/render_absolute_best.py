import numpy as np, glob, os
import imageio.v2 as imageio
os.makedirs('viz', exist_ok=True)
LOGDIR = 'logs/mario_run4_plan2explore'
all_files = glob.glob(f'{LOGDIR}/eval_eps/*.npz') + glob.glob(f'{LOGDIR}/train_eps/*.npz')
all_r = [(f, float(np.load(f)['reward'].sum())) for f in all_files]
all_r.sort(key=lambda x: -x[1])

# Render top 3 overall (could be train or eval)
for i, (f, r) in enumerate(all_r[:3]):
    src = 'train' if 'train_eps' in f else 'eval'
    d = np.load(f)
    frames = d['image']
    big = np.repeat(np.repeat(frames, 4, axis=1), 4, axis=2)
    out = f'viz/ABSOLUTE_TOP{i+1}_{src}_r{r:.1f}.gif'
    imageio.mimsave(out, big, duration=1/15, loop=0)
    print(f'{out}  (r={r:.1f}, ~{r/0.05:.0f} px, {src})')
