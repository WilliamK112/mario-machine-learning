import numpy as np, glob, os
import imageio.v2 as imageio
os.makedirs('viz', exist_ok=True)
files = sorted(glob.glob('logs/mario_run1/eval_eps/*.npz'))
# render every 2nd to save time, keep latest 8
todo = files[::2] + files[-4:]
todo = list(dict.fromkeys(todo))
print(f'rendering {len(todo)} / {len(files)} eval episodes')
for i, f in enumerate(todo):
    d = np.load(f)
    frames = d['image']
    r = float(d['reward'].sum())
    big = np.repeat(np.repeat(frames, 3, axis=1), 3, axis=2)
    idx = files.index(f)
    out = f'viz/eval{idx:02d}_r{r:.1f}.gif'
    imageio.mimsave(out, big, duration=1/30, loop=0)
    print(f'  {out}')
print('done')
