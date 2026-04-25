import numpy as np, glob, os, imageio.v2 as imageio
os.makedirs('viz', exist_ok=True)

LOGDIR = 'logs/mario_run4_plan2explore'
files = glob.glob(f'{LOGDIR}/eval_eps/*.npz') + glob.glob(f'{LOGDIR}/train_eps/*.npz')

def mx(f):
    d = np.load(f)
    return sum(x for x in d['reward'] if 0 < x < 50) / 0.05

scored = sorted([(f, mx(f)) for f in files], key=lambda t: -t[1])
print('Top 5 episodes by real max_x:')
for f, x in scored[:5]:
    src = 'eval' if 'eval_eps' in f else 'train'
    print(f'  [{src}] x={x:.0f}  {os.path.basename(f)}')

f, x = scored[0]
d = np.load(f)
frames = d['image']
big = np.repeat(np.repeat(frames, 4, axis=1), 4, axis=2)
src = 'eval' if 'eval_eps' in f else 'train'
out = f'viz/LONGEST_EVER_{src}_x{x:.0f}.gif'
imageio.mimsave(out, big, duration=1/15, loop=0)
print(f'\nwrote {out}')
