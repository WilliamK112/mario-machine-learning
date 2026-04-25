import glob, os, numpy as np
files = sorted(glob.glob('logs/mario_run1/eval_eps/*.npz'), key=os.path.getmtime)
for i, f in enumerate(files[-6:]):
    r = float(np.load(f)['reward'].sum())
    print(i, os.path.basename(f), f'r={r:.1f}')
