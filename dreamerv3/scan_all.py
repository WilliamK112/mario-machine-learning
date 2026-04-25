import numpy as np, glob, os
LOGDIR = 'logs/mario_run4_plan2explore'
eval_files = sorted(glob.glob(f'{LOGDIR}/eval_eps/*.npz'), key=os.path.getmtime)
train_files = sorted(glob.glob(f'{LOGDIR}/train_eps/*.npz'), key=os.path.getmtime)

def scan(files, tag):
    rs = [(f, float(np.load(f)['reward'].sum()), len(np.load(f)['reward'])) for f in files]
    rs.sort(key=lambda x: -x[1])
    print(f'\n=== TOP 8 {tag} (total {len(files)}) ===')
    for f, r, n in rs[:8]:
        print(f'  r={r:7.2f}  len={n:3d}  {os.path.basename(f)}')
    return rs

scan(eval_files, 'EVAL')
tr = scan(train_files, 'TRAIN')

# Overall max across both
all_r = [(f, float(np.load(f)['reward'].sum()), 'eval') for f in eval_files] + \
        [(f, float(np.load(f)['reward'].sum()), 'train') for f in train_files]
all_r.sort(key=lambda x: -x[1])
print('\n=== ABSOLUTE TOP 3 (any source) ===')
for f, r, src in all_r[:3]:
    print(f'  [{src}] r={r:.2f}  {os.path.basename(f)}')
