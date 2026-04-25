"""Deep-dive on rescue run episodes: x-trajectory, death type, recent-vs-old comparison."""
import numpy as np, glob, os, json

LOGDIR = 'logs/mario_run3_rescue'
eval_files = sorted(glob.glob(f'{LOGDIR}/eval_eps/*.npz'), key=os.path.getmtime)
train_files = sorted(glob.glob(f'{LOGDIR}/train_eps/*.npz'), key=os.path.getmtime)

def analyze(files, label):
    print(f'\n=== {label} (n={len(files)}) ===')
    for i, f in enumerate(files[-8:]):
        d = np.load(f)
        rewards = d['reward']
        n = len(rewards)
        total = float(rewards.sum())
        # reconstruct x from per-step forward_bonus = 0.05 * delta_x
        # death_pen is -200 applied on last step if died
        # So cumulative forward_bonus[:-1] / 0.05 = cumulative delta_x up to step -1
        if rewards[-1] < -100:
            death = True
            forward_total = float(rewards[:-1].sum()) + float(rewards[-1] + 200)
        else:
            death = False
            forward_total = total
        est_x_delta = forward_total / 0.05
        is_flag = total > 400
        tag = 'FLAG' if is_flag else ('DIED' if death else 'TIMEOUT')
        print(f'  ep{i:2d}  len={n:4d}  total={total:7.1f}  ~x_delta={est_x_delta:6.0f}  [{tag}]')

analyze(eval_files, 'EVAL episodes')
analyze(train_files, 'TRAIN episodes')

# Episode-wide max_x estimation via cumulative forward delta (assumes no backward movement)
print('\n=== Max x reached across all rescue data ===')
all_files = eval_files + train_files
best = 0
best_src = ''
for f in all_files:
    d = np.load(f)
    rewards = d['reward']
    # cumulative forward delta approximation
    cum = 0
    peak = 0
    for r in rewards:
        if r > 0:  # positive = forward_bonus
            cum += r
            peak = max(peak, cum)
        elif r < -100:  # death, ignore
            pass
        else:
            # backward/0 — cum doesn't reset since delta_x reward only gives positive
            pass
    # Actually this underestimates because forward_bonus is only delta_x > 0, so cum represents SUM of positive deltas, which is upper bound on final x
    est = cum / 0.05
    if est > best:
        best = est
        best_src = f
print(f'  best estimated forward distance: {best:.0f} px  (file: {os.path.basename(best_src)})')
