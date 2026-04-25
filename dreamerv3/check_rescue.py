import json, glob, os
path = 'logs/mario_run3_rescue/metrics.jsonl'
if not os.path.exists(path):
    print('rescue log not found yet')
    raise SystemExit
lines = open(path).read().strip().split('\n')
print(f'rescue log lines: {len(lines)}')

# latest step
last = json.loads(lines[-1])
print(f"latest step: {last['step']}")

# collect eval returns
evals = []
for line in lines:
    d = json.loads(line)
    if 'eval_return' in d:
        evals.append((d['step'], d['eval_return'], d.get('eval_length', '?')))
print('--- eval history (rescue) ---')
for s, r, l in evals:
    print(f'  step={s}: return={r:.2f}  length={l}  (~{r/0.05:.0f} px)')

# latest training metrics if available
for line in reversed(lines):
    d = json.loads(line)
    if 'actor_entropy' in d:
        print(f'\n--- latest training metrics (step={d["step"]}) ---')
        for k in ['actor_entropy','actor_grad_norm','imag_reward_mean','imag_reward_min','imag_reward_max','reward_loss','fps']:
            if k in d:
                print(f'  {k}: {d[k]:.4f}')
        break

# episode counts
n_eval = len(glob.glob('logs/mario_run3_rescue/eval_eps/*.npz'))
n_train = len(glob.glob('logs/mario_run3_rescue/train_eps/*.npz'))
print(f'\neval eps: {n_eval},  train eps: {n_train}')
