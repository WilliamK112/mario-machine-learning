import json, glob
lines = open('logs/mario_run1/metrics.jsonl').read().strip().split('\n')
print(f'total log lines: {len(lines)}')
last = json.loads(lines[-1])
print('--- latest log entry ---')
for k in ['step','env_step','fps','train_loss','actor_loss','critic_loss','eval_return','eval_length','train_return','train_length']:
    if k in last:
        v = last[k]
        print(f'  {k}: {v:.3f}' if isinstance(v,float) else f'  {k}: {v}')
print('--- all keys in latest ---')
print(sorted(last.keys()))
print('--- eval_return history ---')
evals = []
for line in lines:
    d = json.loads(line)
    if 'eval_return' in d:
        evals.append((d.get('step','?'), d['eval_return'], d.get('eval_length','?')))
for s, r, l in evals:
    print(f'  step={s}: return={r:.2f}  length={l}  (~{r/0.05:.0f} px)')
print(f'num eval eps: {len(glob.glob("logs/mario_run1/eval_eps/*.npz"))}')
print(f'num train eps: {len(glob.glob("logs/mario_run1/train_eps/*.npz"))}')
