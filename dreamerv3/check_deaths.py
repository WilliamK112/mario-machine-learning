import json
lines = open('logs/mario_run3_rescue/metrics.jsonl').read().strip().split('\n')
print('train_return history (looking for negative = death signal):')
for line in lines:
    d = json.loads(line)
    if 'train_return' in d and 'train_length' in d:
        s = d.get('step', '?')
        r = d['train_return']
        l = d['train_length']
        marker = '  <-- DEATH' if r < 50 else ('  (timed out normally)' if l >= 280 else '  (short)')
        print(f'  step={s}  r={r:7.2f}  len={l:6.1f}{marker}')
