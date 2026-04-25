import json
lines = open('logs/mario_run1/metrics.jsonl').read().strip().split('\n')
# find the most recent line that has actor_entropy
for line in reversed(lines):
    d = json.loads(line)
    if 'actor_entropy' in d:
        break
print(f"=== step={d['step']} ===")
keys_interest = ['actor_entropy','actor_grad_norm','imag_reward_mean','imag_reward_std','reward_loss','kl','image_loss','actor_loss','value_loss','fps','post_ent','prior_ent']
for k in keys_interest:
    if k in d:
        print(f'  {k}: {d[k]:.4f}')
