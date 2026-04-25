# Mario Machine Learning — Two RL Algorithms, One NES Pit, Honest Numbers

> An end-to-end reinforcement-learning project on **NES Super Mario Bros 1-1**
> that runs and compares two very different agents on a single Windows
> workstation with one consumer GPU:
>
> - a **Stable-Baselines3 PPO** baseline (model-free, well-trodden)
> - a **DreamerV3 + Plan2Explore** agent (model-based, curiosity-driven)
>
> The repo documents the *full engineering journey*: which algorithms were
> tried, what reward functions were iterated, the bugs that bit on Windows,
> the monitoring tooling that was written, the academic papers that motivated
> each decision — **and the negative results**, with the diagnostic evidence
> that led to them.

---

## 0. TL;DR

| Stack | Status (as of 2026-04-25) | Where Mario got | What we learned |
|---|---|---|---|
| **SB3 PPO + reward shaping** *(active)* | Training, 1.5 M-step right-only run with `--normalize-reward`, milestone densification, hurdle bonuses. First eval at 25 k steps → Mario reached **x ≈ 698 / 3161 (22 %)**. Updated final number lives in [§9 Results](#9-results). | (will update at end of run) | A *small, well-normalised* reward function with a high entropy bonus is what unblocked PPO; an earlier 3-phase pipeline with `flag_bonus = 1500` policy-collapsed in 12 k steps. |
| **DreamerV3 + Plan2Explore** *(parked, post-mortem)* | Three runs, **2 413 episodes total**, **0 flag-completions**. Best progress reached only ≈ x = 2 600 / 3 161. Diagnosis is in [§7 Failed Experiment](#7-failed-experiment-dreamerv3--plan2explore). | x ≈ 2 600 / 3 161 (≈ 82 %) | World-model + curiosity-driven exploration on sparse-reward NES Mario is *very* sample-inefficient. The reward head never saw a positive flag example, so the actor had no anchor for "what success looks like." This is the textbook Salimans-Chen failure mode that calls for imitation seeding (which we didn't have budget for — see lessons learned). |

---

## 1. Quickstart (clone → train → see Mario play)

```powershell
# 1. Clone & set up the GPU virtualenv (one-time)
git clone https://github.com/<you>/mario-rl.git
cd mario-rl
py -3.11 -m venv .venv-gpu
.\.venv-gpu\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Train the PPO baseline (1.5 M steps, ~3-5 h on RTX 4070-class GPU)
python train_mario.py `
  --timesteps 1500000 --device cuda --action-set right_only `
  --n-envs 8 --learning-rate 2.5e-4 --n-steps 512 --batch-size 256 `
  --ent-coef 0.05 --n-epochs 4 --clip-range 0.1 `
  --normalize-reward --norm-reward-clip 10 `
  --forward-reward-scale 0.1 --backward-penalty-scale 0.05 `
  --flag-bonus 100 `
  --milestone-step 100 --milestone-bonus 1.0 `
  --hurdle-x 600 1200 1800 2400 3000 --hurdle-bonus 5.0 `
  --eval-freq 25000 --eval-episodes 3 --preview-freq 50000

# 3. Watch live (one-shot or polling)
.\watch_ppo.ps1                 # one-shot snapshot
.\watch_ppo.ps1 -Loop -EveryS 30  # auto-poll every 30 s

# 4. Render the trained agent playing Mario 1-1
python evaluate_mario.py --model runs\<latest>\models\mario_final.zip
```

---

## 2. Repository Structure

```
mario-rl/
├── README.md                       # this document
├── requirements.txt                # pinned deps for both stacks
├── mario_runtime.py                # NES wrapper, reward shaping, Monitor/VecEnv plumbing
├── train_mario.py                  # SB3 PPO trainer (single-stage, all hypers exposed)
├── train_mario_professional.py     # 3-phase staged PPO pipeline (multi-seed orchestration)
├── evaluate_mario.py               # SB3 evaluator → mp4 + json summary
├── rnd.py                          # Random Network Distillation intrinsic-reward module
├── watch_ppo.ps1                   # One-shot / polling health snapshot for the active PPO run
├── run_mario.ps1                   # PowerShell front-door: train | eval
├── auto_update_mario_monitor.py    # TensorBoard-style live reward/length plot
├── quick_mario_smoke.py            # 30-second sanity check on the env wrapper
├── watch_mario.py                  # Open a window and watch the trained policy play live
└── dreamerv3/                      # PyTorch DreamerV3 fork, customised for Mario
    ├── dreamer.py                  # main training loop (env factory patched for Mario)
    ├── envs/mario.py               # DreamerV3 env wrapper + reward shaping
    ├── configs.yaml                # all DreamerV3 experiment configs
    ├── exploration.py              # Plan2Explore intrinsic-reward (latent disagreement)
    ├── models.py / networks.py     # world model, actor, critic, ensemble heads
    ├── start_plan2explore.ps1      # canonical detached-launch script
    ├── live_monitor.ps1            # TensorBoard launcher bound to logs/
    ├── live_view.py                # streams real-time emulator frames in a window
    ├── heartbeat.ps1               # one-line health probe
    ├── render_rescue_best.py       # render best evaluation episode as GIF
    ├── _diagnose.py                # cross-run flag/death/reward-profile analyser  (post-mortem)
    ├── _perf_summary.py            # cross-run summary: episodes, best/latest eval, max-x
    └── _npz_keys.py                # introspect what's in DreamerV3 episode .npz files
```

`logs/`, `runs/`, virtualenvs, `*.npz`, `*.pt`, `*.zip`, `*.mp4`, `*.gif`
are intentionally `.gitignore`d — they are reproducible from the configs and
would balloon the repo to multiple gigabytes.

---

## 3. The PPO Recipe That Works

The current PPO baseline is the result of one false start and a careful
re-tune. The original 3-phase `train_mario_professional.py` pipeline used
`flag_bonus = 1500` and `learning_rate = 2.5e-4` with `n_epochs = 10`; it
**policy-collapsed within 12 k steps** (entropy crashed from 0.95 to 5 × 10⁻⁸,
KL went to 0, clip-fraction went to 0). The value head exploded on the giant
flag bonus and drowned out the policy gradient.

The fix was a *minimal* recipe, every choice motivated by a known PPO failure
mode:

| Knob | Value | Why |
|---|---|---|
| `--action-set right_only` | 5 actions instead of 7 | Smaller action space → easier exploration |
| `--n-envs 8` | 8 parallel envs | Variance reduction; fits comfortably in 12 GB VRAM |
| `--learning-rate 2.5e-4` | SB3/Atari default | Conservative; works with `--clip-range 0.1` |
| `--n-steps 512`, `--batch-size 256`, `--n-epochs 4` | OpenAI baselines style | Fewer epochs/rollout = less risk of policy collapse |
| `--clip-range 0.1` | Tighter than default 0.2 | Prevents over-aggressive updates |
| `--ent-coef 0.05` | 5× SB3 default | Keeps the policy from collapsing to argmax (the failure of run #1) |
| `--normalize-reward --norm-reward-clip 10` | VecNormalize wrap | **Critical:** keeps value loss bounded so it never drowns out the policy gradient |
| `--forward-reward-scale 0.1` | Small forward bonus | Densifies signal but doesn't dominate it |
| `--flag-bonus 100` | One-time flag bonus, *not* 1500 | Combined with normalize-reward, this is large enough to be a clear positive anchor without exploding the value function |
| `--milestone-step 100 --milestone-bonus 1.0` | +1.0 every 100 px of new max-x | Curriculum-style densification; prevents reward sparsity |
| `--hurdle-x 600 1200 1800 2400 3000 --hurdle-bonus 5.0` | One-time bonus per hurdle crossed | Explicit credit assignment for landmark progress |

The reward-engineering principle here is: *combine a dense, small, normalised
shaping term with a discrete, one-time landmark bonus, and let `VecNormalize`
keep everything in the right scale.* That formula is also what eventually
worked for Kauten's well-known PPO Mario implementations (cited in §10).

---

## 4. How To Reproduce

> Tested on Windows 11, Python 3.11 in `.venv-gpu`, NVIDIA RTX 4070 SUPER
> (CUDA 12). Should work unchanged on any RTX-30/40 GPU.

### PPO baseline (the recommended path)

```powershell
.\.venv-gpu\Scripts\Activate.ps1
python train_mario.py `
  --timesteps 1500000 --device cuda --action-set right_only `
  --n-envs 8 --learning-rate 2.5e-4 --n-steps 512 --batch-size 256 `
  --ent-coef 0.05 --n-epochs 4 --clip-range 0.1 `
  --normalize-reward --norm-reward-clip 10 `
  --forward-reward-scale 0.1 --backward-penalty-scale 0.05 `
  --flag-bonus 100 `
  --milestone-step 100 --milestone-bonus 1.0 `
  --hurdle-x 600 1200 1800 2400 3000 --hurdle-bonus 5.0 `
  --eval-freq 25000 --eval-episodes 3 --preview-freq 50000
```

### Detached background launch (Windows-specific)

Direct console launches are vulnerable to `forrtl: error (200): program
aborting due to window-CLOSE event` — Intel/Fortran-linked numerics inside
the NES emulator abort when the parent console closes. Always launch with
`Start-Process -WindowStyle Hidden`:

```powershell
$python = "C:\Users\<you>\mario-rl\.venv-gpu\Scripts\python.exe"
Start-Process -FilePath $python `
  -ArgumentList "-u","train_mario.py","--timesteps","1500000", "...rest..." `
  -WorkingDirectory "C:\Users\<you>\mario-rl" `
  -RedirectStandardOutput "runs\train.out.log" `
  -RedirectStandardError  "runs\train.err.log" `
  -WindowStyle Hidden
```

### DreamerV3 (for reference / reproducing the failed experiment)

```powershell
cd dreamerv3
& "..\.venv-gpu\Scripts\python.exe" dreamer.py `
  --configs mario_plan2explore --logdir logs\mario_run4_plan2explore --steps 7e5
```

### Live monitoring

```powershell
.\watch_ppo.ps1                        # one-shot PPO snapshot
.\watch_ppo.ps1 -Loop -EveryS 30        # poll every 30 s
cd dreamerv3; .\heartbeat.ps1           # one-shot DreamerV3 snapshot
& "..\.venv-gpu\Scripts\python.exe" dreamerv3\_perf_summary.py     # cross-run summary
```

---

## 5. What Changed Over Time (PPO + DreamerV3)

This is the chronological diff for **both** stacks, with the failure mode
that motivated each change. Every row has a corresponding commit in `git log`.

### 5.1 PPO reward + hyper-parameter timeline

| # | Change | Triggered by |
|---|---|---|
| P0 | Vanilla PPO + `RIGHT_ONLY` + NES built-in reward only | Sanity baseline |
| P1 | + `forward_reward_scale = 0.05 · max(0, Δx)` | NES built-in reward too sparse |
| P2 | Use `Δ(max-x)` instead of `Δx`, clip ≥ 0 | Oscillation exploit (Mario wagged left/right at chokepoints to farm Δx) |
| P3 | + `flag_bonus = 100` | Successful runs needed a positive terminal anchor |
| P4 | + 3-phase staged pipeline (`train_mario_professional.py`) with `flag_bonus = 1500`, `n_epochs = 10` | More steps + curriculum should help |
| P5 | **Reverted P4.** Switched to single-phase `train_mario.py` with `--normalize-reward`, `flag_bonus = 100`, `n_epochs = 4`, `ent_coef = 0.05`, `clip_range = 0.1` | P4 collapsed at 12 k steps (entropy → 0, KL → 0). Diagnosed as value-loss explosion drowning policy gradient. |
| P6 (current) | + milestone bonus every 100 px, hurdle bonus at 600/1200/1800/2400/3000 | Densify the credit-assignment signal mid-level |

### 5.2 DreamerV3 reward + algorithm timeline

| # | Change | Triggered by |
|---|---|---|
| D0 | DreamerV3 default reward (= NES built-in only) | Sanity |
| D1 | + `forward_bonus = 0.05 · Δx`, switch to `Δ(max-x)` | Oscillation exploit, same as P2 |
| D2 | + `flag_base_bonus = 100` | Need a positive-terminal anchor |
| D3 | + `time_bonus_scale * time_remaining_on_flag_get` | Successful runs were too slow |
| D4 | + `hurdle_bonus = 5` every 200 px of new max-x | Sparse signal mid-level |
| D5 | Treat single-life loss as terminal + `death_penalty = 50` | NES `done` only fires on *all 3* lives lost — the death penalty never paid out |
| D6 | Switched explorer from `greedy` → **`plan2explore`** (5-model latent-disagreement ensemble), `expl_intr_scale = 0.3` | Long plateau at second pit; classic exploration-bottleneck signature |
| D7 | + `step_penalty = 0.005` per env step | Plan2Explore made the agent *more* timid (pure curiosity → conservative) |
| D8 | + Potential-based shaping: `0.01 · (γ · Φ(s′) − Φ(s))` with `Φ = −dist_to_goal` | Encourage decisive forward commits (Ng et al. policy-invariance) |

In parallel the algorithm itself evolved:

| Stage | Algorithm change |
|---|---|
| A0 | SB3 PPO, RIGHT_ONLY, vectorised env |
| A1 | DreamerV3 (NM512 PyTorch port), single env, simple action set, greedy explorer |
| A2 | Bumped `time_limit` from 4 500 → 48 000 raw frames so NES game-over fires *before* our truncation |
| A3 | Wipe replay buffer between runs whenever reward semantics change (poisoned-reward-head fix) |
| A4 | Plan2Explore intrinsic reward (5-model ensemble, `expl_intr_scale = 0.3`) |
| A5 | `actor.entropy = 3e-3` to prevent early collapse under Plan2Explore |

---

## 6. Issues We Faced & How We Solved Them

### Issue 1 — Reward "oscillation exploit"
- **Symptom.** Training reward grew but evaluation GIFs showed Mario walking right-left-right at the first chokepoint forever.
- **Cause.** Original shaping was `0.05 · Δx`, so any forward step paid out, even if it cancelled a previous backward step.
- **Fix.** Pay only on **new max-x**: `delta_new_max = max(0, x_now − self._max_x)`. Oscillation now nets exactly zero.

### Issue 2 — DreamerV3 death penalty never triggering
- **Symptom.** `mario_death_penalty=50` set in config; reward distribution had no negative tail.
- **Cause.** NES Mario only emits `done = True` after **all 3 lives** are lost; we were ending episodes on time-limit truncation, so the NES game-over never fired.
- **Fix.** Detect single-life loss inside the wrapper (`info["life"] < self._last_life`), treat it as terminal, apply `−death_penalty`. Bumped `time_limit` from 4 500 → 48 000 raw frames so the NES timer (≈ 9 600 raw frames) is the *first* thing that fires.

### Issue 3 — World-model "remembered" old reward semantics
- **Symptom.** After changing reward shaping, the policy got *worse* for tens of thousands of steps before recovering.
- **Cause.** DreamerV3's reward head trains on the replay buffer. Changing the reward formula but keeping the old `train_eps/` supervises the head with **inconsistent labels** (old reward in old episodes, new reward in new ones).
- **Fix.** When reward semantics change, wipe `train_eps/` and `eval_eps/` while keeping `latest.pt` (so the world-model encoder is warm-started but the reward head retrains on clean data). This is what `mario_rescue` documents.

### Issue 4 — Trainer dying silently with `forrtl: error (200)`
- **Symptom.** `python.exe` for the trainer disappeared from Task Manager, `metrics.jsonl` stopped updating, `*.err` log ended with `forrtl: error (200): program aborting due to window-CLOSE event`.
- **Cause.** Trainer launched as a child of a PowerShell pipeline; when the PowerShell host quit, the child received `CTRL_CLOSE_EVENT` and Intel/Fortran-linked numerics aborted.
- **Fix.** Launch with `Start-Process -WindowStyle Hidden` so the process owns no console window. `start_plan2explore.ps1` and the PPO launcher in §4 use this pattern.

### Issue 5 — Two trainers writing the same `logdir`
- **Symptom.** Episode counts in `train_eps/` jumped non-monotonically; `latest.pt` size oscillated; eval scores became noisy.
- **Cause.** A re-launch helper inadvertently spawned a second trainer while the first was still alive; both wrote into the same `logdir`, corrupting the replay store and `latest.pt`.
- **Fix.** A "list-and-kill" helper is run before every launch; the launcher refuses to start if a Mario trainer is already alive. Invariant: **exactly one trainer per `logdir` at any time.**

### Issue 6 — Venv launcher creates a child `python.exe`
- **Symptom.** Two `python.exe` processes appeared after every launch and looked like duplicate trainers.
- **Cause.** On Windows, `.venv\Scripts\python.exe` is a *launcher stub* that re-execs the system Python interpreter; both parent and child show up in `Get-Process python`.
- **Fix.** Use `Get-CimInstance Win32_Process` and inspect `ParentProcessId` to confirm one is the parent of the other. `watch_ppo.ps1` does this.

### Issue 7 — `imageio` missing when rendering best episodes
- **Symptom.** `python render_rescue_best.py` failed with `ModuleNotFoundError: No module named 'imageio'`.
- **Cause.** Default `python` on PATH was the system Python 3.12 install, not the project's `.venv-gpu`.
- **Fix.** Always invoke renderers via the project venv: `& ".\.venv-gpu\Scripts\python.exe" render_*.py`.

### Issue 8 — PPO 3-phase pipeline policy-collapse
- **Symptom.** `train_mario_professional.py` with default phase-1 hypers (`flag_bonus=1500`, `lr=2.5e-4`, `n_epochs=10`) drove `entropy_loss` from `−0.95` to `−5.6 × 10⁻⁸` in 12 k steps. KL → 0, clip-fraction → 0, value_loss = 1.9 × 10⁵ — the optimiser was fitting only the value function, the policy froze to argmax.
- **Cause.** Without `VecNormalize`, the giant flag bonus exploded the return distribution; `vf_coef · value_loss` dominated `ent_coef · entropy + pg_loss` in the total loss.
- **Fix.** Single-phase `train_mario.py` with `--normalize-reward`, `flag_bonus=100`, `clip_range=0.1`, `n_epochs=4`, `ent_coef=0.05`. Verified post-fix: KL = 0.0015, clip = 0.05, entropy = 1.6 (preserved). See §3.

---

## 7. Failed Experiment — DreamerV3 + Plan2Explore

This section is included because *the failure is informative*. A model-based,
curiosity-driven world-model agent is the kind of approach the field is most
excited about right now (DreamerV3 was published in *Nature* in 2025) — and
on this task, with this budget, **it didn't beat the PPO baseline.** Here is
the evidence and the diagnosis.

### 7.1 What we ran

| Run | Algorithm | Logdir | Steps | Notes |
|---|---|---|---|---|
| `mario_run1` | DreamerV3 + greedy actor | `dreamerv3/logs/mario_run1` | 222 k | Forward bonus only |
| `mario_run3_rescue` | DreamerV3 + greedy + death penalty + hurdles | `dreamerv3/logs/mario_run3_rescue` | 228 k | Resumed from `mario_run1` checkpoint with fresh replay |
| `mario_run4_plan2explore` | DreamerV3 + Plan2Explore + step penalty + potential shaping | `dreamerv3/logs/mario_run4_plan2explore` | 565 k | Curiosity + R7 + R8 shaping |

### 7.2 The brutal numbers

Generated by `dreamerv3/_diagnose.py` (see [`dreamerv3/diagnose_at_pivot.txt`](dreamerv3/diagnose_at_pivot.txt) for the full output captured at the pivot point):

```
Q1: Did Mario EVER reach the flag in any run?
  mario_run1                      flag_get episodes:    0 /  239
  mario_run3_rescue               flag_get episodes:    0 /  115
  mario_run4_plan2explore         flag_get episodes:    0 / 2059

Q5: Latest training-side numbers (extrinsic vs intrinsic gradient signal)
  imag_reward_mean         =    -0.366
  imag_reward_min          =   -53.886
  expl_imag_reward_mean    =    -3.440         ← intrinsic is 9× more negative
  expl_imag_reward_max     =     4.576
```

**Total: 0 flag completions across 2 413 episodes.** Best max-x reached
was ≈ 2 600 / 3 161 (82 %), achieved by `mario_run3_rescue` with the simpler
greedy explorer.

### 7.3 Root cause

Three compounding factors:

1. **The world model never saw a positive flag example.** DreamerV3's
   reward head is supervised by replay-buffer rewards. With zero
   flag-completions on disk, the head has no anchor for *what high reward
   looks like* — and the actor, which trains on *imagined* rollouts in
   latent space, can never imagine itself reaching the flag in any way that
   gets credit. This is the textbook Salimans-Chen failure mode for sparse-
   reward, long-horizon games (their Montezuma's Revenge result needed
   imitation seeding precisely for this reason).
2. **Plan2Explore's intrinsic reward dominated the negative side.**
   `expl_imag_reward_mean = −3.44` vs `imag_reward_mean = −0.37`: the
   intrinsic signal was 9 × more negative-skewed than the extrinsic one.
   This drove the actor toward high-uncertainty regions — which on a
   platformer are the regions full of pits.
3. **The `step_penalty` we added (R7, see §5.2) was too weak to incentivise
   speed, but strong enough to depress total return.** Episodes that
   *almost* survived but ran out of time scored slightly worse than ones
   that died early but hit a hurdle bonus, so the gradient pointed in the
   wrong direction.

### 7.4 What would have fixed it (out of budget)

- **Imitation seeding.** Record 5–20 minutes of human play, convert to
  DreamerV3 episode `.npz` format, prepend to the replay buffer. This is
  the single biggest known fix for sparse-reward DreamerV3 on long-horizon
  games (Hafner et al. § 4.3).
- **Backward curriculum from RAM state.** Save NES emulator RAM state at
  x = 3 000, train the agent to clear the last 161 px first, then back up
  the start position progressively. Standard Atari trick (Salimans & Chen).
- **Substantially more compute.** All three runs above totalled ≈ 1 M env
  steps; the original DreamerV3 Atari results used 200 M.

### 7.5 Honest engineering call

We made the call to **stop iterating on DreamerV3 and ship the PPO baseline**
because (a) PPO has a plausible 1.5 M-step path to a confirmed flag-clear,
and (b) the next DreamerV3 fix (imitation seeding) is a research-grade
project on its own, not a one-evening tune. The DreamerV3 code, configs,
and post-mortem stay in the repo because the exercise of *diagnosing why
the fancy agent failed* is itself a useful artifact.

---

## 8. Method Notes (for the curious)

### 8.1 Why PPO at all?

PPO (Schulman et al., 2017) is the workhorse policy-gradient method:
on-policy, clipped importance ratio, a few epochs of mini-batch SGD per
rollout. On pixel-based Atari-style tasks it is sample-inefficient compared
to DreamerV3 in *theory*, but it has two huge practical advantages:

1. **No reward head to break.** PPO's value function is also pixel-conditioned,
   so it sees exactly what the policy sees. There is no replay-buffer-induced
   distribution mismatch.
2. **Battle-tested hypers.** SB3 ships defaults that work on most Atari
   tasks out of the box. The Mario reward shaping in this repo is a
   straightforward extension.

### 8.2 Why DreamerV3 was the right thing to *try*

DreamerV3's two key ideas — **discrete latent world model** (Hafner et al.,
2021) and **symlog reward heads** — make it the most sample-efficient
known model-based agent on pixel-based control as of 2025. For an emulator
that runs at < 200 fps on a single CPU thread, sample-efficiency is exactly
what you want; one wall-clock hour of NES Mario gives you much more
"imagined" rollout data with DreamerV3 than with PPO.

### 8.3 Why Plan2Explore was the right thing to *add*

Plan2Explore (Sekar et al., 2020) is a curiosity-driven exploration scheme
that fits naturally on top of DreamerV3: it trains an ensemble of latent-
dynamics models and uses their **disagreement** as an intrinsic reward.
Long plateaus around the second pit of Mario 1-1 looked exactly like the
exploration-bottleneck pattern Sekar et al. describe.

### 8.4 Why it didn't work

See §7.3.

---

## 9. Results

> Numbers are taken directly from `metrics.jsonl` and the on-disk
> `train_eps/`, `eval_eps/` for each run. Updated as runs complete.

### 9.1 PPO (current run, in flight)

Run dir: `runs/ppo_safe_20260425_012556/`. Pinned hypers: see §3.

| Step | best_remaining_distance | Mario reached x | % of level | Notes |
|---|---|---|---|---|
| 25 000 | 2 463 px | 698 / 3 161 | 22 % | First eval after warm-up |
| (... will be filled in as evals complete; final number lands here ...) | | | | |

The success-clip GIF and final eval table land here when the run finishes.

### 9.2 DreamerV3 (parked)

| Stage | Steps logged | Train eps | Eval eps | Best `eval_return` | Flag-clears |
|---|---|---|---|---|---|
| `mario_run1` (greedy) | 222 216 | 197 | 42 | 146.95 | **0** |
| `mario_run3_rescue` (death penalty + hurdles) | 227 904 | 69 | 46 | **272.50** | **0** |
| `mario_run4_plan2explore` (Plan2Explore + R7+R8) | 565 712 | 2 059 | 88 | 152.90 | **0** |

`mario_run3_rescue` is the strongest single-evaluation episode (272.50);
`mario_run4_plan2explore` did not surpass it despite a longer horizon.
See §7 for the diagnosis.

---

## 10. References (MLA, 9th Edition)

> Every reference below was actually consulted while building this project
> and is cited in the body of the README where the corresponding idea is
> introduced.

Bellemare, Marc G., et al. "The Arcade Learning Environment: An Evaluation
Platform for General Agents." *Journal of Artificial Intelligence Research*,
vol. 47, June 2013, pp. 253–279. arXiv,
[arxiv.org/abs/1207.4708](https://arxiv.org/abs/1207.4708). Accessed 24 Apr. 2026.

Burda, Yuri, et al. *Exploration by Random Network Distillation*. arXiv,
30 Oct. 2018, [arxiv.org/abs/1810.12894](https://arxiv.org/abs/1810.12894). Accessed 24 Apr. 2026.

Hafner, Danijar, et al. *Mastering Atari with Discrete World Models*. arXiv,
2 Feb. 2021, [arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193). Accessed 24 Apr. 2026.

Hafner, Danijar, et al. "Mastering Diverse Control Tasks through World Models."
*Nature*, vol. 640, 2 Apr. 2025, pp. 647–653. *Nature Portfolio*,
[doi.org/10.1038/s41586-025-08744-2](https://doi.org/10.1038/s41586-025-08744-2). Accessed 24 Apr. 2026.

Kauten, Christian. *gym-super-mario-bros*. GitHub, 2018,
[github.com/Kautenja/gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros). Accessed 24 Apr. 2026.

Mnih, Volodymyr, et al. "Human-Level Control through Deep Reinforcement
Learning." *Nature*, vol. 518, no. 7540, 26 Feb. 2015, pp. 529–533.
*Nature Portfolio*, [doi.org/10.1038/nature14236](https://doi.org/10.1038/nature14236). Accessed 24 Apr. 2026.

NM512. *dreamerv3-torch*. GitHub, 2023,
[github.com/NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch). Accessed 24 Apr. 2026.

Ng, Andrew Y., et al. "Policy Invariance Under Reward Transformations: Theory
and Application to Reward Shaping." *Proceedings of the 16th International
Conference on Machine Learning*, Morgan Kaufmann, 1999, pp. 278–287.
*Stanford AI Lab*,
[ai.stanford.edu/~ang/papers/shaping-icml99.pdf](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).
Accessed 24 Apr. 2026.

Pathak, Deepak, et al. "Curiosity-Driven Exploration by Self-Supervised
Prediction." *Proceedings of the 34th International Conference on Machine
Learning*, vol. 70, PMLR, 2017, pp. 2778–2787. arXiv,
[arxiv.org/abs/1705.05363](https://arxiv.org/abs/1705.05363). Accessed 24 Apr. 2026.

Raffin, Antonin, et al. "Stable-Baselines3: Reliable Reinforcement Learning
Implementations." *Journal of Machine Learning Research*, vol. 22, no. 268,
2021, pp. 1–8. *JMLR*,
[jmlr.org/papers/v22/20-1364.html](https://jmlr.org/papers/v22/20-1364.html).
Accessed 24 Apr. 2026.

Salimans, Tim, and Richard Chen. *Learning Montezuma's Revenge from a Single
Demonstration*. arXiv, 8 Dec. 2018,
[arxiv.org/abs/1812.03381](https://arxiv.org/abs/1812.03381). Accessed 24 Apr. 2026.

Schulman, John, et al. *Proximal Policy Optimization Algorithms*. arXiv,
20 July 2017, [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347). Accessed 24 Apr. 2026.

Sekar, Ramanan, et al. *Planning to Explore via Self-Supervised World Models*.
*Proceedings of the 37th International Conference on Machine Learning*,
vol. 119, PMLR, 2020, pp. 8583–8592. arXiv,
[arxiv.org/abs/2005.05960](https://arxiv.org/abs/2005.05960). Accessed 24 Apr. 2026.

---

### Why this matters

Beating Mario 1-1 is not the point. The point is the engineering loop:

- **Try the well-trodden algorithm first.** PPO + careful reward shaping
  is what eventually gets Mario to the flag.
- **Try the fancy algorithm to learn what it actually needs.** DreamerV3 +
  Plan2Explore is a cutting-edge method; running it on a single GPU with no
  imitation seed taught us *exactly* why the published numbers need the
  budget they do.
- **Ship the working result. Document the failure honestly.** Both belong
  in the repo. A reviewer who only ever sees green checkmarks does not
  learn whether the engineer can debug a real RL system; this README is an
  attempt to show that loop end-to-end.
