"""
render_best_checkpoint.py
=========================

Take a trained PPO checkpoint and render the best evaluation episode as MP4
(and optionally GIF) for the README hero image.

The script tries N seeds × {deterministic, stochastic} and keeps the
*best* episode by the following priority:
    1. flag_get == True  (any clear beats any non-clear)
    2. max_x_pos descending
    3. episode return descending

Output:
    <output-dir>/best.mp4
    <output-dir>/best.gif   (optional, --gif)
    <output-dir>/best.json  (full episode summary)
    <output-dir>/all_runs.json  (summary of every attempted run)

Usage:
    # Render a specific checkpoint, try 5 seeds, also write GIF
    python render_best_checkpoint.py --model runs/ppo_overnight_TS/run/evaluations/best_eval.zip --seeds 5 --gif

    # Auto-detect newest run's best checkpoint
    python render_best_checkpoint.py --auto

    # Auto-detect, but prefer the latest periodic checkpoint over best_eval
    python render_best_checkpoint.py --auto --prefer-latest-step
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from mario_runtime import EnvConfig
from mario_runtime import build_rollout_summary
from mario_runtime import extract_sanitized_x_position
from mario_runtime import load_env_config_for_model
from mario_runtime import make_single_env
from mario_runtime import render_rgb_frame
from mario_runtime import save_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Path to a PPO checkpoint .zip. Use --auto to skip and resolve automatically.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect newest checkpoint under runs/ppo_*",
    )
    parser.add_argument(
        "--prefer-latest-step",
        action="store_true",
        help="With --auto, prefer the newest periodic checkpoint over best_eval.zip.",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to try.")
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=4_000)
    parser.add_argument(
        "--policy-modes",
        type=str,
        default="deterministic,stochastic",
        help="Comma-separated subset of {deterministic,stochastic}.",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--gif", action="store_true", help="Also write a GIF (slower, larger file).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Default: <model_dir>/best_render/",
    )
    return parser.parse_args()


def auto_detect_checkpoint(prefer_latest_step: bool = False) -> Path | None:
    """Find the most relevant PPO checkpoint.

    Priority (default):
        1. Newest run dir's best_eval.zip
        2. Newest run dir's mario_final.zip
        3. Newest run dir's latest mario_ppo_<N>_steps.zip checkpoint

    With `prefer_latest_step=True`, the latest periodic checkpoint wins.
    Useful for inspecting an in-flight run where best_eval is stale.
    """
    runs_root = Path("runs")
    if not runs_root.exists():
        return None
    for run_dir in sorted(runs_root.glob("ppo_*"), reverse=True):
        latest_step = None
        ckpt_dir = run_dir / "run" / "models" / "checkpoints"
        if ckpt_dir.exists():
            steps = sorted(
                ckpt_dir.glob("mario_ppo_*_steps.zip"),
                key=lambda p: int(p.stem.split("_")[-2]),
                reverse=True,
            )
            if steps:
                latest_step = steps[0]

        candidates_in_order = (
            [latest_step] if prefer_latest_step and latest_step else []
        ) + [
            run_dir / "run" / "evaluations" / "best_eval.zip",
            run_dir / "run" / "models" / "mario_final.zip",
        ]
        if not prefer_latest_step and latest_step:
            candidates_in_order.append(latest_step)

        for path in candidates_in_order:
            if path and path.exists():
                return path
    return None


def episode_quality_key(ep: dict) -> tuple[int, int, float]:
    return (1 if ep["flag"] else 0, ep["max_x"], ep["return"])


def run_one(
    *,
    model: PPO,
    config: EnvConfig,
    seed: int,
    deterministic: bool,
    max_steps: int,
) -> dict:
    env = make_single_env(config, seed=seed)
    obs, _info = env.reset(seed=seed)
    frame_stack: deque[np.ndarray] = deque(maxlen=config.frame_stack)
    frame_stack.clear()
    for _ in range(config.frame_stack):
        frame_stack.append(obs)

    frames: list[np.ndarray] = [render_rgb_frame(env)]
    episode_return = 0.0
    episode_length = 0
    episode_max_x = 0
    episode_x_pos = 0
    flag = False
    done = False

    while not done and episode_length < max_steps:
        stacked = np.concatenate(list(frame_stack), axis=2).transpose(2, 0, 1)
        action, _ = model.predict(stacked, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        frame_stack.append(obs)
        frames.append(render_rgb_frame(env))
        episode_return += float(reward)
        episode_length += 1
        episode_x_pos = extract_sanitized_x_position(info, previous_x_pos=episode_x_pos)
        episode_max_x = max(episode_max_x, episode_x_pos)
        if info.get("flag_get", False):
            flag = True
        done = bool(terminated or truncated)

    env.close()
    return {
        "seed": seed,
        "deterministic": deterministic,
        "return": episode_return,
        "length": episode_length,
        "max_x": int(episode_max_x),
        "flag": bool(flag),
        "frames": frames,
    }


def write_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        print(f"  imageio not available; skipping GIF write to {path}")
        return
    duration_s = max(1.0 / fps, 0.01)
    imageio.mimwrite(str(path), frames, format="GIF", duration=duration_s, loop=0)


def main() -> None:
    args = parse_args()

    if args.auto and not args.model:
        detected = auto_detect_checkpoint(prefer_latest_step=args.prefer_latest_step)
        if not detected:
            raise SystemExit("No PPO checkpoint found under runs/. Pass --model explicitly.")
        model_path = detected
        print(f"auto_detected_model={model_path}")
    elif args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(model_path)
    else:
        raise SystemExit("Pass either --model <path> or --auto.")

    output_dir = Path(args.output_dir) if args.output_dir else model_path.parent / "best_render"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_env_config_for_model(
        model_path,
        fallback=EnvConfig(n_envs=1, noop_max=0, end_on_flag=True),
    )
    config.n_envs = 1
    config.noop_max = 0
    config.end_on_flag = True

    model = PPO.load(model_path, device="cpu")

    modes = []
    for mode_token in args.policy_modes.split(","):
        mode_token = mode_token.strip().lower()
        if mode_token == "deterministic":
            modes.append(True)
        elif mode_token == "stochastic":
            modes.append(False)
    if not modes:
        modes = [True]

    all_runs: list[dict] = []
    best: dict | None = None

    for seed_idx in range(args.seeds):
        for deterministic in modes:
            seed = args.seed_base + seed_idx
            print(f"  rolling out seed={seed} deterministic={deterministic} ...")
            ep = run_one(
                model=model,
                config=config,
                seed=seed,
                deterministic=deterministic,
                max_steps=args.max_steps,
            )
            all_runs.append({k: v for k, v in ep.items() if k != "frames"})
            if best is None or episode_quality_key(ep) > episode_quality_key(best):
                best = ep
            print(
                f"    -> return={ep['return']:.1f}  max_x={ep['max_x']}  "
                f"len={ep['length']}  flag={ep['flag']}"
            )

    assert best is not None
    print(
        f"BEST seed={best['seed']} deterministic={best['deterministic']} "
        f"max_x={best['max_x']} flag={best['flag']} return={best['return']:.1f}"
    )

    mp4_path = output_dir / "best.mp4"
    save_video(best["frames"], mp4_path, fps=args.fps)
    print(f"  wrote {mp4_path}")

    if args.gif:
        gif_path = output_dir / "best.gif"
        write_gif(best["frames"], gif_path, fps=args.fps)
        if gif_path.exists():
            print(f"  wrote {gif_path}")

    summary = build_rollout_summary(
        episodes=1,
        deterministic=best["deterministic"],
        episode_returns=[best["return"]],
        episode_lengths=[best["length"]],
        episode_flags=[best["flag"]],
        max_x_positions=[best["max_x"]],
        video_path=str(mp4_path),
    )
    summary["seed"] = best["seed"]
    summary["model"] = str(model_path)
    (output_dir / "best.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    (output_dir / "all_runs.json").write_text(
        json.dumps(all_runs, indent=2),
        encoding="utf-8",
    )
    print(f"  wrote {output_dir / 'all_runs.json'}")


if __name__ == "__main__":
    main()
