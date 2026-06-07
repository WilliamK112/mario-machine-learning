"""
render_best_checkpoint.py
=========================

Take a trained PPO checkpoint and render the best evaluation episode as MP4
(and optionally GIF) for the README hero image.

The script tries N seeds × {deterministic, stochastic} and keeps one
episode (one MP4 = one rollout, no concatenation). Ranking is controlled
by ``--rank-by``:

* ``default``: among clears prefer higher max_x, then higher return; if none
  clear, prefer higher max_x then return.
* ``fastest_clear``: among clears prefer **fewest env steps** (shortest
  time-to-flag), then higher return; if none clear, same tie-break as
  ``default``.
* ``fastest_last_life``: rank by **fewest env steps** in the **first contiguous
  visit** to the configured ``(world, stage)`` measured from **episode reset**
  (e.g. 1-2 from spawn until the first frame that is no longer 1-2, usually
  before a warp to 1-1). The MP4 is **only that segment** — one continuous
  run on that stage, not mixed with later worlds.

  (The final life can start on 1-1 after a warp, so we do **not** restrict the
  stage span to the last life — that slice would often be empty.)

* ``fastest_target_stage_clear``: like ``fastest_last_life``, but only rollouts
  where Mario **actually leaves** the configured stage (e.g. 1-2 pipe exit —
  RAM stage/world changes) are eligible. Episodes that time out still on 1-2
  are rejected. Optional ``--require-flag-on-target-stage`` tightens to NES
  ``flag_get`` while still on that stage (often false for underground pipes).
  Use ``--max-deaths-on-target`` (default 0) to cap deaths while on that stage.

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
import functools
import json
import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import cv2
from stable_baselines3 import PPO

from mario_runtime import EnvConfig
from mario_runtime import build_rollout_summary
from mario_runtime import effective_goal_line_x
from mario_runtime import extract_sanitized_x_position
from mario_runtime import load_env_config_for_model
from mario_runtime import make_single_env
from mario_runtime import render_rgb_frame
from mario_runtime import save_video
from mario_runtime import stage_order_index


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
    parser.add_argument(
        "--use-train-noop",
        action="store_true",
        help="Keep noop_max from train_config instead of forcing deterministic no-op reset.",
    )
    parser.add_argument(
        "--noop-max",
        type=int,
        default=0,
        help="Override reset no-op count unless --use-train-noop is set.",
    )
    parser.add_argument(
        "--export-max-width",
        type=int,
        default=0,
        help="If >0, downscale every RGB frame so width<=this (height keeps aspect, INTER_AREA). 0=full NES res.",
    )
    parser.add_argument("--gif", action="store_true", help="Also write a GIF (slower, larger file).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Default: <model_dir>/best_render/",
    )
    parser.add_argument(
        "--rank-by",
        choices=("default", "fastest_clear", "fastest_last_life", "fastest_target_stage_clear"),
        default="default",
        help="How to pick the single episode to render (see module docstring).",
    )
    parser.add_argument(
        "--last-life-video",
        action="store_true",
        help=(
            "Trim the saved MP4/GIF to frames after the last life loss (final life only). "
            "Implied when --rank-by fastest_last_life."
        ),
    )
    parser.add_argument(
        "--world",
        type=int,
        default=None,
        help="Override SMB world from train_config (e.g. 1). Omit to use saved training config.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="Override SMB stage from train_config (e.g. 2 for 1-2 only). Omit to use saved training config.",
    )
    parser.add_argument(
        "--require-flag-on-target-stage",
        action="store_true",
        help="With fastest_target_stage_clear: require flag_get while still on configured world/stage. "
        "Often never happens on 1-2 (pipe exit); leave off for underground.",
    )
    parser.add_argument(
        "--max-deaths-on-target",
        type=int,
        default=0,
        help="With fastest_target_stage_clear: max allowed life losses while Mario was on the target stage.",
    )
    parser.add_argument(
        "--stage-exit-tail-frames",
        type=int,
        default=30,
        help="With fastest_target_stage_clear: append this many frames after first leaving the target stage "
        "(pipe / transition) so the MP4 shows the exit.",
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


def episode_quality_key_default(ep: dict) -> tuple[int, int, float] | tuple[int, int, int, float]:
    if "furthest_world_stage_index" in ep:
        return (
            int(ep.get("furthest_world_stage_index", 0)),
            int(ep.get("stage_clears", 0)),
            int(ep.get("max_x", 0)),
            float(ep["return"]),
        )
    return (1 if ep["flag"] else 0, ep["max_x"], ep["return"])


def episode_quality_key_fastest_clear(ep: dict) -> tuple[int, int, float]:
    """Prefer flag clears with minimum step count; shorter length -> larger key."""
    if ep["flag"]:
        return (2, -int(ep["length"]), int(round(ep["return"] * 1000)))
    return (1, ep["max_x"], int(round(ep["return"] * 1000)))


def episode_quality_key_fastest_last_life(ep: dict) -> tuple[int, int, int, float]:
    """Prefer shortest first contiguous visit to target stage from episode start."""
    steps = int(ep.get("target_stage_segment_steps_full", -1))
    if steps >= 0:
        return (
            2,
            -steps,
            int(ep.get("flag_on_target_stage", False)),
            int(round(ep["return"] * 1000)),
        )
    return (0, ep["max_x"], ep["length"], int(round(ep["return"] * 1000)))


def episode_quality_key_target_stage_clear(
    ep: dict,
    *,
    max_deaths_on_target: int,
    require_flag_on_target_stage: bool,
) -> tuple[int, int, int, float]:
    """Shortest first target-stage visit among episodes that leave the stage (pipe/clear)."""
    if require_flag_on_target_stage and not ep.get("flag_on_target_stage"):
        return (0, 0, 0, float(ep["return"]))
    if not ep.get("exited_target_stage"):
        return (0, 0, 0, float(ep["return"]))
    if int(ep.get("deaths_while_on_target", 0)) > int(max_deaths_on_target):
        return (0, 0, 0, float(ep["return"]))
    steps = int(ep.get("target_stage_last_attempt_steps", -1))
    if steps < 0:
        steps = int(ep.get("target_stage_segment_steps_full", -1))
    if steps < 0:
        return (0, 0, 0, float(ep["return"]))
    return (
        3,
        -steps,
        -int(ep.get("deaths_while_on_target", 0)),
        int(round(ep["return"] * 1000)),
    )


def select_quality_key(rank_by: str):
    if rank_by == "fastest_clear":
        return episode_quality_key_fastest_clear
    if rank_by == "fastest_last_life":
        return episode_quality_key_fastest_last_life
    if rank_by == "fastest_target_stage_clear":
        raise ValueError("use functools.partial(episode_quality_key_target_stage_clear, ...)")
    return episode_quality_key_default


def _set_rollout_seeds(seed: int) -> None:
    """Make a single stochastic rollout reproducible for a given env seed (CPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _life_count(info: dict) -> int | None:
    if "life" in info:
        v = info["life"]
        return int(v) if v is not None else None
    if "lives" in info:
        v = info["lives"]
        return int(v) if v is not None else None
    return None


def _unwrap_supermario_env(env: Any) -> Any | None:
    e = env
    for _ in range(64):
        if e.__class__.__name__ == "SuperMarioBrosEnv":
            return e
        if not hasattr(e, "env"):
            return None
        nxt = e.env
        if nxt is e:
            return None
        e = nxt
    return None


def _ws_pair(info: dict, fallback: tuple[int, int]) -> tuple[int, int]:
    w, s = info.get("world"), info.get("stage")
    if w is None or s is None:
        return fallback
    return (int(w), int(s))


def first_target_segment_slice(
    meta: list[tuple[int, int]],
    target: tuple[int, int],
    lo: int,
    hi: int,
) -> tuple[int, int] | None:
    """First contiguous [i, j) frame indices with meta == target, searching meta[lo:hi)."""
    tw, ts = target
    k = max(0, lo)
    n = min(hi, len(meta))
    while k < n:
        if meta[k] == (tw, ts):
            i = k
            k += 1
            while k < n and meta[k] == (tw, ts):
                k += 1
            return (i, k)
        k += 1
    return None


def run_one(
    *,
    model: PPO,
    config: EnvConfig,
    seed: int,
    deterministic: bool,
    max_steps: int,
) -> dict:
    _set_rollout_seeds(seed)
    env = make_single_env(config, seed=seed)
    obs, reset_info = env.reset(seed=seed)
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
    flag_on_target_stage = False
    done = False
    stage_clears = 0
    goal_x = effective_goal_line_x(config)
    tw, ts = int(config.world), int(config.stage)
    deaths_while_on_target = 0
    # Frame indices (post-step) right after each life loss while still on (tw, ts); used to
    # export only the final successful attempt without earlier failed lives in the same visit.
    on_target_attempt_starts: list[int] = []

    core = _unwrap_supermario_env(env)
    if core is not None:
        ws0 = (int(core._world), int(core._stage))
    else:
        ws0 = (int(config.world), int(config.stage))
    meta_ws: list[tuple[int, int]] = [ws0]
    furthest_ws = ws0

    prev_life: int | None = _life_count(reset_info)
    life_losses = 0
    last_death_step = 0
    trim_frame_start = 0

    while not done and episode_length < max_steps:
        stacked = np.concatenate(list(frame_stack), axis=2).transpose(2, 0, 1)
        action, _ = model.predict(stacked, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        frame_stack.append(obs)
        frames.append(render_rgb_frame(env))
        episode_return += float(reward)
        episode_length += 1
        episode_x_pos = extract_sanitized_x_position(
            info,
            previous_x_pos=episode_x_pos,
            goal_line_x=goal_x,
        )
        episode_max_x = max(episode_max_x, episode_x_pos)
        step_ws = _ws_pair(info, meta_ws[-1])
        if (
            step_ws != meta_ws[-1]
            and stage_order_index(step_ws) > stage_order_index(meta_ws[-1])
        ):
            stage_clears += 1
        if stage_order_index(step_ws) > stage_order_index(furthest_ws):
            furthest_ws = step_ws
        meta_ws.append(step_ws)
        if info.get("flag_get", False):
            flag = True
            if step_ws == (int(config.world), int(config.stage)):
                flag_on_target_stage = True
        done = bool(terminated or truncated)

        cur_life = _life_count(info)
        if cur_life is not None:
            if prev_life is not None and cur_life < prev_life:
                life_losses += 1
                last_death_step = episode_length
                trim_frame_start = len(frames) - 1
                if len(meta_ws) >= 2 and meta_ws[-2] == (tw, ts):
                    deaths_while_on_target += 1
                    on_target_attempt_starts.append(len(frames) - 1)
            prev_life = cur_life

    env.close()
    last_life_steps = episode_length - last_death_step
    seg_full = first_target_segment_slice(meta_ws, (tw, ts), 0, len(meta_ws))
    seg_last = first_target_segment_slice(meta_ws, (tw, ts), trim_frame_start, len(meta_ws))

    def _steps(slc: tuple[int, int] | None) -> int:
        if slc is None or slc[1] <= slc[0]:
            return -1
        return int(slc[1] - slc[0] - 1)

    exited_target_stage = bool(seg_full is not None and seg_full[1] < len(meta_ws))

    la_raw = int(seg_full[0]) if seg_full is not None else 0
    if seg_full is not None and on_target_attempt_starts:
        sf1 = int(seg_full[1])
        for cand in reversed(on_target_attempt_starts):
            if cand < sf1:
                la_raw = int(cand)
                break
    la_eff = la_raw
    if seg_full is not None:
        sf0, sf1 = int(seg_full[0]), int(seg_full[1])
        if sf1 > sf0:
            la_eff = max(sf0, min(la_raw, sf1 - 1))
        else:
            la_eff = sf0
    target_stage_last_attempt_steps = -1
    if seg_full is not None and seg_full[1] > la_eff:
        target_stage_last_attempt_steps = int(seg_full[1] - la_eff - 1)

    return {
        "seed": seed,
        "deterministic": deterministic,
        "return": episode_return,
        "length": episode_length,
        "last_life_steps": int(last_life_steps),
        "life_losses": int(life_losses),
        "deaths_while_on_target": int(deaths_while_on_target),
        "exited_target_stage": bool(exited_target_stage),
        "trim_frame_start": int(trim_frame_start),
        "max_x": int(episode_max_x),
        "flag": bool(flag),
        "flag_on_target_stage": bool(flag_on_target_stage),
        "stage_clears": int(stage_clears),
        "furthest_world_stage": list(furthest_ws),
        "furthest_world_stage_index": int(stage_order_index(furthest_ws)),
        "final_world_stage": list(meta_ws[-1]) if meta_ws else None,
        "stage_segment_full": list(seg_full) if seg_full else None,
        "stage_segment_last_life": list(seg_last) if seg_last else None,
        "target_stage_segment_steps_full": _steps(seg_full),
        "target_stage_segment_steps_last_life": _steps(seg_last),
        "last_attempt_start_frame_on_target": int(la_eff),
        "target_stage_last_attempt_steps": int(target_stage_last_attempt_steps),
        "frames": frames,
    }


def downscale_frame_rgb(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    new_w = max_width
    new_h = max(1, int(round(h * (max_width / w))))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def downscale_frames(frames: list[np.ndarray], max_width: int) -> list[np.ndarray]:
    if max_width <= 0:
        return frames
    return [downscale_frame_rgb(f, max_width) for f in frames]


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
    if not args.use_train_noop:
        config.noop_max = int(args.noop_max)
    if config.whole_game:
        config.end_on_flag = False
    if args.world is not None:
        config.world = int(args.world)
    if args.stage is not None:
        config.stage = int(args.stage)

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
    if args.rank_by == "fastest_target_stage_clear":
        qkey = functools.partial(
            episode_quality_key_target_stage_clear,
            max_deaths_on_target=args.max_deaths_on_target,
            require_flag_on_target_stage=args.require_flag_on_target_stage,
        )
    else:
        qkey = select_quality_key(args.rank_by)
    trim_video = bool(
        args.last_life_video
        or args.rank_by == "fastest_last_life"
        or args.rank_by == "fastest_target_stage_clear"
    )

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
            qualified = True
            if args.rank_by == "fastest_target_stage_clear":
                qualified = qkey(ep)[0] >= 3
            if qualified:
                if best is None or qkey(ep) > qkey(best):
                    best = ep
            print(
                f"    -> return={ep['return']:.1f}  max_x={ep['max_x']}  "
                f"len={ep['length']}  last_life={ep['last_life_steps']}  "
                f"furthest={ep.get('furthest_world_stage')}  "
                f"stage_clears={ep.get('stage_clears')}  "
                f"target_seg={ep['target_stage_segment_steps_full']}  "
                f"deaths={ep['life_losses']}  deaths_on_tgt={ep['deaths_while_on_target']}  "
                f"exited_tgt={ep['exited_target_stage']}  "
                f"last_att_steps={ep['target_stage_last_attempt_steps']}  "
                f"flag={ep['flag']}  flag_on_tgt={ep['flag_on_target_stage']}"
            )

    if best is None:
        raise SystemExit(
            "No rollout matched the filters for --rank-by fastest_target_stage_clear "
            f"(need exited_target_stage=True, deaths_while_on_target<={args.max_deaths_on_target}, "
            f"require_flag_on_target_stage={args.require_flag_on_target_stage}). "
            "Increase --seeds, set --max-deaths-on-target higher, or drop --require-flag-on-target-stage."
        )
    print(
        f"BEST rank_by={args.rank_by} seed={best['seed']} "
        f"deterministic={best['deterministic']} max_x={best['max_x']} "
        f"furthest={best.get('furthest_world_stage')} "
        f"stage_clears={best.get('stage_clears')} "
        f"flag={best['flag']} flag_on_target_stage={best['flag_on_target_stage']} "
        f"len={best['length']} "
        f"last_life_steps={best['last_life_steps']} life_losses={best['life_losses']} "
        f"target_stage_steps={best['target_stage_segment_steps_full']} "
        f"exited_target_stage={best['exited_target_stage']} "
        f"deaths_on_target={best['deaths_while_on_target']} "
        f"return={best['return']:.1f}"
    )

    frames_src = best["frames"]
    stage_seg = best.get("stage_segment_full")
    if args.rank_by in ("fastest_last_life", "fastest_target_stage_clear"):
        if stage_seg and stage_seg[1] > stage_seg[0]:
            i0 = int(stage_seg[0])
            i1 = int(stage_seg[1])
            if args.rank_by == "fastest_target_stage_clear":
                la = int(best.get("last_attempt_start_frame_on_target", 0))
                i0 = max(i0, la)
                i1 = min(i1 + max(0, args.stage_exit_tail_frames), len(frames_src))
            frames_src = frames_src[i0:i1]
            print(
                f"  video: world {config.world}-{config.stage} "
                + (
                    "last attempt on stage only + exit tail"
                    if args.rank_by == "fastest_target_stage_clear"
                    else "first visit"
                )
                + (
                    f" (+{args.stage_exit_tail_frames} tail)"
                    if args.rank_by == "fastest_target_stage_clear"
                    else ""
                )
                + f", slice [{i0}:{i1}] ({len(frames_src)} frames)"
            )
        else:
            t0 = int(best.get("trim_frame_start", 0))
            frames_src = frames_src[t0:]
            print(
                f"  video: no target-stage span; fallback trim to final life "
                f"({len(frames_src)} frames, skipped first {t0})"
            )
    elif trim_video:
        t0 = int(best.get("trim_frame_start", 0))
        frames_src = frames_src[t0:]
        print(
            f"  video: trimmed to final life ({len(frames_src)} frames, "
            f"skipped first {t0} frames from full episode)"
        )


    out_frames = downscale_frames(frames_src, args.export_max_width)
    if args.export_max_width > 0:
        print(f"  export: downscaled to max_width={args.export_max_width} px (cv2.INTER_AREA)")

    mp4_path = output_dir / "best.mp4"
    save_video(out_frames, mp4_path, fps=args.fps)
    n_frames = len(out_frames)
    playback_s = n_frames / max(args.fps, 1)
    print(
        f"  wrote {mp4_path}  ({n_frames} frames @ {args.fps} fps -> playback {playback_s:.3f}s)"
    )

    if args.gif:
        gif_path = output_dir / "best.gif"
        write_gif(out_frames, gif_path, fps=args.fps)
        if gif_path.exists():
            print(f"  wrote {gif_path}")

    if (
        args.rank_by == "fastest_last_life"
        and stage_seg
        and stage_seg[1] > stage_seg[0]
    ):
        report_length = max(0, int(stage_seg[1]) - int(stage_seg[0]) - 1)
    elif (
        args.rank_by == "fastest_target_stage_clear"
        and stage_seg
        and stage_seg[1] > stage_seg[0]
    ):
        report_length = max(0, int(best.get("target_stage_last_attempt_steps", 0)))
    elif trim_video:
        report_length = int(best["last_life_steps"])
    else:
        report_length = int(best["length"])
    stage_metric_kwargs: dict[str, Any] = {}
    if config.whole_game:
        furthest_ws = tuple(best["furthest_world_stage"]) if best.get("furthest_world_stage") else None
        final_ws = tuple(best["final_world_stage"]) if best.get("final_world_stage") else None
        stage_metric_kwargs = {
            "episode_stage_clears": [int(best.get("stage_clears", 0))],
            "episode_furthest_world_stages": [furthest_ws],
            "episode_final_world_stages": [final_ws],
        }

    summary = build_rollout_summary(
        episodes=1,
        deterministic=best["deterministic"],
        episode_returns=[best["return"]],
        episode_lengths=[report_length],
        episode_flags=[
            best["flag_on_target_stage"]
            if args.rank_by != "fastest_target_stage_clear"
            else best["exited_target_stage"]
        ],
        max_x_positions=[best["max_x"]],
        video_path=str(mp4_path),
        video_fps=args.fps,
        video_num_frames=n_frames,
        goal_line_x=effective_goal_line_x(config),
        **stage_metric_kwargs,
    )
    summary["seed"] = best["seed"]
    summary["model"] = str(model_path)
    summary["rank_by"] = args.rank_by
    summary["last_life_steps"] = best["last_life_steps"]
    summary["life_losses"] = best["life_losses"]
    summary["full_episode_length"] = best["length"]
    summary["video_trimmed_to_last_life"] = trim_video
    summary["trim_frame_start"] = best.get("trim_frame_start", 0)
    summary["flag_on_target_stage"] = best["flag_on_target_stage"]
    summary["stage_segment_full"] = best.get("stage_segment_full")
    summary["stage_segment_last_life"] = best.get("stage_segment_last_life")
    summary["target_stage_segment_steps_full"] = best.get("target_stage_segment_steps_full")
    summary["target_stage_segment_steps_last_life"] = best.get(
        "target_stage_segment_steps_last_life"
    )
    summary["exited_target_stage"] = best.get("exited_target_stage")
    summary["deaths_while_on_target"] = best.get("deaths_while_on_target")
    summary["last_attempt_start_frame_on_target"] = best.get(
        "last_attempt_start_frame_on_target"
    )
    summary["target_stage_last_attempt_steps"] = best.get("target_stage_last_attempt_steps")
    summary["stage_clears"] = best.get("stage_clears")
    summary["furthest_world_stage"] = best.get("furthest_world_stage")
    summary["final_world_stage"] = best.get("final_world_stage")
    (output_dir / "best.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    (output_dir / "all_runs.json").write_text(
        json.dumps(all_runs, indent=2),
        encoding="utf-8",
    )
    print(f"  wrote {output_dir / 'all_runs.json'}")


if __name__ == "__main__":
    main()
