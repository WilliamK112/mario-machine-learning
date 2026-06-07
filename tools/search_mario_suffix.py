from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario_runtime import ACTION_SET_MAP
from mario_runtime import EnvConfig
from mario_runtime import effective_goal_line_x
from mario_runtime import extract_sanitized_x_position
from mario_runtime import load_env_config_for_model
from mario_runtime import make_single_env
from mario_runtime import render_rgb_frame
from mario_runtime import save_video
from mario_runtime import stage_order_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a Mario policy and search small scripted suffix guards near a stuck point. "
            "This is for diagnosis/rescue; it does not modify model weights."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--export-max-width", type=int, default=512)
    parser.add_argument("--use-train-noop", action="store_true")
    parser.add_argument("--noop-max", type=int, default=0)
    parser.add_argument(
        "--initial-noops-exact",
        type=int,
        default=0,
        help="Run this many exact NOOP actions after reset before the policy starts.",
    )
    parser.add_argument(
        "--trigger-x",
        type=int,
        nargs="*",
        default=[1900, 2050, 2200, 2300, 2380, 2450],
        help="Start applying suffix guard once sanitized x reaches any of these values.",
    )
    parser.add_argument(
        "--down-x",
        type=int,
        nargs="*",
        default=[2300, 2380, 2420, 2460, 2480, 2500],
        help="Press DOWN once sanitized x reaches any of these values.",
    )
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Search timed action suffixes after trigger-x instead of simple dynamic guards.",
    )
    parser.add_argument("--script-coast1", type=int, nargs="*", default=[0, 4, 8, 12, 16, 20])
    parser.add_argument("--script-jump", type=int, nargs="*", default=[0, 3, 5, 7, 9, 12])
    parser.add_argument("--script-coast2", type=int, nargs="*", default=[0, 4, 8, 12, 16, 24])
    parser.add_argument("--script-down-steps", type=int, nargs="*", default=[12, 20, 32, 48, 64])
    parser.add_argument(
        "--script-down-x",
        type=int,
        nargs="*",
        default=[0],
        help=(
            "For scripted candidates, force DOWN once sanitized x reaches this value after "
            "the script has started. Use 0 to disable the x-triggered override."
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="If >0, shuffle and test at most this many candidates.",
    )
    parser.add_argument(
        "--rank-by",
        choices=("closest", "survival"),
        default="closest",
        help=(
            "How to rank non-clearing candidates. 'closest' prioritizes max_x after clear/exit; "
            "'survival' preserves the older behavior of preferring fewer deaths before max_x."
        ),
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=1,
        help="Save videos/traces for the top K candidates under the selected ranking.",
    )
    parser.add_argument("--no-video", action="store_true", help="Skip frame capture during search.")
    parser.add_argument(
        "--demo-output",
        default="",
        help="Optional .npz path for saving the best candidate as BC training data.",
    )
    return parser.parse_args()


def set_rollout_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ws_pair(info: dict[str, Any], fallback: tuple[int, int]) -> tuple[int, int]:
    try:
        return int(info.get("world", fallback[0])), int(info.get("stage", fallback[1]))
    except (TypeError, ValueError):
        return fallback


def life_count(info: dict[str, Any]) -> int | None:
    for key in ("life", "lives"):
        value = info.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def downscale_frames(frames: list[np.ndarray], max_width: int) -> list[np.ndarray]:
    if max_width <= 0:
        return frames
    import cv2

    out = []
    for frame in frames:
        h, w = frame.shape[:2]
        if w <= max_width:
            out.append(frame)
            continue
        scale = max_width / float(w)
        out.append(
            cv2.resize(frame, (max_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        )
    return out


def action_indices(action_set: str) -> dict[str, int]:
    actions = ACTION_SET_MAP[action_set]
    labels = {"+".join(action): idx for idx, action in enumerate(actions)}
    # Fall back conservatively if a custom action set is used.
    return {
        "noop": labels.get("NOOP", 0),
        "right": labels.get("right", 1 if len(actions) > 1 else 0),
        "right_b": labels.get("right+B", labels.get("right", 1 if len(actions) > 1 else 0)),
        "right_a": labels.get("right+A", labels.get("right", 1 if len(actions) > 1 else 0)),
        "right_ab": labels.get("right+A+B", labels.get("right+B", 1 if len(actions) > 1 else 0)),
        "down": labels.get("down", labels.get("NOOP", 0)),
        "up": labels.get("up", labels.get("NOOP", 0)),
        "left": labels.get("left", labels.get("NOOP", 0)),
    }


def has_button(action_set: str, action_idx: int, button: str) -> bool:
    actions = ACTION_SET_MAP[action_set]
    if action_idx < 0 or action_idx >= len(actions):
        return False
    return button in actions[action_idx]


def guarded_action(
    *,
    model_action: int,
    x_pos: int,
    y_pos: int,
    action_set: str,
    ids: dict[str, int],
    trigger_x: int,
    down_x: int,
    mode: str,
) -> int:
    if x_pos < trigger_x:
        return model_action
    if x_pos >= down_x:
        return ids["down"]
    if mode == "model":
        return model_action
    if mode == "right":
        return ids["right"]
    if mode == "right_b":
        return ids["right_b"]
    if mode == "suppress_jump":
        return ids["right_b"] if has_button(action_set, model_action, "A") else model_action
    if mode == "no_jump_on_platform":
        if y_pos >= 120 and has_button(action_set, model_action, "A"):
            return ids["right_b"]
        return model_action
    if mode == "coast_then_down":
        return ids["right_b"] if x_pos < down_x else ids["down"]
    return model_action


def scripted_action(
    *,
    model_action: int,
    x_pos: int,
    ids: dict[str, int],
    trigger_x: int,
    down_x: int,
    script: list[int],
    script_start_step: int | None,
    step: int,
) -> tuple[int, int | None]:
    if script_start_step is None:
        if x_pos < trigger_x:
            return model_action, None
        script_start_step = step
    if down_x > 0 and x_pos >= down_x:
        return ids["down"], script_start_step
    idx = step - script_start_step
    if 0 <= idx < len(script):
        return int(script[idx]), script_start_step
    return model_action, script_start_step


def run_candidate(
    *,
    model: PPO,
    config: EnvConfig,
    seed: int,
    deterministic: bool,
    max_steps: int,
    trigger_x: int,
    down_x: int,
    mode: str,
    script: list[int] | None = None,
    capture_video: bool = True,
    capture_demo: bool = False,
) -> dict[str, Any]:
    set_rollout_seeds(seed)
    env = make_single_env(config, seed=seed)
    obs, info = env.reset(seed=seed)
    frame_stack: deque[np.ndarray] = deque(maxlen=config.frame_stack)
    for _ in range(config.frame_stack):
        frame_stack.append(obs)

    ids = action_indices(config.action_set)
    frames: list[np.ndarray] = [render_rgb_frame(env)] if capture_video else []
    goal_x = effective_goal_line_x(config)
    tw, ts = int(config.world), int(config.stage)
    cur_ws = ws_pair(info, (tw, ts))
    furthest_ws = cur_ws
    prev_life = life_count(info)
    deaths_on_target = 0
    x_pos = 0
    max_x = 0
    stage_clears = 0
    flag = False
    flag_on_target = False
    exited_target = False
    records: list[dict[str, Any]] = []
    demo_observations: list[np.ndarray] = []
    demo_actions: list[int] = []
    done = False
    total_return = 0.0
    script_start_step: int | None = None

    for step in range(1, max_steps + 1):
        stacked = np.concatenate(list(frame_stack), axis=2).transpose(2, 0, 1)
        model_action, _ = model.predict(stacked, deterministic=deterministic)
        y_pos = int(info.get("y_pos", 0) or 0)
        if script is not None:
            action, script_start_step = scripted_action(
                model_action=int(model_action),
                x_pos=x_pos,
                ids=ids,
                trigger_x=trigger_x,
                down_x=down_x,
                script=script,
                script_start_step=script_start_step,
                step=step,
            )
        else:
            action = guarded_action(
                model_action=int(model_action),
                x_pos=x_pos,
                y_pos=y_pos,
                action_set=config.action_set,
                ids=ids,
                trigger_x=trigger_x,
                down_x=down_x,
                mode=mode,
            )
        if capture_demo:
            demo_observations.append(stacked.copy())
            demo_actions.append(int(action))
        obs, reward, terminated, truncated, info = env.step(action)
        frame_stack.append(obs)
        if capture_video:
            frames.append(render_rgb_frame(env))
        total_return += float(reward)
        x_pos = extract_sanitized_x_position(info, previous_x_pos=x_pos, goal_line_x=goal_x)
        max_x = max(max_x, x_pos)
        next_ws = ws_pair(info, cur_ws)
        if stage_order_index(next_ws) > stage_order_index(cur_ws):
            stage_clears += 1
        if stage_order_index(next_ws) > stage_order_index(furthest_ws):
            furthest_ws = next_ws
        if cur_ws == (tw, ts) and next_ws != (tw, ts):
            exited_target = True
        cur_ws = next_ws
        if info.get("flag_get", False):
            flag = True
            if cur_ws == (tw, ts):
                flag_on_target = True
        cur_life = life_count(info)
        if cur_life is not None and prev_life is not None and cur_life < prev_life and cur_ws == (tw, ts):
            deaths_on_target += 1
        if cur_life is not None:
            prev_life = cur_life
        records.append(
            {
                "step": step,
                "x": int(x_pos),
                "y": int(info.get("y_pos", 0) or 0),
                "action": int(action),
                "model_action": int(model_action),
                "world": cur_ws[0],
                "stage": cur_ws[1],
                "flag_get": bool(info.get("flag_get", False)),
                "life": cur_life,
            }
        )
        done = bool(terminated or truncated)
        if done or exited_target:
            break

    env.close()
    return {
        "seed": seed,
        "deterministic": deterministic,
        "trigger_x": trigger_x,
        "down_x": down_x,
        "mode": mode,
        "script_start_step": script_start_step,
        "script": script,
        "length": len(records),
        "return": total_return,
        "max_x": int(max_x),
        "remaining": max(0, int(goal_x) - int(max_x)),
        "flag": bool(flag),
        "flag_on_target_stage": bool(flag_on_target),
        "exited_target_stage": bool(exited_target),
        "stage_clears": int(stage_clears),
        "deaths_on_target": int(deaths_on_target),
        "furthest_world_stage": list(furthest_ws),
        "final_world_stage": list(cur_ws),
        "frames": frames,
        "records": records,
        "demo_observations": demo_observations,
        "demo_actions": demo_actions,
    }


def quality(ep: dict[str, Any], rank_by: str = "closest") -> tuple[int, int, int, int, float]:
    if rank_by == "survival":
        return (
            1 if ep["exited_target_stage"] else 0,
            1 if ep["flag"] else 0,
            -int(ep["deaths_on_target"]),
            int(ep["max_x"]),
            float(ep["return"]),
        )
    return (
        1 if ep["exited_target_stage"] else 0,
        1 if ep["flag"] else 0,
        int(ep["max_x"]),
        -int(ep["deaths_on_target"]),
        float(ep["return"]),
    )


def save_episode_artifacts(
    *,
    episode: dict[str, Any],
    output_dir: Path,
    stem: str,
    fps: int,
    export_max_width: int,
) -> None:
    lite = {
        k: v
        for k, v in episode.items()
        if k not in ("frames", "records", "demo_observations", "demo_actions")
    }
    with (output_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(lite, f, indent=2)
    with (output_dir / f"{stem}_trace.json").open("w", encoding="utf-8") as f:
        json.dump(episode["records"], f, indent=2)
    if episode["frames"]:
        frames = downscale_frames(episode["frames"], int(export_max_width))
        save_video(frames, output_dir / f"{stem}.mp4", fps=int(fps))


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_env_config_for_model(
        model_path,
        fallback=EnvConfig(n_envs=1, noop_max=0, end_on_flag=True),
    )
    config.n_envs = 1
    config.world = int(args.world)
    config.stage = int(args.stage)
    if not args.use_train_noop:
        config.noop_max = int(args.noop_max)
    config.initial_noops_exact = int(args.initial_noops_exact)
    model = PPO.load(model_path, device="cpu")

    ids = action_indices(config.action_set)
    modes = ["model", "right", "right_b", "suppress_jump", "no_jump_on_platform", "coast_then_down"]
    scripts: list[tuple[str, list[int]]] = []
    if args.scripted:
        # Families tuned for the late 1-2 moving-platform/pipe area:
        # coast on right+B, jump to adjust height, coast/land, then hold DOWN.
        for coast1 in args.script_coast1:
            for jump in args.script_jump:
                for coast2 in args.script_coast2:
                    for down in args.script_down_steps:
                        script = (
                            [ids["right_b"]] * coast1
                            + [ids["right_ab"]] * jump
                            + [ids["right_b"]] * coast2
                            + [ids["down"]] * down
                        )
                        name = f"rb{coast1}_jab{jump}_rb{coast2}_d{down}"
                        scripts.append((name, script))
    all_runs: list[dict[str, Any]] = []
    top_episodes: list[dict[str, Any]] = []
    if args.scripted:
        candidates = [
            (trigger_x, down_x, name, script)
            for trigger_x, down_x, (name, script) in itertools.product(
                args.trigger_x, args.script_down_x, scripts
            )
        ]
    else:
        candidates = [
            (trigger_x, down_x, mode, None)
            for trigger_x, down_x, mode in itertools.product(args.trigger_x, args.down_x, modes)
            if down_x >= trigger_x
        ]
    random.Random(12345).shuffle(candidates)
    if args.max_candidates and args.max_candidates > 0:
        candidates = candidates[: int(args.max_candidates)]
    for trigger_x, down_x, mode, script in candidates:
        ep = run_candidate(
            model=model,
            config=config,
            seed=int(args.seed),
            deterministic=bool(args.deterministic),
            max_steps=int(args.max_steps),
            trigger_x=int(trigger_x),
            down_x=int(down_x),
            mode=mode,
            script=script,
            capture_video=not bool(args.no_video),
            capture_demo=bool(args.demo_output),
        )
        lite = {
            k: v
            for k, v in ep.items()
            if k not in ("frames", "records", "demo_observations", "demo_actions")
        }
        all_runs.append(lite)
        top_episodes.append(ep)
        top_episodes = sorted(
            top_episodes,
            key=lambda candidate: quality(candidate, str(args.rank_by)),
            reverse=True,
        )[: max(1, int(args.save_top_k))]
        print(
            f"trigger={trigger_x} down={down_x} mode={mode} "
            f"exit={ep['exited_target_stage']} flag={ep['flag']} "
            f"deaths={ep['deaths_on_target']} max_x={ep['max_x']} rem={ep['remaining']} "
            f"len={ep['length']}"
        )

    if not top_episodes:
        raise SystemExit("No candidates ran.")
    best = top_episodes[0]

    with (output_dir / "all_runs.json").open("w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)
    for idx, episode in enumerate(top_episodes, start=1):
        save_episode_artifacts(
            episode=episode,
            output_dir=output_dir,
            stem=f"top_{idx:02d}",
            fps=int(args.fps),
            export_max_width=int(args.export_max_width),
        )
    save_episode_artifacts(
        episode=best,
        output_dir=output_dir,
        stem="best",
        fps=int(args.fps),
        export_max_width=int(args.export_max_width),
    )
    best_lite = {
        k: v
        for k, v in best.items()
        if k not in ("frames", "records", "demo_observations", "demo_actions")
    }
    print("BEST")
    print(json.dumps(best_lite, indent=2))
    if best.get("frames"):
        print(f"video={output_dir / 'best.mp4'}")
    if args.demo_output:
        demo_path = Path(args.demo_output)
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        records = best.get("records") or []
        np.savez_compressed(
            demo_path,
            observations=np.asarray(best.get("demo_observations") or [], dtype=np.uint8),
            actions=np.asarray(best.get("demo_actions") or [], dtype=np.int64),
            worlds=np.asarray([int(r.get("world", 0)) for r in records], dtype=np.int16),
            stages=np.asarray([int(r.get("stage", 0)) for r in records], dtype=np.int16),
            stage_x=np.asarray([int(r.get("x", 0)) for r in records], dtype=np.int32),
            y_pos=np.asarray([int(r.get("y", 0)) for r in records], dtype=np.int32),
            action_set=np.asarray(config.action_set),
            reached_stop=np.asarray(bool(best.get("flag") or best.get("exited_target_stage"))),
            stage_clears=np.asarray(int(best.get("stage_clears", 0))),
            deaths=np.asarray(int(best.get("deaths_on_target", 0))),
            trigger_x=np.asarray(int(best.get("trigger_x", 0))),
            mode=np.asarray(str(best.get("mode", ""))),
        )
        print(f"demo={demo_path}")


if __name__ == "__main__":
    main()
