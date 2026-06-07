from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mario_runtime import EnvConfig
from mario_runtime import extract_sanitized_x_position
from mario_runtime import goal_line_x_for_world_stage
from mario_runtime import load_env_config_for_model
from mario_runtime import make_single_env
from mario_runtime import render_rgb_frame
from mario_runtime import save_video
from mario_runtime import stage_order_index
from mario_runtime import world_stage_from_info
from mario_runtime import ACTION_SET_MAP
from tools.run_mario_hybrid_route import make_2_2_exit_controller
from tools.run_mario_hybrid_route import make_2_2_late_controller
from tools.run_mario_hybrid_route import make_2_2_mid_controller
from tools.search_mario_suffix import action_indices
from tools.search_mario_suffix import downscale_frames
from tools.search_mario_suffix import life_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a whole-game Mario chain using learned policies only, with no scripted guards."
    )
    parser.add_argument("--level11-model", required=True)
    parser.add_argument("--level12-model", required=True)
    parser.add_argument("--level13-model", default="")
    parser.add_argument("--level14-model", default="")
    parser.add_argument("--level21-model", default="")
    parser.add_argument("--level22-model", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--export-max-width", type=int, default=512)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--initial-noops-exact", type=int, default=0)
    parser.add_argument("--stop-world", type=int, default=1)
    parser.add_argument("--stop-stage", type=int, default=3)
    parser.add_argument("--stop-on-death", action="store_true")
    parser.add_argument("--reset-stack-on-stage-change", action="store_true")
    parser.add_argument(
        "--demo-output",
        default="",
        help="Optional .npz path for recording stacked observations/actions from this rollout.",
    )
    parser.add_argument(
        "--rescue-13-flag-touch-demo",
        action="store_true",
        help=(
            "Data-generation only: when the 1-3 policy reaches the flag base near x=2416, "
            "override a tiny right+B/right+A+B suffix to create a correction demo. "
            "Do not use runs with this flag as policy-only proof."
        ),
    )
    parser.add_argument(
        "--level22-script-handoff-x",
        type=int,
        default=0,
        help=(
            "Teacher recording only: at 2-2 when stage_x reaches this value, replay the verified "
            "hybrid mid/late/exit script through 2-3. Do not use those rollouts as policy-only proof."
        ),
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_rollout_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict_action(model: PPO, frame_stack: deque[np.ndarray], deterministic: bool) -> int:
    stacked = np.concatenate(list(frame_stack), axis=2).transpose(2, 0, 1)
    action, _ = model.predict(stacked, deterministic=deterministic)
    return int(action)


def stacked_observation(frame_stack: deque[np.ndarray]) -> np.ndarray:
    stacked = np.concatenate(list(frame_stack), axis=2).transpose(2, 0, 1)
    return np.ascontiguousarray(stacked)


def apply_initial_noops(env: Any, obs: np.ndarray, info: dict[str, Any], noops: int) -> tuple[np.ndarray, dict[str, Any]]:
    for _ in range(max(0, int(noops))):
        obs, _reward, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            obs, info = env.reset()
    return obs, info


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_rollout_seeds(int(args.seed))
    device = resolve_device(args.device)

    policies: dict[tuple[int, int], PPO] = {
        (1, 1): PPO.load(args.level11_model, device=device),
        (1, 2): PPO.load(args.level12_model, device=device),
    }
    if args.level13_model:
        policies[(1, 3)] = PPO.load(args.level13_model, device=device)
    if args.level14_model:
        policies[(1, 4)] = PPO.load(args.level14_model, device=device)
    if args.level21_model:
        policies[(2, 1)] = PPO.load(args.level21_model, device=device)
    if args.level22_model:
        policies[(2, 2)] = PPO.load(args.level22_model, device=device)

    config = load_env_config_for_model(
        Path(args.level11_model),
        fallback=EnvConfig(n_envs=1, noop_max=0, end_on_flag=False, action_set="complex"),
    )
    config.n_envs = 1
    config.whole_game = True
    config.end_on_flag = False
    config.noop_max = 0
    config.initial_noops_exact = 0
    config.action_set = "complex"
    ids = action_indices(config.action_set)
    action_names_by_id = {
        int(idx): "+".join(buttons)
        for idx, buttons in enumerate(ACTION_SET_MAP[config.action_set])
    }

    env = make_single_env(config, seed=int(args.seed))
    obs, info = env.reset(seed=int(args.seed))
    obs, info = apply_initial_noops(env, obs, info, int(args.initial_noops_exact))
    frame_stack: deque[np.ndarray] = deque(maxlen=config.frame_stack)
    for _ in range(config.frame_stack):
        frame_stack.append(obs)

    stop_ws = (int(args.stop_world), int(args.stop_stage))
    current_ws = world_stage_from_info(info, (1, 1))
    previous_ws = current_ws
    furthest_ws = current_ws
    previous_life = life_count(info)
    deaths = 0
    stage_clears = 0
    stage_x_by_ws: dict[tuple[int, int], int] = {}
    stage_max_by_ws: dict[tuple[int, int], int] = {}
    trace: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    demo_observations: list[np.ndarray] = []
    demo_actions: list[int] = []
    demo_worlds: list[int] = []
    demo_stages: list[int] = []
    demo_stage_x: list[int] = []
    demo_y: list[int] = []
    demo_modes: list[str] = []
    rescue_queue: list[int] = []
    level22_script_queue: list[int] = []
    level22_script_armed = False
    used_scripted_action = False
    reached_stop = stage_order_index(current_ws) >= stage_order_index(stop_ws)
    done_reason = "reached_stop_at_reset" if reached_stop else "max_steps"

    for step in range(1, int(args.max_steps) + 1):
        if current_ws not in policies:
            done_reason = f"missing_policy_{current_ws[0]}-{current_ws[1]}" if current_ws else "missing_policy"
            break
        if not args.no_video:
            frames.append(render_rgb_frame(env))

        action = predict_action(policies[current_ws], frame_stack, bool(args.deterministic))
        mode = f"policy:{current_ws[0]}-{current_ws[1]}"
        current_stage_x_before = stage_x_by_ws.get(current_ws, 0)
        if (
            args.rescue_13_flag_touch_demo
            and current_ws == (1, 3)
            and current_stage_x_before >= 2416
            and not rescue_queue
        ):
            rescue_queue = [ids["right_b"], ids["right_b"]] + [ids["right_ab"]] * 24
        if rescue_queue:
            action = int(rescue_queue.pop(0))
            mode = "rescue_1_3_flag_touch"
            used_scripted_action = True
        elif (
            int(args.level22_script_handoff_x) > 0
            and current_ws == (2, 2)
            and not level22_script_armed
            and current_stage_x_before >= int(args.level22_script_handoff_x)
        ):
            level22_script_armed = True
            level22_script_queue = (
                make_2_2_mid_controller(ids)
                + make_2_2_late_controller(ids)
                + make_2_2_exit_controller(ids)
            )
        if level22_script_queue:
            action = int(level22_script_queue.pop(0))
            mode = "level22_hybrid_script"
            used_scripted_action = True

        if args.demo_output:
            demo_observations.append(stacked_observation(frame_stack))
            demo_actions.append(int(action))
            demo_worlds.append(int(current_ws[0]))
            demo_stages.append(int(current_ws[1]))
            demo_stage_x.append(int(current_stage_x_before))
            demo_y.append(int(info.get("y_pos", 0) or 0))
            demo_modes.append(mode)

        obs, reward, terminated, truncated, info = env.step(int(action))
        next_ws = world_stage_from_info(info, current_ws)
        stage_changed = next_ws is not None and previous_ws is not None and next_ws != previous_ws
        if stage_changed and stage_order_index(next_ws) > stage_order_index(previous_ws):
            stage_clears += stage_order_index(next_ws) - stage_order_index(previous_ws)

        goal_x = goal_line_x_for_world_stage(next_ws, fallback=3200)
        previous_stage_x = 0 if stage_changed else stage_x_by_ws.get(next_ws, 0)
        stage_x = extract_sanitized_x_position(
            info,
            previous_x_pos=previous_stage_x,
            goal_line_x=goal_x,
        )
        if next_ws is not None:
            stage_x_by_ws[next_ws] = int(stage_x)
            stage_max_by_ws[next_ws] = max(stage_max_by_ws.get(next_ws, 0), int(stage_x))
            if furthest_ws is None or stage_order_index(next_ws) > stage_order_index(furthest_ws):
                furthest_ws = next_ws

        current_life = life_count(info)
        died = False
        if previous_life is not None and current_life is not None and current_life < previous_life:
            deaths += previous_life - current_life
            died = True
        if current_life is not None:
            previous_life = current_life

        trace.append(
            {
                "step": step,
                "world": int(next_ws[0]) if next_ws else None,
                "stage": int(next_ws[1]) if next_ws else None,
                "stage_x": int(stage_x),
                "max_stage_x": int(stage_max_by_ws.get(next_ws, stage_x)) if next_ws else int(stage_x),
                "y": int(info.get("y_pos", 0) or 0),
                "action": int(action),
                "mode": mode,
                "reward": float(reward),
                "flag_get": bool(info.get("flag_get", False)),
                "life": current_life,
                "died": bool(died),
                "buttons": action_names_by_id.get(int(action), str(action)),
            }
        )

        current_ws = next_ws
        if current_ws is not None and stage_order_index(current_ws) >= stage_order_index(stop_ws):
            reached_stop = True
            done_reason = "reached_stop"
            break
        if args.stop_on_death and died:
            done_reason = "death"
            break
        if terminated or truncated:
            done_reason = "terminated" if terminated else "truncated"
            break
        if stage_changed and next_ws == (2, 2):
            level22_script_armed = False
            level22_script_queue = []
        if args.reset_stack_on_stage_change and stage_changed:
            frame_stack.clear()
            for _ in range(config.frame_stack):
                frame_stack.append(obs)
        else:
            frame_stack.append(obs)
        previous_ws = current_ws

    video_path = ""
    if frames:
        frames = downscale_frames(frames, max_width=int(args.export_max_width))
        video_path = str((output_dir / "evaluation.mp4").resolve())
        save_video(frames, output_dir / "evaluation.mp4", fps=int(args.fps))
    if args.demo_output and demo_observations:
        demo_path = Path(args.demo_output)
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            demo_path,
            observations=np.asarray(demo_observations, dtype=np.uint8),
            actions=np.asarray(demo_actions, dtype=np.int64),
            worlds=np.asarray(demo_worlds, dtype=np.int16),
            stages=np.asarray(demo_stages, dtype=np.int16),
            stage_x=np.asarray(demo_stage_x, dtype=np.int32),
            y=np.asarray(demo_y, dtype=np.int16),
            modes=np.asarray(demo_modes, dtype="U96"),
            action_set=np.asarray(config.action_set, dtype="U32"),
            seed=np.asarray(int(args.seed), dtype=np.int32),
            deterministic=np.asarray(bool(args.deterministic)),
            reached_stop=np.asarray(bool(reached_stop)),
            stop_world_stage=np.asarray([int(stop_ws[0]), int(stop_ws[1])], dtype=np.int16),
            furthest_world_stage=np.asarray(
                [int(furthest_ws[0]), int(furthest_ws[1])] if furthest_ws else [0, 0],
                dtype=np.int16,
            ),
            stage_clears=np.asarray(int(stage_clears), dtype=np.int16),
            deaths=np.asarray(int(deaths), dtype=np.int16),
            scripted_actions=np.asarray(bool(used_scripted_action)),
        )
    (output_dir / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    summary = {
        "reached_stop": bool(reached_stop),
        "stop_world_stage": [int(stop_ws[0]), int(stop_ws[1])],
        "done_reason": done_reason,
        "steps": len(trace),
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "initial_noops_exact": int(args.initial_noops_exact),
        "stage_clears": int(stage_clears),
        "deaths": int(deaths),
        "furthest_world_stage": [int(furthest_ws[0]), int(furthest_ws[1])] if furthest_ws else None,
        "final_world_stage": [int(current_ws[0]), int(current_ws[1])] if current_ws else None,
        "stage_max_x": {f"{ws[0]}-{ws[1]}": int(value) for ws, value in stage_max_by_ws.items()},
        "models": {
            "1-1": str(Path(args.level11_model).resolve()),
            "1-2": str(Path(args.level12_model).resolve()),
            "1-3": str(Path(args.level13_model).resolve()) if args.level13_model else "",
            "1-4": str(Path(args.level14_model).resolve()) if args.level14_model else "",
            "2-1": str(Path(args.level21_model).resolve()) if args.level21_model else "",
            "2-2": str(Path(args.level22_model).resolve()) if args.level22_model else "",
        },
        "policy_only": True,
        "scripted_actions": bool(used_scripted_action),
        "demo_output": str(Path(args.demo_output).resolve()) if args.demo_output else "",
        "video": video_path,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    env.close()
    return 0 if reached_stop else 2


if __name__ == "__main__":
    raise SystemExit(main())
