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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mario_runtime import ACTION_SET_MAP
from mario_runtime import EnvConfig
from mario_runtime import extract_sanitized_x_position
from mario_runtime import goal_line_x_for_world_stage
from mario_runtime import load_env_config_for_model
from mario_runtime import make_single_env
from mario_runtime import render_rgb_frame
from mario_runtime import save_video
from mario_runtime import stage_order_index
from mario_runtime import world_stage_from_info
from tools.search_mario_branch import clear_needs_reset
from tools.search_mario_suffix import action_indices
from tools.search_mario_suffix import downscale_frames
from tools.search_mario_suffix import life_count


def align_demo_fields(demo: dict[str, Any]) -> dict[str, Any]:
    """Ensure observation/action metadata arrays share the same length before saving."""
    keys = ("observations", "actions", "worlds", "stages", "stage_x", "y", "modes")
    lengths = [len(demo[key]) for key in keys if key in demo]
    if not lengths:
        return demo
    n = min(lengths)
    if len(set(lengths)) > 1:
        print(
            f"warning: demo length mismatch {dict(zip([k for k in keys if k in demo], lengths))}; truncating to {n}",
            flush=True,
        )
    return {key: demo[key][:n] for key in demo}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Branch-search a short scripted correction from a policy-only whole-game chain state."
    )
    parser.add_argument("--level11-model", required=True)
    parser.add_argument("--level12-model", required=True)
    parser.add_argument("--level13-model", default="")
    parser.add_argument("--level14-model", default="")
    parser.add_argument("--level21-model", default="")
    parser.add_argument("--level22-model", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--demo-output", default="")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--initial-noops-exact", type=int, default=16)
    parser.add_argument("--reset-stack-on-stage-change", action="store_true")
    parser.add_argument("--branch-world", type=int, default=1)
    parser.add_argument("--branch-stage", type=int, default=4)
    parser.add_argument("--branch-x", type=int, nargs="+", default=[1980, 2000, 2020, 2030])
    parser.add_argument("--stop-world", type=int, default=2)
    parser.add_argument("--stop-stage", type=int, default=1)
    parser.add_argument("--max-prefix-steps", type=int, default=5000)
    parser.add_argument("--continue-steps", type=int, default=180)
    parser.add_argument("--max-candidates", type=int, default=400)
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--save-top-k", type=int, default=5)
    parser.add_argument(
        "--candidate-script-from-demo",
        default="",
        help="Optional .npz demo whose target-stage actions are inserted as the first candidate script.",
    )
    parser.add_argument(
        "--candidate-script-stage-x-min",
        type=int,
        default=0,
        help="Start the demo-derived candidate at the first target-stage sample with stage_x >= this value.",
    )
    parser.add_argument(
        "--pre-branch-script-name",
        default="",
        help=(
            "Optional known correction script to replay from the first branch point before "
            "starting the candidate search. This is for second-stage searches such as "
            "2-2 water rescue: policy reaches x~1058, known script reaches x~2800, "
            "then candidates search the final tail."
        ),
    )
    parser.add_argument(
        "--pre-branch-continue-steps",
        type=int,
        default=0,
        help="How many steps to replay the pre-branch script plus policy continuation.",
    )
    parser.add_argument(
        "--subbranch-x",
        type=int,
        default=0,
        help="If >0 with --pre-branch-script-name, search candidates from the first x >= this value.",
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--export-max-width", type=int, default=512)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_rollout_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stacked_observation(frame_stack: deque[np.ndarray]) -> np.ndarray:
    stacked = np.concatenate(list(frame_stack), axis=2).transpose(2, 0, 1)
    return np.ascontiguousarray(stacked)


def predict_action(model: PPO, frame_stack: deque[np.ndarray], deterministic: bool) -> int:
    action, _ = model.predict(stacked_observation(frame_stack), deterministic=deterministic)
    return int(action)


def apply_initial_noops(env: Any, obs: np.ndarray, info: dict[str, Any], noops: int) -> tuple[np.ndarray, dict[str, Any]]:
    for _ in range(max(0, int(noops))):
        obs, _reward, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            obs, info = env.reset()
    return obs, info


def make_tail_scripts(ids: dict[str, int], max_candidates: int, offset: int) -> list[tuple[str, list[int]]]:
    scripts: list[tuple[str, list[int]]] = []

    def tail(style: str, length: int) -> list[int]:
        if style == "right":
            return [ids["right"]] * length
        if style == "right_a":
            return [ids["right_a"]] * length
        if style == "water_swim":
            pat = [ids["right"]] * 4 + [ids["right_a"]] * 8
            return [pat[i % len(pat)] for i in range(length)]
        if style == "water_low":
            pat = [ids["right"]] * 8 + [ids["right_a"]] * 4
            return [pat[i % len(pat)] for i in range(length)]
        if style == "water_high":
            pat = [ids["right_a"]] * 10 + [ids["right"]] * 2
            return [pat[i % len(pat)] for i in range(length)]
        if style == "water_climb":
            pat = [ids["right_ab"]] * 6 + [ids["right_a"]] * 6 + [ids["right"]] * 2
            return [pat[i % len(pat)] for i in range(length)]
        if style == "pipe_down":
            pat = [ids["right"]] * 8 + [ids["down"]] * 12
            return [pat[i % len(pat)] for i in range(length)]
        if style == "pipe_up":
            pat = [ids["right_a"]] * 8 + [ids["up"]] * 8 + [ids["right"]] * 4
            return [pat[i % len(pat)] for i in range(length)]
        if style == "right_ab":
            return [ids["right_ab"]] * length
        if style == "pulse":
            pat = [ids["right_b"]] * 4 + [ids["right_ab"]] * 8
            return [pat[i % len(pat)] for i in range(length)]
        if style == "stair":
            pat = [ids["right_ab"]] * 12 + [ids["right_b"]] * 4
            return [pat[i % len(pat)] for i in range(length)]
        return [ids["right_b"]] * length

    wait_actions = [
        ("noop", ids["noop"]),
        ("right", ids["right"]),
        ("left", ids["left"]),
        ("down", ids["down"]),
    ]
    for wait_name, wait_action in wait_actions:
        for wait, rb1, jump, rb2, style in itertools.product(
            [0, 2, 4, 6, 8, 10, 12, 16],
            [0, 2, 4, 6, 8],
            [0, 4, 8, 12, 16],
            [0, 4, 8, 12],
            [
                "right",
                "right_a",
                "water_swim",
                "water_low",
                "water_high",
                "water_climb",
                "pipe_down",
                "pipe_up",
                "right_b",
                "right_ab",
                "pulse",
                "stair",
            ],
        ):
            script = (
                [wait_action] * wait
                + [ids["right_b"]] * rb1
                + [ids["right_ab"]] * jump
                + [ids["right_b"]] * rb2
                + tail(style, 100)
            )
            scripts.append((f"{wait_name}{wait}_rb{rb1}_jab{jump}_rb{rb2}_{style}", script))
    random.Random(20260517).shuffle(scripts)
    if offset:
        scripts = scripts[int(offset) :]
    if max_candidates > 0:
        scripts = scripts[: int(max_candidates)]
    return scripts


def quality(ep: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        1 if ep["reached_stop"] else 0,
        -int(ep["deaths"]),
        int(ep["stage_clears"]),
        int(ep["max_x"]),
        -int(ep["steps_after_branch"]),
    )


def save_episode_artifacts(ep: dict[str, Any], output_dir: Path, stem: str, fps: int, export_max_width: int) -> None:
    lite = {k: v for k, v in ep.items() if k not in ("frames", "trace", "demo")}
    (output_dir / f"{stem}.json").write_text(json.dumps(lite, indent=2), encoding="utf-8")
    (output_dir / f"{stem}_trace.json").write_text(json.dumps(ep["trace"], indent=2), encoding="utf-8")
    if ep.get("frames"):
        frames = downscale_frames(ep["frames"], int(export_max_width))
        save_video(frames, output_dir / f"{stem}.mp4", fps=int(fps))


def run_candidate(
    *,
    env: Any,
    policies: dict[tuple[int, int], PPO],
    deterministic: bool,
    branch_state: dict[str, Any],
    script_name: str,
    script: list[int],
    stop_ws: tuple[int, int],
    no_video: bool,
    action_names_by_id: dict[int, str],
) -> dict[str, Any]:
    env.unwrapped._restore()
    clear_needs_reset(env)
    frame_stack: deque[np.ndarray] = deque(
        (frame.copy() for frame in branch_state["frame_stack"]),
        maxlen=branch_state["frame_stack"].maxlen,
    )
    current_ws = tuple(branch_state["world_stage"])
    previous_ws = current_ws
    previous_life = branch_state["life"]
    deaths = 0
    stage_clears = 0
    stage_x = int(branch_state["stage_x"])
    max_x = int(branch_state["max_x"])
    trace = list(branch_state["trace"])
    frames = [] if no_video else list(branch_state["frames"])
    demo_obs = list(branch_state["demo_obs"])
    demo_actions = list(branch_state["demo_actions"])
    demo_worlds = list(branch_state["demo_worlds"])
    demo_stages = list(branch_state["demo_stages"])
    demo_stage_x = list(branch_state["demo_stage_x"])
    demo_y = list(branch_state["demo_y"])
    demo_modes = list(branch_state["demo_modes"])
    reached_stop = False
    done_reason = "continue_exhausted"

    for offset in range(1, int(branch_state["continue_steps"]) + 1):
        if current_ws not in policies:
            done_reason = f"missing_policy_{current_ws[0]}-{current_ws[1]}"
            break
        if not no_video:
            frames.append(render_rgb_frame(env))
        if offset <= len(script):
            action = int(script[offset - 1])
            mode = f"branch:{script_name}"
        else:
            action = predict_action(policies[current_ws], frame_stack, deterministic)
            mode = f"policy:{current_ws[0]}-{current_ws[1]}"
        demo_obs.append(stacked_observation(frame_stack))
        demo_actions.append(action)
        demo_worlds.append(int(current_ws[0]))
        demo_stages.append(int(current_ws[1]))
        demo_stage_x.append(int(stage_x))
        demo_y.append(int(branch_state.get("last_y", 0)))
        demo_modes.append(mode)

        obs, reward, terminated, truncated, info = env.step(action)
        next_ws = world_stage_from_info(info, current_ws)
        stage_changed = next_ws is not None and next_ws != previous_ws
        if stage_changed and stage_order_index(next_ws) > stage_order_index(previous_ws):
            stage_clears += stage_order_index(next_ws) - stage_order_index(previous_ws)
        goal_x = goal_line_x_for_world_stage(next_ws, fallback=3200)
        prev_x = 0 if stage_changed else stage_x
        stage_x = extract_sanitized_x_position(info, previous_x_pos=prev_x, goal_line_x=goal_x)
        if next_ws == tuple(branch_state["world_stage"]):
            max_x = max(max_x, int(stage_x))
        current_life = life_count(info)
        died = False
        if previous_life is not None and current_life is not None and current_life < previous_life:
            deaths += previous_life - current_life
            died = True
        if current_life is not None:
            previous_life = current_life
        trace.append(
            {
                "phase": "candidate",
                "offset": offset,
                "world": int(next_ws[0]) if next_ws else None,
                "stage": int(next_ws[1]) if next_ws else None,
                "stage_x": int(stage_x),
                "max_x": int(max_x),
                "y": int(info.get("y_pos", 0) or 0),
                "action": action,
                "buttons": action_names_by_id.get(action, str(action)),
                "mode": mode,
                "reward": float(reward),
                "life": current_life,
                "died": died,
            }
        )
        current_ws = next_ws
        if current_ws is not None and stage_order_index(current_ws) >= stage_order_index(stop_ws):
            reached_stop = True
            done_reason = "reached_stop"
            break
        if died:
            done_reason = "death"
            break
        if terminated or truncated:
            done_reason = "terminated" if terminated else "truncated"
            break
        frame_stack.append(obs)
        previous_ws = current_ws
        branch_state["last_y"] = int(info.get("y_pos", 0) or 0)

    return {
        "script_name": script_name,
        "branch_world_stage": list(branch_state["world_stage"]),
        "branch_x": int(branch_state["stage_x"]),
        "branch_y": int(branch_state["branch_y"]),
        "reached_stop": bool(reached_stop),
        "done_reason": done_reason,
        "stage_clears": int(stage_clears),
        "deaths": int(deaths),
        "max_x": int(max_x),
        "final_world_stage": list(current_ws) if current_ws else None,
        "steps_after_branch": int(offset),
        "frames": frames,
        "trace": trace,
        "demo": {
            "observations": demo_obs,
            "actions": demo_actions,
            "worlds": demo_worlds,
            "stages": demo_stages,
            "stage_x": demo_stage_x,
            "y": demo_y,
            "modes": demo_modes,
        },
    }


def advance_branch_state_with_known_script(
    *,
    env: Any,
    policies: dict[tuple[int, int], PPO],
    deterministic: bool,
    branch_state: dict[str, Any],
    script_name: str,
    script: list[int],
    target_x: int,
    continue_steps: int,
    no_video: bool,
    action_names_by_id: dict[int, str],
) -> dict[str, Any] | None:
    env.unwrapped._restore()
    clear_needs_reset(env)
    frame_stack: deque[np.ndarray] = deque(
        (frame.copy() for frame in branch_state["frame_stack"]),
        maxlen=branch_state["frame_stack"].maxlen,
    )
    current_ws = tuple(branch_state["world_stage"])
    previous_ws = current_ws
    previous_life = branch_state["life"]
    stage_x = int(branch_state["stage_x"])
    max_x = int(branch_state["max_x"])
    trace = list(branch_state["trace"])
    frames = [] if no_video else list(branch_state["frames"])
    demo_obs = list(branch_state["demo_obs"])
    demo_actions = list(branch_state["demo_actions"])
    demo_worlds = list(branch_state["demo_worlds"])
    demo_stages = list(branch_state["demo_stages"])
    demo_stage_x = list(branch_state["demo_stage_x"])
    demo_y = list(branch_state["demo_y"])
    demo_modes = list(branch_state["demo_modes"])
    last_y = int(branch_state.get("last_y", branch_state.get("branch_y", 0)))
    target_ws = tuple(branch_state["world_stage"])

    for offset in range(1, max(1, int(continue_steps)) + 1):
        if current_ws not in policies:
            break
        if not no_video:
            frames.append(render_rgb_frame(env))
        if offset <= len(script):
            action = int(script[offset - 1])
            mode = f"prebranch:{script_name}"
        else:
            action = predict_action(policies[current_ws], frame_stack, deterministic)
            mode = f"policy:{current_ws[0]}-{current_ws[1]}"

        demo_obs.append(stacked_observation(frame_stack))
        demo_actions.append(action)
        demo_worlds.append(int(current_ws[0]))
        demo_stages.append(int(current_ws[1]))
        demo_stage_x.append(int(stage_x))
        demo_y.append(int(last_y))
        demo_modes.append(mode)

        obs, reward, terminated, truncated, info = env.step(action)
        next_ws = world_stage_from_info(info, current_ws)
        stage_changed = next_ws is not None and next_ws != previous_ws
        goal_x = goal_line_x_for_world_stage(next_ws, fallback=3200)
        prev_x = 0 if stage_changed else stage_x
        stage_x = extract_sanitized_x_position(info, previous_x_pos=prev_x, goal_line_x=goal_x)
        if next_ws == target_ws:
            max_x = max(max_x, int(stage_x))
        current_life = life_count(info)
        died = False
        if previous_life is not None and current_life is not None and current_life < previous_life:
            died = True
        if current_life is not None:
            previous_life = current_life
        last_y = int(info.get("y_pos", 0) or 0)
        trace.append(
            {
                "phase": "prebranch",
                "offset": offset,
                "world": int(next_ws[0]) if next_ws else None,
                "stage": int(next_ws[1]) if next_ws else None,
                "stage_x": int(stage_x),
                "max_x": int(max_x),
                "y": last_y,
                "action": action,
                "buttons": action_names_by_id.get(action, str(action)),
                "mode": mode,
                "reward": float(reward),
                "life": current_life,
                "died": died,
            }
        )
        current_ws = next_ws
        if current_ws == target_ws and int(stage_x) >= int(target_x):
            env.unwrapped._backup()
            return {
                "world_stage": current_ws,
                "stage_x": int(stage_x),
                "max_x": int(max_x),
                "branch_y": last_y,
                "last_y": last_y,
                "life": current_life,
                "frame_stack": deque((frame.copy() for frame in frame_stack), maxlen=frame_stack.maxlen),
                "trace": list(trace),
                "frames": list(frames),
                "demo_obs": list(demo_obs),
                "demo_actions": list(demo_actions),
                "demo_worlds": list(demo_worlds),
                "demo_stages": list(demo_stages),
                "demo_stage_x": list(demo_stage_x),
                "demo_y": list(demo_y),
                "demo_modes": list(demo_modes),
                "continue_steps": int(branch_state["continue_steps"]),
                "target_branch_x": int(target_x),
                "pre_branch_script_name": script_name,
            }
        if died or terminated or truncated:
            break
        frame_stack.append(obs)
        previous_ws = current_ws

    return None


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
        int(idx): "+".join(buttons) if buttons else "NOOP"
        for idx, buttons in enumerate(ACTION_SET_MAP[config.action_set])
    }
    scripts = make_tail_scripts(ids, int(args.max_candidates), int(args.candidate_offset))

    env = make_single_env(config, seed=int(args.seed))
    obs, info = env.reset(seed=int(args.seed))
    obs, info = apply_initial_noops(env, obs, info, int(args.initial_noops_exact))
    frame_stack: deque[np.ndarray] = deque(maxlen=config.frame_stack)
    for _ in range(config.frame_stack):
        frame_stack.append(obs)

    current_ws = world_stage_from_info(info, (1, 1))
    previous_ws = current_ws
    previous_life = life_count(info)
    stage_x_by_ws: dict[tuple[int, int], int] = {}
    stage_max_by_ws: dict[tuple[int, int], int] = {}
    trace: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    demo_obs: list[np.ndarray] = []
    demo_actions: list[int] = []
    demo_worlds: list[int] = []
    demo_stages: list[int] = []
    demo_stage_x: list[int] = []
    demo_y: list[int] = []
    demo_modes: list[str] = []
    target_ws = (int(args.branch_world), int(args.branch_stage))
    stop_ws = (int(args.stop_world), int(args.stop_stage))
    if args.candidate_script_from_demo:
        demo = np.load(args.candidate_script_from_demo)
        required = {"actions", "worlds", "stages", "stage_x"}
        missing = required.difference(demo.files)
        if missing:
            raise ValueError(
                f"--candidate-script-from-demo missing fields {sorted(missing)}: {args.candidate_script_from_demo}"
            )
        keep = (demo["worlds"] == target_ws[0]) & (demo["stages"] == target_ws[1])
        if int(args.candidate_script_stage_x_min) > 0:
            keep &= demo["stage_x"] >= int(args.candidate_script_stage_x_min)
        candidate_demo_actions = [int(action) for action in np.asarray(demo["actions"])[keep]]
        if not candidate_demo_actions:
            raise ValueError(
                f"No demo actions for stage {target_ws[0]}-{target_ws[1]} "
                f"at x>={int(args.candidate_script_stage_x_min)}"
            )
        candidate_demo_actions = candidate_demo_actions[: int(args.continue_steps)]
        scripts.insert(
            0,
            (
                f"demo_tail_x{int(args.candidate_script_stage_x_min)}",
                candidate_demo_actions,
            ),
        )
    branch_points = sorted(int(x) for x in args.branch_x)
    branch_state: dict[str, Any] | None = None

    for step in range(1, int(args.max_prefix_steps) + 1):
        if current_ws not in policies:
            break
        if not args.no_video:
            frames.append(render_rgb_frame(env))
        action = predict_action(policies[current_ws], frame_stack, bool(args.deterministic))
        mode = f"policy:{current_ws[0]}-{current_ws[1]}"
        current_stage_x_before = stage_x_by_ws.get(current_ws, 0)
        demo_obs.append(stacked_observation(frame_stack))
        demo_actions.append(action)
        demo_worlds.append(int(current_ws[0]))
        demo_stages.append(int(current_ws[1]))
        demo_stage_x.append(int(current_stage_x_before))
        demo_y.append(int(info.get("y_pos", 0) or 0))
        demo_modes.append(mode)

        obs, reward, terminated, truncated, info = env.step(action)
        next_ws = world_stage_from_info(info, current_ws)
        stage_changed = next_ws is not None and next_ws != previous_ws
        goal_x = goal_line_x_for_world_stage(next_ws, fallback=3200)
        previous_stage_x = 0 if stage_changed else stage_x_by_ws.get(next_ws, 0)
        stage_x = extract_sanitized_x_position(info, previous_x_pos=previous_stage_x, goal_line_x=goal_x)
        if next_ws is not None:
            stage_x_by_ws[next_ws] = int(stage_x)
            stage_max_by_ws[next_ws] = max(stage_max_by_ws.get(next_ws, 0), int(stage_x))
        current_life = life_count(info)
        died = False
        if previous_life is not None and current_life is not None and current_life < previous_life:
            died = True
        if current_life is not None:
            previous_life = current_life
        trace.append(
            {
                "phase": "prefix",
                "step": step,
                "world": int(next_ws[0]) if next_ws else None,
                "stage": int(next_ws[1]) if next_ws else None,
                "stage_x": int(stage_x),
                "max_stage_x": int(stage_max_by_ws.get(next_ws, stage_x)) if next_ws else int(stage_x),
                "y": int(info.get("y_pos", 0) or 0),
                "action": action,
                "buttons": action_names_by_id.get(action, str(action)),
                "mode": mode,
                "reward": float(reward),
                "life": current_life,
                "died": died,
            }
        )
        current_ws = next_ws
        if current_ws == target_ws and int(stage_x) >= branch_points[0]:
            actual_target = next(x for x in branch_points if int(stage_x) >= x)
            env.unwrapped._backup()
            branch_state = {
                "world_stage": current_ws,
                "stage_x": int(stage_x),
                "max_x": int(stage_max_by_ws.get(current_ws, stage_x)),
                "branch_y": int(info.get("y_pos", 0) or 0),
                "last_y": int(info.get("y_pos", 0) or 0),
                "life": current_life,
                "frame_stack": deque((frame.copy() for frame in frame_stack), maxlen=frame_stack.maxlen),
                "trace": list(trace),
                "frames": list(frames),
                "demo_obs": list(demo_obs),
                "demo_actions": list(demo_actions),
                "demo_worlds": list(demo_worlds),
                "demo_stages": list(demo_stages),
                "demo_stage_x": list(demo_stage_x),
                "demo_y": list(demo_y),
                "demo_modes": list(demo_modes),
                "continue_steps": int(args.continue_steps),
                "target_branch_x": int(actual_target),
            }
            print(f"BRANCH x={stage_x} y={info.get('y_pos', 0)} candidates={len(scripts)}")
            break
        if died or terminated or truncated:
            break
        if args.reset_stack_on_stage_change and stage_changed:
            frame_stack.clear()
            for _ in range(config.frame_stack):
                frame_stack.append(obs)
        else:
            frame_stack.append(obs)
        previous_ws = current_ws

    all_runs: list[dict[str, Any]] = []
    top: list[dict[str, Any]] = []
    if branch_state is not None:
        if args.pre_branch_script_name:
            all_scripts = dict(make_tail_scripts(ids, 0, 0))
            pre_script = all_scripts.get(str(args.pre_branch_script_name))
            if pre_script is None:
                raise ValueError(f"Unknown --pre-branch-script-name: {args.pre_branch_script_name}")
            subbranch_x = int(args.subbranch_x)
            if subbranch_x <= 0:
                raise ValueError("--subbranch-x must be >0 when --pre-branch-script-name is used")
            continue_steps = int(args.pre_branch_continue_steps) or len(pre_script)
            print(
                f"PREBRANCH {args.pre_branch_script_name} target_x={subbranch_x} "
                f"continue_steps={continue_steps}"
            )
            branch_state = advance_branch_state_with_known_script(
                env=env,
                policies=policies,
                deterministic=bool(args.deterministic),
                branch_state=branch_state,
                script_name=str(args.pre_branch_script_name),
                script=pre_script,
                target_x=subbranch_x,
                continue_steps=continue_steps,
                no_video=bool(args.no_video),
                action_names_by_id=action_names_by_id,
            )
            if branch_state is None:
                print("PREBRANCH_FAILED")

    if branch_state is not None:
        for idx, (script_name, script) in enumerate(scripts, start=1):
            ep = run_candidate(
                env=env,
                policies=policies,
                deterministic=bool(args.deterministic),
                branch_state=branch_state,
                script_name=script_name,
                script=script,
                stop_ws=stop_ws,
                no_video=bool(args.no_video),
                action_names_by_id=action_names_by_id,
            )
            lite = {k: v for k, v in ep.items() if k not in ("frames", "trace", "demo")}
            lite["candidate_index"] = idx + int(args.candidate_offset)
            all_runs.append(lite)
            top.append(ep)
            top = sorted(top, key=quality, reverse=True)[: max(1, int(args.save_top_k))]
            if idx % 50 == 0:
                best = top[0]
                print(f"progress {idx}/{len(scripts)} best reached={best['reached_stop']} max_x={best['max_x']}")
            if ep["reached_stop"] and ep["deaths"] == 0:
                print("SUCCESS")
                print(json.dumps(lite, indent=2))
                break

    (output_dir / "all_runs.json").write_text(json.dumps(all_runs, indent=2), encoding="utf-8")
    best_lite: dict[str, Any] = {}
    for idx, ep in enumerate(top, start=1):
        save_episode_artifacts(ep, output_dir, f"top_{idx:02d}", int(args.fps), int(args.export_max_width))
    if top:
        save_episode_artifacts(top[0], output_dir, "best", int(args.fps), int(args.export_max_width))
        best_lite = {k: v for k, v in top[0].items() if k not in ("frames", "trace", "demo")}
        if args.demo_output:
            Path(args.demo_output).parent.mkdir(parents=True, exist_ok=True)
            demo = align_demo_fields(top[0]["demo"])
            np.savez_compressed(
                args.demo_output,
                observations=np.asarray(demo["observations"], dtype=np.uint8),
                actions=np.asarray(demo["actions"], dtype=np.int64),
                worlds=np.asarray(demo["worlds"], dtype=np.int16),
                stages=np.asarray(demo["stages"], dtype=np.int16),
                stage_x=np.asarray(demo["stage_x"], dtype=np.int32),
                y=np.asarray(demo["y"], dtype=np.int16),
                modes=np.asarray(demo["modes"]),
                action_set=np.asarray(config.action_set),
                seed=np.asarray(int(args.seed)),
                deterministic=np.asarray(bool(args.deterministic)),
                reached_stop=np.asarray(bool(best_lite.get("reached_stop", False))),
                stop_world_stage=np.asarray([int(args.stop_world), int(args.stop_stage)], dtype=np.int16),
                stage_clears=np.asarray(int(best_lite.get("stage_clears", 0)), dtype=np.int16),
                deaths=np.asarray(int(best_lite.get("deaths", 0)), dtype=np.int16),
                scripted_actions=np.asarray(True),
            )
    summary = {
        "branch_found": branch_state is not None,
        "runs": len(all_runs),
        "best": best_lite,
        "demo_output": str(Path(args.demo_output).resolve()) if args.demo_output else "",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    env.close()
    return 0 if best_lite.get("reached_stop", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
