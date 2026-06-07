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

from mario_runtime import EnvConfig
from mario_runtime import effective_goal_line_x
from mario_runtime import extract_sanitized_x_position
from mario_runtime import load_env_config_for_model
from mario_runtime import make_single_env
from mario_runtime import render_rgb_frame
from mario_runtime import save_video
from mario_runtime import stage_order_index
from search_mario_suffix import action_indices
from search_mario_suffix import downscale_frames
from search_mario_suffix import life_count
from search_mario_suffix import ws_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reach a late Mario state with a PPO/script approach, backup the emulator, "
            "then branch-search short final controller scripts from that exact state."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--whole-game", action="store_true")
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-approach-steps", type=int, default=420)
    parser.add_argument("--branch-x", type=int, default=2380)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--export-max-width", type=int, default=512)
    parser.add_argument("--use-train-noop", action="store_true")
    parser.add_argument("--noop-max", type=int, default=0)
    parser.add_argument("--trigger-x", type=int, nargs="*", default=[2050, 2200, 2300, 2380])
    parser.add_argument("--script-coast1", type=int, nargs="*", default=[0, 4, 8, 12, 16, 20])
    parser.add_argument("--script-jump", type=int, nargs="*", default=[0, 3, 5, 7, 9, 12])
    parser.add_argument("--script-coast2", type=int, nargs="*", default=[0, 4, 8, 12, 16, 24])
    parser.add_argument("--script-down-steps", type=int, nargs="*", default=[12, 20, 32, 48, 64])
    parser.add_argument("--max-approach-candidates", type=int, default=300)
    parser.add_argument("--final-rb1", type=int, nargs="*", default=[0, 2, 4, 6])
    parser.add_argument("--final-jump", type=int, nargs="*", default=[0, 2, 4, 5, 6])
    parser.add_argument("--final-rb2", type=int, nargs="*", default=[0, 2, 4, 8, 12, 16])
    parser.add_argument("--final-down", type=int, nargs="*", default=[20, 40, 64, 96])
    parser.add_argument("--final-tail", type=int, nargs="*", default=[0, 60, 120])
    parser.add_argument(
        "--final-tail-style",
        nargs="*",
        default=["right_b"],
        help="Tail controller after DOWN: right_b, right_ab, pulse_jump, or stair_jump.",
    )
    parser.add_argument("--max-final-candidates", type=int, default=600)
    parser.add_argument("--save-top-k", type=int, default=5)
    parser.add_argument(
        "--demo-output",
        default="",
        help="Optional .npz path for saving the best branch episode as BC training data.",
    )
    return parser.parse_args()


def set_rollout_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clear_needs_reset(env: Any) -> None:
    cur = env
    while cur is not None:
        if hasattr(cur, "needs_reset"):
            cur.needs_reset = False
        cur = getattr(cur, "env", None)
    if hasattr(env.unwrapped, "done"):
        env.unwrapped.done = False


def make_approach_scripts(ids: dict[str, int], args: argparse.Namespace) -> list[tuple[str, list[int]]]:
    scripts: list[tuple[str, list[int]]] = []
    for coast1 in args.script_coast1:
        for jump in args.script_jump:
            for coast2 in args.script_coast2:
                for down in args.script_down_steps:
                    script = (
                        [ids["right_b"]] * int(coast1)
                        + [ids["right_ab"]] * int(jump)
                        + [ids["right_b"]] * int(coast2)
                        + [ids["down"]] * int(down)
                    )
                    scripts.append((f"rb{coast1}_jab{jump}_rb{coast2}_d{down}", script))
    return scripts


def make_final_scripts(ids: dict[str, int], args: argparse.Namespace) -> list[tuple[str, list[int]]]:
    def tail_actions(style: str, length: int) -> list[int]:
        if length <= 0:
            return []
        if style == "right_ab":
            return [ids["right_ab"]] * int(length)
        if style == "pulse_jump":
            pattern = [ids["right_b"]] * 8 + [ids["right_ab"]] * 4
            return [pattern[i % len(pattern)] for i in range(int(length))]
        if style == "stair_jump":
            pattern = [ids["right_b"]] * 4 + [ids["right_ab"]] * 8
            return [pattern[i % len(pattern)] for i in range(int(length))]
        return [ids["right_b"]] * int(length)

    scripts: list[tuple[str, list[int]]] = []
    for rb1, jump, rb2, down, tail, tail_style in itertools.product(
        args.final_rb1,
        args.final_jump,
        args.final_rb2,
        args.final_down,
        args.final_tail,
        args.final_tail_style,
    ):
        script = (
            [ids["right_b"]] * int(rb1)
            + [ids["right_ab"]] * int(jump)
            + [ids["right_b"]] * int(rb2)
            + [ids["down"]] * int(down)
            + tail_actions(str(tail_style), int(tail))
        )
        scripts.append(
            (f"rb{rb1}_jab{jump}_rb{rb2}_down{down}_tail{tail}_{tail_style}", script)
        )
    random.Random(23456).shuffle(scripts)
    if args.max_final_candidates > 0:
        scripts = scripts[: int(args.max_final_candidates)]
    return scripts


def quality(ep: dict[str, Any]) -> tuple[int, int, int, int, float]:
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
    frames = downscale_frames(episode["frames"], int(export_max_width))
    save_video(frames, output_dir / f"{stem}.mp4", fps=int(fps))


def run_final_from_backup(
    *,
    env: Any,
    final_name: str,
    final_script: list[int],
    branch_stack: list[np.ndarray],
    prefix_frames: list[np.ndarray],
    prefix_records: list[dict[str, Any]],
    prefix_demo_observations: list[np.ndarray],
    prefix_demo_actions: list[int],
    branch_info: dict[str, Any],
    branch_x: int,
    branch_ws: tuple[int, int],
    branch_life: int | None,
    target_ws: tuple[int, int],
    goal_x: int,
) -> dict[str, Any]:
    env.unwrapped._restore()
    clear_needs_reset(env)
    frame_stack: deque[np.ndarray] = deque((frame.copy() for frame in branch_stack), maxlen=len(branch_stack))
    frames = list(prefix_frames)
    records = list(prefix_records)
    demo_observations = list(prefix_demo_observations)
    demo_actions = list(prefix_demo_actions)
    x_pos = int(branch_x)
    max_x = int(branch_x)
    cur_ws = branch_ws
    furthest_ws = branch_ws
    prev_life = branch_life
    deaths_on_target = 0
    flag = False
    flag_on_target = False
    exited_target = False
    stage_clears = 0
    total_return = 0.0

    for offset, action in enumerate(final_script, start=1):
        stacked = np.concatenate(list(frame_stack), axis=2).transpose(2, 0, 1)
        demo_observations.append(stacked.copy())
        demo_actions.append(int(action))
        obs, reward, terminated, truncated, info = env.step(int(action))
        frame_stack.append(obs)
        frames.append(render_rgb_frame(env))
        total_return += float(reward)
        x_pos = extract_sanitized_x_position(info, previous_x_pos=x_pos, goal_line_x=goal_x)
        max_x = max(max_x, x_pos)
        next_ws = ws_pair(info, cur_ws)
        if stage_order_index(next_ws) > stage_order_index(cur_ws):
            stage_clears += 1
        if stage_order_index(next_ws) > stage_order_index(furthest_ws):
            furthest_ws = next_ws
        if cur_ws == target_ws and next_ws != target_ws:
            exited_target = True
        cur_ws = next_ws
        if info.get("flag_get", False):
            flag = True
            if cur_ws == target_ws:
                flag_on_target = True
        cur_life = life_count(info)
        if cur_life is not None and prev_life is not None and cur_life < prev_life and cur_ws == target_ws:
            deaths_on_target += 1
        if cur_life is not None:
            prev_life = cur_life
        records.append(
            {
                "phase": "final",
                "final_name": final_name,
                "offset": offset,
                "x": int(x_pos),
                "y": int(info.get("y_pos", 0) or 0),
                "action": int(action),
                "world": cur_ws[0],
                "stage": cur_ws[1],
                "flag_get": bool(info.get("flag_get", False)),
                "life": cur_life,
            }
        )
        if flag or exited_target or terminated or truncated or deaths_on_target > 0:
            break

    return {
        "final_name": final_name,
        "branch_start_x": int(branch_x),
        "branch_start_y": int(branch_info.get("y_pos", 0) or 0),
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)
    config = load_env_config_for_model(
        model_path,
        fallback=EnvConfig(n_envs=1, noop_max=0, end_on_flag=True),
    )
    config.n_envs = 1
    config.world = int(args.world)
    config.stage = int(args.stage)
    if args.whole_game:
        config.whole_game = True
        config.end_on_flag = False
    if not args.use_train_noop:
        config.noop_max = int(args.noop_max)
    model = PPO.load(model_path, device="cpu")
    ids = action_indices(config.action_set)
    approach_scripts = make_approach_scripts(ids, args)
    final_scripts = make_final_scripts(ids, args)
    approach_candidates = [
        (trigger_x, name, script)
        for trigger_x, (name, script) in itertools.product(args.trigger_x, approach_scripts)
    ]
    random.Random(12345).shuffle(approach_candidates)
    if args.max_approach_candidates > 0:
        approach_candidates = approach_candidates[: int(args.max_approach_candidates)]

    goal_x = effective_goal_line_x(config)
    target_ws = (int(args.world), int(args.stage))
    top_episodes: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []
    branch_hits = 0

    for approach_idx, (trigger_x, approach_name, approach_script) in enumerate(approach_candidates, start=1):
        set_rollout_seeds(int(args.seed))
        env = make_single_env(config, seed=int(args.seed))
        obs, info = env.reset(seed=int(args.seed))
        frame_stack: deque[np.ndarray] = deque(maxlen=config.frame_stack)
        for _ in range(config.frame_stack):
            frame_stack.append(obs)
        frames: list[np.ndarray] = [render_rgb_frame(env)]
        records: list[dict[str, Any]] = []
        demo_observations: list[np.ndarray] = []
        demo_actions: list[int] = []
        x_pos = 0
        max_x = 0
        cur_ws = ws_pair(info, target_ws)
        script_start_step: int | None = None
        branch_done = False

        for step in range(1, int(args.max_approach_steps) + 1):
            stacked = np.concatenate(list(frame_stack), axis=2).transpose(2, 0, 1)
            model_action, _ = model.predict(stacked, deterministic=bool(args.deterministic))
            if script_start_step is None:
                if x_pos >= int(trigger_x):
                    script_start_step = step
                action = int(model_action)
            if script_start_step is not None:
                idx = step - script_start_step
                action = int(approach_script[idx]) if idx < len(approach_script) else int(model_action)
            demo_observations.append(stacked.copy())
            demo_actions.append(int(action))
            obs, reward, terminated, truncated, info = env.step(int(action))
            frame_stack.append(obs)
            frames.append(render_rgb_frame(env))
            x_pos = extract_sanitized_x_position(info, previous_x_pos=x_pos, goal_line_x=goal_x)
            max_x = max(max_x, x_pos)
            cur_ws = ws_pair(info, cur_ws)
            cur_life = life_count(info)
            records.append(
                {
                    "phase": "approach",
                    "approach_idx": approach_idx,
                    "approach_name": approach_name,
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
            if x_pos >= int(args.branch_x) and cur_ws == target_ws:
                branch_hits += 1
                print(
                    f"BRANCH approach={approach_idx} trigger={trigger_x} mode={approach_name} "
                    f"x={x_pos} y={info.get('y_pos', 0)} final_candidates={len(final_scripts)}"
                )
                env.unwrapped._backup()
                branch_stack = [frame.copy() for frame in frame_stack]
                prefix_frames = list(frames)
                prefix_records = list(records)
                prefix_demo_observations = list(demo_observations)
                prefix_demo_actions = list(demo_actions)
                for final_name, final_script in final_scripts:
                    ep = run_final_from_backup(
                        env=env,
                        final_name=final_name,
                        final_script=final_script,
                        branch_stack=branch_stack,
                        prefix_frames=prefix_frames,
                        prefix_records=prefix_records,
                        prefix_demo_observations=prefix_demo_observations,
                        prefix_demo_actions=prefix_demo_actions,
                        branch_info=info,
                        branch_x=int(x_pos),
                        branch_ws=cur_ws,
                        branch_life=cur_life,
                        target_ws=target_ws,
                        goal_x=goal_x,
                    )
                    ep.update(
                        {
                            "seed": int(args.seed),
                            "deterministic": bool(args.deterministic),
                            "approach_idx": approach_idx,
                            "trigger_x": int(trigger_x),
                            "approach_name": approach_name,
                            "script_start_step": script_start_step,
                        }
                    )
                    lite = {
                        k: v
                        for k, v in ep.items()
                        if k not in ("frames", "records", "demo_observations", "demo_actions")
                    }
                    all_runs.append(lite)
                    top_episodes.append(ep)
                    top_episodes = sorted(top_episodes, key=quality, reverse=True)[: max(1, int(args.save_top_k))]
                    if ep["flag"] or ep["exited_target_stage"]:
                        print("SUCCESS")
                        print(json.dumps(lite, indent=2))
                        branch_done = True
                        break
                env.unwrapped._restore()
                clear_needs_reset(env)
                branch_done = True
                break
            if terminated or truncated:
                break
        if not branch_done:
            print(
                f"approach={approach_idx} trigger={trigger_x} mode={approach_name} "
                f"no_branch max_x={max_x}"
            )
        env.close()
        if top_episodes and (top_episodes[0]["flag"] or top_episodes[0]["exited_target_stage"]):
            break

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
    if top_episodes:
        save_episode_artifacts(
            episode=top_episodes[0],
            output_dir=output_dir,
            stem="best",
            fps=int(args.fps),
            export_max_width=int(args.export_max_width),
        )
        best_lite = {
            k: v
            for k, v in top_episodes[0].items()
            if k not in ("frames", "records", "demo_observations", "demo_actions")
        }
    else:
        best_lite = {}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "branch_hits": branch_hits,
                "runs": len(all_runs),
                "best": best_lite,
            },
            f,
            indent=2,
        )
    print("SUMMARY")
    print(json.dumps({"branch_hits": branch_hits, "runs": len(all_runs), "best": best_lite}, indent=2))
    if args.demo_output and top_episodes:
        best = top_episodes[0]
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
            final_name=np.asarray(str(best.get("final_name", ""))),
            branch_start_x=np.asarray(int(best.get("branch_start_x", 0))),
        )
        print(f"demo={demo_path}")


if __name__ == "__main__":
    main()
