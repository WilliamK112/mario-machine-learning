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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mario_runtime import EnvConfig
from mario_runtime import extract_sanitized_x_position
from mario_runtime import goal_line_x_for_world_stage
from mario_runtime import load_env_config_for_model
from mario_runtime import make_single_env
from mario_runtime import render_rgb_frame
from mario_runtime import save_video
from mario_runtime import stage_order_index
from mario_runtime import world_stage_from_info
from run_mario_hybrid_route import apply_exact_initial_noops
from run_mario_hybrid_route import make_1_2_controller
from run_mario_hybrid_route import make_1_2_early_controller
from run_mario_hybrid_route import make_1_2_late_controller
from run_mario_hybrid_route import make_1_3_early_controller
from run_mario_hybrid_route import make_1_3_gap_controller
from run_mario_hybrid_route import make_1_3_gap2_controller
from run_mario_hybrid_route import make_1_3_gap3_controller
from run_mario_hybrid_route import make_1_3_exit_controller
from run_mario_hybrid_route import make_1_3_late_controller
from run_mario_hybrid_route import make_1_3_mid_controller
from run_mario_hybrid_route import make_1_4_exit_controller
from run_mario_hybrid_route import make_1_4_opening_controller
from run_mario_hybrid_route import make_2_1_bridge_controller
from run_mario_hybrid_route import make_2_1_exit_controller
from run_mario_hybrid_route import make_2_1_late_controller
from run_mario_hybrid_route import make_2_1_mid_controller
from run_mario_hybrid_route import make_2_1_opening_controller
from run_mario_hybrid_route import make_2_1_pipe_controller
from run_mario_hybrid_route import make_2_1_post_pipe_controller
from run_mario_hybrid_route import make_2_1_tail_controller
from run_mario_hybrid_route import make_2_2_exit_controller
from run_mario_hybrid_route import make_2_2_late_controller
from run_mario_hybrid_route import make_2_2_mid_controller
from run_mario_hybrid_route import make_2_2_opening_controller
from run_mario_hybrid_route import make_2_3_bridge_controller
from run_mario_hybrid_route import make_2_3_exit_controller
from run_mario_hybrid_route import make_2_3_mid_controller
from run_mario_hybrid_route import make_2_3_late_controller
from run_mario_hybrid_route import make_2_3_opening_controller
from run_mario_hybrid_route import make_2_3_tail_controller
from run_mario_hybrid_route import make_2_4_exit_controller
from run_mario_hybrid_route import make_2_4_mid_controller
from run_mario_hybrid_route import make_2_4_opening_controller
from run_mario_hybrid_route import make_3_1_exit_controller
from run_mario_hybrid_route import make_3_1_late_controller
from run_mario_hybrid_route import make_3_1_mid_controller
from run_mario_hybrid_route import make_3_1_opening_controller
from run_mario_hybrid_route import make_3_1_tail_controller
from run_mario_hybrid_route import make_3_2_exit_controller
from run_mario_hybrid_route import make_3_2_late_controller
from run_mario_hybrid_route import make_3_2_mid_controller
from run_mario_hybrid_route import make_3_2_opening_controller
from run_mario_hybrid_route import make_3_3_late_controller
from run_mario_hybrid_route import make_3_3_landing_controller
from run_mario_hybrid_route import make_3_3_mid_controller
from run_mario_hybrid_route import make_3_3_opening_controller
from run_mario_hybrid_route import make_3_3_exit_controller
from run_mario_hybrid_route import make_3_3_tail_controller
from run_mario_hybrid_route import make_3_4_opening_controller
from run_mario_hybrid_route import make_3_4_mid_controller
from run_mario_hybrid_route import make_3_4_exit_controller
from run_mario_hybrid_route import make_4_1_exit_controller
from run_mario_hybrid_route import make_4_1_opening_controller
from run_mario_hybrid_route import make_4_1_mid_controller
from run_mario_hybrid_route import make_4_2_late_controller
from run_mario_hybrid_route import make_4_2_mid_controller
from run_mario_hybrid_route import make_4_2_opening_controller
from run_mario_hybrid_route import make_4_2_bridge_controller
from run_mario_hybrid_route import make_4_2_exit_controller
from run_mario_hybrid_route import make_4_2_exit_approach_controller
from run_mario_hybrid_route import make_4_2_pipe_entry_controller
from run_mario_hybrid_route import make_4_2_safe_bridge_controller
from run_mario_hybrid_route import make_4_2_post_tail_controller
from run_mario_hybrid_route import make_4_2_tail_controller
from run_mario_hybrid_route import make_4_3_opening_controller
from run_mario_hybrid_route import make_4_4_lava1_controller
from run_mario_hybrid_route import make_4_4_maze1_controller
from run_mario_hybrid_route import make_4_4_maze2_controller
from run_mario_hybrid_route import make_4_4_mid_controller
from run_mario_hybrid_route import map_simple_pipe_to_complex
from run_mario_hybrid_route import predict_action
from search_mario_branch import clear_needs_reset
from search_mario_suffix import action_indices
from search_mario_suffix import downscale_frames
from search_mario_suffix import life_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search short whole-game 1-2 rescue scripts from an exact hybrid route state, "
            "then resume the existing PPO/final-pipe controller."
        )
    )
    parser.add_argument("--level11-model", required=True)
    parser.add_argument("--level12-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--use-train-noop", action="store_true")
    parser.add_argument("--noop-max", type=int, default=0)
    parser.add_argument("--initial-noops-exact", type=int, default=None)
    parser.add_argument("--branch-world", type=int, default=1)
    parser.add_argument("--branch-stage", type=int, default=2)
    parser.add_argument(
        "--branch-after-max-x",
        type=int,
        default=None,
        help="Only branch after the current stage has previously reached at least this max x.",
    )
    parser.add_argument(
        "--branch-current-x-max",
        type=int,
        default=None,
        help="Only branch when the current stage x is at most this value.",
    )
    parser.add_argument("--stop-world", type=int, default=1)
    parser.add_argument("--stop-stage", type=int, default=3)
    parser.add_argument("--branch-x", type=int, nargs="*", default=[280, 320, 360, 400, 440])
    parser.add_argument("--max-prefix-steps", type=int, default=900)
    parser.add_argument("--continue-steps", type=int, default=900)
    parser.add_argument(
        "--stall-steps",
        type=int,
        default=180,
        help="Stop a candidate early after this many steps without increasing max_stage_x; 0 disables.",
    )
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument(
        "--candidate-offset",
        type=int,
        default=0,
        help="Skip this many generated candidates before applying --max-candidates; useful for bounded batches.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Write progress.json and flush stdout after this many candidates; 0 disables periodic progress.",
    )
    parser.add_argument("--save-top-k", type=int, default=5)
    parser.add_argument("--no-video", action="store_true", help="Skip RGB frame capture during search for faster sweeps.")
    parser.add_argument(
        "--use-level12-early-controller",
        action="store_true",
        help="Use the known x>=320 1-2 rescue script while reaching later branch points.",
    )
    parser.add_argument(
        "--use-level13-early-controller",
        action="store_true",
        help="Use the known x>=140 1-3 stair60 script while reaching later branch points.",
    )
    parser.add_argument(
        "--use-level14-opening-controller",
        action="store_true",
        help="Use the known x>=160 1-4 opening script while reaching later branch points.",
    )
    parser.add_argument(
        "--use-level21-opening-controller",
        action="store_true",
        help="Use the known x>=245 2-1 opening script while reaching later branch points.",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--export-max-width", type=int, default=256)
    return parser.parse_args()


def set_rollout_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_early_scripts(ids: dict[str, int], max_candidates: int) -> list[tuple[str, list[int]]]:
    scripts: list[tuple[str, list[int]]] = [("resume_model", [])]
    # 4-4 maze route: the first puzzle loops if Mario stays on the bottom
    # corridor. Prioritize high/long jumps so targeted searches around
    # x700-800 can choose the upper path before trying generic lava-gap probes.
    for coast, jump, settle, tail in itertools.product(
        [0, 2, 4, 8, 12],
        [18, 24, 30, 36, 48, 60],
        [0, 4, 8, 12],
        [24, 48, 80, 120],
    ):
        script = (
            [ids["right_b"]] * coast
            + [ids["right_ab"]] * jump
            + [ids["right"]] * settle
            + [ids["right_b"]] * tail
        )
        scripts.append((f"toproute_rb{coast}_jab{jump}_right{settle}_tail{tail}", script))
    for coast, length, tail in itertools.product(
        [0, 4, 8],
        [80, 120, 180, 240],
        [0, 32, 64],
    ):
        pattern = [ids["right_b"]] * 3 + [ids["right_ab"]] * 9
        script = [ids["right_b"]] * coast + [pattern[i % len(pattern)] for i in range(length)] + [ids["right_b"]] * tail
        scripts.append((f"toproute_rb{coast}_stair3{length}_tail{tail}", script))
    # 4-4 starts with Mario flying into the castle's first lava-gap landing.
    # The useful action is often not an immediate jump; he needs to settle on
    # the tiny middle platform, then jump again before sliding into lava.
    for coast, wait, jump, tail in itertools.product(
        [0, 2, 4],
        [0, 2, 4, 6, 8, 10, 12, 16],
        [10, 14, 18, 22, 26, 30, 36],
        [16, 32, 48, 72],
    ):
        script = (
            [ids["right_b"]] * coast
            + [ids["noop"]] * wait
            + [ids["right_ab"]] * jump
            + [ids["right_b"]] * tail
        )
        scripts.append((f"lava1_rb{coast}_wait{wait}_jab{jump}_tail{tail}", script))
    for brake, wait, jump, tail in itertools.product(
        [1, 2, 4],
        [0, 2, 4, 6, 8, 12],
        [10, 14, 18, 24, 30],
        [16, 32, 48],
    ):
        script = (
            [ids["left"]] * brake
            + [ids["noop"]] * wait
            + [ids["right_ab"]] * jump
            + [ids["right_b"]] * tail
        )
        scripts.append((f"lava1_brake{brake}_wait{wait}_jab{jump}_tail{tail}", script))
    # 4-3 reaches the first mushroom/enemy cluster already airborne. The old
    # priority set mostly assumed a ground takeoff, so bounded searches could
    # miss the urgent "release, land, re-jump" timing around x=500-575.
    for release, jump, carry, jump2, tail in itertools.product(
        [0, 2, 4, 6, 8],
        [6, 8, 10, 12, 14, 18],
        [0, 4, 8, 12, 16, 24],
        [0, 4, 8, 12],
        [8, 16, 28, 40, 64],
    ):
        script = (
            [ids["right_b"]] * release
            + [ids["right_ab"]] * jump
            + [ids["right_b"]] * carry
            + [ids["right_ab"]] * jump2
            + [ids["right_b"]] * tail
        )
        scripts.append((f"air_rejump_rb{release}_jab{jump}_rb{carry}_jab{jump2}_tail{tail}", script))
    for brake, wait, jump, carry in itertools.product(
        [1, 2, 3, 4, 6],
        [0, 2, 4, 6, 8, 12],
        [8, 10, 12, 14, 18, 22],
        [16, 28, 40, 64],
    ):
        script = [ids["left"]] * brake + [ids["noop"]] * wait + [ids["right_ab"]] * jump + [ids["right_b"]] * carry
        scripts.append((f"air_brake{brake}_wait{wait}_jab{jump}_tail{carry}", script))
    for slow, jump, slow2, jump2, tail in itertools.product(
        [0, 2, 4, 6, 8],
        [8, 10, 12, 16],
        [4, 8, 12, 16],
        [0, 6, 10, 14],
        [16, 28, 40],
    ):
        script = (
            [ids["right"]] * slow
            + [ids["right_a"]] * jump
            + [ids["right"]] * slow2
            + [ids["right_a"]] * jump2
            + [ids["right_b"]] * tail
        )
        scripts.append((f"air_slow{slow}_ja{jump}_slow{slow2}_ja{jump2}_tail{tail}", script))
    jump_variants = [("jab", ids["right_ab"]), ("ja", ids["right_a"])]
    settle_variants = [("noop", ids["noop"]), ("left", ids["left"]), ("right", ids["right"])]
    for wait, coast, jump, (jump_name, jump_action), (settle_name, settle_action), settle, tail in itertools.product(
        [8, 12, 16],
        [8, 12],
        [14, 18, 22],
        jump_variants,
        settle_variants,
        [8, 16],
        [0, 24],
    ):
        scripts.append(
            (
                f"land_wait{wait}_rb{coast}_{jump_name}{jump}_{settle_name}{settle}_tail{tail}",
                [ids["noop"]] * wait
                + [ids["right_b"]] * coast
                + [jump_action] * jump
                + [settle_action] * settle
                + [ids["right_b"]] * tail,
            )
        )
    for wait, coast, jump1, (settle_name, settle_action), settle, jump2, tail in itertools.product(
        [8, 12, 16],
        [8, 12],
        [14, 18, 22],
        [("noop", ids["noop"]), ("right", ids["right"])],
        [4, 8, 12],
        [6, 10, 14],
        [16, 32],
    ):
        scripts.append(
            (
                f"doubleland_wait{wait}_rb{coast}_jab{jump1}_{settle_name}{settle}_jab{jump2}_tail{tail}",
                [ids["noop"]] * wait
                + [ids["right_b"]] * coast
                + [ids["right_ab"]] * jump1
                + [settle_action] * settle
                + [ids["right_ab"]] * jump2
                + [ids["right_b"]] * tail,
            )
        )
    for wait, coast, jump, tail in itertools.product(
        [60, 90, 120, 150, 180, 220],
        [0, 8, 12, 16],
        [14, 18, 24, 30],
        [0, 16, 32, 60],
    ):
        scripts.append(
            (
                f"phasewait{wait}_rb{coast}_jab{jump}_tail{tail}",
                [ids["noop"]] * wait
                + [ids["right_b"]] * coast
                + [ids["right_ab"]] * jump
                + [ids["right_b"]] * tail,
            )
        )
    phase_patterns = {
        "stair": [ids["right_b"]] * 4 + [ids["right_ab"]] * 8,
        "jump_bias": [ids["right_ab"]] * 12 + [ids["right_b"]] * 2,
        "slow_stair": [ids["right"]] * 4 + [ids["right_a"]] * 8,
    }
    for wait, pattern_name, length in itertools.product(
        [60, 90, 120, 150, 180, 220],
        ["stair", "jump_bias", "slow_stair"],
        [90, 140, 220],
    ):
        pattern = phase_patterns[pattern_name]
        scripts.append(
            (
                f"phasewait{wait}_{pattern_name}{length}",
                [ids["noop"]] * wait + [pattern[i % len(pattern)] for i in range(length)],
            )
        )
    priority_count = len(scripts)
    # 1-3 late platforms often need timing alignment, not only more speed.
    # Put wait/brake/slow timing probes before the shuffled generic library so
    # bounded searches still test them when max_candidates is small.
    for wait, coast, jump, tail in itertools.product(
        [0, 4, 8, 12, 16, 24, 36, 48],
        [0, 4, 8, 12],
        [4, 8, 12, 16, 24],
        [0, 8, 16, 24, 36],
    ):
        scripts.append(
            (
                f"wait{wait}_rb{coast}_jab{jump}_tail{tail}",
                [ids["noop"]] * wait
                + [ids["right_b"]] * coast
                + [ids["right_ab"]] * jump
                + [ids["right_b"]] * tail,
            )
        )
    for brake, wait, jump, tail in itertools.product(
        [2, 4, 6, 8, 12],
        [0, 4, 8, 16, 24],
        [6, 10, 14, 20],
        [8, 16, 28, 40],
    ):
        scripts.append(
            (
                f"brake{brake}_wait{wait}_jab{jump}_tail{tail}",
                [ids["left"]] * brake
                + [ids["noop"]] * wait
                + [ids["right_ab"]] * jump
                + [ids["right_b"]] * tail,
            )
        )
    for slow, jump, slow2, jump2, tail in itertools.product(
        [4, 8, 12, 16],
        [4, 8, 12, 16],
        [0, 4, 8, 12],
        [0, 4, 8, 12],
        [8, 16, 28],
    ):
        scripts.append(
            (
                f"slow{slow}_ja{jump}_slow{slow2}_ja{jump2}_tail{tail}",
                [ids["right"]] * slow
                + [ids["right_a"]] * jump
                + [ids["right"]] * slow2
                + [ids["right_a"]] * jump2
                + [ids["right_b"]] * tail,
            )
        )
    rng_random = random.Random(89123)
    random_actions = [
        ids["right_b"],
        ids["right_b"],
        ids["right_b"],
        ids["right_ab"],
        ids["right_ab"],
        ids["right_a"],
        ids["right"],
        ids["noop"],
        ids["left"],
    ]
    for length in [32, 48, 64, 80, 96, 128, 160]:
        for idx in range(90):
            script = [rng_random.choice(random_actions) for _ in range(length)]
            # Keep a forward bias at the tail so good landings keep converting into x progress.
            script.extend([ids["right_b"]] * 24)
            scripts.append((f"rand{length}_{idx:02d}", script))
    long_patterns = {
        "pulse": [ids["right_b"]] * 8 + [ids["right_ab"]] * 4,
        "stair": [ids["right_b"]] * 4 + [ids["right_ab"]] * 8,
        "hop": [ids["right_b"]] * 12 + [ids["right_ab"]] * 5,
        "jump_run": [ids["right_ab"]] * 10 + [ids["right_b"]] * 10,
        "slow_pulse": [ids["right"]] * 8 + [ids["right_a"]] * 5,
        "slow_stair": [ids["right"]] * 4 + [ids["right_a"]] * 8,
        "slow_jump": [ids["right_a"]] * 10 + [ids["right"]] * 10,
        "stair2": [ids["right_b"]] * 2 + [ids["right_ab"]] * 10,
        "stair1": [ids["right_b"]] + [ids["right_ab"]] * 11,
        "stair3": [ids["right_b"]] * 3 + [ids["right_ab"]] * 9,
        "jump_bias": [ids["right_ab"]] * 12 + [ids["right_b"]] * 2,
        "brake_jump": [ids["left"]] * 2 + [ids["right_ab"]] * 9 + [ids["right_b"]] * 5,
        "noop_jump": [ids["noop"]] * 2 + [ids["right_ab"]] * 9 + [ids["right_b"]] * 5,
        "land_slow": [ids["right"]] * 8 + [ids["noop"]] * 2 + [ids["right_a"]] * 8,
        "tap_brake": [ids["right_b"]] * 6 + [ids["left"]] * 1 + [ids["right_ab"]] * 8,
    }
    for pattern_name, pattern in long_patterns.items():
        for length in [60, 90, 120, 180, 240, 320]:
            script = [pattern[i % len(pattern)] for i in range(length)]
            scripts.append((f"{pattern_name}{length}", script))
    for start_coast, pattern_name, length in itertools.product(
        [0, 8, 16, 24, 36],
        [
            "pulse",
            "stair",
            "hop",
            "jump_run",
            "slow_pulse",
            "slow_stair",
            "slow_jump",
            "stair2",
            "stair1",
            "stair3",
            "jump_bias",
            "brake_jump",
            "noop_jump",
            "land_slow",
            "tap_brake",
        ],
        [90, 140, 220],
    ):
        pattern = long_patterns[pattern_name]
        script = [ids["right_b"]] * start_coast + [pattern[i % len(pattern)] for i in range(length)]
        scripts.append((f"rb{start_coast}_{pattern_name}{length}", script))
    for coast1, jump1, coast2, jump2, coast3 in itertools.product(
        [0, 4, 8, 12, 16],
        [0, 3, 5, 7, 10, 14],
        [0, 4, 8, 12, 16, 24],
        [0, 3, 5, 8, 12],
        [0, 8, 16, 24, 36],
    ):
        script = (
            [ids["right_b"]] * coast1
            + [ids["right_ab"]] * jump1
            + [ids["right_b"]] * coast2
            + [ids["right_ab"]] * jump2
            + [ids["right_b"]] * coast3
        )
        name = f"rb{coast1}_jab{jump1}_rb{coast2}_jab{jump2}_rb{coast3}"
        scripts.append((name, script))
    rng = random.Random(34567)
    fixed_first = scripts[:priority_count]
    shuffled_rest = scripts[priority_count:]
    rng.shuffle(shuffled_rest)
    scripts = fixed_first + shuffled_rest
    if max_candidates > 0:
        scripts = scripts[: max(1, int(max_candidates))]
    return scripts


def quality(ep: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        1 if ep["reached_stop"] else 0,
        stage_order_index(tuple(ep["furthest_world_stage"])),
        int(ep["max_stage_x"]),
        -int(ep["deaths"]),
        -int(ep["steps_after_branch"]),
    )


def save_episode(ep: dict[str, Any], output_dir: Path, stem: str, fps: int, export_max_width: int) -> None:
    lite = {k: v for k, v in ep.items() if k not in ("frames", "records")}
    with (output_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(lite, f, indent=2)
    with (output_dir / f"{stem}_trace.json").open("w", encoding="utf-8") as f:
        json.dump(ep["records"], f, indent=2)
    if ep.get("frames"):
        frames = downscale_frames(ep["frames"], int(export_max_width))
        save_video(frames, output_dir / f"{stem}.mp4", fps=int(fps))


def run_to_branch(
    *,
    env: Any,
    level11_model: PPO,
    level12_model: PPO,
    frame_stack: deque[np.ndarray],
    ids: dict[str, int],
    branch_x: int,
    branch_after_max_x: int | None,
    branch_current_x_max: int | None,
    branch_ws: tuple[int, int],
    deterministic: bool,
    max_prefix_steps: int,
    use_early_controller: bool,
    use_level13_early_controller: bool,
    use_level14_opening_controller: bool,
    use_level21_opening_controller: bool,
    capture_frames: bool,
    seed: int,
    initial_noops_exact: int | None,
) -> dict[str, Any] | None:
    obs, info = env.reset(seed=int(seed))
    if initial_noops_exact is not None:
        obs, info = apply_exact_initial_noops(env, obs, info, int(initial_noops_exact))
    frame_stack.clear()
    for _ in range(frame_stack.maxlen or 1):
        frame_stack.append(obs)

    cur_ws = world_stage_from_info(info, (1, 1)) or (1, 1)
    stage_x = 0
    max_stage_x = 0
    prev_life = life_count(info)
    frames = [render_rgb_frame(env)] if capture_frames else []
    records: list[dict[str, Any]] = []
    early_12 = make_1_2_early_controller(ids)
    late_12 = make_1_2_late_controller(ids)
    early_13 = make_1_3_early_controller(ids)
    mid_13 = make_1_3_mid_controller(ids)
    late_13 = make_1_3_late_controller(ids)
    gap_13 = make_1_3_gap_controller(ids)
    gap2_13 = make_1_3_gap2_controller(ids)
    gap3_13 = make_1_3_gap3_controller(ids)
    exit_13 = make_1_3_exit_controller(ids)
    opening_14 = make_1_4_opening_controller(ids)
    exit_14 = make_1_4_exit_controller(ids)
    opening_21 = make_2_1_opening_controller(ids)
    mid_21 = make_2_1_mid_controller(ids)
    late_21 = make_2_1_late_controller(ids)
    bridge_21 = make_2_1_bridge_controller(ids)
    tail_21 = make_2_1_tail_controller(ids)
    pipe_21 = make_2_1_pipe_controller(ids)
    post_pipe_21 = make_2_1_post_pipe_controller(ids)
    exit_21 = make_2_1_exit_controller(ids)
    opening_22 = make_2_2_opening_controller(ids)
    mid_22 = make_2_2_mid_controller(ids)
    late_22 = make_2_2_late_controller(ids)
    exit_22 = make_2_2_exit_controller(ids)
    opening_23 = make_2_3_opening_controller(ids)
    mid_23 = make_2_3_mid_controller(ids)
    late_23 = make_2_3_late_controller(ids)
    bridge_23 = make_2_3_bridge_controller(ids)
    tail_23 = make_2_3_tail_controller(ids)
    exit_23 = make_2_3_exit_controller(ids)
    opening_24 = make_2_4_opening_controller(ids)
    mid_24 = make_2_4_mid_controller(ids)
    exit_24 = make_2_4_exit_controller(ids)
    opening_31 = make_3_1_opening_controller(ids)
    mid_31 = make_3_1_mid_controller(ids)
    late_31 = make_3_1_late_controller(ids)
    tail_31 = make_3_1_tail_controller(ids)
    exit_31 = make_3_1_exit_controller(ids)
    opening_32 = make_3_2_opening_controller(ids)
    mid_32 = make_3_2_mid_controller(ids)
    late_32 = make_3_2_late_controller(ids)
    exit_32 = make_3_2_exit_controller(ids)
    opening_33 = make_3_3_opening_controller(ids)
    mid_33 = make_3_3_mid_controller(ids)
    late_33 = make_3_3_late_controller(ids)
    tail_33 = make_3_3_tail_controller(ids)
    landing_33 = make_3_3_landing_controller(ids)
    exit_33 = make_3_3_exit_controller(ids)
    opening_34 = make_3_4_opening_controller(ids)
    mid_34 = make_3_4_mid_controller(ids)
    exit_34 = make_3_4_exit_controller(ids)
    opening_41 = make_4_1_opening_controller(ids)
    mid_41 = make_4_1_mid_controller(ids)
    exit_41 = make_4_1_exit_controller(ids)
    opening_42 = make_4_2_opening_controller(ids)
    mid_42 = make_4_2_mid_controller(ids)
    late_42 = make_4_2_late_controller(ids)
    tail_42 = make_4_2_tail_controller(ids)
    post_tail_42 = make_4_2_post_tail_controller(ids)
    bridge_42 = make_4_2_bridge_controller(ids)
    safe_bridge_42 = make_4_2_safe_bridge_controller(ids)
    pipe_entry_42 = make_4_2_pipe_entry_controller(ids)
    exit_approach_42 = make_4_2_exit_approach_controller(ids)
    exit_42 = make_4_2_exit_controller(ids)
    opening_43 = make_4_3_opening_controller(ids)
    lava1_44 = make_4_4_lava1_controller(ids)
    mid_44 = make_4_4_mid_controller(ids)
    maze1_44 = make_4_4_maze1_controller(ids)
    maze2_44 = make_4_4_maze2_controller(ids)
    approach_12, final_12 = make_1_2_controller(ids)
    early_started = False
    early_done = False
    early_idx = 0
    late_started = False
    late_done = False
    late_idx = 0
    approach_started = False
    approach_idx = 0
    final_started = False
    final_idx = 0
    early13_started = False
    early13_done = False
    early13_idx = 0
    mid13_started = False
    mid13_done = False
    mid13_idx = 0
    late13_started = False
    late13_done = False
    late13_idx = 0
    gap13_started = False
    gap13_done = False
    gap13_idx = 0
    gap2_started = False
    gap2_done = False
    gap2_idx = 0
    gap3_started = False
    gap3_done = False
    gap3_idx = 0
    exit13_started = False
    exit13_done = False
    exit13_idx = 0
    opening14_started = False
    opening14_done = False
    opening14_idx = 0
    exit14_started = False
    exit14_done = False
    exit14_idx = 0
    opening21_started = False
    opening21_done = False
    opening21_idx = 0
    mid21_started = False
    mid21_done = False
    mid21_idx = 0
    late21_started = False
    late21_done = False
    late21_idx = 0
    bridge21_started = False
    bridge21_done = False
    bridge21_idx = 0
    tail21_started = False
    tail21_done = False
    tail21_idx = 0
    pipe21_started = False
    pipe21_done = False
    pipe21_idx = 0
    postpipe21_started = False
    postpipe21_done = False
    postpipe21_idx = 0
    exit21_started = False
    exit21_done = False
    exit21_idx = 0
    opening22_started = False
    opening22_done = False
    opening22_idx = 0
    mid22_started = False
    mid22_done = False
    mid22_idx = 0
    late22_started = False
    late22_done = False
    late22_idx = 0
    exit22_started = False
    exit22_done = False
    exit22_idx = 0
    opening23_started = False
    opening23_done = False
    opening23_idx = 0
    mid23_started = False
    mid23_done = False
    mid23_idx = 0
    late23_started = False
    late23_done = False
    late23_idx = 0
    bridge23_started = False
    bridge23_done = False
    bridge23_idx = 0
    tail23_started = False
    tail23_done = False
    tail23_idx = 0
    exit23_started = False
    exit23_done = False
    exit23_idx = 0
    opening24_started = False
    opening24_done = False
    opening24_idx = 0
    mid24_started = False
    mid24_done = False
    mid24_idx = 0
    exit24_started = False
    exit24_done = False
    exit24_idx = 0
    opening31_started = False
    opening31_done = False
    opening31_idx = 0
    mid31_started = False
    mid31_done = False
    mid31_idx = 0
    late31_started = False
    late31_done = False
    late31_idx = 0
    tail31_started = False
    tail31_done = False
    tail31_idx = 0
    exit31_started = False
    exit31_done = False
    exit31_idx = 0
    opening32_started = False
    opening32_done = False
    opening32_idx = 0
    mid32_started = False
    mid32_done = False
    mid32_idx = 0
    late32_started = False
    late32_done = False
    late32_idx = 0
    exit32_started = False
    exit32_done = False
    exit32_idx = 0
    opening33_started = False
    opening33_done = False
    opening33_idx = 0
    mid33_started = False
    mid33_done = False
    mid33_idx = 0
    late33_started = False
    late33_done = False
    late33_idx = 0
    tail33_started = False
    tail33_done = False
    tail33_idx = 0
    landing33_started = False
    landing33_done = False
    landing33_idx = 0
    exit33_started = False
    exit33_done = False
    exit33_idx = 0
    opening34_started = False
    opening34_done = False
    opening34_idx = 0
    mid34_started = False
    mid34_done = False
    mid34_idx = 0
    exit34_started = False
    exit34_done = False
    exit34_idx = 0
    opening41_started = False
    opening41_done = False
    opening41_idx = 0
    mid41_started = False
    mid41_done = False
    mid41_idx = 0
    exit41_started = False
    exit41_done = False
    exit41_idx = 0
    opening42_started = False
    opening42_done = False
    opening42_idx = 0
    mid42_started = False
    mid42_done = False
    mid42_idx = 0
    late42_started = False
    late42_done = False
    late42_idx = 0
    tail42_started = False
    tail42_done = False
    tail42_idx = 0
    posttail42_started = False
    posttail42_done = False
    posttail42_idx = 0
    bridge42_started = False
    bridge42_done = False
    bridge42_idx = 0
    safebridge42_started = False
    safebridge42_done = False
    safebridge42_idx = 0
    pipeentry42_started = False
    pipeentry42_done = False
    pipeentry42_idx = 0
    exitapproach42_started = False
    exitapproach42_done = False
    exitapproach42_idx = 0
    exit42_started = False
    exit42_done = False
    exit42_idx = 0
    opening43_started = False
    opening43_done = False
    opening43_idx = 0
    lava1_44_started = False
    lava1_44_done = False
    lava1_44_idx = 0
    mid44_started = False
    mid44_done = False
    mid44_idx = 0
    maze1_44_started = False
    maze1_44_done = False
    maze1_44_idx = 0
    maze2_44_started = False
    maze2_44_done = False
    maze2_44_idx = 0

    for step in range(1, int(max_prefix_steps) + 1):
        prev_ws = cur_ws
        if cur_ws == (1, 1):
            action = predict_action(level11_model, frame_stack, deterministic)
            mode = "ppo_1_1"
        elif cur_ws == (1, 2):
            if final_started:
                action = final_12[final_idx] if final_idx < len(final_12) else ids["right_b"]
                final_idx += 1
                mode = "script_1_2_final"
            elif stage_x >= 2380:
                final_started = True
                final_idx = 1
                action = final_12[0]
                mode = "script_1_2_final"
            elif approach_started:
                if approach_idx < len(approach_12):
                    action = approach_12[approach_idx]
                else:
                    action = map_simple_pipe_to_complex(predict_action(level12_model, frame_stack, deterministic), ids)
                approach_idx += 1
                mode = "script_1_2_approach"
            elif stage_x >= 2200:
                approach_started = True
                approach_idx = 1
                action = approach_12[0]
                mode = "script_1_2_approach"
            elif use_early_controller and late_started:
                if late_idx < len(late_12):
                    action = late_12[late_idx]
                    late_idx += 1
                    mode = "script_1_2_late"
                else:
                    late_started = False
                    late_done = True
                    action = map_simple_pipe_to_complex(predict_action(level12_model, frame_stack, deterministic), ids)
                    mode = "ppo_1_2"
            elif use_early_controller and not late_done and stage_x >= 1920:
                late_started = True
                late_idx = 1
                action = late_12[0]
                mode = "script_1_2_late"
            elif use_early_controller and early_started:
                if early_idx < len(early_12):
                    action = early_12[early_idx]
                    early_idx += 1
                    mode = "script_1_2_early"
                else:
                    early_started = False
                    early_done = True
                    action = map_simple_pipe_to_complex(predict_action(level12_model, frame_stack, deterministic), ids)
                    mode = "ppo_1_2"
            elif use_early_controller and not early_done and stage_x >= 320:
                early_started = True
                early_idx = 1
                action = early_12[0]
                mode = "script_1_2_early"
            else:
                action = map_simple_pipe_to_complex(predict_action(level12_model, frame_stack, deterministic), ids)
                mode = "ppo_1_2"
        elif cur_ws == (1, 3):
            if use_level13_early_controller and exit13_started:
                if exit13_idx < len(exit_13):
                    action = exit_13[exit13_idx]
                    exit13_idx += 1
                    mode = "script_1_3_exit"
                else:
                    exit13_started = False
                    exit13_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level13_early_controller and not exit13_done and stage_x >= 1800:
                early13_started = False
                early13_done = True
                mid13_started = False
                mid13_done = True
                late13_started = False
                late13_done = True
                gap13_started = False
                gap13_done = True
                gap2_started = False
                gap2_done = True
                gap3_started = False
                gap3_done = True
                exit13_started = True
                exit13_idx = 1
                action = exit_13[0]
                mode = "script_1_3_exit"
            elif use_level13_early_controller and gap3_started:
                if gap3_idx < len(gap3_13):
                    action = gap3_13[gap3_idx]
                    gap3_idx += 1
                    mode = "script_1_3_gap3"
                else:
                    gap3_started = False
                    gap3_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level13_early_controller and not gap3_done and 1500 <= stage_x < 2300:
                early13_started = False
                early13_done = True
                mid13_started = False
                mid13_done = True
                late13_started = False
                late13_done = True
                gap13_started = False
                gap13_done = True
                gap2_started = False
                gap2_done = True
                gap3_started = True
                gap3_idx = 1
                action = gap3_13[0]
                mode = "script_1_3_gap3"
            elif use_level13_early_controller and gap2_started:
                if gap2_idx < len(gap2_13):
                    action = gap2_13[gap2_idx]
                    gap2_idx += 1
                    mode = "script_1_3_gap2"
                else:
                    gap2_started = False
                    gap2_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level13_early_controller and not gap2_done and 1310 <= stage_x < 1800:
                early13_started = False
                early13_done = True
                mid13_started = False
                mid13_done = True
                late13_started = False
                late13_done = True
                gap13_started = False
                gap13_done = True
                gap2_started = True
                gap2_idx = 1
                action = gap2_13[0]
                mode = "script_1_3_gap2"
            elif use_level13_early_controller and gap13_started:
                if gap13_idx < len(gap_13):
                    action = gap_13[gap13_idx]
                    gap13_idx += 1
                    mode = "script_1_3_gap"
                else:
                    gap13_started = False
                    gap13_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level13_early_controller and not gap13_done and not gap2_done and 940 <= stage_x < 1310:
                early13_started = False
                early13_done = True
                mid13_started = False
                mid13_done = True
                late13_started = False
                late13_done = True
                gap13_started = True
                gap13_idx = 1
                action = gap_13[0]
                mode = "script_1_3_gap"
            elif use_level13_early_controller and late13_started:
                if late13_idx < len(late_13):
                    action = late_13[late13_idx]
                    late13_idx += 1
                    mode = "script_1_3_late"
                else:
                    late13_started = False
                    late13_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level13_early_controller and not late13_done and not gap13_done and not gap2_done and 700 <= stage_x < 940:
                early13_started = False
                early13_done = True
                mid13_started = False
                mid13_done = True
                late13_started = True
                late13_idx = 1
                action = late_13[0]
                mode = "script_1_3_late"
            elif use_level13_early_controller and mid13_started:
                if mid13_idx < len(mid_13):
                    action = mid_13[mid13_idx]
                    mid13_idx += 1
                    mode = "script_1_3_mid"
                else:
                    mid13_started = False
                    mid13_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level13_early_controller and not mid13_done and not late13_done and not gap13_done and not gap2_done and 500 <= stage_x < 700:
                early13_started = False
                early13_done = True
                mid13_started = True
                mid13_idx = 1
                action = mid_13[0]
                mode = "script_1_3_mid"
            elif use_level13_early_controller and early13_started:
                if early13_idx < len(early_13):
                    action = early_13[early13_idx]
                    early13_idx += 1
                    mode = "script_1_3_early"
                else:
                    early13_started = False
                    early13_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level13_early_controller and not early13_done and not mid13_done and not late13_done and not gap13_done and not gap2_done and 140 <= stage_x < 500:
                early13_started = True
                early13_idx = 1
                action = early_13[0]
                mode = "script_1_3_early"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (1, 4):
            if use_level14_opening_controller and exit14_started:
                if exit14_idx < len(exit_14):
                    action = exit_14[exit14_idx]
                    exit14_idx += 1
                    mode = "script_1_4_exit"
                else:
                    exit14_started = False
                    exit14_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level14_opening_controller and not exit14_done and stage_x >= 900:
                opening14_started = False
                opening14_done = True
                exit14_started = True
                exit14_idx = 1
                action = exit_14[0]
                mode = "script_1_4_exit"
            elif use_level14_opening_controller and opening14_started:
                if opening14_idx < len(opening_14):
                    action = opening_14[opening14_idx]
                    opening14_idx += 1
                    mode = "script_1_4_opening"
                else:
                    opening14_started = False
                    opening14_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level14_opening_controller and not opening14_done and stage_x >= 160:
                opening14_started = True
                opening14_idx = 1
                action = opening_14[0]
                mode = "script_1_4_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (2, 1):
            if use_level21_opening_controller and exit21_started:
                if exit21_idx < len(exit_21):
                    action = exit_21[exit21_idx]
                    exit21_idx += 1
                    mode = "script_2_1_exit"
                else:
                    exit21_started = False
                    exit21_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not exit21_done and stage_x >= 2720:
                opening21_started = False
                opening21_done = True
                mid21_started = False
                mid21_done = True
                late21_started = False
                late21_done = True
                bridge21_started = False
                bridge21_done = True
                tail21_started = False
                tail21_done = True
                pipe21_started = False
                pipe21_done = True
                postpipe21_started = False
                postpipe21_done = True
                exit21_started = True
                exit21_idx = 1
                action = exit_21[0]
                mode = "script_2_1_exit"
            elif use_level21_opening_controller and postpipe21_started:
                if postpipe21_idx < len(post_pipe_21):
                    action = post_pipe_21[postpipe21_idx]
                    postpipe21_idx += 1
                    mode = "script_2_1_post_pipe"
                else:
                    postpipe21_started = False
                    postpipe21_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not postpipe21_done and stage_x >= 2350:
                opening21_started = False
                opening21_done = True
                mid21_started = False
                mid21_done = True
                late21_started = False
                late21_done = True
                bridge21_started = False
                bridge21_done = True
                tail21_started = False
                tail21_done = True
                pipe21_started = False
                pipe21_done = True
                postpipe21_started = True
                postpipe21_idx = 1
                action = post_pipe_21[0]
                mode = "script_2_1_post_pipe"
            elif use_level21_opening_controller and pipe21_started:
                if pipe21_idx < len(pipe_21):
                    action = pipe_21[pipe21_idx]
                    pipe21_idx += 1
                    mode = "script_2_1_pipe"
                else:
                    pipe21_started = False
                    pipe21_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not pipe21_done and stage_x >= 2000:
                opening21_started = False
                opening21_done = True
                mid21_started = False
                mid21_done = True
                late21_started = False
                late21_done = True
                bridge21_started = False
                bridge21_done = True
                tail21_started = False
                tail21_done = True
                pipe21_started = True
                pipe21_idx = 1
                action = pipe_21[0]
                mode = "script_2_1_pipe"
            elif use_level21_opening_controller and tail21_started:
                if tail21_idx < len(tail_21):
                    action = tail_21[tail21_idx]
                    tail21_idx += 1
                    mode = "script_2_1_tail"
                else:
                    tail21_started = False
                    tail21_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not tail21_done and stage_x >= 1720:
                opening21_started = False
                opening21_done = True
                mid21_started = False
                mid21_done = True
                late21_started = False
                late21_done = True
                bridge21_started = False
                bridge21_done = True
                tail21_started = True
                tail21_idx = 1
                action = tail_21[0]
                mode = "script_2_1_tail"
            elif use_level21_opening_controller and bridge21_started:
                if bridge21_idx < len(bridge_21):
                    action = bridge_21[bridge21_idx]
                    bridge21_idx += 1
                    mode = "script_2_1_bridge"
                else:
                    bridge21_started = False
                    bridge21_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not bridge21_done and stage_x >= 1200:
                opening21_started = False
                opening21_done = True
                mid21_started = False
                mid21_done = True
                late21_started = False
                late21_done = True
                bridge21_started = True
                bridge21_idx = 1
                action = bridge_21[0]
                mode = "script_2_1_bridge"
            elif use_level21_opening_controller and late21_started:
                if late21_idx < len(late_21):
                    action = late_21[late21_idx]
                    late21_idx += 1
                    mode = "script_2_1_late"
                else:
                    late21_started = False
                    late21_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not late21_done and stage_x >= 855:
                opening21_started = False
                opening21_done = True
                mid21_started = False
                mid21_done = True
                late21_started = True
                late21_idx = 1
                action = late_21[0]
                mode = "script_2_1_late"
            elif use_level21_opening_controller and mid21_started:
                if mid21_idx < len(mid_21):
                    action = mid_21[mid21_idx]
                    mid21_idx += 1
                    mode = "script_2_1_mid"
                else:
                    mid21_started = False
                    mid21_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not mid21_done and stage_x >= 605:
                opening21_started = False
                opening21_done = True
                mid21_started = True
                mid21_idx = 1
                action = mid_21[0]
                mode = "script_2_1_mid"
            elif use_level21_opening_controller and opening21_started:
                if opening21_idx < len(opening_21):
                    action = opening_21[opening21_idx]
                    opening21_idx += 1
                    mode = "script_2_1_opening"
                else:
                    opening21_started = False
                    opening21_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not opening21_done and stage_x >= 245:
                opening21_started = True
                opening21_idx = 1
                action = opening_21[0]
                mode = "script_2_1_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (2, 2):
            if use_level21_opening_controller and exit22_started:
                if exit22_idx < len(exit_22):
                    action = exit_22[exit22_idx]
                    exit22_idx += 1
                    mode = "script_2_2_exit"
                else:
                    exit22_started = False
                    exit22_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not exit22_done and stage_x >= 2600:
                opening22_started = False
                opening22_done = True
                mid22_started = False
                mid22_done = True
                late22_started = False
                late22_done = True
                exit22_started = True
                exit22_idx = 1
                action = exit_22[0]
                mode = "script_2_2_exit"
            elif use_level21_opening_controller and late22_started:
                if late22_idx < len(late_22):
                    action = late_22[late22_idx]
                    late22_idx += 1
                    mode = "script_2_2_late"
                else:
                    late22_started = False
                    late22_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not late22_done and stage_x >= 2220:
                opening22_started = False
                opening22_done = True
                mid22_started = False
                mid22_done = True
                late22_started = True
                late22_idx = 1
                action = late_22[0]
                mode = "script_2_2_late"
            elif use_level21_opening_controller and mid22_started:
                if mid22_idx < len(mid_22):
                    action = mid_22[mid22_idx]
                    mid22_idx += 1
                    mode = "script_2_2_mid"
                else:
                    mid22_started = False
                    mid22_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not mid22_done and stage_x >= 1500:
                opening22_started = False
                opening22_done = True
                mid22_started = True
                mid22_idx = 1
                action = mid_22[0]
                mode = "script_2_2_mid"
            elif use_level21_opening_controller and opening22_started:
                if opening22_idx < len(opening_22):
                    action = opening_22[opening22_idx]
                    opening22_idx += 1
                    mode = "script_2_2_opening"
                else:
                    opening22_started = False
                    opening22_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not opening22_done and stage_x >= 0:
                opening22_started = True
                opening22_idx = 1
                action = opening_22[0]
                mode = "script_2_2_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (2, 3):
            if use_level21_opening_controller and exit23_started:
                if exit23_idx < len(exit_23):
                    action = exit_23[exit23_idx]
                    exit23_idx += 1
                    mode = "script_2_3_exit"
                else:
                    exit23_started = False
                    exit23_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not exit23_done and stage_x >= 3300:
                opening23_started = False
                opening23_done = True
                mid23_started = False
                mid23_done = True
                late23_started = False
                late23_done = True
                bridge23_started = False
                bridge23_done = True
                tail23_started = False
                tail23_done = True
                exit23_started = True
                exit23_idx = 1
                action = exit_23[0]
                mode = "script_2_3_exit"
            elif use_level21_opening_controller and tail23_started:
                if tail23_idx < len(tail_23):
                    action = tail_23[tail23_idx]
                    tail23_idx += 1
                    mode = "script_2_3_tail"
                else:
                    tail23_started = False
                    tail23_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not tail23_done and stage_x >= 2750:
                opening23_started = False
                opening23_done = True
                mid23_started = False
                mid23_done = True
                late23_started = False
                late23_done = True
                bridge23_started = False
                bridge23_done = True
                tail23_started = True
                tail23_idx = 1
                action = tail_23[0]
                mode = "script_2_3_tail"
            elif use_level21_opening_controller and bridge23_started:
                if bridge23_idx < len(bridge_23):
                    action = bridge_23[bridge23_idx]
                    bridge23_idx += 1
                    mode = "script_2_3_bridge"
                else:
                    bridge23_started = False
                    bridge23_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not bridge23_done and stage_x >= 2300:
                opening23_started = False
                opening23_done = True
                mid23_started = False
                mid23_done = True
                late23_started = False
                late23_done = True
                bridge23_started = True
                bridge23_idx = 1
                action = bridge_23[0]
                mode = "script_2_3_bridge"
            elif use_level21_opening_controller and late23_started:
                if late23_idx < len(late_23):
                    action = late_23[late23_idx]
                    late23_idx += 1
                    mode = "script_2_3_late"
                else:
                    late23_started = False
                    late23_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not late23_done and stage_x >= 1850:
                opening23_started = False
                opening23_done = True
                mid23_started = False
                mid23_done = True
                late23_started = True
                late23_idx = 1
                action = late_23[0]
                mode = "script_2_3_late"
            elif use_level21_opening_controller and mid23_started:
                if mid23_idx < len(mid_23):
                    action = mid_23[mid23_idx]
                    mid23_idx += 1
                    mode = "script_2_3_mid"
                else:
                    mid23_started = False
                    mid23_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not mid23_done and stage_x >= 1500:
                opening23_started = False
                opening23_done = True
                mid23_started = True
                mid23_idx = 1
                action = mid_23[0]
                mode = "script_2_3_mid"
            elif use_level21_opening_controller and opening23_started:
                if opening23_idx < len(opening_23):
                    action = opening_23[opening23_idx]
                    opening23_idx += 1
                    mode = "script_2_3_opening"
                else:
                    opening23_started = False
                    opening23_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not opening23_done and stage_x >= 80:
                opening23_started = True
                opening23_idx = 1
                action = opening_23[0]
                mode = "script_2_3_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (2, 4):
            if use_level21_opening_controller and exit24_started:
                if exit24_idx < len(exit_24):
                    action = exit_24[exit24_idx]
                    exit24_idx += 1
                    mode = "script_2_4_exit"
                else:
                    exit24_started = False
                    exit24_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not exit24_done and stage_x >= 1700:
                opening24_started = False
                opening24_done = True
                mid24_started = False
                mid24_done = True
                exit24_started = True
                exit24_idx = 1
                action = exit_24[0]
                mode = "script_2_4_exit"
            elif use_level21_opening_controller and mid24_started:
                if mid24_idx < len(mid_24):
                    action = mid_24[mid24_idx]
                    mid24_idx += 1
                    mode = "script_2_4_mid"
                else:
                    mid24_started = False
                    mid24_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not mid24_done and stage_x >= 1260:
                opening24_started = False
                opening24_done = True
                mid24_started = True
                mid24_idx = 1
                action = mid_24[0]
                mode = "script_2_4_mid"
            elif use_level21_opening_controller and opening24_started:
                if opening24_idx < len(opening_24):
                    action = opening_24[opening24_idx]
                    opening24_idx += 1
                    mode = "script_2_4_opening"
                else:
                    opening24_started = False
                    opening24_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not opening24_done and stage_x >= 220:
                opening24_started = True
                opening24_idx = 1
                action = opening_24[0]
                mode = "script_2_4_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (3, 1):
            if use_level21_opening_controller and exit31_started:
                if exit31_idx < len(exit_31):
                    action = exit_31[exit31_idx]
                    exit31_idx += 1
                    mode = "script_3_1_exit"
                else:
                    exit31_started = False
                    exit31_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not exit31_done and stage_x >= 2380:
                opening31_started = False
                opening31_done = True
                mid31_started = False
                mid31_done = True
                late31_started = False
                late31_done = True
                tail31_started = False
                tail31_done = True
                exit31_started = True
                exit31_idx = 1
                action = exit_31[0]
                mode = "script_3_1_exit"
            elif use_level21_opening_controller and tail31_started:
                if tail31_idx < len(tail_31):
                    action = tail_31[tail31_idx]
                    tail31_idx += 1
                    mode = "script_3_1_tail"
                else:
                    tail31_started = False
                    tail31_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not tail31_done and stage_x >= 1780:
                opening31_started = False
                opening31_done = True
                mid31_started = False
                mid31_done = True
                late31_started = False
                late31_done = True
                tail31_started = True
                tail31_idx = 1
                action = tail_31[0]
                mode = "script_3_1_tail"
            elif use_level21_opening_controller and late31_started:
                if late31_idx < len(late_31):
                    action = late_31[late31_idx]
                    late31_idx += 1
                    mode = "script_3_1_late"
                else:
                    late31_started = False
                    late31_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not late31_done and stage_x >= 1470:
                opening31_started = False
                opening31_done = True
                mid31_started = False
                mid31_done = True
                late31_started = True
                late31_idx = 1
                action = late_31[0]
                mode = "script_3_1_late"
            elif use_level21_opening_controller and mid31_started:
                if mid31_idx < len(mid_31):
                    action = mid_31[mid31_idx]
                    mid31_idx += 1
                    mode = "script_3_1_mid"
                else:
                    mid31_started = False
                    mid31_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not mid31_done and stage_x >= 880:
                opening31_started = False
                opening31_done = True
                mid31_started = True
                mid31_idx = 1
                action = mid_31[0]
                mode = "script_3_1_mid"
            elif use_level21_opening_controller and opening31_started:
                if opening31_idx < len(opening_31):
                    action = opening_31[opening31_idx]
                    opening31_idx += 1
                    mode = "script_3_1_opening"
                else:
                    opening31_started = False
                    opening31_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not opening31_done and stage_x >= 480:
                opening31_started = True
                opening31_idx = 1
                action = opening_31[0]
                mode = "script_3_1_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (3, 2):
            if use_level21_opening_controller and exit32_started:
                if exit32_idx < len(exit_32):
                    action = exit_32[exit32_idx]
                    exit32_idx += 1
                    mode = "script_3_2_exit"
                else:
                    exit32_started = False
                    exit32_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not exit32_done and stage_x >= 2700:
                opening32_started = False
                opening32_done = True
                mid32_started = False
                mid32_done = True
                late32_started = False
                late32_done = True
                exit32_started = True
                exit32_idx = 1
                action = exit_32[0]
                mode = "script_3_2_exit"
            elif use_level21_opening_controller and late32_started:
                if late32_idx < len(late_32):
                    action = late_32[late32_idx]
                    late32_idx += 1
                    mode = "script_3_2_late"
                else:
                    late32_started = False
                    late32_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not late32_done and stage_x >= 2000:
                opening32_started = False
                opening32_done = True
                mid32_started = False
                mid32_done = True
                late32_started = True
                late32_idx = 1
                action = late_32[0]
                mode = "script_3_2_late"
            elif use_level21_opening_controller and mid32_started:
                if mid32_idx < len(mid_32):
                    action = mid_32[mid32_idx]
                    mid32_idx += 1
                    mode = "script_3_2_mid"
                else:
                    mid32_started = False
                    mid32_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not mid32_done and stage_x >= 1200:
                opening32_started = False
                opening32_done = True
                mid32_started = True
                mid32_idx = 1
                action = mid_32[0]
                mode = "script_3_2_mid"
            elif use_level21_opening_controller and opening32_started:
                if opening32_idx < len(opening_32):
                    action = opening_32[opening32_idx]
                    opening32_idx += 1
                    mode = "script_3_2_opening"
                else:
                    opening32_started = False
                    opening32_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not opening32_done and stage_x >= 120:
                opening32_started = True
                opening32_idx = 1
                action = opening_32[0]
                mode = "script_3_2_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (3, 3):
            if use_level21_opening_controller and exit33_started:
                if exit33_idx < len(exit_33):
                    action = exit_33[exit33_idx]
                    exit33_idx += 1
                    mode = "script_3_3_exit"
                else:
                    exit33_started = False
                    exit33_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not exit33_done and stage_x >= 1950:
                opening33_started = False
                opening33_done = True
                mid33_started = False
                mid33_done = True
                late33_started = False
                late33_done = True
                tail33_started = False
                tail33_done = True
                landing33_started = False
                landing33_done = True
                exit33_started = True
                exit33_idx = 1
                action = exit_33[0]
                mode = "script_3_3_exit"
            elif use_level21_opening_controller and landing33_started:
                if landing33_idx < len(landing_33):
                    action = landing_33[landing33_idx]
                    landing33_idx += 1
                    mode = "script_3_3_landing"
                else:
                    landing33_started = False
                    landing33_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not landing33_done and stage_x >= 2050:
                opening33_started = False
                opening33_done = True
                mid33_started = False
                mid33_done = True
                late33_started = False
                late33_done = True
                tail33_started = False
                tail33_done = True
                landing33_started = True
                landing33_idx = 1
                action = landing_33[0]
                mode = "script_3_3_landing"
            elif use_level21_opening_controller and tail33_started:
                if tail33_idx < len(tail_33):
                    action = tail_33[tail33_idx]
                    tail33_idx += 1
                    mode = "script_3_3_tail"
                else:
                    tail33_started = False
                    tail33_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not tail33_done and stage_x >= 1650:
                opening33_started = False
                opening33_done = True
                mid33_started = False
                mid33_done = True
                late33_started = False
                late33_done = True
                tail33_started = True
                tail33_idx = 1
                action = tail_33[0]
                mode = "script_3_3_tail"
            elif use_level21_opening_controller and late33_started:
                if late33_idx < len(late_33):
                    action = late_33[late33_idx]
                    late33_idx += 1
                    mode = "script_3_3_late"
                else:
                    late33_started = False
                    late33_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not late33_done and stage_x >= 1100:
                opening33_started = False
                opening33_done = True
                mid33_started = False
                mid33_done = True
                late33_started = True
                late33_idx = 1
                action = late_33[0]
                mode = "script_3_3_late"
            elif use_level21_opening_controller and mid33_started:
                if mid33_idx < len(mid_33):
                    action = mid_33[mid33_idx]
                    mid33_idx += 1
                    mode = "script_3_3_mid"
                else:
                    mid33_started = False
                    mid33_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not mid33_done and stage_x >= 600:
                opening33_started = False
                opening33_done = True
                mid33_started = True
                mid33_idx = 1
                action = mid_33[0]
                mode = "script_3_3_mid"
            elif use_level21_opening_controller and opening33_started:
                if opening33_idx < len(opening_33):
                    action = opening_33[opening33_idx]
                    opening33_idx += 1
                    mode = "script_3_3_opening"
                else:
                    opening33_started = False
                    opening33_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not opening33_done and stage_x >= 200:
                opening33_started = True
                opening33_idx = 1
                action = opening_33[0]
                mode = "script_3_3_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (3, 4):
            if use_level21_opening_controller and exit34_started:
                if exit34_idx < len(exit_34):
                    action = exit_34[exit34_idx]
                    exit34_idx += 1
                    mode = "script_3_4_exit"
                else:
                    exit34_started = False
                    exit34_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not exit34_done and stage_x >= 1450:
                opening34_started = False
                opening34_done = True
                mid34_started = False
                mid34_done = True
                exit34_started = True
                exit34_idx = 1
                action = exit_34[0]
                mode = "script_3_4_exit"
            elif use_level21_opening_controller and mid34_started:
                if mid34_idx < len(mid_34):
                    action = mid_34[mid34_idx]
                    mid34_idx += 1
                    mode = "script_3_4_mid"
                else:
                    mid34_started = False
                    mid34_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not mid34_done and stage_x >= 1400:
                opening34_started = False
                opening34_done = True
                mid34_started = True
                mid34_idx = 1
                action = mid_34[0]
                mode = "script_3_4_mid"
            elif use_level21_opening_controller and opening34_started:
                if opening34_idx < len(opening_34):
                    action = opening_34[opening34_idx]
                    opening34_idx += 1
                    mode = "script_3_4_opening"
                else:
                    opening34_started = False
                    opening34_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not opening34_done and stage_x >= 160:
                opening34_started = True
                opening34_idx = 1
                action = opening_34[0]
                mode = "script_3_4_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (4, 1):
            if use_level21_opening_controller and exit41_started:
                if exit41_idx < len(exit_41):
                    action = exit_41[exit41_idx]
                    exit41_idx += 1
                    mode = "script_4_1_exit"
                else:
                    exit41_started = False
                    exit41_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not exit41_done and stage_x >= 3000:
                opening41_started = False
                opening41_done = True
                mid41_started = False
                mid41_done = True
                exit41_started = True
                exit41_idx = 1
                action = exit_41[0]
                mode = "script_4_1_exit"
            elif use_level21_opening_controller and mid41_started:
                if mid41_idx < len(mid_41):
                    action = mid_41[mid41_idx]
                    mid41_idx += 1
                    mode = "script_4_1_mid"
                else:
                    mid41_started = False
                    mid41_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not mid41_done and stage_x >= 2000:
                opening41_started = False
                opening41_done = True
                mid41_started = True
                mid41_idx = 1
                action = mid_41[0]
                mode = "script_4_1_mid"
            elif use_level21_opening_controller and opening41_started:
                if opening41_idx < len(opening_41):
                    action = opening_41[opening41_idx]
                    opening41_idx += 1
                    mode = "script_4_1_opening"
                else:
                    opening41_started = False
                    opening41_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not opening41_done and stage_x >= 220:
                opening41_started = True
                opening41_idx = 1
                action = opening_41[0]
                mode = "script_4_1_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (4, 2):
            if use_level21_opening_controller and exit42_started:
                if exit42_idx < len(exit_42):
                    action = exit_42[exit42_idx]
                    exit42_idx += 1
                    mode = "script_4_2_exit"
                else:
                    exit42_started = False
                    exit42_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not exit42_done and stage_x >= 2803:
                opening42_started = False
                opening42_done = True
                mid42_started = False
                mid42_done = True
                late42_started = False
                late42_done = True
                tail42_started = False
                tail42_done = True
                posttail42_started = False
                posttail42_done = True
                bridge42_started = False
                bridge42_done = True
                safebridge42_started = False
                safebridge42_done = True
                pipeentry42_started = False
                pipeentry42_done = True
                exitapproach42_started = False
                exitapproach42_done = True
                exit42_started = True
                exit42_idx = 1
                action = exit_42[0]
                mode = "script_4_2_exit"
            elif use_level21_opening_controller and exitapproach42_started:
                if exitapproach42_idx < len(exit_approach_42):
                    action = exit_approach_42[exitapproach42_idx]
                    exitapproach42_idx += 1
                    mode = "script_4_2_exit_approach"
                else:
                    exitapproach42_started = False
                    exitapproach42_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not exitapproach42_done and stage_x >= 2451:
                opening42_started = False
                opening42_done = True
                mid42_started = False
                mid42_done = True
                late42_started = False
                late42_done = True
                tail42_started = False
                tail42_done = True
                posttail42_started = False
                posttail42_done = True
                bridge42_started = False
                bridge42_done = True
                safebridge42_started = False
                safebridge42_done = True
                pipeentry42_started = False
                pipeentry42_done = True
                exitapproach42_started = True
                exitapproach42_idx = 1
                action = exit_approach_42[0]
                mode = "script_4_2_exit_approach"
            elif use_level21_opening_controller and pipeentry42_started:
                if pipeentry42_idx < len(pipe_entry_42):
                    action = pipe_entry_42[pipeentry42_idx]
                    pipeentry42_idx += 1
                    mode = "script_4_2_pipe_entry"
                else:
                    pipeentry42_started = False
                    pipeentry42_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not pipeentry42_done and stage_x >= 2258:
                opening42_started = False
                opening42_done = True
                mid42_started = False
                mid42_done = True
                late42_started = False
                late42_done = True
                tail42_started = False
                tail42_done = True
                posttail42_started = False
                posttail42_done = True
                bridge42_started = False
                bridge42_done = True
                safebridge42_started = False
                safebridge42_done = True
                pipeentry42_started = True
                pipeentry42_idx = 1
                action = pipe_entry_42[0]
                mode = "script_4_2_pipe_entry"
            elif use_level21_opening_controller and safebridge42_started:
                if safebridge42_idx < len(safe_bridge_42):
                    action = safe_bridge_42[safebridge42_idx]
                    safebridge42_idx += 1
                    mode = "script_4_2_safe_bridge"
                else:
                    safebridge42_started = False
                    safebridge42_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not safebridge42_done and stage_x >= 1771:
                opening42_started = False
                opening42_done = True
                mid42_started = False
                mid42_done = True
                late42_started = False
                late42_done = True
                tail42_started = False
                tail42_done = True
                posttail42_started = False
                posttail42_done = True
                bridge42_started = False
                bridge42_done = True
                safebridge42_started = True
                safebridge42_idx = 1
                action = safe_bridge_42[0]
                mode = "script_4_2_safe_bridge"
            elif use_level21_opening_controller and bridge42_started:
                if bridge42_idx < len(bridge_42):
                    action = bridge_42[bridge42_idx]
                    bridge42_idx += 1
                    mode = "script_4_2_bridge"
                else:
                    bridge42_started = False
                    bridge42_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not bridge42_done and stage_x >= 1629:
                opening42_started = False
                opening42_done = True
                mid42_started = False
                mid42_done = True
                late42_started = False
                late42_done = True
                tail42_started = False
                tail42_done = True
                posttail42_started = False
                posttail42_done = True
                bridge42_started = True
                bridge42_idx = 1
                action = bridge_42[0]
                mode = "script_4_2_bridge"
            elif use_level21_opening_controller and posttail42_started:
                if posttail42_idx < len(post_tail_42):
                    action = post_tail_42[posttail42_idx]
                    posttail42_idx += 1
                    mode = "script_4_2_post_tail"
                else:
                    posttail42_started = False
                    posttail42_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not posttail42_done and stage_x >= 1410:
                opening42_started = False
                opening42_done = True
                mid42_started = False
                mid42_done = True
                late42_started = False
                late42_done = True
                tail42_started = False
                tail42_done = True
                posttail42_started = True
                posttail42_idx = 1
                action = post_tail_42[0]
                mode = "script_4_2_post_tail"
            elif use_level21_opening_controller and tail42_started:
                if tail42_idx < len(tail_42):
                    action = tail_42[tail42_idx]
                    tail42_idx += 1
                    mode = "script_4_2_tail"
                else:
                    tail42_started = False
                    tail42_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not tail42_done and stage_x >= 1138:
                opening42_started = False
                opening42_done = True
                mid42_started = False
                mid42_done = True
                late42_started = False
                late42_done = True
                tail42_started = True
                tail42_idx = 1
                action = tail_42[0]
                mode = "script_4_2_tail"
            elif use_level21_opening_controller and late42_started:
                if late42_idx < len(late_42):
                    action = late_42[late42_idx]
                    late42_idx += 1
                    mode = "script_4_2_late"
                else:
                    late42_started = False
                    late42_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not late42_done and stage_x >= 890:
                opening42_started = False
                opening42_done = True
                mid42_started = False
                mid42_done = True
                late42_started = True
                late42_idx = 1
                action = late_42[0]
                mode = "script_4_2_late"
            elif use_level21_opening_controller and mid42_started:
                if mid42_idx < len(mid_42):
                    action = mid_42[mid42_idx]
                    mid42_idx += 1
                    mode = "script_4_2_mid"
                else:
                    mid42_started = False
                    mid42_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not mid42_done and stage_x >= 428:
                opening42_started = False
                opening42_done = True
                mid42_started = True
                mid42_idx = 1
                action = mid_42[0]
                mode = "script_4_2_mid"
            elif use_level21_opening_controller and opening42_started:
                if opening42_idx < len(opening_42):
                    action = opening_42[opening42_idx]
                    opening42_idx += 1
                    mode = "script_4_2_opening"
                else:
                    opening42_started = False
                    opening42_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not opening42_done and stage_x >= 42:
                opening42_started = True
                opening42_idx = 1
                action = opening_42[0]
                mode = "script_4_2_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (4, 3):
            if use_level21_opening_controller and opening43_started:
                if opening43_idx < len(opening_43):
                    action = opening_43[opening43_idx]
                    opening43_idx += 1
                    mode = "script_4_3_opening"
                else:
                    opening43_started = False
                    opening43_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not opening43_done and stage_x >= 161:
                opening43_started = True
                opening43_idx = 1
                action = opening_43[0]
                mode = "script_4_3_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (4, 4):
            if use_level21_opening_controller and maze2_44_started:
                if maze2_44_idx < len(maze2_44):
                    action = maze2_44[maze2_44_idx]
                    maze2_44_idx += 1
                    mode = "script_4_4_maze2"
                else:
                    maze2_44_started = False
                    maze2_44_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif (
                use_level21_opening_controller
                and not maze2_44_done
                and max_stage_x >= 1000
                and 569 <= stage_x <= 700
            ):
                maze1_44_started = False
                maze1_44_done = True
                maze2_44_started = True
                maze2_44_idx = 1
                action = maze2_44[0]
                mode = "script_4_4_maze2"
            elif use_level21_opening_controller and maze1_44_started:
                if maze1_44_idx < len(maze1_44):
                    action = maze1_44[maze1_44_idx]
                    maze1_44_idx += 1
                    mode = "script_4_4_maze1"
                else:
                    maze1_44_started = False
                    maze1_44_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif (
                use_level21_opening_controller
                and not maze1_44_done
                and max_stage_x >= 1000
                and 194 <= stage_x <= 220
            ):
                maze1_44_started = True
                maze1_44_idx = 1
                action = maze1_44[0]
                mode = "script_4_4_maze1"
            elif use_level21_opening_controller and mid44_started:
                if mid44_idx < len(mid_44):
                    action = mid_44[mid44_idx]
                    mid44_idx += 1
                    mode = "script_4_4_mid"
                else:
                    mid44_started = False
                    mid44_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not mid44_done and stage_x >= 568:
                lava1_44_started = False
                lava1_44_done = True
                mid44_started = True
                mid44_idx = 1
                action = mid_44[0]
                mode = "script_4_4_mid"
            elif use_level21_opening_controller and lava1_44_started:
                if lava1_44_idx < len(lava1_44):
                    action = lava1_44[lava1_44_idx]
                    lava1_44_idx += 1
                    mode = "script_4_4_lava1"
                else:
                    lava1_44_started = False
                    lava1_44_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif use_level21_opening_controller and not lava1_44_done and stage_x >= 127:
                lava1_44_started = True
                lava1_44_idx = 1
                action = lava1_44[0]
                mode = "script_4_4_lava1"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        else:
            action = ids["right_b"]
            mode = "default_right_b"

        obs, reward, terminated, truncated, info = env.step(int(action))
        frame_stack.append(obs)
        if capture_frames:
            frames.append(render_rgb_frame(env))
        cur_ws = world_stage_from_info(info, prev_ws) or prev_ws
        if cur_ws != prev_ws:
            stage_x = 0
            max_stage_x = 0
            early_started = False
            early_done = False
            early_idx = 0
            late_started = False
            late_done = False
            late_idx = 0
            approach_started = False
            approach_idx = 0
            final_started = False
            final_idx = 0
            early13_started = False
            early13_done = False
            early13_idx = 0
            mid13_started = False
            mid13_done = False
            mid13_idx = 0
            late13_started = False
            late13_done = False
            late13_idx = 0
            gap13_started = False
            gap13_done = False
            gap13_idx = 0
            gap2_started = False
            gap2_done = False
            gap2_idx = 0
            gap3_started = False
            gap3_done = False
            gap3_idx = 0
            exit13_started = False
            exit13_done = False
            exit13_idx = 0
            opening14_started = False
            opening14_done = False
            opening14_idx = 0
            exit14_started = False
            exit14_done = False
            exit14_idx = 0
            opening21_started = False
            opening21_done = False
            opening21_idx = 0
            mid21_started = False
            mid21_done = False
            mid21_idx = 0
            late21_started = False
            late21_done = False
            late21_idx = 0
            bridge21_started = False
            bridge21_done = False
            bridge21_idx = 0
            tail21_started = False
            tail21_done = False
            tail21_idx = 0
            pipe21_started = False
            pipe21_done = False
            pipe21_idx = 0
            postpipe21_started = False
            postpipe21_done = False
            postpipe21_idx = 0
            exit21_started = False
            exit21_done = False
            exit21_idx = 0
            opening22_started = False
            opening22_done = False
            opening22_idx = 0
            mid22_started = False
            mid22_done = False
            mid22_idx = 0
            late22_started = False
            late22_done = False
            late22_idx = 0
            exit22_started = False
            exit22_done = False
            exit22_idx = 0
            opening23_started = False
            opening23_done = False
            opening23_idx = 0
            mid23_started = False
            mid23_done = False
            mid23_idx = 0
            late23_started = False
            late23_done = False
            late23_idx = 0
            bridge23_started = False
            bridge23_done = False
            bridge23_idx = 0
            tail23_started = False
            tail23_done = False
            tail23_idx = 0
            exit23_started = False
            exit23_done = False
            exit23_idx = 0
            opening24_started = False
            opening24_done = False
            opening24_idx = 0
            mid24_started = False
            mid24_done = False
            mid24_idx = 0
            exit24_started = False
            exit24_done = False
            exit24_idx = 0
            opening31_started = False
            opening31_done = False
            opening31_idx = 0
            mid31_started = False
            mid31_done = False
            mid31_idx = 0
            late31_started = False
            late31_done = False
            late31_idx = 0
            tail31_started = False
            tail31_done = False
            tail31_idx = 0
            exit31_started = False
            exit31_done = False
            exit31_idx = 0
            opening32_started = False
            opening32_done = False
            opening32_idx = 0
            mid32_started = False
            mid32_done = False
            mid32_idx = 0
            late32_started = False
            late32_done = False
            late32_idx = 0
            exit32_started = False
            exit32_done = False
            exit32_idx = 0
            opening33_started = False
            opening33_done = False
            opening33_idx = 0
            mid33_started = False
            mid33_done = False
            mid33_idx = 0
            late33_started = False
            late33_done = False
            late33_idx = 0
            tail33_started = False
            tail33_done = False
            tail33_idx = 0
            landing33_started = False
            landing33_done = False
            landing33_idx = 0
            exit33_started = False
            exit33_done = False
            exit33_idx = 0
            opening34_started = False
            opening34_done = False
            opening34_idx = 0
            mid34_started = False
            mid34_done = False
            mid34_idx = 0
            exit34_started = False
            exit34_done = False
            exit34_idx = 0
            opening41_started = False
            opening41_done = False
            opening41_idx = 0
            mid41_started = False
            mid41_done = False
            mid41_idx = 0
            exit41_started = False
            exit41_done = False
            exit41_idx = 0
            opening42_started = False
            opening42_done = False
            opening42_idx = 0
            mid42_started = False
            mid42_done = False
            mid42_idx = 0
            late42_started = False
            late42_done = False
            late42_idx = 0
            tail42_started = False
            tail42_done = False
            tail42_idx = 0
            posttail42_started = False
            posttail42_done = False
            posttail42_idx = 0
            bridge42_started = False
            bridge42_done = False
            bridge42_idx = 0
            safebridge42_started = False
            safebridge42_done = False
            safebridge42_idx = 0
            pipeentry42_started = False
            pipeentry42_done = False
            pipeentry42_idx = 0
            exitapproach42_started = False
            exitapproach42_done = False
            exitapproach42_idx = 0
            exit42_started = False
            exit42_done = False
            exit42_idx = 0
            opening43_started = False
            opening43_done = False
            opening43_idx = 0
            lava1_44_started = False
            lava1_44_done = False
            lava1_44_idx = 0
            mid44_started = False
            mid44_done = False
            mid44_idx = 0
            maze1_44_started = False
            maze1_44_done = False
            maze1_44_idx = 0
            maze2_44_started = False
            maze2_44_done = False
            maze2_44_idx = 0
        else:
            goal_x = goal_line_x_for_world_stage(cur_ws, fallback=3200)
            stage_x = extract_sanitized_x_position(info, previous_x_pos=stage_x, goal_line_x=goal_x)
            max_stage_x = max(max_stage_x, stage_x)
        cur_life = life_count(info)
        life_lost = cur_life is not None and prev_life is not None and cur_life < prev_life
        if cur_life is not None:
            prev_life = cur_life
        records.append(
            {
                "phase": "prefix",
                "step": int(step),
                "world": cur_ws[0],
                "stage": cur_ws[1],
                "stage_x": int(stage_x),
                "max_stage_x": int(max_stage_x),
                "y": int(info.get("y_pos", 0) or 0),
                "action": int(action),
                "mode": mode,
                "life": cur_life,
                "life_lost": bool(life_lost),
                "reward": float(reward),
            }
        )
        branch_after_ok = branch_after_max_x is None or max_stage_x >= int(branch_after_max_x)
        branch_current_ok = branch_current_x_max is None or stage_x <= int(branch_current_x_max)
        if cur_ws == branch_ws and stage_x >= int(branch_x) and branch_after_ok and branch_current_ok:
            env.unwrapped._backup()
            return {
                "branch_x": int(stage_x),
                "branch_y": int(info.get("y_pos", 0) or 0),
                "world_stage": cur_ws,
                "life": cur_life,
                "frames": list(frames),
                "records": list(records),
                "frame_stack": deque(frame_stack, maxlen=frame_stack.maxlen),
            }
        if terminated or truncated or life_lost:
            return None
    return None


def run_candidate(
    *,
    env: Any,
    level12_model: PPO,
    ids: dict[str, int],
    branch_state: dict[str, Any],
    script_name: str,
    script: list[int],
    deterministic: bool,
    continue_steps: int,
    stop_ws: tuple[int, int],
    capture_frames: bool,
    stall_steps: int,
) -> dict[str, Any]:
    env.unwrapped._restore()
    clear_needs_reset(env)
    frame_stack: deque[np.ndarray] = deque(branch_state["frame_stack"], maxlen=branch_state["frame_stack"].maxlen)
    frames = list(branch_state["frames"])
    records = list(branch_state["records"])
    cur_ws = tuple(branch_state["world_stage"])
    stage_x = int(branch_state["branch_x"])
    max_stage_x = int(stage_x)
    prev_life = branch_state["life"]
    deaths = 0
    furthest_ws = cur_ws
    reached_stop = False
    last_improve_offset = 0
    approach_12, final_12 = make_1_2_controller(ids)
    approach_started = False
    approach_idx = 0
    final_started = False
    final_idx = 0

    for offset in range(1, int(continue_steps) + 1):
        prev_ws = cur_ws
        if offset <= len(script):
            action = int(script[offset - 1])
            mode = f"script_{script_name}"
        elif cur_ws == (1, 2):
            if final_started:
                action = final_12[final_idx] if final_idx < len(final_12) else ids["right_b"]
                final_idx += 1
                mode = "script_1_2_final"
            elif stage_x >= 2380:
                final_started = True
                final_idx = 1
                action = final_12[0]
                mode = "script_1_2_final"
            elif approach_started:
                if approach_idx < len(approach_12):
                    action = approach_12[approach_idx]
                else:
                    action = map_simple_pipe_to_complex(
                        predict_action(level12_model, frame_stack, deterministic),
                        ids,
                    )
                approach_idx += 1
                mode = "script_1_2_approach"
            elif stage_x >= 2200:
                approach_started = True
                approach_idx = 1
                action = approach_12[0]
                mode = "script_1_2_approach"
            else:
                action = map_simple_pipe_to_complex(predict_action(level12_model, frame_stack, deterministic), ids)
                mode = "ppo_1_2"
        else:
            action = ids["right_b"]
            mode = "default_right_b"

        obs, reward, terminated, truncated, info = env.step(int(action))
        frame_stack.append(obs)
        if capture_frames:
            frames.append(render_rgb_frame(env))
        cur_ws = world_stage_from_info(info, prev_ws) or prev_ws
        if cur_ws != prev_ws:
            stage_x = 0
            max_stage_x = 0
            approach_started = False
            approach_idx = 0
            final_started = False
            final_idx = 0
        else:
            goal_x = goal_line_x_for_world_stage(cur_ws, fallback=3200)
            stage_x = extract_sanitized_x_position(info, previous_x_pos=stage_x, goal_line_x=goal_x)
            if stage_x > max_stage_x:
                max_stage_x = stage_x
                last_improve_offset = int(offset)
        if stage_order_index(cur_ws) > stage_order_index(furthest_ws):
            furthest_ws = cur_ws
        cur_life = life_count(info)
        life_lost = cur_life is not None and prev_life is not None and cur_life < prev_life
        if life_lost:
            deaths += 1
        if cur_life is not None:
            prev_life = cur_life
        records.append(
            {
                "phase": "candidate",
                "script_name": script_name,
                "offset": int(offset),
                "world": cur_ws[0],
                "stage": cur_ws[1],
                "stage_x": int(stage_x),
                "max_stage_x": int(max_stage_x),
                "y": int(info.get("y_pos", 0) or 0),
                "action": int(action),
                "mode": mode,
                "life": cur_life,
                "life_lost": bool(life_lost),
                "reward": float(reward),
            }
        )
        if stage_order_index(cur_ws) >= stage_order_index(stop_ws):
            reached_stop = True
            break
        if int(stall_steps) > 0 and int(offset) - last_improve_offset >= int(stall_steps):
            break
        if terminated or truncated or life_lost:
            break

    return {
        "script_name": script_name,
        "script_len": len(script),
        "branch_start_x": int(branch_state["branch_x"]),
        "branch_start_y": int(branch_state["branch_y"]),
        "max_stage_x": int(max_stage_x),
        "furthest_world_stage": list(furthest_ws),
        "reached_stop": bool(reached_stop),
        "stop_world_stage": list(stop_ws),
        "deaths": int(deaths),
        "steps_after_branch": int(len(records) - len(branch_state["records"])),
        "final_world_stage": list(cur_ws),
        "final_stage_x": int(stage_x),
        "frames": frames,
        "records": records,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_rollout_seeds(int(args.seed))

    level11_model = PPO.load(args.level11_model, device="cpu")
    level12_model = PPO.load(args.level12_model, device="cpu")
    config = load_env_config_for_model(
        Path(args.level11_model),
        fallback=EnvConfig(n_envs=1, noop_max=0, end_on_flag=False),
    )
    config.n_envs = 1
    config.whole_game = True
    config.end_on_flag = False
    if args.initial_noops_exact is not None:
        config.noop_max = 0
    elif not args.use_train_noop:
        config.noop_max = int(args.noop_max)
    config.action_set = "complex"
    ids = action_indices(config.action_set)
    candidate_offset = max(0, int(args.candidate_offset))
    candidate_window = int(args.max_candidates)
    candidate_count_to_build = candidate_window
    if candidate_window > 0:
        candidate_count_to_build = candidate_offset + candidate_window
    scripts = make_early_scripts(ids, candidate_count_to_build)
    if candidate_offset > 0:
        scripts = scripts[candidate_offset:]
    if candidate_window > 0:
        scripts = scripts[:candidate_window]
    branch_ws = (int(args.branch_world), int(args.branch_stage))
    stop_ws = (int(args.stop_world), int(args.stop_stage))

    all_runs: list[dict[str, Any]] = []
    top: list[dict[str, Any]] = []
    branch_hits = 0
    best_quality: tuple[int, int, int, int, int] | None = None

    def write_progress(done: bool = False) -> None:
        best_lite = {k: v for k, v in top[0].items() if k not in ("frames", "records")} if top else {}
        progress = {
            "done": bool(done),
            "branch_hits": int(branch_hits),
            "runs": len(all_runs),
            "candidate_offset": int(candidate_offset),
            "candidate_window": int(candidate_window),
            "best": best_lite,
        }
        with (output_dir / "progress.json").open("w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)

    for target_x in args.branch_x:
        set_rollout_seeds(int(args.seed))
        env = make_single_env(config, seed=int(args.seed))
        obs, _info = env.reset(seed=int(args.seed))
        frame_stack: deque[np.ndarray] = deque(maxlen=config.frame_stack)
        for _ in range(config.frame_stack):
            frame_stack.append(obs)
        branch_state = run_to_branch(
            env=env,
            level11_model=level11_model,
            level12_model=level12_model,
            frame_stack=frame_stack,
            ids=ids,
            branch_x=int(target_x),
            branch_after_max_x=args.branch_after_max_x,
            branch_current_x_max=args.branch_current_x_max,
            branch_ws=branch_ws,
            deterministic=bool(args.deterministic),
            max_prefix_steps=int(args.max_prefix_steps),
            use_early_controller=bool(args.use_level12_early_controller),
            use_level13_early_controller=bool(args.use_level13_early_controller),
            use_level14_opening_controller=bool(args.use_level14_opening_controller),
            use_level21_opening_controller=bool(args.use_level21_opening_controller),
            capture_frames=not bool(args.no_video),
            seed=int(args.seed),
            initial_noops_exact=args.initial_noops_exact,
        )
        if branch_state is None:
            print(f"branch_x={target_x} no_branch")
            env.close()
            continue
        branch_hits += 1
        print(
            f"BRANCH target_x={target_x} actual_x={branch_state['branch_x']} "
            f"y={branch_state['branch_y']} candidates={len(scripts)}",
            flush=True,
        )
        for script_name, script in scripts:
            ep = run_candidate(
                env=env,
                level12_model=level12_model,
                ids=ids,
                branch_state=branch_state,
                script_name=script_name,
                script=script,
                deterministic=bool(args.deterministic),
                continue_steps=int(args.continue_steps),
                stop_ws=stop_ws,
                capture_frames=not bool(args.no_video),
                stall_steps=int(args.stall_steps),
            )
            lite = {k: v for k, v in ep.items() if k not in ("frames", "records")}
            all_runs.append(lite)
            top.append(ep)
            top = sorted(top, key=quality, reverse=True)[: max(1, int(args.save_top_k))]
            ep_quality = quality(ep)
            if best_quality is None or ep_quality > best_quality:
                best_quality = ep_quality
                print(
                    "IMPROVE "
                    f"runs={len(all_runs)} branch_x={lite['branch_start_x']} "
                    f"script={lite['script_name']} max_x={lite['max_stage_x']} "
                    f"furthest={lite['furthest_world_stage']} deaths={lite['deaths']} "
                    f"reached={lite['reached_stop']}",
                    flush=True,
                )
                with (output_dir / "best_progress.json").open("w", encoding="utf-8") as f:
                    json.dump(lite, f, indent=2)
            progress_every = int(args.progress_every)
            if progress_every > 0 and len(all_runs) % progress_every == 0:
                write_progress(done=False)
                print(f"PROGRESS runs={len(all_runs)}", flush=True)
            if ep["reached_stop"]:
                print("SUCCESS")
                print(json.dumps(lite, indent=2))
                break
        env.close()
        if top and top[0]["reached_stop"]:
            break

    for idx, ep in enumerate(top, start=1):
        save_episode(ep, output_dir, f"top_{idx:02d}", int(args.fps), int(args.export_max_width))
    if top:
        save_episode(top[0], output_dir, "best", int(args.fps), int(args.export_max_width))
        best_lite = {k: v for k, v in top[0].items() if k not in ("frames", "records")}
    else:
        best_lite = {}
    with (output_dir / "all_runs.json").open("w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)
    summary = {"branch_hits": int(branch_hits), "runs": len(all_runs), "best": best_lite}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_progress(done=True)
    print("SUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
