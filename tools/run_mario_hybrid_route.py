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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario_runtime import EnvConfig
from mario_runtime import extract_sanitized_x_position
from mario_runtime import goal_line_x_for_world_stage
from mario_runtime import load_env_config_for_model
from mario_runtime import make_single_env
from mario_runtime import render_rgb_frame
from mario_runtime import save_video
from mario_runtime import stage_order_index
from mario_runtime import world_stage_from_info
from search_mario_suffix import action_indices
from search_mario_suffix import downscale_frames
from search_mario_suffix import life_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a whole-game hybrid Mario route: PPO policies for broad movement plus "
            "verified scripted guards for brittle level exits."
        )
    )
    parser.add_argument("--level11-model", required=True)
    parser.add_argument("--level12-model", required=True)
    parser.add_argument(
        "--level43-model",
        default=None,
        help="Optional focused 4-3 policy. When omitted, use the current scripted 4-3 opening baseline.",
    )
    parser.add_argument(
        "--level43-start-noops",
        type=int,
        default=0,
        help="Run this many NOOP actions at the start of 4-3 before the optional focused 4-3 policy.",
    )
    parser.add_argument(
        "--level13-policy",
        choices=("right_b", "level11"),
        default="right_b",
        help="Fallback policy for 1-3 after scripted guards; level11 reuses the overworld PPO.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--export-max-width", type=int, default=512)
    parser.add_argument("--no-video", action="store_true", help="Skip RGB rendering/video output for fast route sweeps.")
    parser.add_argument("--use-train-noop", action="store_true")
    parser.add_argument("--noop-max", type=int, default=0)
    parser.add_argument(
        "--initial-noops-exact",
        type=int,
        default=None,
        help="Disable random reset noops and run exactly this many NOOP actions before the route starts.",
    )
    parser.add_argument("--stop-world", type=int, default=1)
    parser.add_argument("--stop-stage", type=int, default=3)
    parser.add_argument("--stop-on-death", action="store_true")
    parser.add_argument(
        "--level12-extra-jump-hold",
        type=int,
        default=0,
        help="Repeat 1-2 right+jump actions this many extra env steps; 0 preserves the best whole-game baseline.",
    )
    parser.add_argument(
        "--reset-stack-on-stage-change",
        action="store_true",
        help="Reinitialize the frame stack after a stage transition so the next stage policy sees clean stage frames.",
    )
    parser.add_argument(
        "--demo-output",
        default=None,
        help=(
            "Optional .npz path for behavior-cloning data. Saves stacked observations, "
            "chosen complex-action ids, world/stage/x metadata, and controller mode labels."
        ),
    )
    return parser.parse_args()


def set_rollout_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pulse_jump_tail(ids: dict[str, int], length: int) -> list[int]:
    pattern = [ids["right_b"]] * 8 + [ids["right_ab"]] * 4
    return [pattern[i % len(pattern)] for i in range(int(length))]


def make_1_2_controller(ids: dict[str, int]) -> tuple[list[int], list[int]]:
    # Found by exact-state branch search:
    # approach_name=rb16_jab5_rb8_d12, final_name=rb8_jab4_rb2_down20_tail220_pulse_jump.
    approach = (
        [ids["right_b"]] * 16
        + [ids["right_ab"]] * 5
        + [ids["right_b"]] * 8
        + [ids["down"]] * 12
    )
    final = (
        [ids["right_b"]] * 8
        + [ids["right_ab"]] * 4
        + [ids["right_b"]] * 2
        + [ids["down"]] * 20
        + pulse_jump_tail(ids, 220)
    )
    return approach, final


def make_1_2_early_controller(ids: dict[str, int]) -> list[int]:
    # Found by whole-game branch search from x>=320:
    # rb0_jab0_rb8_jab5_rb16 gets past the first stair/goomba cluster.
    return [ids["right_b"]] * 8 + [ids["right_ab"]] * 5 + [ids["right_b"]] * 16


def make_1_2_late_controller(ids: dict[str, int]) -> list[int]:
    # Found by whole-game branch search from x>=1920:
    # rb0_jab0_rb0_jab3_rb16 preserves the pipe room route into the final controller.
    return [ids["right_ab"]] * 3 + [ids["right_b"]] * 16


def make_1_3_early_controller(ids: dict[str, int]) -> list[int]:
    # Found by whole-game branch search from x>=140:
    # stair60 gets through the first 1-3 platform/gap section to about x=770.
    pattern = [ids["right_b"]] * 4 + [ids["right_ab"]] * 8
    return [pattern[i % len(pattern)] for i in range(60)]


def make_1_3_mid_controller(ids: dict[str, int]) -> list[int]:
    # Found by whole-game branch search from x>=500:
    # tap_brake60 gets past the next 1-3 platform gap to about x=922.
    pattern = [ids["right_b"]] * 6 + [ids["left"]] + [ids["right_ab"]] * 8
    return [pattern[i % len(pattern)] for i in range(60)]


def make_1_3_late_controller(ids: dict[str, int]) -> list[int]:
    # Found by whole-game branch search from x>=700:
    # stair3120 gets through the next 1-3 platform series to about x=1259.
    pattern = [ids["right_b"]] * 3 + [ids["right_ab"]] * 9
    return [pattern[i % len(pattern)] for i in range(120)]


def make_1_3_gap_controller(ids: dict[str, int]) -> list[int]:
    # Found by exact-state branch search from x>=940:
    # rb8_jab3_rb4_jab12_rb24 carries the platform gap route to about x=1495.
    return [ids["right_b"]] * 8 + [ids["right_ab"]] * 3 + [ids["right_b"]] * 4 + [ids["right_ab"]] * 12 + [ids["right_b"]] * 24


def make_1_3_gap2_controller(ids: dict[str, int]) -> list[int]:
    # Found by exact-state branch search from x>=1310:
    # rb4_jab5_rb8_jab8_rb0 extends the 1-3 route to about x=1737.
    return [ids["right_b"]] * 4 + [ids["right_ab"]] * 5 + [ids["right_b"]] * 8 + [ids["right_ab"]] * 8


def make_1_3_gap3_controller(ids: dict[str, int]) -> list[int]:
    # Found by exact-state random branch search from x>=1500:
    # rand32_81 extends the 1-3 route to about x=1962.
    return [
        ids["right_b"],
        ids["right_a"],
        ids["noop"],
        ids["right_ab"],
        ids["right_b"],
        ids["right"],
        ids["right_ab"],
        ids["right"],
        ids["right_a"],
        ids["right_ab"],
        ids["right_ab"],
        ids["right_b"],
        ids["left"],
        ids["noop"],
        ids["right_b"],
        ids["right_b"],
        ids["noop"],
        ids["right_ab"],
        ids["left"],
        ids["right"],
        ids["left"],
        ids["right_ab"],
        ids["right_ab"],
        ids["noop"],
        ids["right_ab"],
        ids["right_ab"],
        ids["left"],
        ids["left"],
        ids["right_b"],
        ids["right_ab"],
        ids["right_b"],
        ids["right"],
    ] + [ids["right_b"]] * 24


def make_1_3_exit_controller(ids: dict[str, int]) -> list[int]:
    # Found by exact-state branch search from x>=1800:
    # rb8_jump_bias140 exits 1-3 cleanly to 1-4.
    pattern = [ids["right_ab"]] * 12 + [ids["right_b"]] * 2
    return [ids["right_b"]] * 8 + [pattern[i % len(pattern)] for i in range(140)]


def make_1_4_opening_controller(ids: dict[str, int]) -> list[int]:
    # Found by exact-state branch search from x>=160:
    # rand128_87 carries 1-4 past the opening lava pits/fire bars to about x=1216.
    names = [
        "right_a", "noop", "right_a", "right", "right", "right_b", "right", "left",
        "right_b", "right_ab", "noop", "right_b", "left", "left", "right_a", "right_b",
        "right_b", "right_b", "right_b", "left", "right_b", "right", "right_a",
        "right_a", "right_b", "right_ab", "right_b", "right_ab", "left", "right_ab",
        "right_a", "right_ab", "left", "right_b", "right", "right", "right_ab",
        "right", "right_b", "noop", "right_b", "right_b", "right_b", "right_ab",
        "right_ab", "right_b", "right_b", "right_b", "right_b", "left", "right_b",
        "right_b", "right_b", "right_ab", "right_ab", "right_b", "right_b", "right",
        "right_b", "right_a", "right_b", "right_b", "right_a", "noop", "right_b",
        "noop", "right_b", "right_ab", "right", "right_ab", "left", "right_b",
        "right", "right_ab", "right_a", "right_b", "left", "left", "left",
        "right_ab", "left", "right_a", "noop", "right_b", "right", "right_b",
        "right_b", "right_ab", "left", "right_b", "right", "right_b", "right_b",
        "right_b", "right_b", "right_ab", "right_ab", "right", "right_b", "right_b",
        "right_ab", "right_b", "right", "right_a", "left", "right", "right_b",
        "right", "right_b", "right_ab", "right", "right_ab", "right_b", "right_a",
        "noop",
    ]
    return [ids[name] for name in names]


def make_1_4_exit_controller(ids: dict[str, int]) -> list[int]:
    # Found by exact-state branch search from x>=901:
    # stair3240 exits 1-4 cleanly to 2-1.
    pattern = [ids["right_b"]] * 3 + [ids["right_ab"]] * 9
    return [pattern[i % len(pattern)] for i in range(240)]


def make_2_1_opening_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=245:
    # rb0_hop140 gets past the first 2-1 goomba/stair obstacle to about x=641.
    pattern = [ids["right_b"]] * 12 + [ids["right_ab"]] * 5
    return [pattern[i % len(pattern)] for i in range(140)]


def make_2_1_mid_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=605:
    # rand32_51 extends 2-1 progress to about x=915.
    names = [
        "left", "noop", "right_ab", "noop", "right_ab", "noop", "right", "noop",
        "left", "left", "right_a", "right_b", "left", "right_b", "left",
        "right_ab", "right_ab", "right_ab", "right_ab", "right_ab", "noop",
        "right_b", "right_ab", "right_a", "right_b", "right_a", "right_ab",
        "right_a", "right_ab", "right_ab", "right_b", "right_ab", "right_b",
        "right_b", "right_b", "right_b", "right_b", "right_b", "right_b",
        "right_b", "right_b", "right_b", "right_b",
    ]
    return [ids[name] for name in names]


def make_2_1_late_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=855:
    # rb4_jab10_rb8_jab12_rb16 extends 2-1 progress to about x=1359.
    return (
        [ids["right_b"]] * 4
        + [ids["right_ab"]] * 10
        + [ids["right_b"]] * 8
        + [ids["right_ab"]] * 12
        + [ids["right_b"]] * 16
    )


def make_2_1_bridge_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=1203:
    # rb8_jump_bias140 extends 2-1 progress to about x=1827.
    pattern = [ids["right_ab"]] * 12 + [ids["right_b"]] * 2
    return [ids["right_b"]] * 8 + [pattern[i % len(pattern)] for i in range(140)]


def make_2_1_tail_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=1727:
    # rb0_stair390 reaches x=2067 without dying, but still needs an exit segment.
    pattern = [ids["right_b"]] * 3 + [ids["right_ab"]] * 9
    return [pattern[i % len(pattern)] for i in range(90)]


def make_2_1_pipe_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=2004:
    # rand48_11 gets through the pipe/plant cluster to about x=2460.
    names = [
        "noop", "right_ab", "left", "right_ab", "right_a", "right_b", "right_b",
        "left", "right", "right_b", "right_b", "right_ab", "right", "right",
        "right_b", "right_a", "right_b", "right_ab", "left", "right", "right_ab",
        "right_b", "right_ab", "noop", "right_b", "right_b", "right_b", "right_b",
        "right_a", "right", "left", "right", "noop", "right_a", "left", "right_b",
        "noop", "left", "right", "right_b", "left", "right_b", "noop", "right",
        "right",
    ]
    return [ids[name] for name in names]


def make_2_1_post_pipe_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=2360:
    # slow8_ja8_slow4_ja8_tail28 reaches about x=2803 without dying.
    return (
        [ids["right"]] * 8
        + [ids["right_a"]] * 8
        + [ids["right"]] * 4
        + [ids["right_a"]] * 8
        + [ids["right_b"]] * 28
    )


def make_2_1_exit_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=2729:
    # rand128_80 exits 2-1 cleanly to 2-2.
    names = [
        "noop", "right_ab", "right_ab", "right_b", "left", "right", "right_a",
        "noop", "right_a", "right_b", "left", "right_ab", "right_b", "right_ab",
        "right", "right_b", "right_b", "right_a", "right_b", "noop", "noop",
        "left", "right", "right", "right_ab", "right", "noop", "right_b",
        "right_b", "right_b", "right", "right_b", "left", "right_ab", "noop",
        "right_b", "right_b", "right_a", "right_b", "right_a", "right_ab",
        "right_b", "right", "left", "noop", "right_a", "right_a", "left",
        "left", "right_ab", "right_b", "right_ab", "right_b", "left", "right_a",
        "right_a", "right_ab", "right_b", "noop", "right_b", "right_ab",
        "right_a", "right_ab", "noop", "left", "right_ab", "left", "right_b",
        "right", "right", "right_b", "right_b", "right_b", "right", "right_b",
        "right_b", "right_b", "right_a", "right_b", "right_ab", "right",
        "right_b", "left", "right_ab", "right", "right_ab", "noop", "right",
        "right", "right_b", "right_b", "right_ab", "noop", "right_b", "left",
        "right_b", "right_b", "left", "right", "right_b", "right_b", "right_b",
        "right", "right_ab", "left", "right_b", "right", "left", "right_b",
        "right_a", "left", "noop",
    ]
    return [ids[name] for name in names]


def make_2_2_opening_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from 2-2 start:
    # stair3240 reaches about x=1810 in the underwater stage.
    pattern = [ids["right_b"]] * 3 + [ids["right_ab"]] * 9
    return [pattern[i % len(pattern)] for i in range(240)]


def make_2_2_mid_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=1501:
    # rb16_slow_stair220 extends underwater 2-2 progress to about x=2226.
    pattern = [ids["right"]] * 4 + [ids["right_a"]] * 8
    return [ids["right_b"]] * 16 + [pattern[i % len(pattern)] for i in range(220)]


def make_2_2_late_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=2224:
    # rand96_31 extends underwater 2-2 progress to about x=2754.
    names = [
        "right_b", "right_a", "noop", "right_a", "right_b", "right_b", "right_b",
        "right_ab", "right_ab", "right_b", "noop", "left", "right_b", "noop",
        "right_b", "right_b", "noop", "right_ab", "right_a", "right_ab", "right_a",
        "right_b", "right_b", "right_b", "right_b", "left", "right_b", "right_ab",
        "right", "right_a", "right_ab", "right_b", "right_ab", "right_ab", "left",
        "left", "noop", "right_b", "right", "right_a", "right_ab", "right_a",
        "right_ab", "right", "noop", "right_a", "right", "right_a", "right_b",
        "right_b", "right_ab", "right_b", "left", "right_a", "right_a", "right_b",
        "right", "noop", "right_a", "right_b", "noop", "right_b", "right_ab",
        "right", "left", "noop", "right_b", "right_ab", "right_b", "right_b",
        "right_b", "noop", "right_ab", "left", "noop", "right_b", "noop",
        "right_b", "noop", "right_b", "right", "right_a", "right_b", "right_ab",
        "noop", "noop", "right_ab", "noop", "right_b", "right_b", "right_a",
        "right_ab", "noop", "noop", "right_ab", "noop", "right_b", "right_b",
        "right_b", "right_b", "right_b", "right_b", "right_b", "right_b",
        "right_b", "right_b", "right_b", "right_b", "right_b", "right_b",
        "right_b", "right_b", "right_b", "right_b", "right_b", "right_b",
        "right_b", "right_b", "right_b", "right_b",
    ]
    return [ids[name] for name in names]


def make_2_2_exit_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=2600:
    # slow_jump180 exits underwater 2-2 cleanly to 2-3.
    pattern = [ids["right_a"]] * 10 + [ids["right"]] * 10
    return [pattern[i % len(pattern)] for i in range(180)]


def make_2_3_opening_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=89:
    # rb0_stair2140 gets through the opening 2-3 fish/gap section to about x=1787.
    pattern = [ids["right_b"]] * 2 + [ids["right_ab"]] * 10
    return [pattern[i % len(pattern)] for i in range(140)]


def make_2_3_mid_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=1511:
    # rb0_jab5_rb4_jab8_rb24 extends 2-3 progress to about x=1979.
    return [ids["right_ab"]] * 5 + [ids["right_b"]] * 4 + [ids["right_ab"]] * 8 + [ids["right_b"]] * 24


def make_2_3_late_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=1859:
    # slow4_ja4_slow4_ja8_tail16 extends 2-3 progress to about x=2466.
    return [ids["right"]] * 4 + [ids["right_a"]] * 4 + [ids["right"]] * 4 + [ids["right_a"]] * 8 + [ids["right_b"]] * 16


def make_2_3_bridge_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=2310:
    # rb8_jab7_rb16_jab12_rb16 extends 2-3 progress to about x=2898.
    return (
        [ids["right_b"]] * 8
        + [ids["right_ab"]] * 7
        + [ids["right_b"]] * 16
        + [ids["right_ab"]] * 12
        + [ids["right_b"]] * 16
    )


def make_2_3_tail_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=2754:
    # rand128_80 extends 2-3 progress to about x=3410.
    return make_2_1_exit_controller(ids)


def make_2_3_exit_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=3301:
    # rb0_stair390 climbs the final staircase and reaches 2-4.
    return make_2_1_tail_controller(ids)


def make_named_rand_controller(ids: dict[str, int], length: int, target_idx: int) -> list[int]:
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
    for candidate_length in [32, 48, 64, 80, 96, 128, 160]:
        for idx in range(90):
            script = [rng_random.choice(random_actions) for _ in range(candidate_length)]
            script.extend([ids["right_b"]] * 24)
            if candidate_length == int(length) and idx == int(target_idx):
                return script
    raise ValueError(f"Unknown rand controller rand{length}_{target_idx:02d}")


def make_2_4_opening_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=223:
    # rand160_36 gets through the first 2-4 castle hazards to about x=1472.
    return make_named_rand_controller(ids, 160, 36)


def make_2_4_mid_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=1260:
    # rb0_stair390 extends the 2-4 castle route to about x=1767.
    return make_2_1_tail_controller(ids)


def make_2_4_exit_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=1710:
    # rand80_69 reaches 3-1 from the castle mid-route.
    return make_named_rand_controller(ids, 80, 69)


def make_repeated_pattern(pattern: list[int], length: int) -> list[int]:
    return [pattern[i % len(pattern)] for i in range(int(length))]


def make_3_1_opening_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=480:
    # noop_jump240 clears the first pipe/enemy trap and reaches about x=983.
    pattern = [ids["noop"]] * 2 + [ids["right_ab"]] * 9 + [ids["right_b"]] * 5
    return make_repeated_pattern(pattern, 240)


def make_3_1_mid_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=886:
    # rand128_87 extends 3-1 progress to about x=1479.
    return make_named_rand_controller(ids, 128, 87)


def make_3_1_late_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=1471:
    # rand128_14 extends 3-1 progress to about x=1801.
    return make_named_rand_controller(ids, 128, 14)


def make_3_1_tail_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=1789:
    # rand80_57 extends 3-1 progress to about x=2431.
    return make_named_rand_controller(ids, 80, 57)


def make_3_1_exit_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=2383:
    # stair3240 reaches 3-2 from the 3-1 tail route.
    pattern = [ids["right_b"]] * 3 + [ids["right_ab"]] * 9
    return make_repeated_pattern(pattern, 240)


def make_3_2_opening_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=125:
    # slow_jump180 clears the first 3-2 hazards and reaches about x=1303.
    return make_2_2_exit_controller(ids)


def make_3_2_mid_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=1201:
    # rand128_29 extends 3-2 progress to about x=2069.
    return make_named_rand_controller(ids, 128, 29)


def make_3_2_late_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=2003:
    # rb0_hop140 extends 3-2 progress to about x=2977.
    return make_2_1_opening_controller(ids)


def make_3_2_exit_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=2703:
    # rb8_jump_bias140 reaches 3-3 from the 3-2 tail route.
    return make_1_3_exit_controller(ids)


def make_3_3_opening_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=209:
    # slow4_ja8_slow4_ja12_tail8 reaches about x=760.
    return [ids["right"]] * 4 + [ids["right_a"]] * 8 + [ids["right"]] * 4 + [ids["right_a"]] * 12 + [ids["right_b"]] * 8


def make_3_3_mid_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=604:
    # rb0_slow_stair220 extends 3-3 progress to about x=1336.
    pattern = [ids["right"]] * 4 + [ids["right_a"]] * 8
    return make_repeated_pattern(pattern, 220)


def make_3_3_late_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=1105:
    # rb8_stair3140 extends 3-3 progress to about x=1815.
    pattern = [ids["right_b"]] * 3 + [ids["right_ab"]] * 9
    return [ids["right_b"]] * 8 + make_repeated_pattern(pattern, 140)


def make_3_3_tail_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=1659:
    # rb8_jump_bias140 extends 3-3 progress to about x=2223.
    return make_1_3_exit_controller(ids)


def make_3_3_landing_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=2055:
    # brake12_wait8_jab14_tail28 extends 3-3 progress to about x=2249.
    return [ids["left"]] * 12 + [ids["noop"]] * 8 + [ids["right_ab"]] * 14 + [ids["right_b"]] * 28


def make_3_3_exit_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=1959:
    # rand80_83 reaches 3-4 cleanly from the late 3-3 platform route.
    return make_named_rand_controller(ids, 80, 83)


def make_3_4_opening_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=170:
    # rb0_slow_stair220 clears the 3-4 opening to about x=1582.
    return make_3_3_mid_controller(ids)


def make_3_4_mid_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=1400:
    # rand96_35 extends 3-4 progress to about x=1674.
    return make_named_rand_controller(ids, 96, 35)


def make_3_4_exit_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=1450:
    # slow_pulse240 reaches 4-1 cleanly from the 3-4 castle mid-route.
    pattern = [ids["right"]] * 8 + [ids["right_a"]] * 5
    return make_repeated_pattern(pattern, 240)


def make_4_1_opening_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=224:
    # rb24_noop_jump220 clears the opening pipe/spike sequence to about x=2107.
    pattern = [ids["noop"]] * 2 + [ids["right_ab"]] * 9 + [ids["right_b"]] * 5
    return [ids["right_b"]] * 24 + make_repeated_pattern(pattern, 220)


def make_4_1_mid_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=2002:
    # rb36_jump_run220 extends 4-1 progress to about x=3410.
    pattern = [ids["right_ab"]] * 10 + [ids["right_b"]] * 10
    return [ids["right_b"]] * 36 + make_repeated_pattern(pattern, 220)


def make_4_1_exit_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=3009:
    # rb0_stair390 reaches 4-2 cleanly from the 4-1 mid-route.
    return make_2_1_tail_controller(ids)


def make_4_2_opening_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=42:
    # rb8_jab3_rb8_jab12_rb16 clears the first 4-2 underground gap.
    return [ids["right_b"]] * 8 + [ids["right_ab"]] * 3 + [ids["right_b"]] * 8 + [ids["right_ab"]] * 12 + [ids["right_b"]] * 16


def make_4_2_mid_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=428:
    # rand32_41 reached the next 4-2 section near x=974 in the exact route state.
    names = [
        "right", "right_b", "right_b", "right_ab", "right_ab", "right_ab",
        "right_ab", "right_b", "left", "right_ab", "right_ab", "right_a",
        "right_b", "right_b", "left", "right", "right_b", "right_ab",
        "right_ab", "right_b", "right_b", "right_b", "right_b", "noop",
        "left", "right", "right_ab", "left", "right_b", "noop", "right_ab",
        "right_ab",
    ]
    return [ids[name] for name in names] + [ids["right_b"]] * 24


def make_4_2_late_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=890:
    # rb0_jab14_rb8_jab0_rb24 reaches x=1138 with zero deaths, preserving a safe 4-2 prefix.
    return [ids["right_ab"]] * 14 + [ids["right_b"]] * 32


def make_4_2_tail_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=1138:
    # slow12_ja8_slow4_ja8_tail28 reaches x=1410 with zero deaths.
    return [ids["right"]] * 12 + [ids["right_a"]] * 8 + [ids["right"]] * 4 + [ids["right_a"]] * 8 + [ids["right_b"]] * 28


def make_4_2_post_tail_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=1410:
    # rb8_jab3_rb16_jab3_rb8 reaches x=1652 with zero deaths.
    return [ids["right_b"]] * 8 + [ids["right_ab"]] * 3 + [ids["right_b"]] * 16 + [ids["right_ab"]] * 3 + [ids["right_b"]] * 8


def make_4_2_bridge_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=1629:
    # brake2_wait16_jab14_tail40 reaches about x=1891; later search must replace its death tail.
    return [ids["left"]] * 2 + [ids["noop"]] * 16 + [ids["right_ab"]] * 14 + [ids["right_b"]] * 40


def make_4_2_safe_bridge_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=1771:
    # rand64_04 reaches x=2258 with zero deaths.
    names = [
        "right_b", "right_ab", "noop", "noop", "right_b", "noop", "right_b",
        "right_a", "right_b", "left", "right_b", "right", "right_b", "left",
        "right_ab", "right_b", "left", "noop", "right_b", "right_ab", "right",
        "right_b", "right_a", "right_a", "right_b", "right_b", "right_ab",
        "left", "right_b", "right_ab", "right_ab", "right_ab", "right",
        "right_ab", "right", "noop", "right_b", "right_b", "left", "right_b",
        "right_ab", "right_ab", "right_b", "right_b", "left", "right_b",
        "right", "right", "noop", "noop", "right_ab", "noop", "left",
        "right_a", "right_ab", "right_b", "right_ab", "right_b", "right_ab",
        "right_b", "right_a", "noop", "right_b", "right_a",
    ]
    return [ids[name] for name in names] + [ids["right_b"]] * 24


def make_4_2_pipe_entry_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=2258:
    # rb8_jab10_rb8_jab3_rb8 reaches x=2451 with zero deaths.
    return [ids["right_b"]] * 8 + [ids["right_ab"]] * 10 + [ids["right_b"]] * 8 + [ids["right_ab"]] * 3 + [ids["right_b"]] * 8


def make_4_2_exit_approach_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=2451:
    # rand48_41 reaches x=2803 with zero deaths.
    names = [
        "right_a", "right_b", "right_b", "right", "right_a", "right_b",
        "right_ab", "right_b", "right", "right_b", "right_ab", "right_b",
        "right_b", "right_b", "left", "right_ab", "noop", "right_b", "noop",
        "right_ab", "right_b", "right_ab", "left", "right_b", "right_ab",
        "right_ab", "right_a", "right", "right_ab", "right_ab", "left",
        "right_a", "right_b", "left", "right_b", "noop", "left", "left",
        "right_b", "right_a", "right_b", "right_a", "right_a", "right_b",
        "right_b", "right_ab", "right_ab", "right_b",
    ]
    return [ids[name] for name in names] + [ids["right_b"]] * 24


def make_4_2_exit_controller(ids: dict[str, int]) -> list[int]:
    # Found by bounded branch search from x>=2803:
    # rand128_36 exits 4-2 cleanly to 4-3 with zero deaths.
    names = [
        "right_b", "left", "noop", "right_b", "right_ab", "right_b",
        "right_ab", "right_b", "right_b", "right_a", "right_ab", "right_a",
        "right_ab", "right_b", "right_ab", "right_a", "right", "right_b",
        "right", "right_a", "right_b", "right", "right_b", "right_a",
        "right_b", "right", "noop", "right_b", "right", "left", "right_b",
        "noop", "right_ab", "left", "right_b", "right_b", "right_b",
        "right_ab", "right_ab", "right_a", "right_ab", "right_b", "right_ab",
        "right_b", "left", "noop", "right", "left", "noop", "right_b",
        "right_ab", "left", "right", "right_b", "right", "left", "right_ab",
        "right_a", "left", "noop", "right_ab", "right_ab", "right_b",
        "right_b", "right_ab", "right_ab", "right_b", "right_ab", "left",
        "right_b", "right_b", "right_b", "right_b", "right_b", "right_b",
        "right_b", "right_b", "right_b", "right_ab", "right_b", "right_b",
        "right", "noop", "right_ab", "right_b", "right", "right_ab",
        "right_ab", "right_b", "left", "right_b", "right_ab", "right_b",
        "right_b", "right_b", "left", "right_a", "right_b", "right_b",
        "right_ab", "right_b", "left", "right_b", "right_ab", "right_a",
        "right_ab", "right_ab", "left", "right_ab", "right", "left", "left",
        "right_a", "right_ab", "right_ab", "right", "right", "right_b",
        "right_ab", "right_b", "right_a", "right_b", "right_b", "left",
        "right", "right", "right_ab", "right_b",
    ]
    return [ids[name] for name in names] + [ids["right_b"]] * 24


def make_4_3_opening_controller(ids: dict[str, int]) -> list[int]:
    # Found by exact-state branch search from 4-3 x>=323:
    # air_rejump_rb0_jab8_rb4_jab8_tail8 improves the opening mushroom
    # section from x575 to about x750 in the current full-chain route state.
    # Then x>=654 air_rejump_rb2_jab6_rb4_jab4_tail8 extends the same
    # route to about x901.
    # Then x>=857 air_rejump_rb0_jab8_rb4_jab4_tail8 extends it to
    # about x1289.
    # Then x>=1013 air_rejump_rb0_jab6_rb4_jab8_tail8 extends it to
    # about x1553.
    # Then x>=1385 air_rejump_rb0_jab6_rb8_jab4_tail8 extends it to
    # about x1822.
    # Then x>=1553 air_rejump_rb0_jab6_rb4_jab4_tail8 extends it to
    # about x2105.
    # Then x>=1829 air_rejump_rb0_jab8_rb8_jab4_tail8 extends it to
    # about x2249.
    # Then x>=2021 air_rejump_rb0_jab6_rb4_jab4_tail8 reaches about
    # x2338 without a death in the bounded search.
    # Then x>=2338 air_rejump_rb0_jab6_rb0_jab0_tail8 exits cleanly
    # to 4-4.
    return (
        [ids["noop"]] * 12
        + [ids["right_b"]] * 12
        + [ids["right_ab"]] * 9
        + [ids["right_b"]] * 4
        + [ids["right_ab"]] * 8
        + [ids["right_b"]] * 8
        + [ids["right_b"]] * 5
        + [ids["right_b"]] * 2
        + [ids["right_ab"]] * 6
        + [ids["right_b"]] * 4
        + [ids["right_ab"]] * 4
        + [ids["right_b"]] * 8
        + [ids["right_b"]] * 4
        + [ids["right_ab"]] * 8
        + [ids["right_b"]] * 4
        + [ids["right_ab"]] * 7
        + [ids["right_b"]] * 4
        + [ids["right_ab"]] * 8
        + [ids["right_b"]] * 8
        + [ids["right_b"]] * 5
        + [ids["right_ab"]] * 6
        + [ids["right_b"]] * 8
        + [ids["right_ab"]] * 6
        + [ids["right_b"]] * 4
        + [ids["right_ab"]] * 4
        + [ids["right_b"]] * 8
        + [ids["right_b"]]
        + [ids["right_ab"]] * 8
        + [ids["right_b"]] * 8
        + [ids["right_ab"]] * 6
        + [ids["right_b"]] * 4
        + [ids["right_ab"]] * 4
        + [ids["right_b"]] * 8
        + [ids["right_b"]] * 10
        + [ids["right_ab"]] * 6
        + [ids["right_b"]] * 8
    )


def make_4_4_lava1_controller(ids: dict[str, int]) -> list[int]:
    # Found by exact-state branch search from 4-4 x>=127:
    # lava1_brake1_wait0_jab10_tail16 gets across the first lava gap pair,
    # improving the default 4-4 opening from x132 to about x198.
    # Then from x>=149, lava1_brake4_wait2_jab10_tail16 keeps the landing low
    # enough to clear the next lava gap and reaches about x627 without a death.
    return (
        [ids["left"]]
        + [ids["right_ab"]] * 5
        + [ids["left"]] * 4
        + [ids["noop"]] * 2
        + [ids["right_ab"]] * 10
        + [ids["right_b"]] * 16
    )


def make_4_4_mid_controller(ids: dict[str, int]) -> list[int]:
    # Found by exact-state branch search from 4-4 x>=568:
    # lava1_rb2_wait4_jab10_tail16 escapes the x626/y79 stall and reaches
    # about x1038 without a death.
    return [ids["right_b"]] * 2 + [ids["noop"]] * 4 + [ids["right_ab"]] * 10 + [ids["right_b"]] * 16


def make_4_4_maze1_controller(ids: dict[str, int]) -> list[int]:
    # Found by post-loop exact-state branch search after the first 4-4 maze
    # sends the bottom route back to x194. toproute_rb0_jab18_right4_tail24
    # takes the upper path and reaches about x628 in the repeated maze section.
    return [ids["right_ab"]] * 18 + [ids["right"]] * 4 + [ids["right_b"]] * 24


def make_4_4_maze2_controller(ids: dict[str, int]) -> list[int]:
    # Found by post-loop branch search from x>=569 after maze1:
    # toproute_rb2_jab18_right12_tail24 reaches about x1055 without a death.
    return [ids["right_b"]] * 2 + [ids["right_ab"]] * 18 + [ids["right"]] * 12 + [ids["right_b"]] * 24


def make_1_1_finish_controller(ids: dict[str, int]) -> list[int]:
    # Found by whole-game exact-state branch search from x>=2200:
    # final_name=rb8_jab6_rb4_down0_tail140_stair_jump.
    stair_tail_pattern = [ids["right_b"]] * 4 + [ids["right_ab"]] * 8
    stair_tail = [stair_tail_pattern[i % len(stair_tail_pattern)] for i in range(140)]
    return [ids["right_b"]] * 8 + [ids["right_ab"]] * 6 + [ids["right_b"]] * 4 + stair_tail


def predict_action(model: PPO, frame_stack: deque[np.ndarray], deterministic: bool) -> int:
    stacked = np.concatenate(list(frame_stack), axis=2).transpose(2, 0, 1)
    action, _ = model.predict(stacked, deterministic=deterministic)
    return int(action)


def stacked_observation(frame_stack: deque[np.ndarray]) -> np.ndarray:
    stacked = np.concatenate(list(frame_stack), axis=2).transpose(2, 0, 1)
    return np.ascontiguousarray(stacked)


def map_simple_pipe_to_complex(action: int, ids: dict[str, int]) -> int:
    # simple_pipe index 5 means DOWN, while complex index 5 means stand A.
    # Indices 0-4 and 6 retain the same semantics.
    if int(action) == 5:
        return ids["down"]
    return int(action)


def apply_exact_initial_noops(env: Any, obs: np.ndarray, info: dict[str, Any], noops: int) -> tuple[np.ndarray, dict[str, Any]]:
    for _ in range(max(0, int(noops))):
        obs, _reward, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            obs, info = env.reset()
    return obs, info


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_rollout_seeds(int(args.seed))

    level11_model = PPO.load(args.level11_model, device="cpu")
    level12_model = PPO.load(args.level12_model, device="cpu")
    level43_model = PPO.load(args.level43_model, device="cpu") if args.level43_model else None

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
    # Use complex as the shared superset: the 1-1 model trained on SIMPLE keeps
    # indices 0-6 unchanged, while the 1-2 simple_pipe model remaps DOWN below.
    config.action_set = "complex"
    ids = action_indices(config.action_set)
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

    env = make_single_env(config, seed=int(args.seed))
    obs, info = env.reset(seed=int(args.seed))
    if args.initial_noops_exact is not None:
        obs, info = apply_exact_initial_noops(env, obs, info, int(args.initial_noops_exact))
    frame_stack: deque[np.ndarray] = deque(maxlen=config.frame_stack)
    for _ in range(config.frame_stack):
        frame_stack.append(obs)

    frames: list[np.ndarray] = [] if args.no_video else [render_rgb_frame(env)]
    records: list[dict[str, Any]] = []
    demo_observations: list[np.ndarray] = []
    demo_actions: list[int] = []
    demo_worlds: list[int] = []
    demo_stages: list[int] = []
    demo_stage_x: list[int] = []
    demo_y: list[int] = []
    demo_modes: list[str] = []
    cur_ws = world_stage_from_info(info, (1, 1)) or (1, 1)
    stage_x = 0
    max_stage_x = 0
    prev_life = life_count(info)
    deaths = 0
    furthest_ws = cur_ws
    stage_clears = 0
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
    level43_noops_remaining = max(0, int(args.level43_start_noops))
    hold12_action: int | None = None
    hold12_remaining = 0
    extra_jump_hold_12 = max(0, int(args.level12_extra_jump_hold))
    stop_ws = (int(args.stop_world), int(args.stop_stage))
    reached_stop = False

    for step in range(1, int(args.max_steps) + 1):
        prev_ws = cur_ws
        goal_x = goal_line_x_for_world_stage(cur_ws, fallback=3200)

        if cur_ws == (1, 2) and hold12_remaining > 0 and hold12_action is not None:
            action = int(hold12_action)
            hold12_remaining -= 1
            mode = "hold_1_2_jump"
        elif cur_ws == (1, 1):
            action = predict_action(level11_model, frame_stack, bool(args.deterministic))
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
                action = approach_12[approach_idx] if approach_idx < len(approach_12) else predict_action(
                    level12_model, frame_stack, bool(args.deterministic)
                )
                approach_idx += 1
                mode = "script_1_2_approach"
            elif stage_x >= 2200:
                approach_started = True
                approach_idx = 1
                action = approach_12[0]
                mode = "script_1_2_approach"
            elif late_started:
                if late_idx < len(late_12):
                    action = late_12[late_idx]
                    late_idx += 1
                    mode = "script_1_2_late"
                else:
                    late_started = False
                    late_done = True
                    action = map_simple_pipe_to_complex(
                        predict_action(level12_model, frame_stack, bool(args.deterministic)),
                        ids,
                    )
                    mode = "ppo_1_2"
            elif not late_done and stage_x >= 1920:
                late_started = True
                late_idx = 1
                action = late_12[0]
                mode = "script_1_2_late"
            elif early_started:
                if early_idx < len(early_12):
                    action = early_12[early_idx]
                    early_idx += 1
                    mode = "script_1_2_early"
                else:
                    early_started = False
                    early_done = True
                    action = map_simple_pipe_to_complex(
                        predict_action(level12_model, frame_stack, bool(args.deterministic)),
                        ids,
                    )
                    mode = "ppo_1_2"
            elif not early_done and stage_x >= 320:
                early_started = True
                early_idx = 1
                action = early_12[0]
                mode = "script_1_2_early"
            else:
                action = map_simple_pipe_to_complex(
                    predict_action(level12_model, frame_stack, bool(args.deterministic)),
                    ids,
                )
                mode = "ppo_1_2"
        elif cur_ws == (1, 3):
            if exit13_started:
                if exit13_idx < len(exit_13):
                    action = exit_13[exit13_idx]
                    exit13_idx += 1
                    mode = "script_1_3_exit"
                else:
                    exit13_started = False
                    exit13_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not exit13_done and stage_x >= 1800:
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
            elif gap3_started:
                if gap3_idx < len(gap3_13):
                    action = gap3_13[gap3_idx]
                    gap3_idx += 1
                    mode = "script_1_3_gap3"
                else:
                    gap3_started = False
                    gap3_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not gap3_done and 1500 <= stage_x < 2300:
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
            elif gap2_started:
                if gap2_idx < len(gap2_13):
                    action = gap2_13[gap2_idx]
                    gap2_idx += 1
                    mode = "script_1_3_gap2"
                else:
                    gap2_started = False
                    gap2_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not gap2_done and 1310 <= stage_x < 1800:
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
            elif gap13_started:
                if gap13_idx < len(gap_13):
                    action = gap_13[gap13_idx]
                    gap13_idx += 1
                    mode = "script_1_3_gap"
                else:
                    gap13_started = False
                    gap13_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not gap13_done and not gap2_done and 940 <= stage_x < 1310:
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
            elif late13_started:
                if late13_idx < len(late_13):
                    action = late_13[late13_idx]
                    late13_idx += 1
                    mode = "script_1_3_late"
                else:
                    late13_started = False
                    late13_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not late13_done and not gap13_done and not gap2_done and 700 <= stage_x < 940:
                early13_started = False
                early13_done = True
                mid13_started = False
                mid13_done = True
                late13_started = True
                late13_idx = 1
                action = late_13[0]
                mode = "script_1_3_late"
            elif mid13_started:
                if mid13_idx < len(mid_13):
                    action = mid_13[mid13_idx]
                    mid13_idx += 1
                    mode = "script_1_3_mid"
                else:
                    mid13_started = False
                    mid13_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not mid13_done and not late13_done and not gap13_done and not gap2_done and 500 <= stage_x < 700:
                early13_started = False
                early13_done = True
                mid13_started = True
                mid13_idx = 1
                action = mid_13[0]
                mode = "script_1_3_mid"
            elif early13_started:
                if early13_idx < len(early_13):
                    action = early_13[early13_idx]
                    early13_idx += 1
                    mode = "script_1_3_early"
                else:
                    early13_started = False
                    early13_done = True
                    if args.level13_policy == "level11":
                        action = predict_action(level11_model, frame_stack, bool(args.deterministic))
                        mode = "ppo_1_3_level11"
                    else:
                        action = ids["right_b"]
                        mode = "default_right_b"
            elif not early13_done and not mid13_done and not late13_done and not gap13_done and not gap2_done and 140 <= stage_x < 500:
                early13_started = True
                early13_idx = 1
                action = early_13[0]
                mode = "script_1_3_early"
            else:
                if args.level13_policy == "level11":
                    action = predict_action(level11_model, frame_stack, bool(args.deterministic))
                    mode = "ppo_1_3_level11"
                else:
                    action = ids["right_b"]
                    mode = "default_right_b"
        elif cur_ws == (1, 4):
            if exit14_started:
                if exit14_idx < len(exit_14):
                    action = exit_14[exit14_idx]
                    exit14_idx += 1
                    mode = "script_1_4_exit"
                else:
                    exit14_started = False
                    exit14_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not exit14_done and stage_x >= 900:
                opening14_started = False
                opening14_done = True
                exit14_started = True
                exit14_idx = 1
                action = exit_14[0]
                mode = "script_1_4_exit"
            elif opening14_started:
                if opening14_idx < len(opening_14):
                    action = opening_14[opening14_idx]
                    opening14_idx += 1
                    mode = "script_1_4_opening"
                else:
                    opening14_started = False
                    opening14_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not opening14_done and stage_x >= 160:
                opening14_started = True
                opening14_idx = 1
                action = opening_14[0]
                mode = "script_1_4_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (2, 1):
            if exit21_started:
                if exit21_idx < len(exit_21):
                    action = exit_21[exit21_idx]
                    exit21_idx += 1
                    mode = "script_2_1_exit"
                else:
                    exit21_started = False
                    exit21_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not exit21_done and stage_x >= 2720:
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
            elif postpipe21_started:
                if postpipe21_idx < len(post_pipe_21):
                    action = post_pipe_21[postpipe21_idx]
                    postpipe21_idx += 1
                    mode = "script_2_1_post_pipe"
                else:
                    postpipe21_started = False
                    postpipe21_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not postpipe21_done and stage_x >= 2350:
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
            elif pipe21_started:
                if pipe21_idx < len(pipe_21):
                    action = pipe_21[pipe21_idx]
                    pipe21_idx += 1
                    mode = "script_2_1_pipe"
                else:
                    pipe21_started = False
                    pipe21_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not pipe21_done and stage_x >= 2000:
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
            elif tail21_started:
                if tail21_idx < len(tail_21):
                    action = tail_21[tail21_idx]
                    tail21_idx += 1
                    mode = "script_2_1_tail"
                else:
                    tail21_started = False
                    tail21_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not tail21_done and stage_x >= 1720:
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
            elif bridge21_started:
                if bridge21_idx < len(bridge_21):
                    action = bridge_21[bridge21_idx]
                    bridge21_idx += 1
                    mode = "script_2_1_bridge"
                else:
                    bridge21_started = False
                    bridge21_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not bridge21_done and stage_x >= 1200:
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
            elif late21_started:
                if late21_idx < len(late_21):
                    action = late_21[late21_idx]
                    late21_idx += 1
                    mode = "script_2_1_late"
                else:
                    late21_started = False
                    late21_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not late21_done and stage_x >= 855:
                opening21_started = False
                opening21_done = True
                mid21_started = False
                mid21_done = True
                late21_started = True
                late21_idx = 1
                action = late_21[0]
                mode = "script_2_1_late"
            elif mid21_started:
                if mid21_idx < len(mid_21):
                    action = mid_21[mid21_idx]
                    mid21_idx += 1
                    mode = "script_2_1_mid"
                else:
                    mid21_started = False
                    mid21_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not mid21_done and stage_x >= 605:
                opening21_started = False
                opening21_done = True
                mid21_started = True
                mid21_idx = 1
                action = mid_21[0]
                mode = "script_2_1_mid"
            elif opening21_started:
                if opening21_idx < len(opening_21):
                    action = opening_21[opening21_idx]
                    opening21_idx += 1
                    mode = "script_2_1_opening"
                else:
                    opening21_started = False
                    opening21_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not opening21_done and stage_x >= 245:
                opening21_started = True
                opening21_idx = 1
                action = opening_21[0]
                mode = "script_2_1_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (2, 2):
            if exit22_started:
                if exit22_idx < len(exit_22):
                    action = exit_22[exit22_idx]
                    exit22_idx += 1
                    mode = "script_2_2_exit"
                else:
                    exit22_started = False
                    exit22_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not exit22_done and stage_x >= 2600:
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
            elif late22_started:
                if late22_idx < len(late_22):
                    action = late_22[late22_idx]
                    late22_idx += 1
                    mode = "script_2_2_late"
                else:
                    late22_started = False
                    late22_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not late22_done and stage_x >= 2220:
                opening22_started = False
                opening22_done = True
                mid22_started = False
                mid22_done = True
                late22_started = True
                late22_idx = 1
                action = late_22[0]
                mode = "script_2_2_late"
            elif mid22_started:
                if mid22_idx < len(mid_22):
                    action = mid_22[mid22_idx]
                    mid22_idx += 1
                    mode = "script_2_2_mid"
                else:
                    mid22_started = False
                    mid22_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not mid22_done and stage_x >= 1500:
                opening22_started = False
                opening22_done = True
                mid22_started = True
                mid22_idx = 1
                action = mid_22[0]
                mode = "script_2_2_mid"
            elif opening22_started:
                if opening22_idx < len(opening_22):
                    action = opening_22[opening22_idx]
                    opening22_idx += 1
                    mode = "script_2_2_opening"
                else:
                    opening22_started = False
                    opening22_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not opening22_done and stage_x >= 0:
                opening22_started = True
                opening22_idx = 1
                action = opening_22[0]
                mode = "script_2_2_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (2, 3):
            if exit23_started:
                if exit23_idx < len(exit_23):
                    action = exit_23[exit23_idx]
                    exit23_idx += 1
                    mode = "script_2_3_exit"
                else:
                    exit23_started = False
                    exit23_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not exit23_done and stage_x >= 3300:
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
            elif tail23_started:
                if tail23_idx < len(tail_23):
                    action = tail_23[tail23_idx]
                    tail23_idx += 1
                    mode = "script_2_3_tail"
                else:
                    tail23_started = False
                    tail23_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not tail23_done and stage_x >= 2750:
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
            elif bridge23_started:
                if bridge23_idx < len(bridge_23):
                    action = bridge_23[bridge23_idx]
                    bridge23_idx += 1
                    mode = "script_2_3_bridge"
                else:
                    bridge23_started = False
                    bridge23_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not bridge23_done and stage_x >= 2300:
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
            elif late23_started:
                if late23_idx < len(late_23):
                    action = late_23[late23_idx]
                    late23_idx += 1
                    mode = "script_2_3_late"
                else:
                    late23_started = False
                    late23_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not late23_done and stage_x >= 1850:
                opening23_started = False
                opening23_done = True
                mid23_started = False
                mid23_done = True
                late23_started = True
                late23_idx = 1
                action = late_23[0]
                mode = "script_2_3_late"
            elif mid23_started:
                if mid23_idx < len(mid_23):
                    action = mid_23[mid23_idx]
                    mid23_idx += 1
                    mode = "script_2_3_mid"
                else:
                    mid23_started = False
                    mid23_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not mid23_done and stage_x >= 1500:
                opening23_started = False
                opening23_done = True
                mid23_started = True
                mid23_idx = 1
                action = mid_23[0]
                mode = "script_2_3_mid"
            elif opening23_started:
                if opening23_idx < len(opening_23):
                    action = opening_23[opening23_idx]
                    opening23_idx += 1
                    mode = "script_2_3_opening"
                else:
                    opening23_started = False
                    opening23_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not opening23_done and stage_x >= 80:
                opening23_started = True
                opening23_idx = 1
                action = opening_23[0]
                mode = "script_2_3_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (2, 4):
            if exit24_started:
                if exit24_idx < len(exit_24):
                    action = exit_24[exit24_idx]
                    exit24_idx += 1
                    mode = "script_2_4_exit"
                else:
                    exit24_started = False
                    exit24_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not exit24_done and stage_x >= 1700:
                opening24_started = False
                opening24_done = True
                mid24_started = False
                mid24_done = True
                exit24_started = True
                exit24_idx = 1
                action = exit_24[0]
                mode = "script_2_4_exit"
            elif mid24_started:
                if mid24_idx < len(mid_24):
                    action = mid_24[mid24_idx]
                    mid24_idx += 1
                    mode = "script_2_4_mid"
                else:
                    mid24_started = False
                    mid24_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not mid24_done and stage_x >= 1260:
                opening24_started = False
                opening24_done = True
                mid24_started = True
                mid24_idx = 1
                action = mid_24[0]
                mode = "script_2_4_mid"
            elif opening24_started:
                if opening24_idx < len(opening_24):
                    action = opening_24[opening24_idx]
                    opening24_idx += 1
                    mode = "script_2_4_opening"
                else:
                    opening24_started = False
                    opening24_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not opening24_done and stage_x >= 220:
                opening24_started = True
                opening24_idx = 1
                action = opening_24[0]
                mode = "script_2_4_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (3, 1):
            if exit31_started:
                if exit31_idx < len(exit_31):
                    action = exit_31[exit31_idx]
                    exit31_idx += 1
                    mode = "script_3_1_exit"
                else:
                    exit31_started = False
                    exit31_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not exit31_done and stage_x >= 2380:
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
            elif tail31_started:
                if tail31_idx < len(tail_31):
                    action = tail_31[tail31_idx]
                    tail31_idx += 1
                    mode = "script_3_1_tail"
                else:
                    tail31_started = False
                    tail31_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not tail31_done and stage_x >= 1780:
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
            elif late31_started:
                if late31_idx < len(late_31):
                    action = late_31[late31_idx]
                    late31_idx += 1
                    mode = "script_3_1_late"
                else:
                    late31_started = False
                    late31_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not late31_done and stage_x >= 1470:
                opening31_started = False
                opening31_done = True
                mid31_started = False
                mid31_done = True
                late31_started = True
                late31_idx = 1
                action = late_31[0]
                mode = "script_3_1_late"
            elif mid31_started:
                if mid31_idx < len(mid_31):
                    action = mid_31[mid31_idx]
                    mid31_idx += 1
                    mode = "script_3_1_mid"
                else:
                    mid31_started = False
                    mid31_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not mid31_done and stage_x >= 880:
                opening31_started = False
                opening31_done = True
                mid31_started = True
                mid31_idx = 1
                action = mid_31[0]
                mode = "script_3_1_mid"
            elif opening31_started:
                if opening31_idx < len(opening_31):
                    action = opening_31[opening31_idx]
                    opening31_idx += 1
                    mode = "script_3_1_opening"
                else:
                    opening31_started = False
                    opening31_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not opening31_done and stage_x >= 480:
                opening31_started = True
                opening31_idx = 1
                action = opening_31[0]
                mode = "script_3_1_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (3, 2):
            if exit32_started:
                if exit32_idx < len(exit_32):
                    action = exit_32[exit32_idx]
                    exit32_idx += 1
                    mode = "script_3_2_exit"
                else:
                    exit32_started = False
                    exit32_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not exit32_done and stage_x >= 2700:
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
            elif late32_started:
                if late32_idx < len(late_32):
                    action = late_32[late32_idx]
                    late32_idx += 1
                    mode = "script_3_2_late"
                else:
                    late32_started = False
                    late32_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not late32_done and stage_x >= 2000:
                opening32_started = False
                opening32_done = True
                mid32_started = False
                mid32_done = True
                late32_started = True
                late32_idx = 1
                action = late_32[0]
                mode = "script_3_2_late"
            elif mid32_started:
                if mid32_idx < len(mid_32):
                    action = mid_32[mid32_idx]
                    mid32_idx += 1
                    mode = "script_3_2_mid"
                else:
                    mid32_started = False
                    mid32_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not mid32_done and stage_x >= 1200:
                opening32_started = False
                opening32_done = True
                mid32_started = True
                mid32_idx = 1
                action = mid_32[0]
                mode = "script_3_2_mid"
            elif opening32_started:
                if opening32_idx < len(opening_32):
                    action = opening_32[opening32_idx]
                    opening32_idx += 1
                    mode = "script_3_2_opening"
                else:
                    opening32_started = False
                    opening32_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not opening32_done and stage_x >= 120:
                opening32_started = True
                opening32_idx = 1
                action = opening_32[0]
                mode = "script_3_2_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (3, 3):
            if exit33_started:
                if exit33_idx < len(exit_33):
                    action = exit_33[exit33_idx]
                    exit33_idx += 1
                    mode = "script_3_3_exit"
                else:
                    exit33_started = False
                    exit33_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not exit33_done and stage_x >= 1950:
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
            elif landing33_started:
                if landing33_idx < len(landing_33):
                    action = landing_33[landing33_idx]
                    landing33_idx += 1
                    mode = "script_3_3_landing"
                else:
                    landing33_started = False
                    landing33_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not landing33_done and stage_x >= 2050:
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
            elif tail33_started:
                if tail33_idx < len(tail_33):
                    action = tail_33[tail33_idx]
                    tail33_idx += 1
                    mode = "script_3_3_tail"
                else:
                    tail33_started = False
                    tail33_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not tail33_done and stage_x >= 1650:
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
            elif late33_started:
                if late33_idx < len(late_33):
                    action = late_33[late33_idx]
                    late33_idx += 1
                    mode = "script_3_3_late"
                else:
                    late33_started = False
                    late33_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not late33_done and stage_x >= 1100:
                opening33_started = False
                opening33_done = True
                mid33_started = False
                mid33_done = True
                late33_started = True
                late33_idx = 1
                action = late_33[0]
                mode = "script_3_3_late"
            elif mid33_started:
                if mid33_idx < len(mid_33):
                    action = mid_33[mid33_idx]
                    mid33_idx += 1
                    mode = "script_3_3_mid"
                else:
                    mid33_started = False
                    mid33_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not mid33_done and stage_x >= 600:
                opening33_started = False
                opening33_done = True
                mid33_started = True
                mid33_idx = 1
                action = mid_33[0]
                mode = "script_3_3_mid"
            elif opening33_started:
                if opening33_idx < len(opening_33):
                    action = opening_33[opening33_idx]
                    opening33_idx += 1
                    mode = "script_3_3_opening"
                else:
                    opening33_started = False
                    opening33_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not opening33_done and stage_x >= 200:
                opening33_started = True
                opening33_idx = 1
                action = opening_33[0]
                mode = "script_3_3_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (3, 4):
            if exit34_started:
                if exit34_idx < len(exit_34):
                    action = exit_34[exit34_idx]
                    exit34_idx += 1
                    mode = "script_3_4_exit"
                else:
                    exit34_started = False
                    exit34_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not exit34_done and stage_x >= 1450:
                opening34_started = False
                opening34_done = True
                mid34_started = False
                mid34_done = True
                exit34_started = True
                exit34_idx = 1
                action = exit_34[0]
                mode = "script_3_4_exit"
            elif mid34_started:
                if mid34_idx < len(mid_34):
                    action = mid_34[mid34_idx]
                    mid34_idx += 1
                    mode = "script_3_4_mid"
                else:
                    mid34_started = False
                    mid34_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not mid34_done and stage_x >= 1400:
                opening34_started = False
                opening34_done = True
                mid34_started = True
                mid34_idx = 1
                action = mid_34[0]
                mode = "script_3_4_mid"
            elif opening34_started:
                if opening34_idx < len(opening_34):
                    action = opening_34[opening34_idx]
                    opening34_idx += 1
                    mode = "script_3_4_opening"
                else:
                    opening34_started = False
                    opening34_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not opening34_done and stage_x >= 160:
                opening34_started = True
                opening34_idx = 1
                action = opening_34[0]
                mode = "script_3_4_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (4, 1):
            if exit41_started:
                if exit41_idx < len(exit_41):
                    action = exit_41[exit41_idx]
                    exit41_idx += 1
                    mode = "script_4_1_exit"
                else:
                    exit41_started = False
                    exit41_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not exit41_done and stage_x >= 3000:
                opening41_started = False
                opening41_done = True
                mid41_started = False
                mid41_done = True
                exit41_started = True
                exit41_idx = 1
                action = exit_41[0]
                mode = "script_4_1_exit"
            elif mid41_started:
                if mid41_idx < len(mid_41):
                    action = mid_41[mid41_idx]
                    mid41_idx += 1
                    mode = "script_4_1_mid"
                else:
                    mid41_started = False
                    mid41_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not mid41_done and stage_x >= 2000:
                opening41_started = False
                opening41_done = True
                mid41_started = True
                mid41_idx = 1
                action = mid_41[0]
                mode = "script_4_1_mid"
            elif opening41_started:
                if opening41_idx < len(opening_41):
                    action = opening_41[opening41_idx]
                    opening41_idx += 1
                    mode = "script_4_1_opening"
                else:
                    opening41_started = False
                    opening41_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not opening41_done and stage_x >= 220:
                opening41_started = True
                opening41_idx = 1
                action = opening_41[0]
                mode = "script_4_1_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (4, 2):
            if exit42_started:
                if exit42_idx < len(exit_42):
                    action = exit_42[exit42_idx]
                    exit42_idx += 1
                    mode = "script_4_2_exit"
                else:
                    exit42_started = False
                    exit42_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not exit42_done and stage_x >= 2803:
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
            elif exitapproach42_started:
                if exitapproach42_idx < len(exit_approach_42):
                    action = exit_approach_42[exitapproach42_idx]
                    exitapproach42_idx += 1
                    mode = "script_4_2_exit_approach"
                else:
                    exitapproach42_started = False
                    exitapproach42_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not exitapproach42_done and stage_x >= 2451:
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
            elif pipeentry42_started:
                if pipeentry42_idx < len(pipe_entry_42):
                    action = pipe_entry_42[pipeentry42_idx]
                    pipeentry42_idx += 1
                    mode = "script_4_2_pipe_entry"
                else:
                    pipeentry42_started = False
                    pipeentry42_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not pipeentry42_done and stage_x >= 2258:
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
            elif safebridge42_started:
                if safebridge42_idx < len(safe_bridge_42):
                    action = safe_bridge_42[safebridge42_idx]
                    safebridge42_idx += 1
                    mode = "script_4_2_safe_bridge"
                else:
                    safebridge42_started = False
                    safebridge42_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not safebridge42_done and stage_x >= 1771:
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
            elif bridge42_started:
                if bridge42_idx < len(bridge_42):
                    action = bridge_42[bridge42_idx]
                    bridge42_idx += 1
                    mode = "script_4_2_bridge"
                else:
                    bridge42_started = False
                    bridge42_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not bridge42_done and stage_x >= 1629:
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
            elif posttail42_started:
                if posttail42_idx < len(post_tail_42):
                    action = post_tail_42[posttail42_idx]
                    posttail42_idx += 1
                    mode = "script_4_2_post_tail"
                else:
                    posttail42_started = False
                    posttail42_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not posttail42_done and stage_x >= 1410:
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
            elif tail42_started:
                if tail42_idx < len(tail_42):
                    action = tail_42[tail42_idx]
                    tail42_idx += 1
                    mode = "script_4_2_tail"
                else:
                    tail42_started = False
                    tail42_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not tail42_done and stage_x >= 1138:
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
            elif late42_started:
                if late42_idx < len(late_42):
                    action = late_42[late42_idx]
                    late42_idx += 1
                    mode = "script_4_2_late"
                else:
                    late42_started = False
                    late42_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not late42_done and stage_x >= 890:
                opening42_started = False
                opening42_done = True
                mid42_started = False
                mid42_done = True
                late42_started = True
                late42_idx = 1
                action = late_42[0]
                mode = "script_4_2_late"
            elif mid42_started:
                if mid42_idx < len(mid_42):
                    action = mid_42[mid42_idx]
                    mid42_idx += 1
                    mode = "script_4_2_mid"
                else:
                    mid42_started = False
                    mid42_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not mid42_done and stage_x >= 428:
                opening42_started = False
                opening42_done = True
                mid42_started = True
                mid42_idx = 1
                action = mid_42[0]
                mode = "script_4_2_mid"
            elif opening42_started:
                if opening42_idx < len(opening_42):
                    action = opening_42[opening42_idx]
                    opening42_idx += 1
                    mode = "script_4_2_opening"
                else:
                    opening42_started = False
                    opening42_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not opening42_done and stage_x >= 42:
                opening42_started = True
                opening42_idx = 1
                action = opening_42[0]
                mode = "script_4_2_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (4, 3):
            if level43_noops_remaining > 0:
                action = ids["noop"]
                level43_noops_remaining -= 1
                mode = "noop_4_3_start_delay"
            elif level43_model is not None:
                action = predict_action(level43_model, frame_stack, bool(args.deterministic))
                mode = "ppo_4_3_focused"
            elif opening43_started:
                if opening43_idx < len(opening_43):
                    action = opening_43[opening43_idx]
                    opening43_idx += 1
                    mode = "script_4_3_opening"
                else:
                    opening43_started = False
                    opening43_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not opening43_done and stage_x >= 161:
                opening43_started = True
                opening43_idx = 1
                action = opening_43[0]
                mode = "script_4_3_opening"
            else:
                action = ids["right_b"]
                mode = "default_right_b"
        elif cur_ws == (4, 4):
            if maze2_44_started:
                if maze2_44_idx < len(maze2_44):
                    action = maze2_44[maze2_44_idx]
                    maze2_44_idx += 1
                    mode = "script_4_4_maze2"
                else:
                    maze2_44_started = False
                    maze2_44_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not maze2_44_done and max_stage_x >= 1000 and 569 <= stage_x <= 700:
                maze1_44_started = False
                maze1_44_done = True
                maze2_44_started = True
                maze2_44_idx = 1
                action = maze2_44[0]
                mode = "script_4_4_maze2"
            elif maze1_44_started:
                if maze1_44_idx < len(maze1_44):
                    action = maze1_44[maze1_44_idx]
                    maze1_44_idx += 1
                    mode = "script_4_4_maze1"
                else:
                    maze1_44_started = False
                    maze1_44_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not maze1_44_done and max_stage_x >= 1000 and 194 <= stage_x <= 220:
                maze1_44_started = True
                maze1_44_idx = 1
                action = maze1_44[0]
                mode = "script_4_4_maze1"
            elif mid44_started:
                if mid44_idx < len(mid_44):
                    action = mid_44[mid44_idx]
                    mid44_idx += 1
                    mode = "script_4_4_mid"
                else:
                    mid44_started = False
                    mid44_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not mid44_done and stage_x >= 568:
                lava1_44_started = False
                lava1_44_done = True
                mid44_started = True
                mid44_idx = 1
                action = mid_44[0]
                mode = "script_4_4_mid"
            elif lava1_44_started:
                if lava1_44_idx < len(lava1_44):
                    action = lava1_44[lava1_44_idx]
                    lava1_44_idx += 1
                    mode = "script_4_4_lava1"
                else:
                    lava1_44_started = False
                    lava1_44_done = True
                    action = ids["right_b"]
                    mode = "default_right_b"
            elif not lava1_44_done and stage_x >= 127:
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

        if (
            cur_ws == (1, 2)
            and extra_jump_hold_12 > 0
            and hold12_remaining <= 0
            and action in (ids["right_a"], ids["right_ab"])
        ):
            hold12_action = int(action)
            hold12_remaining = extra_jump_hold_12

        if args.demo_output:
            demo_observations.append(stacked_observation(frame_stack))
            demo_actions.append(int(action))
            demo_worlds.append(int(cur_ws[0]))
            demo_stages.append(int(cur_ws[1]))
            demo_stage_x.append(int(stage_x))
            demo_y.append(int(info.get("y_pos", 0) or 0))
            demo_modes.append(str(mode))

        obs, reward, terminated, truncated, info = env.step(int(action))
        frame_stack.append(obs)
        if not args.no_video:
            frames.append(render_rgb_frame(env))
        cur_ws = world_stage_from_info(info, prev_ws) or prev_ws
        if cur_ws != prev_ws:
            if stage_order_index(cur_ws) > stage_order_index(prev_ws):
                stage_clears += 1
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
            level43_noops_remaining = max(0, int(args.level43_start_noops))
            hold12_action = None
            hold12_remaining = 0
            if args.reset_stack_on_stage_change:
                frame_stack.clear()
                for _ in range(config.frame_stack):
                    frame_stack.append(obs)
        else:
            stage_x = extract_sanitized_x_position(info, previous_x_pos=stage_x, goal_line_x=goal_x)
            max_stage_x = max(max_stage_x, stage_x)
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
                "step": step,
                "world": cur_ws[0],
                "stage": cur_ws[1],
                "stage_x": int(stage_x),
                "max_stage_x": int(max_stage_x),
                "y": int(info.get("y_pos", 0) or 0),
                "action": int(action),
                "mode": mode,
                "reward": float(reward),
                "flag_get": bool(info.get("flag_get", False)),
                "life": cur_life,
                "life_lost": bool(life_lost),
            }
        )
        if stage_order_index(cur_ws) >= stage_order_index(stop_ws):
            reached_stop = True
            # Keep a short tail so the video visibly lands in the new stage.
            for _ in range(30):
                obs, reward, terminated, truncated, info = env.step(ids["right_b"])
                frame_stack.append(obs)
                if not args.no_video:
                    frames.append(render_rgb_frame(env))
            break
        if terminated or truncated or (args.stop_on_death and life_lost):
            break

    env.close()
    video_path = output_dir / "best.mp4"
    if not args.no_video:
        frames = downscale_frames(frames, int(args.export_max_width))
        save_video(frames, video_path, fps=int(args.fps))
    summary = {
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "reached_stop": bool(reached_stop),
        "stop_world_stage": list(stop_ws),
        "furthest_world_stage": list(furthest_ws),
        "stage_clears": int(stage_clears),
        "deaths": int(deaths),
        "steps": len(records),
        "video": None if args.no_video else str(video_path),
        "level11_model": str(args.level11_model),
        "level12_model": str(args.level12_model),
        "level43_model": str(args.level43_model) if args.level43_model else None,
        "level43_start_noops": int(args.level43_start_noops),
        "controller_1_1": {
            "policy": "ppo_1_1",
            "initial_noops_exact": args.initial_noops_exact,
        },
        "controller_1_2": {
            "early_trigger_x": 320,
            "early": "rb0_jab0_rb8_jab5_rb16",
            "late_trigger_x": 1920,
            "late": "rb0_jab0_rb0_jab3_rb16",
            "approach_trigger_x": 2200,
            "final_trigger_x": 2380,
            "approach": "rb16_jab5_rb8_d12",
            "final": "rb8_jab4_rb2_down20_tail220_pulse_jump",
            "extra_jump_hold": int(extra_jump_hold_12),
            "reset_stack_on_stage_change": bool(args.reset_stack_on_stage_change),
        },
        "controller_1_3": {
            "early_trigger_x": 140,
            "early": "stair60",
            "mid_trigger_x": 500,
            "mid": "tap_brake60",
            "late_trigger_x": 700,
            "late": "stair3120",
            "gap_trigger_x": 940,
            "gap": "rb8_jab3_rb4_jab12_rb24",
            "gap2_trigger_x": 1310,
            "gap2": "rb4_jab5_rb8_jab8_rb0",
            "gap3_trigger_x": 1500,
            "gap3": "rand32_81",
            "exit_trigger_x": 1800,
            "exit": "rb8_jump_bias140",
            "fallback_policy": args.level13_policy,
        },
        "controller_1_4": {
            "opening_trigger_x": 160,
            "opening": "rand128_87",
            "exit_trigger_x": 900,
            "exit": "stair3240",
            "fallback_policy": "right_b",
        },
        "controller_2_1": {
            "opening_trigger_x": 245,
            "opening": "rb0_hop140",
            "mid_trigger_x": 605,
            "mid": "rand32_51",
            "late_trigger_x": 855,
            "late": "rb4_jab10_rb8_jab12_rb16",
            "bridge_trigger_x": 1200,
            "bridge": "rb8_jump_bias140",
            "tail_trigger_x": 1720,
            "tail": "rb0_stair390",
            "pipe_trigger_x": 2000,
            "pipe": "rand48_11",
            "post_pipe_trigger_x": 2350,
            "post_pipe": "slow8_ja8_slow4_ja8_tail28",
            "exit_trigger_x": 2720,
            "exit": "rand128_80",
            "fallback_policy": "right_b",
        },
        "controller_2_2": {
            "opening_trigger_x": 0,
            "opening": "stair3240",
            "mid_trigger_x": 1500,
            "mid": "rb16_slow_stair220",
            "late_trigger_x": 2220,
            "late": "rand96_31",
            "exit_trigger_x": 2600,
            "exit": "slow_jump180",
            "fallback_policy": "right_b",
        },
        "controller_2_3": {
            "opening_trigger_x": 80,
            "opening": "rb0_stair2140",
            "mid_trigger_x": 1500,
            "mid": "rb0_jab5_rb4_jab8_rb24",
            "late_trigger_x": 1850,
            "late": "slow4_ja4_slow4_ja8_tail16",
            "bridge_trigger_x": 2300,
            "bridge": "rb8_jab7_rb16_jab12_rb16",
            "tail_trigger_x": 2750,
            "tail": "rand128_80",
            "exit_trigger_x": 3300,
            "exit": "rb0_stair390",
            "fallback_policy": "right_b",
        },
        "controller_2_4": {
            "opening_trigger_x": 220,
            "opening": "rand160_36",
            "mid_trigger_x": 1260,
            "mid": "rb0_stair390",
            "exit_trigger_x": 1700,
            "exit": "rand80_69",
            "fallback_policy": "right_b",
        },
        "controller_3_1": {
            "opening_trigger_x": 480,
            "opening": "noop_jump240",
            "mid_trigger_x": 880,
            "mid": "rand128_87",
            "late_trigger_x": 1470,
            "late": "rand128_14",
            "tail_trigger_x": 1780,
            "tail": "rand80_57",
            "exit_trigger_x": 2380,
            "exit": "stair3240",
            "fallback_policy": "right_b",
        },
        "controller_3_2": {
            "opening_trigger_x": 120,
            "opening": "slow_jump180",
            "mid_trigger_x": 1200,
            "mid": "rand128_29",
            "late_trigger_x": 2000,
            "late": "rb0_hop140",
            "exit_trigger_x": 2700,
            "exit": "rb8_jump_bias140",
            "fallback_policy": "right_b",
        },
        "controller_3_3": {
            "opening_trigger_x": 200,
            "opening": "slow4_ja8_slow4_ja12_tail8",
            "mid_trigger_x": 600,
            "mid": "rb0_slow_stair220",
            "late_trigger_x": 1100,
            "late": "rb8_stair3140",
            "tail_trigger_x": 1650,
            "tail": "rb8_jump_bias140",
            "landing_trigger_x": 2050,
            "landing": "brake12_wait8_jab14_tail28",
            "exit_trigger_x": 1950,
            "exit": "rand80_83",
            "fallback_policy": "right_b",
        },
        "controller_3_4": {
            "opening_trigger_x": 160,
            "opening": "rb0_slow_stair220",
            "mid_trigger_x": 1400,
            "mid": "rand96_35",
            "exit_trigger_x": 1450,
            "exit": "slow_pulse240",
            "fallback_policy": "right_b",
        },
        "controller_4_1": {
            "opening_trigger_x": 220,
            "opening": "rb24_noop_jump220",
            "mid_trigger_x": 2000,
            "mid": "rb36_jump_run220",
            "exit_trigger_x": 3000,
            "exit": "rb0_stair390",
            "fallback_policy": "right_b",
        },
        "controller_4_2": {
            "opening_trigger_x": 42,
            "opening": "rb8_jab3_rb8_jab12_rb16",
            "mid_trigger_x": 428,
            "mid": "rand32_41",
            "late_trigger_x": 890,
            "late": "rb0_jab14_rb8_jab0_rb24",
            "tail_trigger_x": 1138,
            "tail": "slow12_ja8_slow4_ja8_tail28",
            "post_tail_trigger_x": 1410,
            "post_tail": "rb8_jab3_rb16_jab3_rb8",
            "bridge_trigger_x": 1629,
            "bridge": "brake2_wait16_jab14_tail40",
            "safe_bridge_trigger_x": 1771,
            "safe_bridge": "rand64_04",
            "pipe_entry_trigger_x": 2258,
            "pipe_entry": "rb8_jab10_rb8_jab3_rb8",
            "exit_approach_trigger_x": 2451,
            "exit_approach": "rand48_41",
            "exit_trigger_x": 2803,
            "exit": "rand128_36",
            "fallback_policy": "right_b",
        },
        "controller_4_3": {
            "opening_trigger_x": 161,
            "opening": "wait12_rb12_jab9_rb4_jab8_tail15_jab6_rb4_jab4_tail12_jab8_rb4_jab7_rb4_jab8_tail13_jab6_rb8_jab6_rb4_jab4_tail9_jab8_rb8_jab6_rb4_jab4_tail18_jab6_tail8",
            "fallback_policy": "right_b",
        },
        "controller_4_4": {
            "lava1_trigger_x": 127,
            "lava1": "lava1_brake1_wait0_then_x149_lava1_brake4_wait2_jab10_tail16",
            "mid_trigger_x": 568,
            "mid": "lava1_rb2_wait4_jab10_tail16",
            "maze1_trigger": "after_max_x_1000_current_x_194_220",
            "maze1": "toproute_rb0_jab18_right4_tail24",
            "maze2_trigger": "after_maze1_x_569_700",
            "maze2": "toproute_rb2_jab18_right12_tail24",
            "fallback_policy": "right_b",
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "trace.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    if args.demo_output:
        demo_path = Path(args.demo_output)
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            demo_path,
            observations=np.stack(demo_observations).astype(np.uint8, copy=False),
            actions=np.asarray(demo_actions, dtype=np.int64),
            worlds=np.asarray(demo_worlds, dtype=np.int16),
            stages=np.asarray(demo_stages, dtype=np.int16),
            stage_x=np.asarray(demo_stage_x, dtype=np.int32),
            y=np.asarray(demo_y, dtype=np.int16),
            modes=np.asarray(demo_modes, dtype="U64"),
            action_set=np.asarray(config.action_set),
            seed=np.asarray(int(args.seed), dtype=np.int64),
            deterministic=np.asarray(bool(args.deterministic)),
            reached_stop=np.asarray(bool(reached_stop)),
            stop_world_stage=np.asarray(stop_ws, dtype=np.int16),
            furthest_world_stage=np.asarray(furthest_ws, dtype=np.int16),
            stage_clears=np.asarray(int(stage_clears), dtype=np.int16),
            deaths=np.asarray(int(deaths), dtype=np.int16),
        )
        summary["demo_output"] = str(demo_path)
        summary["demo_steps"] = int(len(demo_actions))
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
