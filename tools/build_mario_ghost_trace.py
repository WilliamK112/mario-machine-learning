"""Convert a Mario route trace into a compact ghost trace for reward shaping."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mario_runtime import stage_order_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a .npz ghost route trace from tools/run_mario_hybrid_route.py trace.json.",
    )
    parser.add_argument("--trace", required=True, help="Input trace.json path.")
    parser.add_argument("--output", required=True, help="Output ghost .npz path.")
    parser.add_argument("--source-name", default="", help="Human-readable source label.")
    parser.add_argument("--source-url", default="", help="Optional source URL.")
    parser.add_argument(
        "--progress-stride",
        type=int,
        default=10_000,
        help="Virtual route-progress spacing between stages.",
    )
    return parser.parse_args()


def int_field(row: dict[str, Any], name: str, default: int = 0) -> int:
    value = row.get(name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def main() -> None:
    args = parse_args()
    trace_path = Path(args.trace)
    output_path = Path(args.output)
    rows = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Trace is empty or not a list: {trace_path}")

    steps: list[int] = []
    worlds: list[int] = []
    stages: list[int] = []
    stage_x: list[int] = []
    max_stage_x: list[int] = []
    y_pos: list[int] = []
    actions: list[int] = []
    stage_indices: list[int] = []
    route_progress: list[int] = []
    modes: list[str] = []

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Trace row {idx} is not an object.")
        world = int_field(row, "world", 1)
        stage = int_field(row, "stage", 1)
        x_value = max(0, int_field(row, "stage_x", 0))
        stage_index = stage_order_index((world, stage))
        steps.append(int_field(row, "step", idx + 1))
        worlds.append(world)
        stages.append(stage)
        stage_x.append(x_value)
        max_stage_x.append(max(0, int_field(row, "max_stage_x", x_value)))
        y_pos.append(int_field(row, "y", 0))
        actions.append(int_field(row, "action", 0))
        stage_indices.append(stage_index)
        route_progress.append(stage_index * int(args.progress_stride) + x_value)
        modes.append(str(row.get("mode", ""))[:96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        steps=np.asarray(steps, dtype=np.int32),
        worlds=np.asarray(worlds, dtype=np.int16),
        stages=np.asarray(stages, dtype=np.int16),
        stage_x=np.asarray(stage_x, dtype=np.int32),
        max_stage_x=np.asarray(max_stage_x, dtype=np.int32),
        y=np.asarray(y_pos, dtype=np.int16),
        actions=np.asarray(actions, dtype=np.int16),
        stage_indices=np.asarray(stage_indices, dtype=np.int16),
        route_progress=np.asarray(route_progress, dtype=np.float32),
        modes=np.asarray(modes, dtype="U96"),
        source_name=np.asarray(args.source_name, dtype="U256"),
        source_url=np.asarray(args.source_url, dtype="U512"),
        progress_stride=np.asarray(int(args.progress_stride), dtype=np.int32),
    )

    unique_stages = []
    seen: set[tuple[int, int]] = set()
    for world, stage in zip(worlds, stages, strict=False):
        ws = (int(world), int(stage))
        if ws not in seen:
            seen.add(ws)
            unique_stages.append([ws[0], ws[1]])
    summary = {
        "trace": str(trace_path),
        "output": str(output_path),
        "source_name": args.source_name,
        "source_url": args.source_url,
        "progress_stride": int(args.progress_stride),
        "steps": len(steps),
        "first_world_stage": [int(worlds[0]), int(stages[0])],
        "last_world_stage": [int(worlds[-1]), int(stages[-1])],
        "furthest_world_stage": max(unique_stages, key=lambda ws: stage_order_index(tuple(ws))),
        "max_route_progress": int(max(route_progress)),
        "world_stages": unique_stages,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
