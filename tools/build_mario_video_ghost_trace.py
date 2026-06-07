from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mario_runtime import goal_line_x_for_world_stage
from mario_runtime import stage_order_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a pace-based Mario ghost trace from a reference video plus stage timing segments. "
            "This creates the .npz format consumed by train_mario.py --ghost-trace-path."
        )
    )
    parser.add_argument("--video", default="", help="Optional local reference video path.")
    parser.add_argument("--segments-json", required=True, help="Stage timing manifest JSON.")
    parser.add_argument("--output", required=True, help="Output ghost .npz path.")
    parser.add_argument("--source-name", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--trace-fps", type=float, default=15.0)
    parser.add_argument("--progress-stride", type=int, default=10_000)
    parser.add_argument("--contact-sheet", default="", help="Optional PNG contact sheet output.")
    parser.add_argument("--contact-sheet-cols", type=int, default=5)
    parser.add_argument("--contact-sheet-width", type=int, default=256)
    parser.add_argument(
        "--write-template",
        action="store_true",
        help="Write a starter segments JSON and exit. Existing files are not overwritten.",
    )
    return parser.parse_args()


def parse_time(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        raise ValueError("empty timestamp")
    if ":" not in text:
        return float(text)
    parts = [float(part) for part in text.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60.0 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600.0 + minutes * 60.0 + seconds
    raise ValueError(f"Unsupported timestamp: {value!r}")


def load_segments(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        segments = data.get("segments", [])
    else:
        segments = data
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{path} must contain a non-empty segments list")
    parsed: list[dict[str, Any]] = []
    for idx, raw in enumerate(segments):
        if not isinstance(raw, dict):
            raise ValueError(f"segment {idx} is not an object")
        world = int(raw.get("world", 1))
        stage = int(raw.get("stage", idx + 1))
        start_time = parse_time(raw.get("start_time", raw.get("start", 0.0)))
        end_time = parse_time(raw.get("end_time", raw.get("end", 0.0)))
        if end_time <= start_time:
            raise ValueError(f"segment {idx} has end_time <= start_time")
        start_x = int(raw.get("start_x", 0))
        end_x = int(raw.get("end_x", goal_line_x_for_world_stage((world, stage), fallback=3200)))
        parsed.append(
            {
                "world": world,
                "stage": stage,
                "start_time": float(start_time),
                "end_time": float(end_time),
                "start_x": start_x,
                "end_x": end_x,
                "label": str(raw.get("label", f"{world}-{stage}")),
            }
        )
    return parsed


def write_template(path: Path) -> None:
    if path.exists():
        print(json.dumps({"ok": False, "reason": "template_exists", "path": str(path)}, indent=2))
        return
    template = {
        "notes": (
            "Fill start/end times from the reference video. Times can be seconds or MM:SS.mmm. "
            "end_x defaults to mario_runtime.STAGE_FLAG_LINE_X when omitted."
        ),
        "segments": [
            {"world": 1, "stage": 1, "start_time": "0:00.000", "end_time": "0:00.000"},
            {"world": 1, "stage": 2, "start_time": "0:00.000", "end_time": "0:00.000"},
            {"world": 1, "stage": 3, "start_time": "0:00.000", "end_time": "0:00.000"},
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "template": str(path.resolve())}, indent=2))


def build_arrays(segments: list[dict[str, Any]], trace_fps: float, progress_stride: int) -> dict[str, np.ndarray]:
    if trace_fps <= 0:
        raise ValueError("--trace-fps must be positive")
    step_dt = 1.0 / float(trace_fps)
    rows: list[dict[str, Any]] = []
    step = 0
    for segment in segments:
        duration = segment["end_time"] - segment["start_time"]
        samples = max(1, int(math.ceil(duration * trace_fps)))
        max_x = 0
        for local_idx in range(samples):
            t = segment["start_time"] + local_idx * step_dt
            alpha = min(1.0, max(0.0, (t - segment["start_time"]) / duration))
            x_value = int(round(segment["start_x"] + alpha * (segment["end_x"] - segment["start_x"])))
            max_x = max(max_x, x_value)
            ws = (int(segment["world"]), int(segment["stage"]))
            rows.append(
                {
                    "step": step,
                    "time": t,
                    "world": ws[0],
                    "stage": ws[1],
                    "stage_x": x_value,
                    "max_stage_x": max_x,
                    "stage_index": stage_order_index(ws),
                    "route_progress": stage_order_index(ws) * int(progress_stride) + x_value,
                    "mode": f"video_pace:{segment['label']}",
                }
            )
            step += 1
    return {
        "steps": np.asarray([row["step"] for row in rows], dtype=np.int32),
        "times": np.asarray([row["time"] for row in rows], dtype=np.float32),
        "worlds": np.asarray([row["world"] for row in rows], dtype=np.int16),
        "stages": np.asarray([row["stage"] for row in rows], dtype=np.int16),
        "stage_x": np.asarray([row["stage_x"] for row in rows], dtype=np.int32),
        "max_stage_x": np.asarray([row["max_stage_x"] for row in rows], dtype=np.int32),
        "y": np.zeros(len(rows), dtype=np.int16),
        "actions": np.full(len(rows), -1, dtype=np.int16),
        "stage_indices": np.asarray([row["stage_index"] for row in rows], dtype=np.int16),
        "route_progress": np.asarray([row["route_progress"] for row in rows], dtype=np.float32),
        "modes": np.asarray([row["mode"] for row in rows], dtype="U96"),
    }


def read_frame_at(cap: cv2.VideoCapture, time_seconds: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_seconds)) * 1000.0)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def write_contact_sheet(video_path: Path, segments: list[dict[str, Any]], output: Path, cols: int, width: int) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    samples: list[tuple[str, float]] = []
    for segment in segments:
        samples.append((f"{segment['label']} start", segment["start_time"]))
        mid = (segment["start_time"] + segment["end_time"]) / 2.0
        samples.append((f"{segment['label']} mid", mid))
        samples.append((f"{segment['label']} end", segment["end_time"]))
    tiles: list[np.ndarray] = []
    for label, timestamp in samples:
        frame = read_frame_at(cap, timestamp)
        if frame is None:
            continue
        scale = float(width) / float(frame.shape[1])
        height = max(1, int(round(frame.shape[0] * scale)))
        tile = cv2.resize(frame, (int(width), height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 34), (0, 0, 0), thickness=-1)
        cv2.putText(
            tile,
            f"{label} {timestamp:.2f}s",
            (6, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    cap.release()
    if not tiles:
        raise ValueError("No frames could be read for the contact sheet")
    cols = max(1, int(cols))
    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = int(width)
    rows = int(math.ceil(len(tiles) / cols))
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        canvas[row * tile_h : row * tile_h + tile.shape[0], col * tile_w : col * tile_w + tile.shape[1]] = tile
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def main() -> int:
    args = parse_args()
    segments_path = Path(args.segments_json)
    if args.write_template:
        write_template(segments_path)
        return 0

    segments = load_segments(segments_path)
    arrays = build_arrays(segments, float(args.trace_fps), int(args.progress_stride))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **arrays,
        source_name=np.asarray(args.source_name, dtype="U256"),
        source_url=np.asarray(args.source_url, dtype="U512"),
        source_video=np.asarray(args.video, dtype="U512"),
        progress_stride=np.asarray(int(args.progress_stride), dtype=np.int32),
        trace_fps=np.asarray(float(args.trace_fps), dtype=np.float32),
    )

    if args.video and args.contact_sheet:
        write_contact_sheet(
            Path(args.video),
            segments,
            Path(args.contact_sheet),
            cols=int(args.contact_sheet_cols),
            width=int(args.contact_sheet_width),
        )

    unique_stages: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for segment in segments:
        ws = (int(segment["world"]), int(segment["stage"]))
        if ws not in seen:
            seen.add(ws)
            unique_stages.append([ws[0], ws[1]])
    summary = {
        "output": str(output_path.resolve()),
        "segments_json": str(segments_path.resolve()),
        "source_name": args.source_name,
        "source_url": args.source_url,
        "source_video": args.video,
        "trace_fps": float(args.trace_fps),
        "progress_stride": int(args.progress_stride),
        "steps": int(len(arrays["steps"])),
        "first_world_stage": unique_stages[0],
        "last_world_stage": unique_stages[-1],
        "furthest_world_stage": max(unique_stages, key=lambda ws: stage_order_index(tuple(ws))),
        "max_route_progress": int(np.max(arrays["route_progress"])),
        "world_stages": unique_stages,
        "contact_sheet": str(Path(args.contact_sheet).resolve()) if args.contact_sheet else "",
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
