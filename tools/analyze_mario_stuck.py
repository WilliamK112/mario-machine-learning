from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario_runtime import (
    ACTION_SET_MAP,
    EnvConfig,
    STAGE_FLAG_LINE_X,
    effective_goal_line_x,
    load_env_config_for_model,
    make_single_env,
    render_rgb_frame,
    sanitize_x_position,
    stack_observations,
    stage_order_index,
    world_stage_from_info,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out a Mario policy and capture where it stalls or dies."
    )
    parser.add_argument("--model", required=True, help="Path to PPO .zip checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory for traces/screenshots.")
    parser.add_argument("--target-world", type=int, default=1)
    parser.add_argument("--target-stage", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=9000)
    parser.add_argument(
        "--policy-modes",
        default="deterministic,stochastic",
        help="Comma separated: deterministic,stochastic.",
    )
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--last-seconds", type=float, default=1.0)
    parser.add_argument(
        "--steps-per-second",
        type=float,
        default=15.0,
        help="Agent steps per playback second. frame_skip=4 is about 15.",
    )
    parser.add_argument("--stuck-seconds", type=float, default=5.0)
    parser.add_argument("--stuck-x-span", type=int, default=24)
    parser.add_argument("--stuck-y-span", type=int, default=80)
    parser.add_argument("--stuck-forward-epsilon", type=int, default=8)
    parser.add_argument("--frame-buffer", type=int, default=1200)
    parser.add_argument(
        "--noop-max",
        type=int,
        default=None,
        help="Override reset no-op count. Omit to use the model's train_config value.",
    )
    parser.add_argument(
        "--initial-noops-exact",
        type=int,
        default=0,
        help="Run this many NOOP actions after reset before evaluating the policy.",
    )
    parser.add_argument("--contact-cols", type=int, default=5)
    parser.add_argument("--export-max-width", type=int, default=512)
    parser.add_argument(
        "--eval-action-bias-jump",
        type=float,
        default=0.0,
        help="Temporary eval-only logit bias added to actions containing A.",
    )
    parser.add_argument(
        "--eval-action-bias-down",
        type=float,
        default=0.0,
        help="Temporary eval-only logit bias added to DOWN actions.",
    )
    parser.add_argument(
        "--eval-action-bias-noop",
        type=float,
        default=0.0,
        help="Temporary eval-only logit bias added to NOOP actions.",
    )
    parser.add_argument(
        "--eval-action-bias-left",
        type=float,
        default=0.0,
        help="Temporary eval-only logit bias added to LEFT actions.",
    )
    return parser.parse_args()


def action_label(action_set: str, action: int) -> str:
    actions = ACTION_SET_MAP.get(action_set, ACTION_SET_MAP["right_only"])
    if action < 0 or action >= len(actions):
        return str(action)
    buttons = actions[action]
    if not buttons:
        return "NOOP"
    return "+".join(buttons)


def action_indices_containing(action_set: str, button: str) -> list[int]:
    actions = ACTION_SET_MAP.get(action_set, ACTION_SET_MAP["right_only"])
    return [i for i, buttons in enumerate(actions) if button in buttons]


def noop_action_indices(action_set: str) -> list[int]:
    actions = ACTION_SET_MAP.get(action_set, ACTION_SET_MAP["right_only"])
    return [i for i, buttons in enumerate(actions) if buttons == ["NOOP"]]


def apply_eval_action_bias(
    model: PPO,
    *,
    action_set: str,
    jump: float,
    down: float,
    noop: float,
    left: float,
) -> dict[str, Any]:
    requested = {
        "jump": (action_indices_containing(action_set, "A"), float(jump)),
        "down": (action_indices_containing(action_set, "down"), float(down)),
        "noop": (noop_action_indices(action_set), float(noop)),
        "left": (action_indices_containing(action_set, "left"), float(left)),
    }
    applied: dict[str, Any] = {}
    with torch.no_grad():
        for name, (indices, bias_value) in requested.items():
            if bias_value == 0.0:
                continue
            for idx in indices:
                model.policy.action_net.bias[idx] += bias_value
            applied[name] = {
                "bias": bias_value,
                "indices": [int(idx) for idx in indices],
            }
    return applied


def life_count(info: dict[str, Any]) -> int | None:
    for key in ("life", "lives"):
        value = info.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [as_jsonable(v) for v in value]
    if isinstance(value, list):
        return [as_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    return value


def goal_for_stage(world_stage: tuple[int, int] | None, fallback: int = 3200) -> int:
    if world_stage is None:
        return int(fallback)
    return int(STAGE_FLAG_LINE_X.get((int(world_stage[0]), int(world_stage[1])), fallback))


def sanitized_stage_x(
    info: dict[str, Any],
    *,
    world_stage: tuple[int, int] | None,
    previous_by_stage: dict[tuple[int, int], int],
    fallback_goal: int,
) -> int:
    if world_stage is None:
        return 0
    previous = previous_by_stage.get(world_stage, 0)
    goal_x = goal_for_stage(world_stage, fallback=fallback_goal)
    x_pos = sanitize_x_position(
        info.get("x_pos", previous),
        previous_x_pos=previous,
        flag_get=bool(info.get("flag_get", False)),
        goal_line_x=goal_x,
    )
    previous_by_stage[world_stage] = int(x_pos)
    return int(x_pos)


def stage_pair(record: dict[str, Any]) -> tuple[int, int] | None:
    world = record.get("world")
    stage = record.get("stage")
    if world is None or stage is None:
        return None
    return int(world), int(stage)


def is_active_play_record(record: dict[str, Any]) -> bool:
    if bool(record.get("done", False)):
        return False
    time_left = record.get("time")
    if time_left is not None:
        try:
            if int(time_left) <= 0:
                return False
        except (TypeError, ValueError):
            pass
    life = record.get("life")
    if life is not None:
        try:
            if int(life) >= 10:
                return False
        except (TypeError, ValueError):
            pass
    return True


def find_stuck_windows(
    records: list[dict[str, Any]],
    *,
    target: tuple[int, int],
    window_steps: int,
    x_span_limit: int,
    y_span_limit: int,
    forward_epsilon: int,
) -> list[dict[str, Any]]:
    if window_steps <= 1:
        return []

    windows: list[dict[str, Any]] = []
    indices = [
        idx
        for idx, rec in enumerate(records)
        if stage_pair(rec) == target and is_active_play_record(rec)
    ]
    if len(indices) < window_steps:
        return []

    # Analyze only contiguous target-stage spans so a stage transition cannot fake a stall.
    spans: list[list[int]] = []
    current: list[int] = []
    prev = -2
    for idx in indices:
        if idx == prev + 1:
            current.append(idx)
        else:
            if current:
                spans.append(current)
            current = [idx]
        prev = idx
    if current:
        spans.append(current)

    for span in spans:
        if len(span) < window_steps:
            continue
        for offset in range(0, len(span) - window_steps + 1):
            window_indices = span[offset : offset + window_steps]
            chunk = [records[idx] for idx in window_indices]
            xs = [int(rec.get("x", 0)) for rec in chunk]
            ys = [int(rec.get("y", 0)) for rec in chunk if rec.get("y") is not None]
            actions = Counter(str(rec.get("action_label", rec.get("action"))) for rec in chunk)
            x_span = max(xs) - min(xs) if xs else 0
            y_span = max(ys) - min(ys) if ys else 0
            net_forward = xs[-1] - xs[0] if xs else 0
            dominant_action, dominant_count = actions.most_common(1)[0]
            dominant_ratio = dominant_count / float(len(chunk))
            if (
                x_span <= x_span_limit
                and net_forward <= forward_epsilon
                and (y_span <= y_span_limit or dominant_ratio >= 0.70)
            ):
                windows.append(
                    {
                        "start_step": int(chunk[0]["step"]),
                        "end_step": int(chunk[-1]["step"]),
                        "start_record_index": int(window_indices[0]),
                        "end_record_index": int(window_indices[-1]),
                        "x_min": int(min(xs)),
                        "x_max": int(max(xs)),
                        "x_span": int(x_span),
                        "y_min": int(min(ys)) if ys else None,
                        "y_max": int(max(ys)) if ys else None,
                        "y_span": int(y_span),
                        "net_forward": int(net_forward),
                        "dominant_action": dominant_action,
                        "dominant_action_ratio": float(dominant_ratio),
                        "action_counts": dict(actions),
                    }
                )

    # Keep the most relevant windows: latest first, then strongest stall.
    windows.sort(key=lambda item: (item["end_record_index"], -item["x_span"]), reverse=True)
    return windows[:20]


def annotate_frame(frame: np.ndarray, record: dict[str, Any], line2: str = "") -> np.ndarray:
    out = np.array(frame, copy=True)
    text1 = (
        f"step={record.get('step')} ws={record.get('world')}-{record.get('stage')} "
        f"x={record.get('x')} y={record.get('y')} a={record.get('action_label')}"
    )
    text2 = line2 or f"life={record.get('life')} status={record.get('status')} time={record.get('time')}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 39), (0, 0, 0), thickness=-1)
    cv2.putText(out, text1, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    cv2.putText(out, text2, (4, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)
    return out


def resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / float(frame.shape[1])
    size = (max_width, max(1, int(round(frame.shape[0] * scale))))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def save_contact_sheet(
    *,
    records: list[dict[str, Any]],
    frame_by_index: dict[int, np.ndarray],
    indices: list[int],
    output_path: Path,
    cols: int,
    max_width: int,
    note: str,
) -> list[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated: list[np.ndarray] = []
    saved_paths: list[str] = []
    for idx in indices:
        frame = frame_by_index.get(idx)
        if frame is None or idx < 0 or idx >= len(records):
            continue
        img = annotate_frame(frame, records[idx], line2=note)
        img = resize_frame(img, max_width)
        frame_path = output_path.parent / f"frame_{idx:06d}.png"
        imageio.imwrite(frame_path, img)
        saved_paths.append(str(frame_path.resolve()))
        annotated.append(img)

    if not annotated:
        return saved_paths

    cols = max(1, int(cols))
    rows = int(np.ceil(len(annotated) / float(cols)))
    h = max(img.shape[0] for img in annotated)
    w = max(img.shape[1] for img in annotated)
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, img in enumerate(annotated):
        r, c = divmod(i, cols)
        sheet[r * h : r * h + img.shape[0], c * w : c * w + img.shape[1]] = img
    imageio.imwrite(output_path, sheet)
    return saved_paths


def write_trace_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "step",
        "world",
        "stage",
        "x",
        "raw_x",
        "y",
        "action",
        "action_label",
        "reward",
        "life",
        "status",
        "time",
        "flag_get",
        "done",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def rollout_once(
    *,
    model: PPO,
    config: EnvConfig,
    model_path: Path,
    output_dir: Path,
    seed: int,
    deterministic: bool,
    target: tuple[int, int],
    max_steps: int,
    last_frame_count: int,
    stuck_window_steps: int,
    stuck_x_span: int,
    stuck_y_span: int,
    stuck_forward_epsilon: int,
    frame_buffer_size: int,
    contact_cols: int,
    export_max_width: int,
    initial_noops_exact: int,
) -> dict[str, Any]:
    run_name = f"{'det' if deterministic else 'sto'}_seed_{seed}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = make_single_env(config, seed=seed)
    obs, reset_info = env.reset(seed=seed)
    frame_stack: deque[np.ndarray] = deque(maxlen=config.frame_stack)
    for _ in range(config.frame_stack):
        frame_stack.append(obs)

    action_set = config.action_set
    fallback_ws = (int(config.world), int(config.stage))
    current_ws = world_stage_from_info(reset_info, fallback_ws)
    furthest_ws = current_ws
    previous_x_by_stage: dict[tuple[int, int], int] = {}
    fallback_goal = effective_goal_line_x(config)
    target_goal_x = goal_for_stage(target, fallback=2528)
    target_max_x = 0
    target_reached = False
    target_exited = False
    target_first_idx: int | None = None
    target_last_idx: int | None = None
    stage_clears = 0
    prev_life = life_count(reset_info)
    life_losses = 0
    target_death_indices: list[int] = []
    event_idx: int | None = None
    event_reason = ""
    captured_events: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    frame_buffer: deque[tuple[int, np.ndarray]] = deque(maxlen=frame_buffer_size)
    recent_frames: deque[tuple[int, np.ndarray]] = deque(maxlen=max(1, last_frame_count))

    done = False
    step = 0
    while not done and step < max_steps:
        stacked_obs = stack_observations(frame_stack)
        action, _state = model.predict(stacked_obs, deterministic=deterministic)
        action_int = int(action)
        obs, reward, terminated, truncated, info = env.step(action_int)
        frame_stack.append(obs)
        done = bool(terminated or truncated)
        step += 1

        ws = world_stage_from_info(info, current_ws)
        x_pos = sanitized_stage_x(
            info,
            world_stage=ws,
            previous_by_stage=previous_x_by_stage,
            fallback_goal=fallback_goal,
        )
        raw_x = info.get("x_pos")
        try:
            raw_x_json = int(raw_x)
        except (TypeError, ValueError):
            raw_x_json = raw_x
        y_pos = info.get("y_pos")
        try:
            y_json = int(y_pos)
        except (TypeError, ValueError):
            y_json = y_pos

        if (
            ws is not None
            and current_ws is not None
            and ws != current_ws
            and stage_order_index(ws) > stage_order_index(current_ws)
        ):
            stage_clears += 1
            if current_ws == target:
                target_exited = True
        if stage_order_index(ws) > stage_order_index(furthest_ws):
            furthest_ws = ws
        current_ws = ws

        if ws == target:
            target_reached = True
            target_max_x = max(target_max_x, int(x_pos))
            if target_first_idx is None:
                target_first_idx = len(records)
            target_last_idx = len(records)

        cur_life = life_count(info)
        record_index = len(records)
        record = {
            "step": int(step),
            "world": int(ws[0]) if ws is not None else None,
            "stage": int(ws[1]) if ws is not None else None,
            "x": int(x_pos),
            "raw_x": raw_x_json,
            "y": y_json,
            "action": action_int,
            "action_label": action_label(action_set, action_int),
            "reward": float(reward),
            "life": cur_life,
            "status": str(info.get("status")) if info.get("status") is not None else None,
            "time": int(info["time"]) if info.get("time") is not None else None,
            "flag_get": bool(info.get("flag_get", False)),
            "done": bool(done),
        }
        records.append(record)
        frame = None
        try:
            frame = render_rgb_frame(env)
            frame_buffer.append((record_index, frame))
            recent_frames.append((record_index, frame))
        except Exception:
            pass

        life_lost = cur_life is not None and prev_life is not None and cur_life < prev_life
        if life_lost:
            life_losses += 1
            if ws == target:
                target_death_indices.append(record_index)
                event_idx = record_index
                event_reason = "life_loss_on_target"
                captured_events.append(
                    {
                        "reason": "life_loss_on_target",
                        "record_index": int(record_index),
                        "frame_indices": [int(idx) for idx, _frame in recent_frames],
                        "frames": [(idx, np.array(_frame, copy=True)) for idx, _frame in recent_frames],
                    }
                )
        if cur_life is not None:
            prev_life = cur_life

        if done and ws == target and event_idx is None:
            event_idx = record_index
            event_reason = "done_on_target"
            captured_events.append(
                {
                    "reason": "done_on_target",
                    "record_index": int(record_index),
                    "frame_indices": [int(idx) for idx, _frame in recent_frames],
                    "frames": [(idx, np.array(_frame, copy=True)) for idx, _frame in recent_frames],
                }
            )

    env.close()

    stuck_windows = find_stuck_windows(
        records,
        target=target,
        window_steps=stuck_window_steps,
        x_span_limit=stuck_x_span,
        y_span_limit=stuck_y_span,
        forward_epsilon=stuck_forward_epsilon,
    )
    if stuck_windows and (
        event_idx is None or int(stuck_windows[0]["end_record_index"]) > int(event_idx)
    ):
        event_idx = int(stuck_windows[0]["end_record_index"])
        event_reason = "latest_stuck_window"
    if event_idx is None and target_last_idx is not None:
        active_target_indices = [
            idx
            for idx, rec in enumerate(records)
            if stage_pair(rec) == target and is_active_play_record(rec)
        ]
        event_idx = int(active_target_indices[-1] if active_target_indices else target_last_idx)
        event_reason = "last_target_frame"
    if event_idx is None and records:
        event_idx = len(records) - 1
        event_reason = "episode_end"

    frame_by_index = {idx: frame for idx, frame in frame_buffer}
    start_idx = max(0, int(event_idx or 0) - max(1, last_frame_count) + 1)
    last_indices = list(range(start_idx, int(event_idx or 0) + 1))
    if not any(idx in frame_by_index for idx in last_indices):
        for captured in reversed(captured_events):
            if int(captured["record_index"]) == int(event_idx or -1):
                frame_by_index = {idx: frame for idx, frame in captured["frames"]}
                last_indices = list(captured["frame_indices"])
                break
    screenshot_dir = run_dir / "last_second"
    screenshot_paths = save_contact_sheet(
        records=records,
        frame_by_index=frame_by_index,
        indices=last_indices,
        output_path=screenshot_dir / "contact_sheet.png",
        cols=contact_cols,
        max_width=export_max_width,
        note=f"event={event_reason}; 1-2 goal_x={target_goal_x}",
    )

    action_counts_target = Counter(
        rec["action_label"] for rec in records if stage_pair(rec) == target
    )
    action_counts_event = Counter(
        records[idx]["action_label"]
        for idx in last_indices
        if 0 <= idx < len(records)
    )
    target_records = [
        rec
        for rec in records
        if stage_pair(rec) == target and is_active_play_record(rec)
    ]
    target_x_tail = [int(rec["x"]) for rec in target_records[-90:]]

    captured_event_summaries: list[dict[str, Any]] = []
    for captured_idx, captured in enumerate(captured_events):
        captured_frame_by_index = {idx: frame for idx, frame in captured["frames"]}
        event_dir = run_dir / "events" / f"{captured_idx:02d}_{captured['reason']}_{captured['record_index']}"
        paths = save_contact_sheet(
            records=records,
            frame_by_index=captured_frame_by_index,
            indices=list(captured["frame_indices"]),
            output_path=event_dir / "contact_sheet.png",
            cols=contact_cols,
            max_width=export_max_width,
            note=f"event={captured['reason']}; 1-2 goal_x={target_goal_x}",
        )
        captured_event_summaries.append(
            {
                "reason": captured["reason"],
                "record_index": int(captured["record_index"]),
                "record": records[int(captured["record_index"])],
                "contact_sheet": str((event_dir / "contact_sheet.png").resolve()),
                "frames": paths,
            }
        )

    write_trace_csv(records, run_dir / "trace.csv")
    target_remaining = max(0, int(target_goal_x) - int(target_max_x))
    summary = {
        "model": str(model_path.resolve()),
        "seed": int(seed),
        "initial_noops_exact": int(initial_noops_exact),
        "deterministic": bool(deterministic),
        "steps": int(step),
        "done": bool(done),
        "stage_clears": int(stage_clears),
        "life_losses": int(life_losses),
        "furthest_world_stage": list(furthest_ws) if furthest_ws is not None else None,
        "furthest_world_stage_index": int(stage_order_index(furthest_ws)),
        "target_world_stage": list(target),
        "target_stage_reached": bool(target_reached),
        "target_stage_exited": bool(target_exited),
        "target_first_record_index": target_first_idx,
        "target_last_record_index": target_last_idx,
        "target_stage_goal_x": int(target_goal_x),
        "target_stage_max_x": int(target_max_x),
        "target_remaining_distance": int(target_remaining),
        "target_tail_x_min": int(min(target_x_tail)) if target_x_tail else None,
        "target_tail_x_max": int(max(target_x_tail)) if target_x_tail else None,
        "target_tail_x_span": (
            int(max(target_x_tail) - min(target_x_tail)) if target_x_tail else None
        ),
        "target_death_record_indices": target_death_indices,
        "selected_event": {
            "reason": event_reason,
            "record_index": int(event_idx) if event_idx is not None else None,
            "record": records[int(event_idx)] if event_idx is not None and records else None,
        },
        "action_counts_target": dict(action_counts_target),
        "action_counts_last_second": dict(action_counts_event),
        "stuck_windows": stuck_windows[:5],
        "captured_events": captured_event_summaries,
        "trace_csv": str((run_dir / "trace.csv").resolve()),
        "last_second_contact_sheet": str((screenshot_dir / "contact_sheet.png").resolve()),
        "last_second_frames": screenshot_paths,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(as_jsonable(summary), indent=2),
        encoding="utf-8",
    )
    return summary


def parse_modes(raw_modes: str) -> list[bool]:
    modes: list[bool] = []
    for token in raw_modes.split(","):
        token = token.strip().lower()
        if token == "deterministic":
            modes.append(True)
        elif token == "stochastic":
            modes.append(False)
    return modes or [True]


def quality_key(summary: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(summary.get("furthest_world_stage_index") or 0),
        int(summary.get("target_stage_exited") or False),
        int(summary.get("target_stage_max_x") or 0),
        -int(summary.get("target_remaining_distance") or 999999),
    )


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_env_config_for_model(
        model_path,
        fallback=EnvConfig(n_envs=1, noop_max=0, end_on_flag=True),
    )
    config.n_envs = 1
    if args.noop_max is not None:
        config.noop_max = int(args.noop_max)
    config.initial_noops_exact = int(args.initial_noops_exact)

    model = PPO.load(model_path, device=args.device)
    eval_action_bias = apply_eval_action_bias(
        model,
        action_set=config.action_set,
        jump=float(args.eval_action_bias_jump),
        down=float(args.eval_action_bias_down),
        noop=float(args.eval_action_bias_noop),
        left=float(args.eval_action_bias_left),
    )
    if eval_action_bias:
        print(f"eval_action_bias={json.dumps(eval_action_bias, sort_keys=True)}")
    target = (int(args.target_world), int(args.target_stage))
    last_frame_count = max(1, int(round(args.last_seconds * args.steps_per_second)))
    stuck_window_steps = max(2, int(round(args.stuck_seconds * args.steps_per_second)))

    summaries: list[dict[str, Any]] = []
    for seed_idx in range(max(1, int(args.seeds))):
        seed = int(args.seed_base) + seed_idx
        for deterministic in parse_modes(args.policy_modes):
            print(f"rollout seed={seed} deterministic={deterministic}")
            summary = rollout_once(
                model=model,
                config=config,
                model_path=model_path,
                output_dir=output_dir,
                seed=seed,
                deterministic=deterministic,
                target=target,
                max_steps=int(args.max_steps),
                last_frame_count=last_frame_count,
                stuck_window_steps=stuck_window_steps,
                stuck_x_span=int(args.stuck_x_span),
                stuck_y_span=int(args.stuck_y_span),
                stuck_forward_epsilon=int(args.stuck_forward_epsilon),
                frame_buffer_size=int(args.frame_buffer),
                contact_cols=int(args.contact_cols),
                export_max_width=int(args.export_max_width),
                initial_noops_exact=int(args.initial_noops_exact),
            )
            summaries.append(summary)
            summary["eval_action_bias"] = eval_action_bias
            print(
                "  "
                f"furthest={summary['furthest_world_stage']} "
                f"1-2_max_x={summary['target_stage_max_x']} "
                f"remaining={summary['target_remaining_distance']} "
                f"event={summary['selected_event']['reason']} "
                f"at={summary['selected_event']['record']}"
            )

    best = max(summaries, key=quality_key) if summaries else None
    master = {
        "model": str(model_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "target_world_stage": [int(args.target_world), int(args.target_stage)],
        "target_stage_goal_x": goal_for_stage(target, fallback=2528),
        "eval_action_bias": eval_action_bias,
        "num_rollouts": len(summaries),
        "best_run": best,
        "rollouts": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(as_jsonable(master), indent=2),
        encoding="utf-8",
    )
    if best:
        print("BEST")
        print(json.dumps(as_jsonable(best), indent=2))


if __name__ == "__main__":
    main()
