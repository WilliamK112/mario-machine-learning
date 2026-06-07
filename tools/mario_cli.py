from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv-gpu" / "Scripts" / "python.exe"


def emit(data: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(json_safe(data), indent=2, sort_keys=True))
    else:
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"{key}: {value}")
        else:
            print(data)


def json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def process_snapshot() -> list[dict[str, Any]]:
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -match 'train_mario.py|watch_mario_all_levels.ps1' } | "
        "Select-Object ProcessId,ParentProcessId,Name,WorkingSetSize,CommandLine | "
        "ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout = result.stdout.strip()
    if not stdout:
        return []
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return [{"error": "could_not_parse_process_json", "raw": stdout}]
    if isinstance(payload, dict):
        payload = [payload]
    return [
        {
            "pid": int(item.get("ProcessId", 0)),
            "parent_pid": int(item.get("ParentProcessId", 0)),
            "name": item.get("Name"),
            "memory_gb": round(float(item.get("WorkingSetSize", 0)) / (1024**3), 3),
            "command": item.get("CommandLine"),
        }
        for item in payload
    ]


def latest_summaries(limit: int = 5) -> list[dict[str, Any]]:
    runs_dir = REPO / "runs"
    files = sorted(
        runs_dir.glob("mario_all_levels*/**/summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    summaries: list[dict[str, Any]] = []
    for path in files[:limit]:
        data = read_json(path)
        if not data:
            continue
        summaries.append(
            {
                "path": str(path),
                "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                "num_timesteps": data.get("num_timesteps"),
                "best_world_stage": data.get("best_world_stage"),
                "best_world_stage_index": data.get("best_world_stage_index"),
                "best_remaining_distance": data.get("best_remaining_distance"),
                "best_max_x": data.get("best_max_x"),
                "average_stage_clears": data.get("average_stage_clears"),
                "best_stage_clears": data.get("best_stage_clears"),
            }
        )
    return summaries


def command_status(args: argparse.Namespace) -> int:
    data = {
        "repo": str(REPO),
        "processes": process_snapshot(),
        "latest_summaries": latest_summaries(args.limit),
    }
    emit(data, as_json=args.json)
    return 0


def command_smoke_env(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(REPO))
    from mario_runtime import ACTION_SET_MAP, EnvConfig, make_single_env, world_stage_from_info

    cfg = EnvConfig(
        world=args.world,
        stage=args.stage,
        whole_game=args.whole_game,
        action_set=args.action_set,
        n_envs=1,
        noop_max=args.noop_max,
        end_on_flag=False if args.whole_game else args.end_on_flag,
    )
    env = make_single_env(cfg, seed=args.seed)
    obs, info = env.reset(seed=args.seed)
    actions = ACTION_SET_MAP.get(args.action_set, ACTION_SET_MAP["right_only"])
    action = min(max(args.action, 0), len(actions) - 1)
    trace: list[dict[str, Any]] = []
    total_reward = 0.0
    done = False
    for step in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = bool(terminated or truncated)
        ws = world_stage_from_info(info, (args.world, args.stage))
        trace.append(
            {
                "step": step + 1,
                "world_stage": list(ws) if ws is not None else None,
                "x_pos": info.get("x_pos"),
                "y_pos": info.get("y_pos"),
                "life": info.get("life"),
                "flag_get": bool(info.get("flag_get", False)),
                "reward": float(reward),
                "done": done,
            }
        )
        if done:
            break
    env.close()
    data = {
        "ok": True,
        "obs_shape": list(getattr(obs, "shape", [])),
        "steps": len(trace),
        "total_reward": total_reward,
        "final": trace[-1] if trace else {},
        "trace_tail": trace[-5:],
    }
    emit(data, as_json=args.json)
    return 0


def run_subprocess(cmd: list[str]) -> int:
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO, text=True)
    return int(proc.returncode)


def command_smoke_train(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir) if args.run_dir else (
        REPO / "runs" / f"smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    cmd = [
        str(PYTHON),
        "train_mario.py",
        "--run-dir",
        str(run_dir),
        "--timesteps",
        str(args.timesteps),
        "--device",
        "cpu",
        "--n-envs",
        "1",
        "--action-set",
        args.action_set,
        "--checkpoint-freq",
        str(max(1, args.timesteps)),
        "--eval-freq",
        "0",
        "--preview-freq",
        "0",
        "--no-preview-video",
    ]
    if args.whole_game:
        cmd += ["--whole-game", "--no-end-on-flag"]
    return run_subprocess(cmd)


def command_analyze_stuck(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir) if args.output_dir else (
        REPO
        / "runs"
        / "diagnostics"
        / f"stuck_1-2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    cmd = [
        str(PYTHON),
        "tools/analyze_mario_stuck.py",
        "--model",
        args.model,
        "--output-dir",
        str(output_dir),
        "--target-world",
        str(args.target_world),
        "--target-stage",
        str(args.target_stage),
        "--seeds",
        str(args.seeds),
        "--seed-base",
        str(args.seed_base),
        "--policy-modes",
        args.policy_modes,
        "--max-steps",
        str(args.max_steps),
        "--device",
        args.device,
    ]
    return run_subprocess(cmd)


def command_plot_progress(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("mario_all_levels*/**/summary.json")):
        data = read_json(path)
        if not data:
            continue
        rows.append(
            {
                "path": str(path),
                "num_timesteps": data.get("num_timesteps", ""),
                "per_stage_metric": bool(data.get("episode_goal_line_xs")),
                "episode_goal_line_xs": json.dumps(data.get("episode_goal_line_xs", [])),
                "best_world_stage": "-".join(map(str, data.get("best_world_stage") or [])),
                "best_world_stage_index": data.get("best_world_stage_index", ""),
                "best_remaining_distance": data.get("best_remaining_distance", ""),
                "best_max_x": data.get("best_max_x", ""),
                "average_stage_clears": data.get("average_stage_clears", ""),
                "best_stage_clears": data.get("best_stage_clears", ""),
                "average_return": data.get("average_return", ""),
            }
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "num_timesteps",
                "per_stage_metric",
                "episode_goal_line_xs",
                "best_world_stage",
                "best_world_stage_index",
                "best_remaining_distance",
                "best_max_x",
                "average_stage_clears",
                "best_stage_clears",
                "average_return",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    emit({"rows": len(rows), "csv": str(out.resolve())}, as_json=args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stable local commands for the Mario RL repo.")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Show live training processes and newest summaries.")
    status.add_argument("--limit", type=int, default=5)
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=command_status)

    smoke_env = sub.add_parser("smoke-env", help="Reset and step the Mario env.")
    smoke_env.add_argument("--world", type=int, default=1)
    smoke_env.add_argument("--stage", type=int, default=1)
    smoke_env.add_argument("--whole-game", action="store_true")
    smoke_env.add_argument("--action-set", default="simple_pipe")
    smoke_env.add_argument("--seed", type=int, default=123)
    smoke_env.add_argument("--steps", type=int, default=16)
    smoke_env.add_argument("--action", type=int, default=1)
    smoke_env.add_argument("--noop-max", type=int, default=0)
    smoke_env.add_argument("--end-on-flag", action="store_true")
    smoke_env.add_argument("--json", action="store_true")
    smoke_env.set_defaults(func=command_smoke_env)

    smoke_train = sub.add_parser("smoke-train", help="Run a tiny CPU PPO training loop.")
    smoke_train.add_argument("--timesteps", type=int, default=128)
    smoke_train.add_argument("--run-dir", default="")
    smoke_train.add_argument("--whole-game", action="store_true")
    smoke_train.add_argument("--action-set", default="simple_pipe")
    smoke_train.set_defaults(func=command_smoke_train)

    analyze = sub.add_parser("analyze-stuck", help="Run stuck/death analysis for a checkpoint.")
    analyze.add_argument("--model", required=True)
    analyze.add_argument("--output-dir", default="")
    analyze.add_argument("--target-world", type=int, default=1)
    analyze.add_argument("--target-stage", type=int, default=2)
    analyze.add_argument("--seeds", type=int, default=1)
    analyze.add_argument("--seed-base", type=int, default=9201)
    analyze.add_argument("--policy-modes", default="deterministic,stochastic")
    analyze.add_argument("--max-steps", type=int, default=12000)
    analyze.add_argument("--device", default="cpu")
    analyze.set_defaults(func=command_analyze_stuck)

    plot = sub.add_parser("plot-progress", help="Export eval/preview summary metrics to CSV.")
    plot.add_argument("--runs-dir", default=str(REPO / "runs"))
    plot.add_argument("--out", default=str(REPO / "analysis" / "mario_progress.csv"))
    plot.add_argument("--json", action="store_true")
    plot.set_defaults(func=command_plot_progress)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
