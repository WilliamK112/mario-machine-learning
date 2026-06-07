"""Bounded 2-2 chain-conditioned BC, then full policy-chain gate (background-safe)."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = "bc_teacher_2-2_chain_prefix_suffix_anchor"
GATE_ROOT = ROOT / "analysis"


def run_dir() -> Path:
    return ROOT / "runs" / f"{RUN_NAME}_{RUN_TAG}"

FROZEN = {
    "1-1": ROOT / "runs/bc_teacher_1-1_aug_corrections_jumpbias15_20260517_1639/bc_policy.zip",
    "1-2": ROOT / "runs/bc_teacher_1-2_direct_tail_linearhead_20260517_1813/best_rollout_resaved.zip",
    "1-3": ROOT / "runs/bc_teacher_1-3_gap_focus_bias_sweep_20260517_1746/rightb_plus_0p8.zip",
    "1-4": ROOT / "runs/bc_teacher_1-4_policy_tail_rescue_20260517_1828/bc_policy.zip",
    "2-1": ROOT / "runs/bc_teacher_2-1_from_hybrid_resetstack_20260517_1838/bc_policy.zip",
    "2-2_baseline": ROOT / "runs/bc_teacher_2-2_from_hybrid_resetstack_20260517_1849/bc_policy.zip",
}

DEMOS = [
    ROOT
    / "analysis/teacher_demos_policy_branch/policy_chain_2-2_self_prefix_failure_resetstack_20260518_0002.npz",
    ROOT
    / "analysis/teacher_demos_policy_branch/teacher_policy_chain_2-2_suffix_x1150_to_clear_20260517_2307.npz",
]

BASELINE_CHAIN_22_X = 1235
EPOCHS = 18
BATCH_SIZE = 64
LEARNING_RATE = "2e-5"
ENTROPY_COEF = "0.0002"
ANCHOR_KL_COEF = "0.1"
ANCHOR_STAGE_X_MIN = 0
ANCHOR_STAGE_X_MAX = 1100
OVERSAMPLE_STAGE_X_MIN = 1000
OVERSAMPLE_STAGE_X_MAX = 1400
OVERSAMPLE_FACTOR = 10
ROLLOUT_EVAL_EVERY = 5
ROLLOUT_EVAL_EPISODES = 3
ROLLOUT_EVAL_STEPS = 3400


def run(cmd: list[str], log_name: str) -> int:
    log_dir = run_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(proc.returncode)


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def chain_gate(level22_model: Path, label: str) -> dict:
    out_dir = GATE_ROOT / f"policy_chain_gate_{label}_{RUN_TAG}"
    cmd = [
        str(PYTHON),
        "tools/run_mario_policy_chain.py",
        "--level11-model",
        str(FROZEN["1-1"]),
        "--level12-model",
        str(FROZEN["1-2"]),
        "--level13-model",
        str(FROZEN["1-3"]),
        "--level14-model",
        str(FROZEN["1-4"]),
        "--level21-model",
        str(FROZEN["2-1"]),
        "--level22-model",
        str(level22_model),
        "--output-dir",
        str(out_dir),
        "--seed",
        "47",
        "--device",
        "cpu",
        "--deterministic",
        "--initial-noops-exact",
        "16",
        "--stop-world",
        "2",
        "--stop-stage",
        "3",
        "--stop-on-death",
        "--reset-stack-on-stage-change",
        "--max-steps",
        "5200",
        "--no-video",
    ]
    code = run(cmd, f"gate_{label}.log")
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return {"ok": False, "code": code, "output_dir": str(out_dir)}
    summary = load_summary(summary_path)
    summary["gate_output_dir"] = str(out_dir)
    summary["gate_code"] = code
    summary["ok"] = True
    return summary


def better_than_baseline(summary: dict) -> bool:
    if int(summary.get("deaths", 99)) > 0:
        return False
    stage_x = summary.get("stage_max_x") or {}
    x22 = int(stage_x.get("2-2", 0))
    furthest = summary.get("furthest_world_stage") or [0, 0]
    if tuple(furthest) >= (2, 3):
        return True
    return x22 > BASELINE_CHAIN_22_X


def main() -> int:
    lock_path = ROOT / "runs" / ".run_22_chain_bc_and_gate.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            age_s = datetime.now().timestamp() - lock_path.stat().st_mtime
        except OSError:
            age_s = 0.0
        if age_s < 6 * 3600:
            print(f"another orchestrator appears active ({lock_path}, age={age_s:.0f}s)", flush=True)
            return 0
    lock_path.write_text(RUN_TAG, encoding="utf-8")

    for path in [*FROZEN.values(), *DEMOS, FROZEN["2-2_baseline"]]:
        if not Path(path).exists():
            print(f"missing required path: {path}", flush=True)
            return 2

    output_dir = run_dir()
    demo_args = []
    for demo_path in DEMOS:
        demo_args.extend(["--demo", str(demo_path)])

    train_cmd = [
        str(PYTHON),
        "tools/train_mario_bc.py",
        *demo_args,
        "--output-dir",
        str(output_dir),
        "--device",
        "auto",
        "--seed",
        "47",
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        str(BATCH_SIZE),
        "--learning-rate",
        str(LEARNING_RATE),
        "--entropy-coef",
        str(ENTROPY_COEF),
        "--init-model",
        str(FROZEN["2-2_baseline"]),
        "--anchor-model",
        str(FROZEN["2-2_baseline"]),
        "--anchor-kl-coef",
        str(ANCHOR_KL_COEF),
        "--anchor-stage-x-min",
        str(ANCHOR_STAGE_X_MIN),
        "--anchor-stage-x-max",
        str(ANCHOR_STAGE_X_MAX),
        "--train-scope",
        "policy_head",
        "--only-world-stage",
        "2-2",
        "--oversample-stage-x-min",
        str(OVERSAMPLE_STAGE_X_MIN),
        "--oversample-stage-x-max",
        str(OVERSAMPLE_STAGE_X_MAX),
        "--oversample-factor",
        str(OVERSAMPLE_FACTOR),
        "--rollout-eval-every",
        str(ROLLOUT_EVAL_EVERY),
        "--rollout-eval-episodes",
        str(ROLLOUT_EVAL_EPISODES),
        "--rollout-eval-steps",
        str(ROLLOUT_EVAL_STEPS),
    ]
    print(f"training -> {output_dir}", flush=True)
    train_code = run(train_cmd, "train_stdout.log")
    summary_path = output_dir / "summary.json"
    if train_code != 0 or not summary_path.exists():
        print(f"training failed code={train_code}", flush=True)
        return train_code or 1

    train_summary = load_summary(summary_path)
    best_rollout = output_dir / "best_rollout.zip"
    bc_policy = output_dir / "bc_policy.zip"
    candidates: list[tuple[str, Path]] = []
    if best_rollout.exists():
        candidates.append(("best_rollout", best_rollout))
    if bc_policy.exists():
        candidates.append(("bc_policy", bc_policy))

    report: dict = {
        "run_dir": str(output_dir),
        "train_summary": train_summary,
        "baseline_chain_22_x": BASELINE_CHAIN_22_X,
        "gates": {},
        "promoted": None,
    }

    baseline_gate = chain_gate(FROZEN["2-2_baseline"], "baseline22")
    report["gates"]["baseline22"] = baseline_gate

    best_label = ""
    best_model = Path()
    best_summary: dict | None = None
    for label, model_path in candidates:
        gate_summary = chain_gate(model_path, label)
        report["gates"][label] = gate_summary
        if gate_summary.get("ok") is False:
            continue
        if better_than_baseline(gate_summary):
            if best_summary is None:
                best_label, best_model, best_summary = label, model_path, gate_summary
            else:
                old_x = int((best_summary.get("stage_max_x") or {}).get("2-2", 0))
                new_x = int((gate_summary.get("stage_max_x") or {}).get("2-2", 0))
                old_f = tuple(best_summary.get("furthest_world_stage") or [0, 0])
                new_f = tuple(gate_summary.get("furthest_world_stage") or [0, 0])
                if new_f > old_f or (new_f == old_f and new_x > old_x):
                    best_label, best_model, best_summary = label, model_path, gate_summary

    if best_summary is not None:
        report["promoted"] = {
            "label": best_label,
            "model": str(best_model),
            "gate": best_summary,
        }

    report_path = output_dir / "chain_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if lock_path.exists():
        lock_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
