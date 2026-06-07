"""Round 4: record v3b chain failure -> branch search -> BC -> full-chain gate."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


def _paths_for_tag(tag: str) -> None:
    global TAG, PREFIX_DEMO, SEARCH_OUT, TAIL_DEMO, BC_RUN, LOG_DIR
    TAG = tag
    PREFIX_DEMO = ROOT / f"analysis/teacher_demos_policy_branch/policy_chain_2-2_self_prefix_v3b_resetstack_{TAG}.npz"
    SEARCH_OUT = ROOT / f"analysis/policy_chain_branch_2-2_v3b_fail_x1563_search_{TAG}"
    TAIL_DEMO = ROOT / f"analysis/teacher_demos_policy_branch/teacher_policy_chain_2-2_v3b_fail_x1563_tail_{TAG}.npz"
    BC_RUN = ROOT / f"runs/bc_teacher_2-2_chain_v3b_prefix_search_tail_{TAG}"
    LOG_DIR = ROOT / f"runs/orchestrator_round4_{TAG}"


TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
_paths_for_tag(TAG)

PREFIX_DEMO: Path
SEARCH_OUT: Path
TAIL_DEMO: Path
BC_RUN: Path
LOG_DIR: Path

FROZEN_11_21 = {
    "1-1": ROOT / "runs/bc_teacher_1-1_aug_corrections_jumpbias15_20260517_1639/bc_policy.zip",
    "1-2": ROOT / "runs/bc_teacher_1-2_direct_tail_linearhead_20260517_1813/best_rollout_resaved.zip",
    "1-3": ROOT / "runs/bc_teacher_1-3_gap_focus_bias_sweep_20260517_1746/rightb_plus_0p8.zip",
    "1-4": ROOT / "runs/bc_teacher_1-4_policy_tail_rescue_20260517_1828/bc_policy.zip",
    "2-1": ROOT / "runs/bc_teacher_2-1_from_hybrid_resetstack_20260517_1838/bc_policy.zip",
}
LEVEL22_V3B = ROOT / "runs/bc_teacher_2-2_chain_prefix_v3b_tail_20260518_021236/best_rollout.zip"
LEVEL22_FROZEN = ROOT / "runs/bc_teacher_2-2_from_hybrid_resetstack_20260517_1849/bc_policy.zip"
PIPE_CLEAR_DEMO = ROOT / "analysis/teacher_demos_policy_branch/teacher_policy_chain_2-2_real_x1230_to_x1900_tail_20260517_2040.npz"

BASELINE_CHAIN_22_X = 1563


def run(cmd: list[str], log_name: str) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / log_name
    with log_path.open("w", encoding="utf-8") as log_file:
        return int(
            subprocess.run(cmd, cwd=str(ROOT), stdout=log_file, stderr=subprocess.STDOUT, check=False).returncode
        )


def chain_models(level22: Path) -> list[str]:
    args = [
        str(PYTHON),
        "tools/run_mario_policy_chain.py",
        "--level11-model",
        str(FROZEN_11_21["1-1"]),
        "--level12-model",
        str(FROZEN_11_21["1-2"]),
        "--level13-model",
        str(FROZEN_11_21["1-3"]),
        "--level14-model",
        str(FROZEN_11_21["1-4"]),
        "--level21-model",
        str(FROZEN_11_21["2-1"]),
        "--level22-model",
        str(level22),
    ]
    return args


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="", help="Reuse a fixed run tag (for resume).")
    parser.add_argument("--skip-prefix-record", action="store_true")
    parser.add_argument("--skip-search", action="store_true")
    args = parser.parse_args()
    if args.tag:
        _paths_for_tag(args.tag)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    for path in [*FROZEN_11_21.values(), LEVEL22_V3B, LEVEL22_FROZEN, PIPE_CLEAR_DEMO]:
        if not path.exists():
            print(f"missing: {path}", flush=True)
            return 2

    if not args.skip_prefix_record:
        PREFIX_DEMO.parent.mkdir(parents=True, exist_ok=True)
    record_cmd = [
        *chain_models(LEVEL22_V3B),
        "--output-dir",
        str(ROOT / f"analysis/policy_chain_record_v3b_prefix_{TAG}"),
        "--demo-output",
        str(PREFIX_DEMO),
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
    if args.skip_prefix_record:
        if not PREFIX_DEMO.exists():
            print(f"missing prefix demo: {PREFIX_DEMO}", flush=True)
            return 1
        print(f"skip prefix record; using {PREFIX_DEMO}", flush=True)
    else:
        print("recording policy-chain prefix demo...", flush=True)
        record_code = run(record_cmd, "01_record_prefix.log")
        if not PREFIX_DEMO.exists():
            print(f"prefix demo recording failed (exit={record_code}, missing {PREFIX_DEMO})", flush=True)
            return 1
        if record_code not in (0, 2):
            print(f"prefix demo recording error exit={record_code}", flush=True)
            return record_code

    if args.skip_search:
        if not TAIL_DEMO.exists():
            print(f"missing tail demo: {TAIL_DEMO}", flush=True)
            return 1
        print(f"skip search; using {TAIL_DEMO}", flush=True)
    else:
        SEARCH_OUT.mkdir(parents=True, exist_ok=True)
    search_cmd = [
        str(PYTHON),
        "tools/search_mario_policy_chain_branch.py",
        "--level11-model",
        str(FROZEN_11_21["1-1"]),
        "--level12-model",
        str(FROZEN_11_21["1-2"]),
        "--level13-model",
        str(FROZEN_11_21["1-3"]),
        "--level14-model",
        str(FROZEN_11_21["1-4"]),
        "--level21-model",
        str(FROZEN_11_21["2-1"]),
        "--level22-model",
        str(LEVEL22_V3B),
        "--output-dir",
        str(SEARCH_OUT),
        "--demo-output",
        str(TAIL_DEMO),
        "--seed",
        "47",
        "--deterministic",
        "--initial-noops-exact",
        "16",
        "--reset-stack-on-stage-change",
        "--branch-world",
        "2",
        "--branch-stage",
        "2",
        "--branch-x",
        "1545",
        "1555",
        "1563",
        "1575",
        "1585",
        "--stop-world",
        "2",
        "--stop-stage",
        "3",
        "--continue-steps",
        "900",
        "--max-candidates",
        "800",
        "--candidate-script-from-demo",
        str(PIPE_CLEAR_DEMO),
        "--candidate-script-stage-x-min",
        "1100",
        "--no-video",
    ]
    if not args.skip_search:
        print("branch search...", flush=True)
        search_code = run(search_cmd, "02_branch_search.log")
        search_summary = SEARCH_OUT / "summary.json"
        if not search_summary.exists():
            print(f"branch search failed code={search_code}", flush=True)
            return search_code or 1
        if not TAIL_DEMO.exists():
            print("no tail demo produced", flush=True)
            return 1

    train_cmd = [
        str(PYTHON),
        "tools/train_mario_bc.py",
        "--demo",
        str(PREFIX_DEMO),
        "--demo",
        str(TAIL_DEMO),
        "--output-dir",
        str(BC_RUN),
        "--device",
        "auto",
        "--seed",
        "47",
        "--epochs",
        "18",
        "--batch-size",
        "64",
        "--learning-rate",
        "2e-5",
        "--entropy-coef",
        "0.0002",
        "--init-model",
        str(LEVEL22_V3B),
        "--anchor-model",
        str(LEVEL22_V3B),
        "--anchor-kl-coef",
        "0.1",
        "--anchor-stage-x-min",
        "0",
        "--anchor-stage-x-max",
        "1200",
        "--train-scope",
        "policy_head",
        "--only-world-stage",
        "2-2",
        "--oversample-stage-x-min",
        "1450",
        "--oversample-stage-x-max",
        "1700",
        "--oversample-factor",
        "10",
        "--rollout-eval-every",
        "5",
        "--rollout-eval-episodes",
        "3",
        "--rollout-eval-steps",
        "3400",
    ]
    print(f"BC training -> {BC_RUN}", flush=True)
    if run(train_cmd, "03_bc_train.log") != 0 or not (BC_RUN / "summary.json").exists():
        print("BC training failed", flush=True)
        return 1

    gates: dict[str, dict] = {}
    for label, model in (
        ("frozen22", LEVEL22_FROZEN),
        ("v3b_base", LEVEL22_V3B),
        ("best_rollout", BC_RUN / "best_rollout.zip"),
        ("bc_policy", BC_RUN / "bc_policy.zip"),
    ):
        out = ROOT / "analysis" / f"policy_chain_gate_{label}_{TAG}"
        cmd = [
            *chain_models(model),
            "--output-dir",
            str(out),
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
        run(cmd, f"04_gate_{label}.log")
        summary_path = out / "summary.json"
        gates[label] = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    promoted = None
    for label in ("best_rollout", "bc_policy"):
        g = gates.get(label) or {}
        if int(g.get("deaths", 99)) > 0:
            continue
        x22 = int((g.get("stage_max_x") or {}).get("2-2", 0))
        furthest = tuple(g.get("furthest_world_stage") or [0, 0])
        if furthest >= (2, 3) or x22 > BASELINE_CHAIN_22_X:
            promoted = {"label": label, "model": str(BC_RUN / f"{label}.zip"), "gate": g}
            break

    search_summary_path = SEARCH_OUT / "summary.json"
    search_summary_data = (
        json.loads(search_summary_path.read_text(encoding="utf-8"))
        if search_summary_path.exists()
        else {}
    )
    report = {
        "tag": TAG,
        "prefix_demo": str(PREFIX_DEMO),
        "tail_demo": str(TAIL_DEMO),
        "search_dir": str(SEARCH_OUT),
        "bc_run": str(BC_RUN),
        "search_summary": search_summary_data,
        "gates": gates,
        "baseline_chain_22_x": BASELINE_CHAIN_22_X,
        "promoted": promoted,
    }
    report_path = BC_RUN / "round4_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
