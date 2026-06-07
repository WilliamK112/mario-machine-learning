"""Autopilot: try multiple search/BC strategies until policy-only chain clears 2-2->2-3 or exhaust."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = ROOT / f"runs/autopilot_22_to_23_{TAG}"
REPORT_PATH = LOG_DIR / "autopilot_report.json"

FROZEN_11_21 = {
    "1-1": ROOT / "runs/bc_teacher_1-1_aug_corrections_jumpbias15_20260517_1639/bc_policy.zip",
    "1-2": ROOT / "runs/bc_teacher_1-2_direct_tail_linearhead_20260517_1813/best_rollout_resaved.zip",
    "1-3": ROOT / "runs/bc_teacher_1-3_gap_focus_bias_sweep_20260517_1746/rightb_plus_0p8.zip",
    "1-4": ROOT / "runs/bc_teacher_1-4_policy_tail_rescue_20260517_1828/bc_policy.zip",
    "2-1": ROOT / "runs/bc_teacher_2-1_from_hybrid_resetstack_20260517_1838/bc_policy.zip",
}
LEVEL22_V3B = ROOT / "runs/bc_teacher_2-2_chain_prefix_v3b_tail_20260518_021236/best_rollout.zip"
LEVEL22_FROZEN = ROOT / "runs/bc_teacher_2-2_from_hybrid_resetstack_20260517_1849/bc_policy.zip"
PREFIX_V3B = ROOT / "analysis/teacher_demos_policy_branch/policy_chain_2-2_self_prefix_v3b_resetstack_20260518_205545.npz"
HYBRID_23 = ROOT / "analysis/teacher_demos_reset_stack/teacher_1-1_to_2-3_resetstack_noop16_20260517_1847.npz"
FINAL_TAIL_CLEAR = ROOT / "analysis/teacher_demos_policy_branch/teacher_policy_chain_2-2_final_tail_x1058_20260517_1940.npz"
PIPE_CLEAR = ROOT / "analysis/teacher_demos_policy_branch/teacher_policy_chain_2-2_real_x1230_to_x1900_tail_20260517_2040.npz"
BASELINE_CHAIN_X = 1563


def log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with (LOG_DIR / "autopilot.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cmd(cmd: list[str], name: str) -> int:
    path = LOG_DIR / f"{name}.log"
    with path.open("w", encoding="utf-8") as f:
        return int(subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, check=False).returncode)


def chain_gate(level22: Path, label: str) -> dict:
    out = ROOT / "analysis" / f"policy_chain_gate_{label}_{TAG}"
    cmd = [
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
    run_cmd(cmd, f"gate_{label}")
    sp = out / "summary.json"
    return json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}


def gate_ok(summary: dict) -> bool:
    if int(summary.get("deaths", 99)) > 0:
        return False
    x22 = int((summary.get("stage_max_x") or {}).get("2-2", 0))
    furthest = tuple(summary.get("furthest_world_stage") or [0, 0])
    return furthest >= (2, 3) or x22 > BASELINE_CHAIN_X


def extract_hybrid_22_slice() -> Path:
    out = ROOT / f"analysis/teacher_demos_policy_branch/hybrid_22_slice_from_23_resetstack_{TAG}.npz"
    if out.exists():
        return out
    src = np.load(HYBRID_23)
    keep = (src["worlds"] == 2) & (src["stages"] == 2)
    n = int(keep.sum())
    if n == 0:
        raise RuntimeError("no 2-2 frames in hybrid resetstack demo")
    np.savez_compressed(
        out,
        observations=src["observations"][keep],
        actions=src["actions"][keep],
        worlds=src["worlds"][keep],
        stages=src["stages"][keep],
        stage_x=src["stage_x"][keep],
        y=src["y"][keep] if "y" in src.files else np.zeros(n, dtype=np.int16),
        modes=src["modes"][keep] if "modes" in src.files else np.array(["hybrid"] * n),
        action_set=src["action_set"] if "action_set" in src.files else np.asarray("complex"),
        reached_stop=np.asarray(True),
        stage_clears=np.asarray(1),
        deaths=np.asarray(0),
        scripted_actions=np.asarray(True),
    )
    return out


def search_pass(
    name: str,
    out_dir: Path,
    tail_demo: Path,
    *,
    branch_x: list[int],
    continue_steps: int,
    max_candidates: int,
    pre_branch: str = "",
    subbranch_x: int = 0,
    candidate_demo: Path | None = None,
    candidate_x_min: int = 0,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
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
        str(out_dir),
        "--demo-output",
        str(tail_demo),
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
        *[str(x) for x in branch_x],
        "--stop-world",
        "2",
        "--stop-stage",
        "3",
        "--continue-steps",
        str(continue_steps),
        "--max-candidates",
        str(max_candidates),
        "--no-video",
    ]
    if pre_branch:
        cmd.extend(
            [
                "--pre-branch-script-name",
                pre_branch,
                "--subbranch-x",
                str(subbranch_x),
                "--pre-branch-continue-steps",
                "500",
            ]
        )
    if candidate_demo is not None:
        cmd.extend(
            [
                "--candidate-script-from-demo",
                str(candidate_demo),
                "--candidate-script-stage-x-min",
                str(candidate_x_min),
            ]
        )
    log(f"SEARCH {name} ...")
    run_cmd(cmd, f"search_{name}")
    sp = out_dir / "summary.json"
    if not sp.exists():
        return {"ok": False, "name": name}
    data = json.loads(sp.read_text(encoding="utf-8"))
    data["ok"] = True
    data["name"] = name
    data["tail_demo"] = str(tail_demo)
    return data


def bc_and_gate(demo_paths: list[Path], label: str) -> dict:
    bc_run = ROOT / f"runs/bc_teacher_2-2_autopilot_{label}_{TAG}"
    demo_args: list[str] = []
    for p in demo_paths:
        demo_args.extend(["--demo", str(p)])
    cmd = [
        str(PYTHON),
        "tools/train_mario_bc.py",
        *demo_args,
        "--output-dir",
        str(bc_run),
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
        "1400",
        "--oversample-stage-x-max",
        "3200",
        "--oversample-factor",
        "8",
        "--rollout-eval-every",
        "5",
        "--rollout-eval-episodes",
        "3",
        "--rollout-eval-steps",
        "3400",
    ]
    log(f"BC {label} ...")
    code = run_cmd(cmd, f"bc_{label}")
    if code != 0 or not (bc_run / "summary.json").exists():
        return {"ok": False, "label": label}
    best = bc_run / "best_rollout.zip"
    bc = bc_run / "bc_policy.zip"
    results = {"ok": True, "label": label, "bc_run": str(bc_run), "gates": {}}
    for glabel, model in (("best_rollout", best), ("bc_policy", bc)):
        if model.exists():
            results["gates"][glabel] = chain_gate(model, f"{label}_{glabel}")
    return results


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"tag": TAG, "baseline_chain_x": BASELINE_CHAIN_X, "searches": [], "bc_runs": [], "promoted": None}

    if not PREFIX_V3B.exists():
        log("recording prefix ...")
        out = ROOT / f"analysis/policy_chain_record_v3b_{TAG}"
        run_cmd(
            [
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
                str(LEVEL22_V3B),
                "--output-dir",
                str(out),
                "--demo-output",
                str(PREFIX_V3B),
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
            ],
            "record_prefix",
        )

    prefix = PREFIX_V3B if PREFIX_V3B.exists() else None
    if prefix is None:
        log("FATAL: no prefix demo")
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 1

    hybrid22 = extract_hybrid_22_slice()

    search_plans = [
        {
            "name": "water_prebranch",
            "out": ROOT / f"analysis/search_autopilot_water_prebranch_{TAG}",
            "tail": ROOT / f"analysis/teacher_demos_policy_branch/tail_autopilot_water_prebranch_{TAG}.npz",
            "branch_x": [1058],
            "continue_steps": 600,
            "max_candidates": 600,
            "pre_branch": "right6_rb8_jab0_rb12_water_low",
            "subbranch_x": 2700,
        },
        {
            "name": "fail_x1563_long",
            "out": ROOT / f"analysis/search_autopilot_fail1563_{TAG}",
            "tail": ROOT / f"analysis/teacher_demos_policy_branch/tail_autopilot_fail1563_{TAG}.npz",
            "branch_x": [1545, 1555, 1563, 1575],
            "continue_steps": 1200,
            "max_candidates": 1000,
            "candidate_demo": FINAL_TAIL_CLEAR,
            "candidate_x_min": 1100,
        },
        {
            "name": "fail_x1563_pipe_seed",
            "out": ROOT / f"analysis/search_autopilot_fail1563_pipe_{TAG}",
            "tail": ROOT / f"analysis/teacher_demos_policy_branch/tail_autopilot_fail1563_pipe_{TAG}.npz",
            "branch_x": [1545, 1555, 1563],
            "continue_steps": 1200,
            "max_candidates": 800,
            "candidate_demo": PIPE_CLEAR,
            "candidate_x_min": 1100,
        },
    ]

    for plan in search_plans:
        s = search_pass(
            plan["name"],
            plan["out"],
            plan["tail"],
            branch_x=plan["branch_x"],
            continue_steps=plan["continue_steps"],
            max_candidates=plan["max_candidates"],
            pre_branch=plan.get("pre_branch", ""),
            subbranch_x=int(plan.get("subbranch_x", 0)),
            candidate_demo=plan.get("candidate_demo"),
            candidate_x_min=int(plan.get("candidate_x_min", 0)),
        )
        report["searches"].append(s)
        if s.get("ok") and s.get("best", {}).get("reached_stop"):
            log(f"SEARCH SUCCESS {plan['name']}")
            break

    bc_plans = []
    for s in report["searches"]:
        if s.get("ok") and Path(s.get("tail_demo", "")).exists():
            bc_plans.append((f"tail_{s['name']}", [prefix, Path(s["tail_demo"])]))

    bc_plans.extend(
        [
            ("hybrid22_finaltail", [prefix, hybrid22, FINAL_TAIL_CLEAR]),
            ("hybrid22_pipe", [prefix, hybrid22, PIPE_CLEAR]),
            ("finaltail_only", [prefix, FINAL_TAIL_CLEAR]),
        ]
    )

    for label, demos in bc_plans:
        res = bc_and_gate(demos, label)
        report["bc_runs"].append(res)
        if not res.get("ok"):
            continue
        for glabel, gsummary in res.get("gates", {}).items():
            if gate_ok(gsummary):
                model = Path(res["bc_run"]) / f"{glabel}.zip"
                report["promoted"] = {
                    "label": label,
                    "gate_label": glabel,
                    "model": str(model),
                    "gate": gsummary,
                }
                log(f"PROMOTED {label}/{glabel}")
                REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
                return 0

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log("autopilot finished without promotion")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
