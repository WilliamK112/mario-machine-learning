"""Round 2: record chain teacher with in-run 2-2 script handoff -> BC -> policy-only gate."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG = ROOT / f"runs/round2_script_teacher_{TAG}"

FROZEN = {
    "1-1": ROOT / "runs/bc_teacher_1-1_aug_corrections_jumpbias15_20260517_1639/bc_policy.zip",
    "1-2": ROOT / "runs/bc_teacher_1-2_direct_tail_linearhead_20260517_1813/best_rollout_resaved.zip",
    "1-3": ROOT / "runs/bc_teacher_1-3_gap_focus_bias_sweep_20260517_1746/rightb_plus_0p8.zip",
    "1-4": ROOT / "runs/bc_teacher_1-4_policy_tail_rescue_20260517_1828/bc_policy.zip",
    "2-1": ROOT / "runs/bc_teacher_2-1_from_hybrid_resetstack_20260517_1838/bc_policy.zip",
}
V3B = ROOT / "runs/bc_teacher_2-2_chain_prefix_v3b_tail_20260518_021236/best_rollout.zip"
PREFIX = ROOT / "analysis/teacher_demos_policy_branch/policy_chain_2-2_self_prefix_v3b_resetstack_20260518_205545.npz"
TEACHER = ROOT / f"analysis/teacher_demos_policy_branch/chain_script_handoff_22_to_23_{TAG}.npz"
BC_RUN = ROOT / f"runs/bc_teacher_2-2_round2_script_handoff_{TAG}"


def run(cmd: list[str], name: str) -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    with (LOG / f"{name}.log").open("w", encoding="utf-8") as f:
        return int(subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, check=False).returncode)


def gate(model: Path, label: str) -> dict:
    out = ROOT / f"analysis/policy_chain_gate_round2_{label}_{TAG}"
    run(
        [
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
            str(model),
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
        ],
        f"gate_{label}",
    )
    sp = out / "summary.json"
    return json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}


def main() -> int:
    for hx in (1400, 1500, 1560):
        demo = TEACHER.with_name(f"chain_script_handoff_22_to_23_x{hx}_{TAG}.npz")
        out = ROOT / f"analysis/record_round2_handoff_x{hx}_{TAG}"
        code = run(
            [
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
                str(V3B),
                "--output-dir",
                str(out),
                "--demo-output",
                str(demo),
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
                "--reset-stack-on-stage-change",
                "--stop-on-death",
                "--level22-script-handoff-x",
                str(hx),
                "--max-steps",
                "5200",
                "--no-video",
            ],
            f"record_x{hx}",
        )
        if not demo.exists():
            continue
        rec = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        x22 = int((rec.get("stage_max_x") or {}).get("2-2", 0))
        if int(rec.get("deaths", 99)) > 0 and not rec.get("reached_stop"):
            continue
        if not rec.get("reached_stop") and rec.get("furthest_world_stage", [0, 0]) < [2, 3] and x22 < 2500:
            continue
        demos = [str(PREFIX), str(demo)] if PREFIX.exists() else [str(demo)]
        run(
            [
                str(PYTHON),
                "tools/train_mario_bc.py",
                *[x for d in demos for x in ("--demo", d)],
                "--output-dir",
                str(BC_RUN.with_name(f"{BC_RUN.name}_x{hx}")),
                "--device",
                "auto",
                "--seed",
                "47",
                "--epochs",
                "22",
                "--batch-size",
                "64",
                "--learning-rate",
                "2e-5",
                "--entropy-coef",
                "0.0001",
                "--init-model",
                str(V3B),
                "--anchor-model",
                str(V3B),
                "--anchor-kl-coef",
                "0.15",
                "--anchor-stage-x-min",
                "0",
                "--anchor-stage-x-max",
                "1400",
                "--train-scope",
                "policy_head",
                "--only-world-stage",
                "2-2",
                "--oversample-stage-x-min",
                "1500",
                "--oversample-stage-x-max",
                "3200",
                "--oversample-factor",
                "12",
                "--rollout-eval-every",
                "4",
                "--rollout-eval-episodes",
                "3",
                "--rollout-eval-steps",
                "3400",
            ],
            f"bc_x{hx}",
        )
        bc_dir = BC_RUN.with_name(f"{BC_RUN.name}_x{hx}")
        for name in ("best_rollout.zip", "bc_policy.zip"):
            mp = bc_dir / name
            if not mp.exists():
                continue
            g = gate(mp, f"x{hx}_{name[:-4]}")
            ok = int(g.get("deaths", 99)) == 0 and (
                tuple(g.get("furthest_world_stage") or [0, 0]) >= (2, 3)
                or int((g.get("stage_max_x") or {}).get("2-2", 0)) > 1563
            )
            report = {"tag": TAG, "handoff_x": hx, "record": rec, "gate": g, "model": str(mp), "promoted": ok}
            (LOG / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            if ok and tuple(g.get("furthest_world_stage") or [0, 0]) >= (2, 3):
                return 0
        if code == 0:
            break
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
