"""
One small MP4: first stochastic rollout that reaches the flag, low resolution.

  python render_to_flag_lite.py

Uses checkpoint 4M (known to clear in-seed) and a short seed sweep.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CKPT = (
    ROOT
    / "runs"
    / "ppo_overnight_20260425_021341"
    / "run"
    / "models"
    / "checkpoints"
    / "mario_ppo_4000000_steps.zip"
)
OUT = ROOT / "runs" / "ppo_overnight_20260425_021341" / "run" / "to_flag_lite"


def main() -> None:
    if not CKPT.exists():
        print(f"missing: {CKPT}", file=sys.stderr)
        sys.exit(1)
    OUT.mkdir(parents=True, exist_ok=True)
    # Stochastic: need a short sweep to land a clear; 30x2000 worked empirically.
    cmd = [
        sys.executable,
        str(ROOT / "render_best_checkpoint.py"),
        "--model",
        str(CKPT),
        "--seeds",
        "30",
        "--seed-base",
        "0",
        "--max-steps",
        "2000",
        "--policy-modes",
        "stochastic",
        "--fps",
        "10",
        "--export-max-width",
        "256",
        "--output-dir",
        str(OUT),
    ]
    print(" ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    best = OUT / "best.json"
    if r.returncode == 0 and best.exists():
        data = json.loads(best.read_text(encoding="utf-8"))
        if not data.get("episode_flags", [False])[0]:
            print(
                "warning: no flag in best episode; re-run (stochastic) or lower --seeds / change checkpoint.",
                file=sys.stderr,
            )
        print(f"ok -> {OUT / 'best.mp4'}")


if __name__ == "__main__":
    main()
