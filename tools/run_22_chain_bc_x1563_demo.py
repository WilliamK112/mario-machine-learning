"""2-2 BC: previous best plus clean x1563 no-death tail into 2-3."""
from __future__ import annotations

from datetime import datetime

import run_22_chain_bc_and_gate as pipeline

pipeline.RUN_NAME = "bc_teacher_2-2_chain_x1563_tail"
pipeline.RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
pipeline.FROZEN["2-2_baseline"] = (
    pipeline.ROOT
    / "runs/bc_teacher_2-2_chain_prefix_v3b_tail_20260518_021236/best_rollout.zip"
)
pipeline.BASELINE_CHAIN_22_X = 1563
pipeline.DEMOS = [
    pipeline.ROOT
    / "analysis/teacher_demos_policy_branch/teacher_policy_chain_2-2_chain_fail_x1235_v3b_aligned_20260518.npz",
    pipeline.ROOT
    / "analysis/teacher_demos_policy_branch/teacher_policy_chain_2-2_fail_x1563_20260518_0329_aligned.npz",
]
pipeline.ANCHOR_STAGE_X_MAX = 1300
pipeline.OVERSAMPLE_STAGE_X_MIN = 1300
pipeline.OVERSAMPLE_STAGE_X_MAX = 1700
pipeline.OVERSAMPLE_FACTOR = 12

if __name__ == "__main__":
    raise SystemExit(pipeline.main())
