"""2-2 BC: policy chain failure prefix + teacher pipe-up tail that reaches 2-3."""
from __future__ import annotations

import run_22_chain_bc_and_gate as pipeline

pipeline.DEMOS = [
    pipeline.ROOT
    / "analysis/teacher_demos_policy_branch/policy_chain_2-2_self_prefix_failure_resetstack_20260518_0002.npz",
    pipeline.ROOT
    / "analysis/teacher_demos_policy_branch/teacher_policy_chain_2-2_real_x1230_to_x1900_tail_20260517_2040.npz",
]
pipeline.RUN_NAME = "bc_teacher_2-2_chain_prefix_real_tail"
pipeline.RUN_TAG = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")

if __name__ == "__main__":
    raise SystemExit(pipeline.main())
