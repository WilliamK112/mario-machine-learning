"""2-2 BC: policy prefix + branch-search v3b tail (x~3011 from chain fail x1232)."""
from __future__ import annotations

import run_22_chain_bc_and_gate as pipeline

pipeline.RUN_NAME = "bc_teacher_2-2_chain_prefix_v3b_tail"
pipeline.RUN_TAG = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
pipeline.DEMOS = [
    pipeline.ROOT
    / "analysis/teacher_demos_policy_branch/policy_chain_2-2_self_prefix_failure_resetstack_20260518_0002.npz",
    pipeline.ROOT
    / "analysis/teacher_demos_policy_branch/teacher_policy_chain_2-2_chain_fail_x1235_v3b_aligned_20260518.npz",
]

if __name__ == "__main__":
    raise SystemExit(pipeline.main())
