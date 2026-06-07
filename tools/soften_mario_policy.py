from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shrink PPO action-head logits so a deterministic BC policy can explore again "
            "during PPO fine-tuning. This preserves learned features/value weights."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--scale",
        type=float,
        default=0.35,
        help="Multiply action_net weight/bias by this value. Smaller means higher entropy.",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (0.0 < float(args.scale) <= 1.0):
        raise ValueError("--scale must be in (0, 1]")
    model = PPO.load(args.model, device=args.device)
    with torch.no_grad():
        model.policy.action_net.weight.mul_(float(args.scale))
        model.policy.action_net.bias.mul_(float(args.scale))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    summary = {
        "source_model": str(Path(args.model).resolve()),
        "output": str(output.resolve()),
        "scale": float(args.scale),
        "note": "action_net weight and bias were multiplied; features/value were unchanged",
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
