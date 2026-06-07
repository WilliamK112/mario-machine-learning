from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario_runtime import EnvConfig
from mario_runtime import config_to_dict
from mario_runtime import evaluation_score_tuple
from mario_runtime import make_vec_env
from mario_runtime import run_policy_preview
from mario_runtime import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Behavior-clone a Mario CnnPolicy from hybrid teacher traces."
    )
    parser.add_argument("--demo", action="append", required=True, help="Teacher .npz demo path. May be repeated.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument(
        "--init-model",
        default="",
        help="Optional PPO zip to initialize from before BC fine-tuning.",
    )
    parser.add_argument(
        "--train-scope",
        default="all",
        choices=("all", "policy_head", "action_head"),
        help="Limit trainable parameters when fine-tuning from --init-model.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for smoke tests.")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--only-world-stage",
        default="",
        help="Optional filter like 1-1 or 4-3 for stage-specific behavior cloning.",
    )
    parser.add_argument(
        "--oversample-stage-x-min",
        type=int,
        default=0,
        help="If >0, oversample teacher frames at or after this stage x position.",
    )
    parser.add_argument(
        "--oversample-stage-x-max",
        type=int,
        default=0,
        help="Inclusive max x for oversampling. 0 means no upper bound.",
    )
    parser.add_argument(
        "--oversample-factor",
        type=int,
        default=1,
        help="Repeat matching samples this many total times. 1 disables oversampling.",
    )
    parser.add_argument(
        "--rollout-eval-every",
        type=int,
        default=0,
        help="If >0, run a deterministic gameplay eval every N epochs and save best_rollout.zip by progress metrics.",
    )
    parser.add_argument("--rollout-eval-episodes", type=int, default=1)
    parser.add_argument("--rollout-eval-steps", type=int, default=3200)
    parser.add_argument(
        "--anchor-model",
        default="",
        help="Optional PPO zip whose action distribution is used as a KL anchor on selected frames.",
    )
    parser.add_argument(
        "--anchor-kl-coef",
        type=float,
        default=0.0,
        help="Weight for KL(anchor || current). 0 disables anchoring.",
    )
    parser.add_argument(
        "--anchor-stage-x-min",
        type=int,
        default=0,
        help="Minimum stage_x for KL anchoring. Frames without stage_x are ignored.",
    )
    parser.add_argument(
        "--anchor-stage-x-max",
        type=int,
        default=0,
        help="Maximum stage_x for KL anchoring. 0 means no upper bound.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_world_stage(raw: str) -> tuple[int, int] | None:
    if not raw:
        return None
    parts = raw.replace("_", "-").split("-")
    if len(parts) != 2:
        raise ValueError("--only-world-stage must look like 1-1")
    return int(parts[0]), int(parts[1])


def load_demos(
    paths: list[str],
    max_samples: int,
    only_world_stage: tuple[int, int] | None,
    oversample_stage_x_min: int,
    oversample_stage_x_max: int,
    oversample_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    stage_x_values: list[np.ndarray] = []
    metadata: dict[str, Any] = {"demos": []}
    for raw_path in paths:
        path = Path(raw_path)
        data = np.load(path)
        obs = data["observations"]
        acts = data["actions"]
        stage_x = data["stage_x"] if "stage_x" in data.files else None
        original_steps = int(len(acts))
        if only_world_stage is not None:
            if "worlds" not in data.files or "stages" not in data.files:
                raise ValueError(f"{path} cannot be stage-filtered; missing worlds/stages")
            keep = (data["worlds"] == only_world_stage[0]) & (data["stages"] == only_world_stage[1])
            obs = obs[keep]
            acts = acts[keep]
            if stage_x is not None:
                stage_x = stage_x[keep]
            if len(acts) == 0:
                raise ValueError(f"{path} has no samples for stage {only_world_stage[0]}-{only_world_stage[1]}")
        if obs.ndim != 4:
            raise ValueError(f"{path} observations must be NCHW, got {obs.shape}")
        if len(obs) != len(acts):
            n = min(len(obs), len(acts))
            print(
                f"warning: {path} observation/action mismatch {len(obs)} vs {len(acts)}; using first {n}",
                flush=True,
            )
            obs = obs[:n]
            acts = acts[:n]
            if stage_x is not None:
                stage_x = stage_x[:n]
        if stage_x is None:
            stage_x_array = np.full(len(acts), -1, dtype=np.int32)
        else:
            stage_x_array = np.asarray(stage_x, dtype=np.int32)
        used_before_oversample = int(len(acts))
        oversampled = 0
        if oversample_factor > 1 and oversample_stage_x_min > 0:
            if stage_x is None:
                raise ValueError(f"{path} cannot be x-oversampled; missing stage_x")
            x_values = stage_x_array
            keep_x = x_values >= int(oversample_stage_x_min)
            if oversample_stage_x_max > 0:
                keep_x &= x_values <= int(oversample_stage_x_max)
            if np.any(keep_x):
                extra_repeats = int(oversample_factor) - 1
                obs = np.concatenate([obs, np.repeat(obs[keep_x], extra_repeats, axis=0)], axis=0)
                acts = np.concatenate([acts, np.repeat(acts[keep_x], extra_repeats, axis=0)], axis=0)
                stage_x_array = np.concatenate(
                    [stage_x_array, np.repeat(stage_x_array[keep_x], extra_repeats, axis=0)],
                    axis=0,
                )
                oversampled = int(np.count_nonzero(keep_x) * extra_repeats)
        observations.append(obs)
        actions.append(acts)
        stage_x_values.append(stage_x_array)
        metadata["demos"].append(
            {
                "path": str(path),
                "steps": original_steps,
                "used_steps": int(len(acts)),
                "used_steps_before_oversample": used_before_oversample,
                "oversampled_steps": oversampled,
                "action_set": str(data["action_set"].item()) if "action_set" in data.files else "",
                "reached_stop": bool(data["reached_stop"].item()) if "reached_stop" in data.files else None,
                "stage_clears": int(data["stage_clears"].item()) if "stage_clears" in data.files else None,
                "deaths": int(data["deaths"].item()) if "deaths" in data.files else None,
            }
        )
    x = np.concatenate(observations, axis=0).astype(np.uint8, copy=False)
    y = np.concatenate(actions, axis=0).astype(np.int64, copy=False)
    stage_x_all = np.concatenate(stage_x_values, axis=0).astype(np.int32, copy=False)
    if max_samples > 0 and len(y) > max_samples:
        x = x[:max_samples]
        y = y[:max_samples]
        stage_x_all = stage_x_all[:max_samples]
    metadata["num_samples"] = int(len(y))
    metadata["unique_actions"] = {str(int(k)): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    return x, y, stage_x_all, metadata


def make_model(
    device: str,
    learning_rate: float,
    seed: int,
    only_world_stage: tuple[int, int] | None,
    init_model: str = "",
) -> tuple[PPO, EnvConfig]:
    if only_world_stage is None:
        world = 1
        stage = 1
        whole_game = True
        end_on_flag = False
    else:
        world, stage = only_world_stage
        whole_game = False
        end_on_flag = True
    config = EnvConfig(
        world=int(world),
        stage=int(stage),
        n_envs=1,
        noop_max=0,
        whole_game=whole_game,
        end_on_flag=end_on_flag,
        action_set="complex",
    )
    env = make_vec_env(config, seed=seed)
    if init_model:
        model = PPO.load(str(init_model), env=env, device=device)
    else:
        model = PPO(
            "CnnPolicy",
            env,
            verbose=0,
            n_steps=256,
            batch_size=256,
            learning_rate=learning_rate,
            ent_coef=0.0,
            n_epochs=1,
            device=device,
            seed=seed,
        )
    return model, config


def configure_train_scope(model: PPO, learning_rate: float, train_scope: str) -> dict[str, Any]:
    trainable_names: list[str] = []
    for name, param in model.policy.named_parameters():
        if train_scope == "all":
            trainable = True
        elif train_scope == "action_head":
            trainable = name.startswith("action_net.")
        else:
            trainable = name.startswith("features_extractor.linear.") or name.startswith("action_net.")
        param.requires_grad = trainable
        if trainable:
            trainable_names.append(name)
    trainable_params = [param for param in model.policy.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError(f"train_scope={train_scope!r} selected no trainable policy parameters")
    model.policy.optimizer = torch.optim.Adam(trainable_params, lr=float(learning_rate))
    return {
        "train_scope": str(train_scope),
        "trainable_parameter_count": int(sum(param.numel() for param in trainable_params)),
        "trainable_names": trainable_names,
    }


def save_evalsafe_model(model: PPO, path: Path, learning_rate: float) -> None:
    """Save a PPO zip that can be loaded later even after partial-parameter BC."""
    params = list(model.policy.parameters())
    old_requires_grad = [param.requires_grad for param in params]
    old_optimizer = model.policy.optimizer
    try:
        for param in params:
            param.requires_grad = True
        model.policy.optimizer = torch.optim.Adam(params, lr=float(learning_rate))
        model.save(str(path))
    finally:
        model.policy.optimizer = old_optimizer
        for param, requires_grad in zip(params, old_requires_grad, strict=False):
            param.requires_grad = requires_grad


def evaluate_accuracy(model: PPO, observations: np.ndarray, actions: np.ndarray, batch_size: int) -> dict[str, float]:
    model.policy.eval()
    correct = 0
    total = 0
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(actions), batch_size):
            end = min(start + batch_size, len(actions))
            obs_tensor = obs_as_tensor(observations[start:end], model.device)
            act_tensor = torch.as_tensor(actions[start:end], device=model.device, dtype=torch.long)
            dist = model.policy.get_distribution(obs_tensor)
            log_prob = dist.log_prob(act_tensor)
            losses.append(float((-log_prob).mean().item()))
            pred = torch.argmax(dist.distribution.probs, dim=1)
            correct += int((pred == act_tensor).sum().item())
            total += int(end - start)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(correct / max(1, total)),
    }


def main() -> None:
    args = parse_args()
    set_seeds(int(args.seed))
    device = resolve_device(str(args.device))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    only_world_stage = parse_world_stage(str(args.only_world_stage))
    observations, actions, stage_x_values, metadata = load_demos(
        list(args.demo),
        int(args.max_samples),
        only_world_stage,
        int(args.oversample_stage_x_min),
        int(args.oversample_stage_x_max),
        int(args.oversample_factor),
    )
    metadata["only_world_stage"] = list(only_world_stage) if only_world_stage else None
    metadata["oversample"] = {
        "stage_x_min": int(args.oversample_stage_x_min),
        "stage_x_max": int(args.oversample_stage_x_max),
        "factor": int(args.oversample_factor),
    }
    indices = np.arange(len(actions))
    np.random.shuffle(indices)
    observations = observations[indices]
    actions = actions[indices]
    stage_x_values = stage_x_values[indices]

    val_count = int(len(actions) * float(args.val_fraction))
    if val_count <= 0 and len(actions) > 1:
        val_count = 1
    val_obs = observations[:val_count]
    val_actions = actions[:val_count]
    train_obs = observations[val_count:]
    train_actions = actions[val_count:]
    train_stage_x = stage_x_values[val_count:]

    anchor_model: PPO | None = None
    if str(args.anchor_model) and float(args.anchor_kl_coef) > 0.0:
        anchor_model = PPO.load(str(args.anchor_model), device=device)
        anchor_model.policy.eval()
        for param in anchor_model.policy.parameters():
            param.requires_grad = False

    model, env_config = make_model(
        device=device,
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        only_world_stage=only_world_stage,
        init_model=str(args.init_model),
    )
    train_scope_metadata = configure_train_scope(
        model,
        learning_rate=float(args.learning_rate),
        train_scope=str(args.train_scope),
    )
    optimizer = model.policy.optimizer
    batch_size = int(args.batch_size)
    metrics: list[dict[str, Any]] = []
    best_rollout_score: tuple[float, ...] | None = None

    for epoch in range(1, int(args.epochs) + 1):
        model.policy.train()
        order = np.arange(len(train_actions))
        np.random.shuffle(order)
        losses: list[float] = []
        anchor_losses: list[float] = []
        accuracies: list[float] = []
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            obs_tensor = obs_as_tensor(train_obs[batch_idx], model.device)
            act_tensor = torch.as_tensor(train_actions[batch_idx], device=model.device, dtype=torch.long)
            dist = model.policy.get_distribution(obs_tensor)
            log_prob = dist.log_prob(act_tensor)
            entropy = dist.entropy().mean()
            loss = -log_prob.mean() - float(args.entropy_coef) * entropy
            if anchor_model is not None and float(args.anchor_kl_coef) > 0.0:
                stage_x_tensor = torch.as_tensor(train_stage_x[batch_idx], device=model.device, dtype=torch.long)
                anchor_mask = stage_x_tensor >= int(args.anchor_stage_x_min)
                if int(args.anchor_stage_x_max) > 0:
                    anchor_mask &= stage_x_tensor <= int(args.anchor_stage_x_max)
                anchor_mask &= stage_x_tensor >= 0
                if bool(anchor_mask.any().item()):
                    current_probs = dist.distribution.probs[anchor_mask].clamp_min(1e-8)
                    with torch.no_grad():
                        anchor_dist = anchor_model.policy.get_distribution(obs_tensor[anchor_mask])
                        anchor_probs = anchor_dist.distribution.probs.clamp_min(1e-8)
                    anchor_kl = (anchor_probs * (anchor_probs.log() - current_probs.log())).sum(dim=1).mean()
                    loss = loss + float(args.anchor_kl_coef) * anchor_kl
                    anchor_losses.append(float(anchor_kl.item()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()

            pred = torch.argmax(dist.distribution.probs.detach(), dim=1)
            losses.append(float(loss.item()))
            accuracies.append(float((pred == act_tensor).float().mean().item()))

        val_metrics = evaluate_accuracy(model, val_obs, val_actions, batch_size) if val_count else {"loss": 0.0, "accuracy": 0.0}
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_accuracy": float(np.mean(accuracies)),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
        }
        if anchor_losses:
            row["anchor_kl_loss"] = float(np.mean(anchor_losses))
        if int(args.rollout_eval_every) > 0 and epoch % int(args.rollout_eval_every) == 0:
            eval_dir = output_dir / "rollout_evals" / f"epoch_{epoch:04d}"
            rollout_summary = run_policy_preview(
                model,
                eval_dir,
                episodes=int(args.rollout_eval_episodes),
                max_steps=int(args.rollout_eval_steps),
                deterministic=True,
                seed=int(args.seed) + epoch,
                config=env_config,
                record_video=False,
            )
            rollout_score = evaluation_score_tuple(rollout_summary)
            is_best_rollout = best_rollout_score is None or rollout_score > best_rollout_score
            if is_best_rollout:
                best_rollout_score = rollout_score
                best_path = output_dir / "best_rollout.zip"
                save_evalsafe_model(model, best_path, float(args.learning_rate))
                write_json(
                    {
                        "epoch": epoch,
                        "best_model": str(best_path),
                        "selection_score": list(rollout_score),
                        "summary": rollout_summary,
                    },
                    output_dir / "best_rollout_summary.json",
                )
            rollout_summary["epoch"] = epoch
            rollout_summary["selection_score"] = list(rollout_score)
            rollout_summary["is_best_rollout"] = is_best_rollout
            write_json(rollout_summary, eval_dir / "summary.json")
            row["rollout_best_max_x"] = int(rollout_summary.get("best_max_x", 0))
            row["rollout_best_remaining_distance"] = int(
                rollout_summary.get("best_remaining_distance", 0)
            )
            row["rollout_flags_cleared"] = int(rollout_summary.get("flags_cleared", 0))
            if "best_world_stage" in rollout_summary:
                row["rollout_best_world_stage"] = rollout_summary["best_world_stage"]
        metrics.append(row)
        print(json.dumps(row), flush=True)

    model_path = output_dir / "bc_policy.zip"
    save_evalsafe_model(model, model_path, float(args.learning_rate))
    summary = {
        "model": str(model_path),
        "device": device,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "entropy_coef": float(args.entropy_coef),
        "init_model": str(args.init_model),
        "anchor_model": str(args.anchor_model),
        "anchor_kl_coef": float(args.anchor_kl_coef),
        "anchor_stage_x_min": int(args.anchor_stage_x_min),
        "anchor_stage_x_max": int(args.anchor_stage_x_max),
        **train_scope_metadata,
        "train_samples": int(len(train_actions)),
        "val_samples": int(len(val_actions)),
        "metadata": metadata,
        "metrics": metrics,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": int(args.seed),
                "env": config_to_dict(env_config),
                "bc": {
                    "demos": list(args.demo),
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "learning_rate": float(args.learning_rate),
                    "entropy_coef": float(args.entropy_coef),
                    "init_model": str(args.init_model),
                    "train_scope": str(args.train_scope),
                    "anchor_model": str(args.anchor_model),
                    "anchor_kl_coef": float(args.anchor_kl_coef),
                    "anchor_stage_x_min": int(args.anchor_stage_x_min),
                    "anchor_stage_x_max": int(args.anchor_stage_x_max),
                    "oversample_stage_x_min": int(args.oversample_stage_x_min),
                    "oversample_stage_x_max": int(args.oversample_stage_x_max),
                    "oversample_factor": int(args.oversample_factor),
                },
            },
            f,
            indent=2,
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
