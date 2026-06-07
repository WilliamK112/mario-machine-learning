from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import VecNormalize

from mario_runtime import ACTION_SET_MAP
from mario_runtime import EvalCheckpointCallback
from mario_runtime import EnvConfig
from mario_runtime import config_to_dict
from mario_runtime import make_vec_env
from mario_runtime import PreviewCallback
from mario_runtime import timestamped_run_dir
from mario_runtime import write_json
from rnd import RNDModule
from rnd import RNDVecEnvWrapper


def jump_action_indices(action_set: str, include_long_jump: bool = False) -> list[int]:
    """Return indices of jump-containing actions (buttons containing 'A') for the given action set.
    When `include_long_jump` is True, also include the extra macro action appended by
    LongJumpActionWrapper (which is always `len(actions)` - the last index).
    """
    actions = ACTION_SET_MAP.get(action_set, ACTION_SET_MAP["right_only"])
    idx = [i for i, buttons in enumerate(actions) if "A" in buttons]
    if include_long_jump:
        idx.append(len(actions))
    return idx


def long_jump_macro_index(action_set: str) -> int:
    """Index of the macro action appended by LongJumpActionWrapper."""
    return len(ACTION_SET_MAP.get(action_set, ACTION_SET_MAP["right_only"]))


def down_action_indices(action_set: str) -> list[int]:
    """Return indices of actions that press DOWN for pipe-entry exploration."""
    actions = ACTION_SET_MAP.get(action_set, ACTION_SET_MAP["right_only"])
    return [i for i, buttons in enumerate(actions) if "down" in buttons]


def noop_action_indices(action_set: str) -> list[int]:
    """Return indices of actions that do not press any directional/jump button."""
    actions = ACTION_SET_MAP.get(action_set, ACTION_SET_MAP["right_only"])
    return [i for i, buttons in enumerate(actions) if buttons == ["NOOP"]]


def left_action_indices(action_set: str) -> list[int]:
    """Return indices of actions that press LEFT."""
    actions = ACTION_SET_MAP.get(action_set, ACTION_SET_MAP["right_only"])
    return [i for i, buttons in enumerate(actions) if "left" in buttons]


def apply_action_logit_bias(model: PPO, indices: list[int], bias_value: float) -> None:
    """Add `bias_value` to selected actions' logit bias on the policy's action head.

    This is an "action prior" that skews rollout sampling toward the given actions without
    breaking PPO's importance sampling (log_prob is still computed from the biased policy).
    PPO is free to learn the bias away if the actions turn out to be suboptimal.
    """
    with torch.no_grad():
        for idx in indices:
            model.policy.action_net.bias[idx] += bias_value


def apply_requested_action_bias(model: PPO, args: argparse.Namespace) -> None:
    if args.action_bias_jump != 0.0:
        jump_idx = jump_action_indices(args.action_set, include_long_jump=args.long_jump_action)
        apply_action_logit_bias(model, jump_idx, args.action_bias_jump)
        print(
            f"[action-bias] Applied +{args.action_bias_jump} logit bias to jump actions "
            f"{jump_idx} (action_set={args.action_set}, long_jump={args.long_jump_action})"
        )
    if args.action_bias_down != 0.0:
        down_idx = down_action_indices(args.action_set)
        apply_action_logit_bias(model, down_idx, args.action_bias_down)
        print(
            f"[action-bias] Applied +{args.action_bias_down} logit bias to DOWN actions "
            f"{down_idx} (action_set={args.action_set})"
        )
    if args.action_bias_noop != 0.0:
        noop_idx = noop_action_indices(args.action_set)
        apply_action_logit_bias(model, noop_idx, args.action_bias_noop)
        print(
            f"[action-bias] Applied {args.action_bias_noop:+g} logit bias to NOOP actions "
            f"{noop_idx} (action_set={args.action_set})"
        )
    if args.action_bias_left != 0.0:
        left_idx = left_action_indices(args.action_set)
        apply_action_logit_bias(model, left_idx, args.action_bias_left)
        print(
            f"[action-bias] Applied {args.action_bias_left:+g} logit bias to LEFT actions "
            f"{left_idx} (action_set={args.action_set})"
        )
    if args.long_jump_action and args.action_bias_long_jump != 0.0:
        macro_idx = long_jump_macro_index(args.action_set)
        apply_action_logit_bias(model, [macro_idx], args.action_bias_long_jump)
        print(
            f"[action-bias] Applied additional +{args.action_bias_long_jump} logit bias to "
            f"long-jump macro action (index {macro_idx})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PPO agent for Super Mario Bros (single stage or whole-game progression).",
    )
    parser.add_argument("--world", type=int, default=1, help="World 1–8 (vanilla SMB).")
    parser.add_argument("--stage", type=int, default=1, help="Stage 1–4 within the world.")
    parser.add_argument(
        "--whole-game",
        action="store_true",
        help="Start from 1-1 without a target and let the ROM naturally advance across worlds/stages.",
    )
    parser.add_argument(
        "--goal-line-x",
        type=int,
        default=0,
        help="Override flag-line x for x-metrics / shaping sanity (0 = use built-in table for world/stage).",
    )
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--resume-model", type=str, default="")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument(
        "--vec-backend",
        type=str,
        default="dummy",
        choices=["dummy", "subproc"],
    )
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--screen-size", type=int, default=84)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument(
        "--initial-noops-exact",
        type=int,
        default=0,
        help="Run this many fixed NOOP actions after reset before training/eval rollouts.",
    )
    parser.add_argument(
        "--action-set",
        type=str,
        default="right_only",
        choices=["right_only", "simple", "simple_pipe", "complex"],
        help="'simple' adds left; 'simple_pipe' keeps 7 actions but swaps stand-jump for down; "
             "'complex' adds left+jump variants and down/up (larger action space).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor. Mario benefits from a lower gamma (~0.9) so the value "
             "function distinguishes 'about to die in pit' from 'past the pit'.",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda for advantage estimation.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=-1.0,
        help="Early-stop PPO epoch when approx KL exceeds this. Use -1 to disable.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--forward-reward-scale", type=float, default=0.15)
    parser.add_argument("--backward-penalty-scale", type=float, default=0.2)
    parser.add_argument("--flag-bonus", type=float, default=1000.0)
    parser.add_argument(
        "--stage-clear-bonus",
        type=float,
        default=0.0,
        help="One-time reward when the env reports a world/stage change. Useful for whole-game/castle clears.",
    )
    parser.add_argument(
        "--progress-reward-mode",
        choices=["delta", "new_max"],
        default="delta",
        help="delta rewards every positive dx; new_max rewards only new per-stage max x progress.",
    )
    parser.add_argument(
        "--jump-window-x-min",
        type=int,
        default=0,
        help="If >0 with --jump-window-bonus, reward forward jump actions after this x.",
    )
    parser.add_argument(
        "--jump-window-x-max",
        type=int,
        default=0,
        help="Inclusive end x for --jump-window-bonus.",
    )
    parser.add_argument(
        "--jump-window-bonus",
        type=float,
        default=0.0,
        help="Per-step bonus for forward jump actions inside the configured x window.",
    )
    parser.add_argument("--airborne-window-x-min", type=int, default=0)
    parser.add_argument("--airborne-window-x-max", type=int, default=0)
    parser.add_argument(
        "--airborne-min-y",
        type=int,
        default=0,
        help="Inside the airborne window, y_pos at or above this value counts as airborne.",
    )
    parser.add_argument(
        "--airborne-bonus",
        type=float,
        default=0.0,
        help="Per-step reward for actually being airborne inside the configured x window.",
    )
    parser.add_argument(
        "--grounded-penalty",
        type=float,
        default=0.0,
        help="Per-step penalty for being below --airborne-min-y inside the configured x window.",
    )
    parser.add_argument(
        "--pipe-entry-x",
        type=int,
        default=0,
        help="If >0, reward DOWN-containing actions after this x position.",
    )
    parser.add_argument(
        "--pipe-entry-bonus",
        type=float,
        default=0.0,
        help="Per-step shaped reward for pressing DOWN after --pipe-entry-x. 0 disables.",
    )
    parser.add_argument(
        "--left-penalty-x-min",
        type=int,
        default=0,
        help="If >0, penalize LEFT-containing actions at or after this x position.",
    )
    parser.add_argument(
        "--left-action-penalty",
        type=float,
        default=0.0,
        help="Per-step shaped penalty for LEFT actions after --left-penalty-x-min.",
    )
    parser.add_argument(
        "--hold-jump-actions-steps",
        type=int,
        default=1,
        help="Repeat existing right+jump actions for this many env steps without changing action-space size.",
    )
    parser.add_argument(
        "--assist-action-window",
        type=int,
        nargs=3,
        action="append",
        default=[],
        metavar=("X_MIN", "X_MAX", "ACTION"),
        help="Curriculum assist: replace policy actions with ACTION while previous x_pos is in [X_MIN, X_MAX].",
    )
    parser.set_defaults(end_on_flag=True)
    parser.add_argument(
        "--no-end-on-flag",
        dest="end_on_flag",
        action="store_false",
        help="Keep the episode alive after a flag clear so the ROM can advance to the next stage.",
    )
    parser.add_argument("--stall-steps", type=int, default=40)
    parser.add_argument("--stall-penalty", type=float, default=2.0)
    parser.add_argument(
        "--end-on-stall-steps",
        type=int,
        default=0,
        help="Truncate episode after this many consecutive non-forward steps. 0 disables.",
    )
    parser.add_argument(
        "--end-on-stall-penalty",
        type=float,
        default=0.0,
        help="One-time penalty applied when the episode truncates due to stall.",
    )
    parser.add_argument(
        "--milestone-step",
        type=int,
        default=0,
        help="Pixel width of each new-distance milestone. 0 disables milestone bonus.",
    )
    parser.add_argument(
        "--milestone-bonus",
        type=float,
        default=0.0,
        help="One-time bonus when Mario first reaches each new milestone within an episode.",
    )
    parser.add_argument(
        "--hurdle-x",
        type=int,
        nargs="*",
        default=[],
        help="One or more x-position thresholds. Mario gets hurdle-bonus the first time he crosses each.",
    )
    parser.add_argument(
        "--hurdle-bonus",
        type=float,
        default=0.0,
        help="One-time bonus granted per hurdle crossed within an episode.",
    )
    parser.add_argument(
        "--time-penalty-per-step",
        type=float,
        default=0.0,
        help="Subtract this from the shaped reward every agent step (encourages shorter episodes). "
        "Typical fine-tune values: 0.01–0.05 with normalize-reward.",
    )
    parser.add_argument(
        "--life-loss-penalty",
        type=float,
        default=0.0,
        help="Subtract this shaped reward when the emulator life counter decreases. 0 disables.",
    )
    parser.add_argument(
        "--end-on-life-loss",
        action="store_true",
        help="Terminate the episode when Mario loses a life. Useful for clean single-stage rescue runs.",
    )
    parser.add_argument(
        "--backtrack-relief-stall-steps",
        type=int,
        default=0,
        help="After this many consecutive env steps without net forward x progress, stop penalizing "
        "backward motion; if --backtrack-bonus-scale > 0, reward backward dx instead. 0 disables.",
    )
    parser.add_argument(
        "--backtrack-bonus-scale",
        type=float,
        default=0.0,
        help="When stalled (see --backtrack-relief-stall-steps), add this times |delta_x| for left moves.",
    )
    parser.add_argument(
        "--ghost-trace-path",
        type=str,
        default="",
        help="Optional .npz ghost route trace. Adds shaped reward for matching or beating the reference pace.",
    )
    parser.add_argument(
        "--ghost-reward-scale",
        type=float,
        default=0.0,
        help="Per-step signed reward scale based on agent progress minus ghost progress. 0 logs only.",
    )
    parser.add_argument(
        "--ghost-near-bonus",
        type=float,
        default=0.0,
        help="Extra per-step bonus when Mario is within --ghost-near-x pixels of the ghost on the same stage.",
    )
    parser.add_argument(
        "--ghost-near-x",
        type=int,
        default=64,
        help="Pixel window used for ghost lead tanh scaling and near bonus.",
    )
    parser.add_argument(
        "--ghost-lead-margin",
        type=int,
        default=0,
        help="Pixels Mario must lead the ghost by before the signed lead reward becomes positive.",
    )
    parser.add_argument(
        "--ghost-progress-stride",
        type=int,
        default=10_000,
        help="Virtual route-progress spacing between stages in the ghost trace.",
    )
    parser.add_argument(
        "--ghost-reward-cap",
        type=float,
        default=5.0,
        help="Absolute per-step cap for ghost shaping reward. 0 disables clipping.",
    )
    parser.set_defaults(ghost_align_to_current_stage=True)
    parser.add_argument(
        "--no-ghost-align-to-current-stage",
        dest="ghost_align_to_current_stage",
        action="store_false",
        help="Start the ghost at trace step 0 instead of aligning to the env's current world/stage.",
    )
    parser.add_argument(
        "--cage-escape-shaping",
        action="store_true",
        help="Preset for cages: sets --backtrack-relief-stall-steps 48 and --backtrack-bonus-scale 0.12 "
        "(overrides those two flags if you pass --cage-escape-shaping). Requires --action-set simple or complex.",
    )
    parser.add_argument(
        "--action-bias-jump",
        type=float,
        default=0.0,
        help="Add this value to the action_net logit bias of every jump-containing action at init. "
             "Positive values (e.g. 1.0) push initial sampling toward jumping; PPO can learn it back.",
    )
    parser.add_argument(
        "--action-bias-down",
        type=float,
        default=0.0,
        help="Add this value to the action_net logit bias of every DOWN-containing action. "
             "Useful for pipe exits such as SMB 1-2 when using an action set that includes DOWN.",
    )
    parser.add_argument(
        "--action-bias-noop",
        type=float,
        default=0.0,
        help="Add this value to NOOP action logits. Negative values discourage standing still.",
    )
    parser.add_argument(
        "--action-bias-left",
        type=float,
        default=0.0,
        help="Add this value to LEFT action logits. Negative values discourage left/right dithering.",
    )
    parser.add_argument(
        "--action-bias-long-jump",
        type=float,
        default=0.0,
        help="Additional logit bias applied to the long-jump macro action only "
             "(on top of --action-bias-jump). Requires --long-jump-action.",
    )
    parser.add_argument(
        "--skip-action-bias-on-resume",
        action="store_true",
        help="Do not add requested action logit biases when loading a checkpoint. Prevents repeated bias accumulation across segmented runs.",
    )
    parser.add_argument(
        "--long-jump-action",
        action="store_true",
        help="Append a macro action that holds (right+A+B) for --long-jump-hold-steps agent steps. "
             "Each held step costs `frame_skip` raw frames, so total raw frames held = hold_steps * frame_skip.",
    )
    parser.add_argument(
        "--long-jump-hold-steps",
        type=int,
        default=4,
        help="Number of inner env steps to hold the base action during a long jump macro.",
    )
    parser.add_argument(
        "--long-jump-base-action",
        type=int,
        default=-1,
        help="Action index within the action set used by the long jump macro. -1 = auto-pick the "
             "last action that includes both 'right' and 'A' (typically right+A+B).",
    )
    parser.add_argument(
        "--normalize-reward",
        action="store_true",
        help="Wrap vec env with VecNormalize(norm_reward=True) so return scale stays in a healthy range for PPO.",
    )
    parser.add_argument(
        "--norm-reward-clip",
        type=float,
        default=10.0,
        help="Absolute clip on normalized rewards when --normalize-reward is set.",
    )
    parser.add_argument(
        "--rnd-coef",
        type=float,
        default=0.0,
        help="RND intrinsic-reward coefficient. 0 disables RND.",
    )
    parser.add_argument(
        "--rnd-lr",
        type=float,
        default=1e-4,
        help="Learning rate for the RND predictor network.",
    )
    parser.add_argument(
        "--rnd-feature-dim",
        type=int,
        default=256,
        help="Output feature dimension of the RND target/predictor networks.",
    )
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-steps", type=int, default=4_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--preview-freq", type=int, default=5_000)
    parser.add_argument("--preview-steps", type=int, default=1024)
    parser.add_argument("--preview-episodes", type=int, default=1)
    parser.add_argument("--preview-fps", type=int, default=15)
    parser.add_argument("--live-preview", action="store_true")
    parser.add_argument(
        "--no-preview-video",
        dest="record_preview_video",
        action="store_false",
        help="Run preview/evaluation logic without saving preview.mp4. Useful with --live-preview to reduce memory.",
    )
    parser.set_defaults(record_preview_video=True)
    return parser.parse_args()


def resolve_vecnormalize_pkl(resume_model_path: Path) -> Path | None:
    """VecNormalize is saved next to checkpoints under models/; best_eval.zip lives in evaluations/."""
    if resume_model_path.stem.endswith("_steps"):
        parts = resume_model_path.stem.split("_")
        if len(parts) >= 2 and parts[-1] == "steps":
            step = parts[-2]
            checkpoint_matches = sorted(
                resume_model_path.parent.glob(f"*_vecnormalize_{step}_steps.pkl"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if checkpoint_matches:
                return checkpoint_matches[0]

    if resume_model_path.name == "best_eval.zip":
        summary_path = resume_model_path.with_name("best_eval_summary.json")
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            step = int(summary.get("num_timesteps", 0))
        except Exception:
            step = 0
        if step > 0:
            run_dir = resume_model_path.parent.parent
            checkpoint_matches = sorted(
                (run_dir / "models" / "checkpoints").glob(
                    f"*_vecnormalize_{step}_steps.pkl"
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if checkpoint_matches:
                return checkpoint_matches[0]

    for candidate in (
        resume_model_path.parent / "vecnormalize.pkl",
        resume_model_path.parent.parent / "vecnormalize.pkl",
        resume_model_path.parent.parent / "models" / "vecnormalize.pkl",
    ):
        if candidate.exists():
            return candidate
    return None


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resume_custom_objects(args: argparse.Namespace) -> dict[str, float | int | None]:
    custom: dict[str, float | int | None] = {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "ent_coef": args.ent_coef,
        "n_epochs": args.n_epochs,
        "clip_range": args.clip_range,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
    }
    custom["target_kl"] = args.target_kl if args.target_kl > 0 else None
    return custom


def build_callbacks(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    config: EnvConfig,
    start_timesteps: int,
) -> list:
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // args.n_envs),
        save_path=str(run_dir / "models" / "checkpoints"),
        name_prefix="mario_ppo",
        save_vecnormalize=True,
        verbose=1,
    )
    callbacks = [checkpoint_callback]
    if args.eval_freq > 0:
        callbacks.append(
            EvalCheckpointCallback(
                eval_freq=args.eval_freq,
                output_dir=run_dir / "evaluations",
                eval_steps=args.eval_steps,
                eval_episodes=args.eval_episodes,
                deterministic=True,
                seed=args.seed,
                start_timesteps=start_timesteps,
                config=config,
                verbose=1,
            )
        )
    if args.preview_freq > 0:
        callbacks.append(
            PreviewCallback(
                preview_freq=args.preview_freq,
                output_dir=run_dir / "previews",
                preview_steps=args.preview_steps,
                preview_episodes=args.preview_episodes,
                preview_fps=args.preview_fps,
                deterministic=True,
                render_human=args.live_preview,
                record_video=args.record_preview_video,
                seed=args.seed,
                start_timesteps=start_timesteps,
                config=config,
                verbose=1,
            )
        )
    return callbacks


def main() -> None:
    args = parse_args()
    if args.cage_escape_shaping:
        args.backtrack_relief_stall_steps = 48
        args.backtrack_bonus_scale = 0.12
        print(
            "cage_escape_shaping: backtrack_relief_stall_steps=48 backtrack_bonus_scale=0.12 "
            "(stalled ~1–2s at 4 frame_skip: reward left, no backward penalty)"
        )
        if args.action_set == "right_only":
            print(
                "WARNING: action_set=right_only cannot press LEFT. Use --action-set simple or complex "
                "or cage-escape shaping cannot physically walk backward."
            )
    if args.whole_game and args.end_on_flag:
        args.end_on_flag = False
        print("whole_game: forcing end_on_flag=False so episodes can advance past flags.")
    resolved_device = resolve_device(args.device)
    resume_model_path = Path(args.resume_model) if args.resume_model else None
    run_dir = Path(args.run_dir) if args.run_dir else timestamped_run_dir()
    models_dir = run_dir / "models"
    logs_dir = run_dir / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    config = EnvConfig(
        world=args.world,
        stage=args.stage,
        whole_game=args.whole_game,
        goal_line_x=args.goal_line_x,
        n_envs=args.n_envs,
        vec_backend=args.vec_backend,
        frame_skip=args.frame_skip,
        frame_stack=args.frame_stack,
        screen_size=args.screen_size,
        noop_max=args.noop_max,
        initial_noops_exact=args.initial_noops_exact,
        end_on_flag=args.end_on_flag,
        action_set=args.action_set,
        forward_reward_scale=args.forward_reward_scale,
        backward_penalty_scale=args.backward_penalty_scale,
        flag_bonus=args.flag_bonus,
        stall_steps=args.stall_steps,
        stall_penalty=args.stall_penalty,
        end_on_stall_steps=args.end_on_stall_steps,
        end_on_stall_penalty=args.end_on_stall_penalty,
        milestone_step=args.milestone_step,
        milestone_bonus=args.milestone_bonus,
        hurdle_x=tuple(args.hurdle_x),
        hurdle_bonus=args.hurdle_bonus,
        time_penalty_per_step=args.time_penalty_per_step,
        life_loss_penalty=args.life_loss_penalty,
        end_on_life_loss=args.end_on_life_loss,
        stage_clear_bonus=args.stage_clear_bonus,
        progress_reward_mode=args.progress_reward_mode,
        jump_window_x_min=args.jump_window_x_min,
        jump_window_x_max=args.jump_window_x_max,
        jump_window_bonus=args.jump_window_bonus,
        airborne_window_x_min=args.airborne_window_x_min,
        airborne_window_x_max=args.airborne_window_x_max,
        airborne_min_y=args.airborne_min_y,
        airborne_bonus=args.airborne_bonus,
        grounded_penalty=args.grounded_penalty,
        pipe_entry_x=args.pipe_entry_x,
        pipe_entry_bonus=args.pipe_entry_bonus,
        left_penalty_x_min=args.left_penalty_x_min,
        left_action_penalty=args.left_action_penalty,
        assist_action_windows=tuple(tuple(item) for item in args.assist_action_window),
        hold_jump_actions_steps=args.hold_jump_actions_steps,
        backtrack_relief_stall_steps=args.backtrack_relief_stall_steps,
        backtrack_bonus_scale=args.backtrack_bonus_scale,
        long_jump_action=args.long_jump_action,
        long_jump_hold_steps=args.long_jump_hold_steps,
        long_jump_base_action=args.long_jump_base_action,
        ghost_trace_path=args.ghost_trace_path,
        ghost_reward_scale=args.ghost_reward_scale,
        ghost_near_bonus=args.ghost_near_bonus,
        ghost_near_x=args.ghost_near_x,
        ghost_lead_margin=args.ghost_lead_margin,
        ghost_progress_stride=args.ghost_progress_stride,
        ghost_reward_cap=args.ghost_reward_cap,
        ghost_align_to_current_stage=args.ghost_align_to_current_stage,
    )

    env = make_vec_env(config, seed=args.seed)
    if args.rnd_coef > 0.0:
        obs_shape = env.observation_space.shape
        env = RNDVecEnvWrapper(
            env,
            RNDModule(
                obs_shape=obs_shape,
                device=resolved_device,
                feature_dim=args.rnd_feature_dim,
                lr=args.rnd_lr,
            ),
            intrinsic_coef=args.rnd_coef,
        )
    if args.normalize_reward:
        vec_pkl = resolve_vecnormalize_pkl(resume_model_path) if resume_model_path else None
        if resume_model_path and vec_pkl is not None:
            print(f"[vecnorm] loading running reward stats from {vec_pkl}")
            env = VecNormalize.load(str(vec_pkl), env)
            env.training = True
        else:
            if resume_model_path:
                print(
                    "[vecnorm] warning: no vecnormalize.pkl next to checkpoint or run/models/; "
                    "starting fresh reward normalisation (not ideal for resume).",
                )
            env = VecNormalize(
                env,
                norm_obs=False,
                norm_reward=True,
                clip_reward=args.norm_reward_clip,
                gamma=0.99,
            )
    start_timesteps = 0
    callbacks = build_callbacks(
        args=args,
        run_dir=run_dir,
        config=config,
        start_timesteps=start_timesteps,
    )

    if resume_model_path:
        model = PPO.load(
            str(resume_model_path),
            env=env,
            device=resolved_device,
            tensorboard_log=str(logs_dir / "tensorboard"),
            custom_objects=resume_custom_objects(args),
        )
        model.verbose = 1
        start_timesteps = int(model.num_timesteps)
        callbacks = build_callbacks(
            args=args,
            run_dir=run_dir,
            config=config,
            start_timesteps=start_timesteps,
        )
    else:
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            ent_coef=args.ent_coef,
            n_epochs=args.n_epochs,
            clip_range=args.clip_range,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            target_kl=args.target_kl if args.target_kl > 0 else None,
            tensorboard_log=str(logs_dir / "tensorboard"),
            device=resolved_device,
        )
    if resume_model_path and args.skip_action_bias_on_resume:
        print("[action-bias] Skipping requested action logit bias on resume.")
    else:
        apply_requested_action_bias(model, args)
    learn_total_timesteps = (
        start_timesteps + args.timesteps if resume_model_path else args.timesteps
    )
    write_json(
        {
            "run_dir": str(run_dir),
            "timesteps": args.timesteps,
            "target_total_timesteps": learn_total_timesteps,
            "start_timesteps": start_timesteps,
            "resume_model": str(resume_model_path) if resume_model_path else "",
            "seed": args.seed,
            "env": config_to_dict(config),
            "ppo": {
                "learning_rate": args.learning_rate,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "ent_coef": args.ent_coef,
                "n_epochs": args.n_epochs,
                "clip_range": args.clip_range,
                "vf_coef": args.vf_coef,
                "max_grad_norm": args.max_grad_norm,
                "target_kl": args.target_kl if args.target_kl > 0 else None,
                "device": resolved_device,
                "requested_device": args.device,
            },
            "rnd": {
                "rnd_coef": args.rnd_coef,
                "rnd_lr": args.rnd_lr,
                "rnd_feature_dim": args.rnd_feature_dim,
            },
            "pipe_entry": {
                "x": args.pipe_entry_x,
                "bonus": args.pipe_entry_bonus,
                "down_indices": down_action_indices(args.action_set),
            },
            "normalize_reward": {
                "enabled": args.normalize_reward,
                "clip": args.norm_reward_clip,
            },
            "action_bias": {
            "jump": args.action_bias_jump,
            "jump_indices": jump_action_indices(args.action_set, include_long_jump=args.long_jump_action) if args.action_bias_jump != 0.0 else [],
            "down": args.action_bias_down,
            "down_indices": down_action_indices(args.action_set) if args.action_bias_down != 0.0 else [],
            "noop": args.action_bias_noop,
            "noop_indices": noop_action_indices(args.action_set) if args.action_bias_noop != 0.0 else [],
            "left": args.action_bias_left,
            "left_indices": left_action_indices(args.action_set) if args.action_bias_left != 0.0 else [],
            "long_jump": args.action_bias_long_jump,
            "long_jump_index": long_jump_macro_index(args.action_set) if args.long_jump_action else None,
            "skip_on_resume": args.skip_action_bias_on_resume,
        },
            "long_jump": {
                "enabled": args.long_jump_action,
                "hold_steps": args.long_jump_hold_steps,
                "base_action": args.long_jump_base_action,
            },
            "evaluation": {
                "eval_freq": args.eval_freq,
                "eval_steps": args.eval_steps,
                "eval_episodes": args.eval_episodes,
            },
            "preview": {
                "preview_freq": args.preview_freq,
                "preview_steps": args.preview_steps,
                "preview_episodes": args.preview_episodes,
                "preview_fps": args.preview_fps,
                "live_preview": args.live_preview,
                "record_preview_video": args.record_preview_video,
            },
        },
        run_dir / "train_config.json",
    )
    model.learn(
        total_timesteps=learn_total_timesteps,
        progress_bar=True,
        callback=CallbackList(callbacks),
        tb_log_name="mario_ppo",
        reset_num_timesteps=not bool(resume_model_path),
    )

    final_model_path = models_dir / "mario_final.zip"
    model.save(final_model_path)
    if args.normalize_reward and isinstance(env, VecNormalize):
        env.save(str(models_dir / "vecnormalize.pkl"))
    write_json(
        {
            "run_dir": str(run_dir),
            "final_model": str(final_model_path),
            "timesteps": args.timesteps,
            "target_total_timesteps": learn_total_timesteps,
            "start_timesteps": start_timesteps,
            "resume_model": str(resume_model_path) if resume_model_path else "",
            "seed": args.seed,
            "env": config_to_dict(config),
            "ppo": {
                "learning_rate": args.learning_rate,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "ent_coef": args.ent_coef,
                "n_epochs": args.n_epochs,
                "clip_range": args.clip_range,
                "vf_coef": args.vf_coef,
                "max_grad_norm": args.max_grad_norm,
                "target_kl": args.target_kl if args.target_kl > 0 else None,
                "device": resolved_device,
                "requested_device": args.device,
            },
            "pipe_entry": {
                "x": args.pipe_entry_x,
                "bonus": args.pipe_entry_bonus,
                "down_indices": down_action_indices(args.action_set),
            },
            "evaluation": {
                "eval_freq": args.eval_freq,
                "eval_steps": args.eval_steps,
                "eval_episodes": args.eval_episodes,
            },
            "preview": {
                "preview_freq": args.preview_freq,
                "preview_steps": args.preview_steps,
                "preview_episodes": args.preview_episodes,
                "preview_fps": args.preview_fps,
                "live_preview": args.live_preview,
                "record_preview_video": args.record_preview_video,
            },
        },
        run_dir / "train_config.json",
    )
    print(f"train_ok run_dir={run_dir} final_model={final_model_path}")
    env.close()


if __name__ == "__main__":
    main()
