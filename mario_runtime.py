from __future__ import annotations

import inspect
import json
import os
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from gymnasium_super_mario_bros.actions import COMPLEX_MOVEMENT
from gymnasium_super_mario_bros.actions import RIGHT_ONLY
from gymnasium_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium_super_mario_bros.smb_env import SuperMarioBrosEnv
from gymnasium_super_mario_bros.smb_env import decode_target
from gymnasium_super_mario_bros.smb_env import rom_path
from nes_py import NESEnv
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecTransposeImage


ACTION_SET_MAP = {
    "right_only": RIGHT_ONLY,
    "simple": SIMPLE_MOVEMENT,
    # Same action-space size as SIMPLE_MOVEMENT, but replaces stand-still jump
    # with DOWN so old 7-action PPO checkpoints can learn pipe exits in 1-2.
    "simple_pipe": [
        ["NOOP"],
        ["right"],
        ["right", "A"],
        ["right", "B"],
        ["right", "A", "B"],
        ["down"],
        ["left"],
    ],
    "complex": COMPLEX_MOVEMENT,
}
DEFAULT_ACTIONS = RIGHT_ONLY
LEVEL_1_1_GOAL_X = 3161
MAX_REASONABLE_X_DELTA = 256

# Approximate flag-pole x (same units as smb_env `_x_position`) for metrics / x sanitization.
# Extend as you unlock more stages. Unknown (world, stage) falls back to 3200.
STAGE_FLAG_LINE_X: dict[tuple[int, int], int] = {
    (1, 1): 3161,
    (1, 2): 2528,
    (1, 3): 2848,
    (1, 4): 3216,
}


def effective_goal_line_x(cfg: "EnvConfig") -> int:
    """Resolved goal-line x for the configured stage (override or table)."""
    if cfg.goal_line_x > 0:
        return int(cfg.goal_line_x)
    if getattr(cfg, "whole_game", False):
        return 3200
    return int(STAGE_FLAG_LINE_X.get((int(cfg.world), int(cfg.stage)), 3200))


def goal_line_x_for_world_stage(
    world_stage: tuple[int, int] | None,
    *,
    fallback: int = 3200,
) -> int:
    if world_stage is None:
        return int(fallback)
    world, stage = world_stage
    return int(STAGE_FLAG_LINE_X.get((int(world), int(stage)), int(fallback)))


def world_stage_from_info(
    info: dict[str, Any],
    fallback: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    world = info.get("world")
    stage = info.get("stage")
    if world is None or stage is None:
        return fallback
    try:
        return int(world), int(stage)
    except (TypeError, ValueError):
        return fallback


class JoypadSpace(gym.Wrapper):
    """Map a reduced discrete action space to NES controller bytes."""

    _button_map = {
        "right": 0b10000000,
        "left": 0b01000000,
        "down": 0b00100000,
        "up": 0b00010000,
        "start": 0b00001000,
        "select": 0b00000100,
        "B": 0b00000010,
        "A": 0b00000001,
        "NOOP": 0b00000000,
    }

    def __init__(self, env: gym.Env, actions: list[list[str]]):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(actions))
        self._action_map: dict[int, int] = {}
        for action, button_list in enumerate(actions):
            byte_action = 0
            for button in button_list:
                byte_action |= self._button_map[button]
            self._action_map[action] = byte_action

    def step(self, action: int):
        return self.env.step(self._action_map[int(action)])


def patch_super_mario_compat() -> None:
    """Patch package mismatches between the emulator and Mario wrapper."""

    if not getattr(SuperMarioBrosEnv, "_cursor_init_patch", False):
        nes_init_params = inspect.signature(NESEnv.__init__).parameters
        if len(nes_init_params) == 2:

            def _patched_init(
                self,
                rom_mode: str = "vanilla",
                render_mode: str | None = None,
                lost_levels: bool = False,
                target: tuple[int, int] | None = None,
            ) -> None:
                rom = rom_path(lost_levels, rom_mode)
                NESEnv.__init__(self, rom)
                self.render_mode = render_mode
                decoded_target = decode_target(target, lost_levels)
                self._target_world, self._target_stage, self._target_area = decoded_target
                self._time_last = 0
                self._x_position_last = 0
                self._x_position_max = 0
                self._x_coin_last = 0
                self._power_level_last = 0
                self._stage_last = 1
                self.reset()
                self._skip_start_screen()
                self._backup()

            SuperMarioBrosEnv.__init__ = _patched_init

        SuperMarioBrosEnv._cursor_init_patch = True

    if not getattr(SuperMarioBrosEnv, "_cursor_step_patch", False):
        original_did_step = SuperMarioBrosEnv._did_step

        def _patched_did_step(self, terminated: bool, truncated: bool = False):
            return original_did_step(self, terminated, truncated)

        SuperMarioBrosEnv._did_step = _patched_did_step
        SuperMarioBrosEnv._cursor_step_patch = True


class LegacyMarioToGymnasium(gym.Wrapper):
    """Convert the old Gym-style Mario env to Gymnasium's API."""

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None and hasattr(self.env, "seed"):
            self.env.seed(seed)
        obs = self.env.reset()
        return obs, {}

    def step(self, action: int):
        obs, reward, done, info = self.env.step(action)
        terminated = bool(done)
        truncated = False
        return obs, float(reward), terminated, truncated, info


class NoopResetEnv(gym.Wrapper):
    """Run random no-op actions after reset to vary initial states."""

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        rng = self.unwrapped.np_random
        if hasattr(rng, "integers"):
            noops = int(rng.integers(1, self.noop_max + 1))
        else:
            noops = int(rng.randint(1, self.noop_max + 1))
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class ExactNoopResetEnv(gym.Wrapper):
    """Run a fixed number of no-op actions after reset for reproducible starts."""

    def __init__(self, env: gym.Env, noop_steps: int = 0) -> None:
        super().__init__(env)
        self.noop_steps = max(0, int(noop_steps))
        self.noop_action = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.noop_steps):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipFrame(gym.Wrapper):
    """Repeat actions for several frames and max-pool the last two observations."""

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros(
            (2, *env.observation_space.shape), dtype=env.observation_space.dtype
        )

    def step(self, action: int):
        total_reward = 0.0
        terminated = False
        truncated = False
        flag_get_detected = False
        info: dict[str, Any] = {}

        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if info.get("flag_get", False):
                flag_get_detected = True
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if terminated or truncated or flag_get_detected:
                break

        max_frame = self._obs_buffer.max(axis=0)
        if flag_get_detected:
            info["flag_get"] = True
        return max_frame, total_reward, terminated, truncated, info


class LongJumpActionWrapper(gym.Wrapper):
    """Expose one extra discrete action that commits to holding a base jump action
    for `hold_steps` consecutive inner-env steps (i.e. `hold_steps * frame_skip` raw frames).

    Placed *above* MaxAndSkipFrame so each inner step still respects frame skipping.
    Rewards are summed over the whole macro and returned as a single agent step, so
    downstream shaping (FastClearRewardWrapper) treats the macro as one decision.
    """

    def __init__(self, env: gym.Env, hold_steps: int = 4, base_action: int = 4) -> None:
        super().__init__(env)
        n = env.action_space.n
        self.action_space = gym.spaces.Discrete(n + 1)
        self._long_jump_action = n
        self._hold_steps = max(1, int(hold_steps))
        self._base_action = int(base_action)

    @property
    def long_jump_action(self) -> int:
        return self._long_jump_action

    def step(self, action: int):
        if int(action) != self._long_jump_action:
            return self.env.step(action)
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        obs = None
        for _ in range(self._hold_steps):
            obs, reward, terminated, truncated, info = self.env.step(self._base_action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        info = dict(info)
        info["long_jump_used"] = True
        return obs, total_reward, terminated, truncated, info


class HoldSelectedActionsWrapper(gym.Wrapper):
    """Repeat selected existing actions without changing the action-space shape."""

    def __init__(
        self,
        env: gym.Env,
        *,
        action_indices: tuple[int, ...],
        hold_steps: int = 1,
    ) -> None:
        super().__init__(env)
        self._action_indices = frozenset(int(idx) for idx in action_indices)
        self._hold_steps = max(1, int(hold_steps))

    def step(self, action: int):
        action_int = int(action)
        if self._hold_steps <= 1 or action_int not in self._action_indices:
            return self.env.step(action_int)
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        obs = None
        for _ in range(self._hold_steps):
            obs, reward, terminated, truncated, info = self.env.step(action_int)
            total_reward += float(reward)
            if terminated or truncated:
                break
        info = dict(info)
        info["held_action"] = action_int
        info["held_action_steps"] = self._hold_steps
        return obs, total_reward, terminated, truncated, info


class AssistActionWindowWrapper(gym.Wrapper):
    """Temporarily replace actions inside x-position windows for curriculum runs."""

    def __init__(
        self,
        env: gym.Env,
        *,
        windows: tuple[tuple[int, int, int], ...],
    ) -> None:
        super().__init__(env)
        self._windows = tuple((int(lo), int(hi), int(action)) for lo, hi, action in windows)
        self._last_x_pos = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        try:
            self._last_x_pos = int(info.get("x_pos", 0))
        except (TypeError, ValueError):
            self._last_x_pos = 0
        return obs, info

    def step(self, action: int):
        original_action = int(action)
        assisted_action = original_action
        for lo, hi, forced_action in self._windows:
            if lo <= self._last_x_pos <= hi:
                assisted_action = forced_action
                break
        obs, reward, terminated, truncated, info = self.env.step(assisted_action)
        info = dict(info)
        if assisted_action != original_action:
            info["assist_action_from"] = original_action
            info["assist_action_to"] = assisted_action
        try:
            self._last_x_pos = int(info.get("x_pos", self._last_x_pos))
        except (TypeError, ValueError):
            pass
        return obs, reward, terminated, truncated, info


class StageSuccessWrapper(gym.Wrapper):
    """Optionally end an episode when Mario reaches the flag."""

    def __init__(self, env: gym.Env, end_on_flag: bool) -> None:
        super().__init__(env)
        self.end_on_flag = end_on_flag

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.end_on_flag and info.get("flag_get", False):
            truncated = True
        return obs, reward, terminated, truncated, info


class FastClearRewardWrapper(gym.Wrapper):
    """Shape rewards toward rapid level completion and whole-game progression."""

    def __init__(
        self,
        env: gym.Env,
        *,
        forward_reward_scale: float = 0.15,
        backward_penalty_scale: float = 0.2,
        flag_bonus: float = 1000.0,
        stall_steps: int = 40,
        stall_penalty: float = 2.0,
        end_on_stall_steps: int = 0,
        end_on_stall_penalty: float = 0.0,
        milestone_step: int = 0,
        milestone_bonus: float = 0.0,
        hurdle_x: tuple[int, ...] = (),
        hurdle_bonus: float = 0.0,
        time_penalty_per_step: float = 0.0,
        life_loss_penalty: float = 0.0,
        end_on_life_loss: bool = False,
        stage_clear_bonus: float = 0.0,
        progress_reward_mode: str = "delta",
        jump_window_x_min: int = 0,
        jump_window_x_max: int = 0,
        jump_window_bonus: float = 0.0,
        airborne_window_x_min: int = 0,
        airborne_window_x_max: int = 0,
        airborne_min_y: int = 0,
        airborne_bonus: float = 0.0,
        grounded_penalty: float = 0.0,
        jump_action_indices: tuple[int, ...] = (),
        down_action_indices: tuple[int, ...] = (),
        left_action_indices: tuple[int, ...] = (),
        pipe_entry_x: int = 0,
        pipe_entry_bonus: float = 0.0,
        left_penalty_x_min: int = 0,
        left_action_penalty: float = 0.0,
        goal_line_x: int = LEVEL_1_1_GOAL_X,
        backtrack_relief_stall_steps: int = 0,
        backtrack_bonus_scale: float = 0.0,
    ) -> None:
        super().__init__(env)
        self._goal_line_x = int(goal_line_x)
        self.forward_reward_scale = forward_reward_scale
        self.backward_penalty_scale = backward_penalty_scale
        self.backtrack_relief_stall_steps = int(backtrack_relief_stall_steps)
        self.backtrack_bonus_scale = float(backtrack_bonus_scale)
        self.flag_bonus = flag_bonus
        self.stall_steps = stall_steps
        self.stall_penalty = stall_penalty
        self.end_on_stall_steps = end_on_stall_steps
        self.end_on_stall_penalty = end_on_stall_penalty
        self.milestone_step = milestone_step
        self.milestone_bonus = milestone_bonus
        self.hurdle_x = tuple(sorted(int(x) for x in hurdle_x if int(x) > 0))
        self.hurdle_bonus = float(hurdle_bonus)
        self.time_penalty_per_step = float(time_penalty_per_step)
        self.life_loss_penalty = float(life_loss_penalty)
        self.end_on_life_loss = bool(end_on_life_loss)
        self.stage_clear_bonus = float(stage_clear_bonus)
        self.progress_reward_mode = str(progress_reward_mode).lower()
        self.jump_window_x_min = int(jump_window_x_min)
        self.jump_window_x_max = int(jump_window_x_max)
        self.jump_window_bonus = float(jump_window_bonus)
        self.airborne_window_x_min = int(airborne_window_x_min)
        self.airborne_window_x_max = int(airborne_window_x_max)
        self.airborne_min_y = int(airborne_min_y)
        self.airborne_bonus = float(airborne_bonus)
        self.grounded_penalty = float(grounded_penalty)
        self.jump_action_indices = frozenset(int(idx) for idx in jump_action_indices)
        self.down_action_indices = frozenset(int(idx) for idx in down_action_indices)
        self.left_action_indices = frozenset(int(idx) for idx in left_action_indices)
        self.pipe_entry_x = int(pipe_entry_x)
        self.pipe_entry_bonus = float(pipe_entry_bonus)
        self.left_penalty_x_min = int(left_penalty_x_min)
        self.left_action_penalty = float(left_action_penalty)
        self._prev_x_pos = 0
        self._max_x_pos = 0
        self._prev_world_stage: tuple[int, int] | None = None
        self._prev_life_count: int | None = None
        self._stall_count = 0
        self._max_milestone_index = -1
        self._hurdles_claimed: set[int] = set()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_x_pos = sanitize_x_position(
            info.get("x_pos", 0),
            goal_line_x=self._goal_line_x,
        )
        self._max_x_pos = self._prev_x_pos
        self._prev_world_stage = world_stage_from_info(info)
        self._prev_life_count = _life_count_from_info(info)
        self._stall_count = 0
        self._max_milestone_index = -1
        self._hurdles_claimed = set()
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        current_world_stage = world_stage_from_info(info, self._prev_world_stage)
        stage_changed = (
            current_world_stage is not None
            and self._prev_world_stage is not None
            and current_world_stage != self._prev_world_stage
        )
        stage_advanced = (
            stage_changed
            and stage_order_index(current_world_stage) > stage_order_index(self._prev_world_stage)
        )
        x_pos = extract_sanitized_x_position(
            info,
            previous_x_pos=0 if stage_changed else self._prev_x_pos,
            goal_line_x=self._goal_line_x,
        )
        delta_x = 0 if stage_changed else x_pos - self._prev_x_pos
        shaped_reward = float(reward)
        current_life_count = _life_count_from_info(info)
        current_y_pos = _int_from_info(info, "y_pos")

        if stage_changed:
            info["stage_changed"] = True
            info["stage_change_from"] = list(self._prev_world_stage)
            info["stage_change_to"] = list(current_world_stage)
            info["stage_change_forward"] = bool(stage_advanced)

        if stage_advanced:
            if self.stage_clear_bonus > 0.0:
                shaped_reward += self.stage_clear_bonus
            info["stage_clear"] = True
            info["stage_clear_bonus"] = self.stage_clear_bonus
            info["stage_clear_from"] = list(self._prev_world_stage)
            info["stage_clear_to"] = list(current_world_stage)
            self._stall_count = 0
            self._max_milestone_index = -1
            self._hurdles_claimed = set()

        if stage_changed:
            pass
        elif self.progress_reward_mode == "new_max":
            new_progress = max(0, x_pos - self._max_x_pos)
            if new_progress > 0:
                shaped_reward += new_progress * self.forward_reward_scale
                self._stall_count = 0
                info["new_max_progress_x"] = int(x_pos)
            else:
                self._stall_count += 1
                if delta_x < 0:
                    stalled_relief = (
                        self.backtrack_relief_stall_steps > 0
                        and self._stall_count >= self.backtrack_relief_stall_steps
                    )
                    if stalled_relief:
                        if self.backtrack_bonus_scale > 0.0:
                            shaped_reward += abs(delta_x) * self.backtrack_bonus_scale
                    else:
                        shaped_reward -= abs(delta_x) * self.backward_penalty_scale
                if self._stall_count >= self.stall_steps:
                    shaped_reward -= self.stall_penalty
        elif delta_x > 0:
            shaped_reward += delta_x * self.forward_reward_scale
            self._stall_count = 0
        else:
            self._stall_count += 1
            if delta_x < 0:
                stalled_relief = (
                    self.backtrack_relief_stall_steps > 0
                    and self._stall_count >= self.backtrack_relief_stall_steps
                )
                if stalled_relief:
                    if self.backtrack_bonus_scale > 0.0:
                        shaped_reward += abs(delta_x) * self.backtrack_bonus_scale
                else:
                    shaped_reward -= abs(delta_x) * self.backward_penalty_scale
            if self._stall_count >= self.stall_steps:
                shaped_reward -= self.stall_penalty

        if (
            self.milestone_step > 0
            and self.milestone_bonus > 0.0
            and x_pos > 0
        ):
            milestone_index = x_pos // self.milestone_step
            if milestone_index > self._max_milestone_index:
                gained = milestone_index - self._max_milestone_index
                shaped_reward += gained * self.milestone_bonus
                self._max_milestone_index = milestone_index
                info["milestone_x"] = int(milestone_index * self.milestone_step)

        if self.hurdle_bonus > 0.0 and self.hurdle_x:
            for threshold in self.hurdle_x:
                if threshold in self._hurdles_claimed:
                    continue
                if x_pos >= threshold:
                    shaped_reward += self.hurdle_bonus
                    self._hurdles_claimed.add(threshold)
                    info.setdefault("hurdle_cleared", []).append(int(threshold))

        if (
            self.jump_window_bonus > 0.0
            and self.jump_window_x_min > 0
            and self.jump_window_x_max >= self.jump_window_x_min
            and self.jump_window_x_min <= x_pos <= self.jump_window_x_max
            and delta_x > 0
            and int(action) in self.jump_action_indices
        ):
            shaped_reward += self.jump_window_bonus
            info["jump_window_bonus"] = self.jump_window_bonus

        if (
            not stage_changed
            and self.airborne_window_x_min > 0
            and self.airborne_window_x_max >= self.airborne_window_x_min
            and self.airborne_window_x_min <= x_pos <= self.airborne_window_x_max
            and current_y_pos is not None
        ):
            if current_y_pos >= self.airborne_min_y:
                if self.airborne_bonus > 0.0:
                    shaped_reward += self.airborne_bonus
                    info["airborne_window_bonus"] = self.airborne_bonus
            elif self.grounded_penalty > 0.0:
                shaped_reward -= self.grounded_penalty
                info["grounded_window_penalty"] = self.grounded_penalty

        if (
            self.pipe_entry_bonus > 0.0
            and self.pipe_entry_x > 0
            and x_pos >= self.pipe_entry_x
            and int(action) in self.down_action_indices
        ):
            shaped_reward += self.pipe_entry_bonus
            info["pipe_entry_action_bonus"] = self.pipe_entry_bonus

        if (
            self.left_action_penalty > 0.0
            and self.left_penalty_x_min > 0
            and x_pos >= self.left_penalty_x_min
            and int(action) in self.left_action_indices
        ):
            shaped_reward -= self.left_action_penalty
            info["left_action_penalty"] = self.left_action_penalty

        if info.get("flag_get", False):
            shaped_reward += self.flag_bonus

        life_lost = (
            self._prev_life_count is not None
            and current_life_count is not None
            and current_life_count < self._prev_life_count
        )
        if life_lost:
            if self.life_loss_penalty > 0.0:
                shaped_reward -= self.life_loss_penalty
            info["life_lost"] = True
            if self.life_loss_penalty > 0.0:
                info["life_loss_penalty"] = self.life_loss_penalty
            if self.end_on_life_loss:
                terminated = True
                info["life_loss_terminated"] = True

        if self.time_penalty_per_step > 0.0:
            shaped_reward -= self.time_penalty_per_step

        if (
            self.end_on_stall_steps > 0
            and self._stall_count >= self.end_on_stall_steps
        ):
            shaped_reward -= self.end_on_stall_penalty
            truncated = True
            info["stall_truncated"] = True

        self._prev_x_pos = x_pos
        self._max_x_pos = x_pos if stage_changed else max(self._max_x_pos, x_pos)
        self._prev_world_stage = current_world_stage
        if current_life_count is not None:
            self._prev_life_count = current_life_count
        return obs, shaped_reward, terminated, truncated, info


class GhostTraceRewardWrapper(gym.Wrapper):
    """Add a small reward for matching or beating a reference route trace."""

    def __init__(
        self,
        env: gym.Env,
        *,
        trace_path: str | os.PathLike[str],
        reward_scale: float = 0.0,
        near_bonus: float = 0.0,
        near_x: int = 64,
        lead_margin: int = 0,
        progress_stride: int = 10_000,
        reward_cap: float = 5.0,
        align_to_current_stage: bool = True,
    ) -> None:
        super().__init__(env)
        path = Path(trace_path)
        if not path.exists():
            raise FileNotFoundError(f"Ghost trace not found: {path}")
        data = np.load(path, allow_pickle=False)
        required = {"worlds", "stages", "stage_x"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Ghost trace {path} is missing fields: {sorted(missing)}")
        self.trace_path = str(path)
        self.reward_scale = float(reward_scale)
        self.near_bonus = float(near_bonus)
        self.near_x = max(1, int(near_x))
        self.lead_margin = int(lead_margin)
        self.progress_stride = int(progress_stride)
        self.reward_cap = float(reward_cap)
        self.align_to_current_stage = bool(align_to_current_stage)
        self._worlds = np.asarray(data["worlds"], dtype=np.int16)
        self._stages = np.asarray(data["stages"], dtype=np.int16)
        self._stage_x = np.asarray(data["stage_x"], dtype=np.int32)
        if len(self._stage_x) == 0:
            raise ValueError(f"Ghost trace {path} is empty.")
        if "route_progress" in data.files:
            self._route_progress = np.asarray(data["route_progress"], dtype=np.float32)
        else:
            self._route_progress = np.asarray(
                [
                    self._route_progress_value((int(w), int(s)), int(x))
                    for w, s, x in zip(self._worlds, self._stages, self._stage_x, strict=False)
                ],
                dtype=np.float32,
            )
        self._first_index_by_stage: dict[tuple[int, int], int] = {}
        for idx, (world, stage) in enumerate(zip(self._worlds, self._stages, strict=False)):
            self._first_index_by_stage.setdefault((int(world), int(stage)), int(idx))
        self._elapsed_steps = 0
        self._episode_start_index = 0
        self._prev_world_stage: tuple[int, int] | None = None
        self._prev_stage_x = 0

    def _route_progress_value(self, world_stage: tuple[int, int] | None, stage_x: int) -> float:
        return float(stage_order_index(world_stage) * self.progress_stride + int(stage_x))

    def _ghost_index(self) -> int:
        return int(min(self._episode_start_index + max(0, self._elapsed_steps - 1), len(self._stage_x) - 1))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        self._elapsed_steps = 0
        self._prev_world_stage = world_stage_from_info(info)
        self._prev_stage_x = extract_sanitized_x_position(
            info,
            previous_x_pos=0,
            goal_line_x=goal_line_x_for_world_stage(self._prev_world_stage, fallback=3200),
        )
        self._episode_start_index = 0
        if self.align_to_current_stage and self._prev_world_stage is not None:
            self._episode_start_index = self._first_index_by_stage.get(self._prev_world_stage, 0)
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        self._elapsed_steps += 1
        current_world_stage = world_stage_from_info(info, self._prev_world_stage)
        if (
            self.align_to_current_stage
            and self._elapsed_steps == 1
            and self._prev_world_stage is None
            and current_world_stage is not None
        ):
            self._episode_start_index = self._first_index_by_stage.get(current_world_stage, 0)
        stage_changed = (
            current_world_stage is not None
            and self._prev_world_stage is not None
            and current_world_stage != self._prev_world_stage
        )
        stage_x = extract_sanitized_x_position(
            info,
            previous_x_pos=0 if stage_changed else self._prev_stage_x,
            goal_line_x=goal_line_x_for_world_stage(current_world_stage, fallback=3200),
        )
        ghost_idx = self._ghost_index()
        ghost_world_stage = (int(self._worlds[ghost_idx]), int(self._stages[ghost_idx]))
        ghost_stage_x = int(self._stage_x[ghost_idx])
        agent_progress = self._route_progress_value(current_world_stage, stage_x)
        ghost_progress = float(self._route_progress[ghost_idx])
        raw_lead = agent_progress - ghost_progress
        adjusted_lead = raw_lead - float(self.lead_margin)
        ghost_reward = self.reward_scale * float(np.tanh(adjusted_lead / float(self.near_x)))

        if self.near_bonus > 0.0 and current_world_stage == ghost_world_stage:
            distance = abs(int(stage_x) - ghost_stage_x)
            if distance <= self.near_x:
                ghost_reward += self.near_bonus * (1.0 - (distance / float(self.near_x)))

        if self.reward_cap > 0.0:
            ghost_reward = float(np.clip(ghost_reward, -self.reward_cap, self.reward_cap))
        shaped_reward = float(reward) + ghost_reward

        info["ghost_index"] = int(ghost_idx)
        info["ghost_world_stage"] = [int(ghost_world_stage[0]), int(ghost_world_stage[1])]
        info["ghost_stage_x"] = int(ghost_stage_x)
        info["ghost_route_progress"] = float(ghost_progress)
        info["ghost_lead"] = float(raw_lead)
        info["ghost_reward"] = float(ghost_reward)
        info["ghost_trace_path"] = self.trace_path

        self._prev_world_stage = current_world_stage
        self._prev_stage_x = int(stage_x)
        return obs, shaped_reward, terminated, truncated, info


@dataclass
class EnvConfig:
    world: int = 1
    stage: int = 1
    whole_game: bool = False
    frame_skip: int = 4
    frame_stack: int = 4
    screen_size: int = 84
    noop_max: int = 30
    initial_noops_exact: int = 0
    n_envs: int = 1
    vec_backend: str = "dummy"
    end_on_flag: bool = True
    action_set: str = "right_only"
    forward_reward_scale: float = 0.15
    backward_penalty_scale: float = 0.2
    flag_bonus: float = 1000.0
    stall_steps: int = 40
    stall_penalty: float = 2.0
    end_on_stall_steps: int = 0
    end_on_stall_penalty: float = 0.0
    milestone_step: int = 0
    milestone_bonus: float = 0.0
    hurdle_x: tuple[int, ...] = ()
    hurdle_bonus: float = 0.0
    time_penalty_per_step: float = 0.0
    life_loss_penalty: float = 0.0
    end_on_life_loss: bool = False
    stage_clear_bonus: float = 0.0
    progress_reward_mode: str = "delta"
    jump_window_x_min: int = 0
    jump_window_x_max: int = 0
    jump_window_bonus: float = 0.0
    airborne_window_x_min: int = 0
    airborne_window_x_max: int = 0
    airborne_min_y: int = 0
    airborne_bonus: float = 0.0
    grounded_penalty: float = 0.0
    pipe_entry_x: int = 0
    pipe_entry_bonus: float = 0.0
    left_penalty_x_min: int = 0
    left_action_penalty: float = 0.0
    assist_action_windows: tuple[tuple[int, int, int], ...] = ()
    hold_jump_actions_steps: int = 1
    long_jump_action: bool = False
    long_jump_hold_steps: int = 4
    long_jump_base_action: int = -1  # -1 = auto: last jump-containing action in the set
    # After this many consecutive steps without net forward x progress, stop penalizing
    # backward motion; optionally reward it (see backtrack_bonus_scale). Helps cages / pipes.
    backtrack_relief_stall_steps: int = 0
    backtrack_bonus_scale: float = 0.0
    # 0 = use STAGE_FLAG_LINE_X for (world, stage); >0 overrides (for custom ROM/layout).
    goal_line_x: int = 0
    ghost_trace_path: str = ""
    ghost_reward_scale: float = 0.0
    ghost_near_bonus: float = 0.0
    ghost_near_x: int = 64
    ghost_lead_margin: int = 0
    ghost_progress_stride: int = 10_000
    ghost_reward_cap: float = 5.0
    ghost_align_to_current_stage: bool = True


def make_single_env(config: EnvConfig, seed: int | None = None):
    patch_super_mario_compat()
    if config.whole_game:
        env = SuperMarioBrosEnv()
    else:
        env = SuperMarioBrosEnv(target=(config.world, config.stage))
    actions = ACTION_SET_MAP.get(config.action_set, DEFAULT_ACTIONS)
    env = JoypadSpace(env, actions)
    env = LegacyMarioToGymnasium(env)
    if config.noop_max > 0:
        env = NoopResetEnv(env, noop_max=config.noop_max)
    if config.frame_skip > 1:
        env = MaxAndSkipFrame(env, skip=config.frame_skip)
    if config.initial_noops_exact > 0:
        env = ExactNoopResetEnv(env, noop_steps=config.initial_noops_exact)
    if config.hold_jump_actions_steps > 1:
        env = HoldSelectedActionsWrapper(
            env,
            action_indices=tuple(
                idx for idx, buttons in enumerate(actions) if "A" in buttons and "right" in buttons
            ),
            hold_steps=config.hold_jump_actions_steps,
        )
    if config.assist_action_windows:
        env = AssistActionWindowWrapper(env, windows=config.assist_action_windows)
    if config.long_jump_action:
        if config.long_jump_base_action >= 0:
            base_idx = config.long_jump_base_action
        else:
            base_idx = max(
                (i for i, btns in enumerate(actions) if "A" in btns and "right" in btns),
                default=0,
            )
        env = LongJumpActionWrapper(
            env,
            hold_steps=config.long_jump_hold_steps,
            base_action=base_idx,
        )
    env = StageSuccessWrapper(env, end_on_flag=config.end_on_flag)
    env = FastClearRewardWrapper(
        env,
        forward_reward_scale=config.forward_reward_scale,
        backward_penalty_scale=config.backward_penalty_scale,
        flag_bonus=config.flag_bonus,
        stall_steps=config.stall_steps,
        stall_penalty=config.stall_penalty,
        end_on_stall_steps=config.end_on_stall_steps,
        end_on_stall_penalty=config.end_on_stall_penalty,
        milestone_step=config.milestone_step,
        milestone_bonus=config.milestone_bonus,
        hurdle_x=config.hurdle_x,
        hurdle_bonus=config.hurdle_bonus,
        time_penalty_per_step=config.time_penalty_per_step,
        life_loss_penalty=config.life_loss_penalty,
        end_on_life_loss=config.end_on_life_loss,
        stage_clear_bonus=config.stage_clear_bonus,
        progress_reward_mode=config.progress_reward_mode,
        jump_window_x_min=config.jump_window_x_min,
        jump_window_x_max=config.jump_window_x_max,
        jump_window_bonus=config.jump_window_bonus,
        airborne_window_x_min=config.airborne_window_x_min,
        airborne_window_x_max=config.airborne_window_x_max,
        airborne_min_y=config.airborne_min_y,
        airborne_bonus=config.airborne_bonus,
        grounded_penalty=config.grounded_penalty,
        jump_action_indices=tuple(
            idx for idx, buttons in enumerate(actions) if "A" in buttons
        ),
        down_action_indices=tuple(
            idx for idx, buttons in enumerate(actions) if "down" in buttons
        ),
        left_action_indices=tuple(
            idx for idx, buttons in enumerate(actions) if "left" in buttons
        ),
        pipe_entry_x=config.pipe_entry_x,
        pipe_entry_bonus=config.pipe_entry_bonus,
        left_penalty_x_min=config.left_penalty_x_min,
        left_action_penalty=config.left_action_penalty,
        goal_line_x=effective_goal_line_x(config),
        backtrack_relief_stall_steps=config.backtrack_relief_stall_steps,
        backtrack_bonus_scale=config.backtrack_bonus_scale,
    )
    if config.ghost_trace_path:
        env = GhostTraceRewardWrapper(
            env,
            trace_path=config.ghost_trace_path,
            reward_scale=config.ghost_reward_scale,
            near_bonus=config.ghost_near_bonus,
            near_x=config.ghost_near_x,
            lead_margin=config.ghost_lead_margin,
            progress_stride=config.ghost_progress_stride,
            reward_cap=config.ghost_reward_cap,
            align_to_current_stage=config.ghost_align_to_current_stage,
        )
    env = Monitor(env)
    env = WarpFrame(env, width=config.screen_size, height=config.screen_size)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_env_factory(config: EnvConfig, *, seed: int):
    def _make_env():
        return make_single_env(config, seed=seed)

    return _make_env


def make_vec_env(config: EnvConfig, seed: int = 42):
    env_fns = [make_env_factory(config, seed=seed + idx) for idx in range(config.n_envs)]
    vec_backend = config.vec_backend.lower()
    if vec_backend == "subproc" and config.n_envs > 1:
        env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        env = DummyVecEnv(env_fns)
    env = VecFrameStack(env, n_stack=config.frame_stack)
    env = VecTransposeImage(env)
    return env


def render_rgb_frame(env: gym.Env) -> np.ndarray:
    frame = env.unwrapped.render(mode="rgb_array")
    if frame is None:
        raise RuntimeError("Environment did not return an rgb_array frame.")
    # Some emulator backends reuse the same frame buffer object across renders.
    # Copy here so recorded videos preserve each distinct frame instead of
    # ending up as a frozen clip of the last/first shared buffer contents.
    return np.array(frame, copy=True)


def save_video(frames: list[np.ndarray], output_path: str | os.PathLike[str], fps: int = 15) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def timestamped_run_dir(base_dir: str | os.PathLike[str] = "runs") -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(data: dict[str, Any], path: str | os.PathLike[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")


def config_to_dict(config: EnvConfig) -> dict[str, Any]:
    return asdict(config)


def config_from_dict(data: dict[str, Any]) -> EnvConfig:
    return EnvConfig(**data)


def find_run_dir_for_model_path(model_path: str | os.PathLike[str]) -> Path | None:
    path = Path(model_path).resolve()
    for candidate in [path.parent] + list(path.parents):
        config_path = candidate / "train_config.json"
        if config_path.exists():
            return candidate
    return None


def load_env_config_for_model(
    model_path: str | os.PathLike[str],
    *,
    fallback: EnvConfig | None = None,
) -> EnvConfig:
    run_dir = find_run_dir_for_model_path(model_path)
    if run_dir is None:
        return fallback or EnvConfig(n_envs=1, noop_max=0, end_on_flag=True)

    config_path = run_dir / "train_config.json"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        env_data = dict(data.get("env") or {})
        if fallback is not None:
            merged = asdict(fallback)
            merged.update(env_data)
            return config_from_dict(merged)
        return config_from_dict(env_data)
    except Exception:
        return fallback or EnvConfig(n_envs=1, noop_max=0, end_on_flag=True)


def stack_observations(frame_stack: deque[np.ndarray]) -> np.ndarray:
    return np.concatenate(list(frame_stack), axis=2).transpose(2, 0, 1)


def sanitize_x_position(
    raw_x_pos: Any,
    *,
    previous_x_pos: int = 0,
    flag_get: bool = False,
    goal_line_x: int = LEVEL_1_1_GOAL_X,
) -> int:
    previous = max(0, int(previous_x_pos))
    max_reasonable = int(goal_line_x) + 256
    if flag_get:
        try:
            xp = int(raw_x_pos)
        except (TypeError, ValueError):
            return int(goal_line_x)
        if 0 < xp <= max_reasonable + 128:
            return xp
        return int(goal_line_x)

    try:
        x_pos = int(raw_x_pos)
    except (TypeError, ValueError):
        return previous

    if x_pos < 0 or x_pos > max_reasonable:
        return previous

    if previous > 0 and x_pos - previous > MAX_REASONABLE_X_DELTA:
        return previous

    return x_pos


def extract_sanitized_x_position(
    info: dict[str, Any],
    *,
    previous_x_pos: int = 0,
    goal_line_x: int = LEVEL_1_1_GOAL_X,
) -> int:
    return sanitize_x_position(
        info.get("x_pos", previous_x_pos),
        previous_x_pos=previous_x_pos,
        flag_get=bool(info.get("flag_get", False)),
        goal_line_x=goal_line_x,
    )


def _life_count_from_info(info: dict[str, Any]) -> int | None:
    for key in ("life", "lives"):
        value = info.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _int_from_info(info: dict[str, Any], key: str) -> int | None:
    value = info.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def stage_order_index(world_stage: tuple[int, int] | None) -> int:
    if world_stage is None:
        return 0
    world, stage = world_stage
    return max(0, (int(world) - 1) * 4 + int(stage))


def build_rollout_summary(
    *,
    episodes: int,
    deterministic: bool,
    episode_returns: list[float],
    episode_lengths: list[int],
    episode_flags: list[bool],
    max_x_positions: list[int],
    episode_stage_clears: list[int] | None = None,
    episode_furthest_world_stages: list[tuple[int, int] | None] | None = None,
    episode_final_world_stages: list[tuple[int, int] | None] | None = None,
    video_path: str | None = None,
    video_fps: int | None = None,
    video_num_frames: int | None = None,
    goal_line_x: int = LEVEL_1_1_GOAL_X,
    episode_goal_line_xs: list[int] | None = None,
) -> dict[str, Any]:
    flags_cleared = sum(1 for cleared in episode_flags if cleared)
    gl = int(goal_line_x)
    if episode_goal_line_xs is not None:
        remaining_distances = [
            0 if cleared else max(0, int(goal_x) - int(x))
            for x, cleared, goal_x in zip(
                max_x_positions,
                episode_flags,
                episode_goal_line_xs,
            )
        ]
    else:
        remaining_distances = [
            0 if cleared else max(0, gl - x)
            for x, cleared in zip(max_x_positions, episode_flags)
        ]
    clear_lengths = [length for length, cleared in zip(episode_lengths, episode_flags) if cleared]
    summary = {
        "goal_line_x": gl,
        "episodes": episodes,
        "deterministic": deterministic,
        "average_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "average_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "flags_cleared": flags_cleared,
        "clear_rate": float(flags_cleared / episodes) if episodes else 0.0,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "episode_flags": episode_flags,
        "max_x_positions": max_x_positions,
        "average_max_x": float(np.mean(max_x_positions)) if max_x_positions else 0.0,
        "median_max_x": float(np.median(max_x_positions)) if max_x_positions else 0.0,
        "best_max_x": int(max(max_x_positions)) if max_x_positions else 0,
        "remaining_distances": remaining_distances,
        "average_remaining_distance": (
            float(np.mean(remaining_distances)) if remaining_distances else float(gl)
        ),
        "median_remaining_distance": (
            float(np.median(remaining_distances))
            if remaining_distances
            else float(gl)
        ),
        "best_remaining_distance": (
            int(min(remaining_distances)) if remaining_distances else gl
        ),
        "average_clear_steps": float(np.mean(clear_lengths)) if clear_lengths else None,
        "best_clear_steps": int(min(clear_lengths)) if clear_lengths else None,
    }
    if episode_goal_line_xs is not None:
        summary["episode_goal_line_xs"] = [int(x) for x in episode_goal_line_xs]
    if episode_stage_clears is not None:
        summary["episode_stage_clears"] = episode_stage_clears
        summary["average_stage_clears"] = (
            float(np.mean(episode_stage_clears)) if episode_stage_clears else 0.0
        )
        summary["best_stage_clears"] = int(max(episode_stage_clears)) if episode_stage_clears else 0
    if episode_furthest_world_stages is not None:
        furthest_serialized = [
            list(ws) if ws is not None else None for ws in episode_furthest_world_stages
        ]
        summary["episode_furthest_world_stages"] = furthest_serialized
        best_ws = max(
            episode_furthest_world_stages,
            key=stage_order_index,
            default=None,
        )
        summary["best_world_stage"] = list(best_ws) if best_ws is not None else None
        summary["best_world_stage_index"] = int(stage_order_index(best_ws))
    if episode_final_world_stages is not None:
        summary["episode_final_world_stages"] = [
            list(ws) if ws is not None else None for ws in episode_final_world_stages
        ]
    if episode_furthest_world_stages is not None and episode_goal_line_xs is not None:
        episode_stage_indices = [stage_order_index(ws) for ws in episode_furthest_world_stages]
        summary["episode_world_stage_indices"] = [int(idx) for idx in episode_stage_indices]
        if episode_stage_indices:
            best_episode_index = max(
                range(len(episode_stage_indices)),
                key=lambda idx: (
                    int(episode_stage_indices[idx]),
                    -int(remaining_distances[idx]),
                    int(max_x_positions[idx]),
                    float(episode_returns[idx]) if idx < len(episode_returns) else 0.0,
                ),
            )
            best_ws = episode_furthest_world_stages[best_episode_index]
            summary["progress_best_episode_index"] = int(best_episode_index)
            summary["best_world_stage"] = list(best_ws) if best_ws is not None else None
            summary["best_world_stage_index"] = int(stage_order_index(best_ws))
            summary["best_remaining_distance"] = int(remaining_distances[best_episode_index])
            summary["best_max_x"] = int(max_x_positions[best_episode_index])
    if video_path:
        summary["video"] = video_path
    if video_fps is not None and video_num_frames is not None:
        fps_d = max(int(video_fps), 1)
        summary["video_fps"] = int(video_fps)
        summary["video_num_frames"] = int(video_num_frames)
        summary["video_playback_seconds"] = float(video_num_frames) / float(fps_d)
    return summary


def run_policy_preview(
    model,
    output_dir: str | os.PathLike[str],
    *,
    episodes: int = 1,
    max_steps: int = 512,
    fps: int = 15,
    deterministic: bool = True,
    render_human: bool = False,
    seed: int = 123,
    config: EnvConfig | None = None,
    record_video: bool = True,
) -> dict[str, Any]:
    preview_config = config or EnvConfig(n_envs=1, noop_max=0, end_on_flag=True)
    env = make_single_env(preview_config, seed=seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_flags: list[bool] = []
    episode_stage_clears: list[int] = []
    episode_furthest_world_stages: list[tuple[int, int] | None] = []
    episode_final_world_stages: list[tuple[int, int] | None] = []
    episode_progress_goal_xs: list[int] = []
    captured_frames: list[np.ndarray] = []
    frame_stack: deque[np.ndarray] = deque(maxlen=preview_config.frame_stack)
    max_x_positions: list[int] = []

    for episode_idx in range(episodes):
        obs, reset_info = env.reset(seed=seed + episode_idx)
        frame_stack.clear()
        for _stack_idx in range(preview_config.frame_stack):
            frame_stack.append(obs)

        done = False
        episode_return = 0.0
        episode_length = 0
        episode_max_x = 0
        stage_clears = 0
        current_ws = world_stage_from_info(
            reset_info,
            (int(preview_config.world), int(preview_config.stage)),
        )
        furthest_ws = current_ws
        goal_x = effective_goal_line_x(preview_config)
        stage_x_positions: dict[tuple[int, int], int] = {}
        stage_max_x_positions: dict[tuple[int, int], int] = {}
        info = reset_info
        if record_video:
            captured_frames.append(render_rgb_frame(env))

        while not done and episode_length < max_steps:
            if render_human:
                env.unwrapped.render(mode="human")

            stacked_obs = stack_observations(frame_stack)
            action, _state = model.predict(stacked_obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            frame_stack.append(obs)
            if record_video:
                captured_frames.append(render_rgb_frame(env))
            episode_return += float(reward)
            episode_length += 1
            current_ws = world_stage_from_info(info, current_ws)
            stage_goal_x = goal_line_x_for_world_stage(current_ws, fallback=goal_x)
            previous_stage_x = (
                stage_x_positions.get(current_ws, 0)
                if current_ws is not None
                else 0
            )
            stage_x_pos = extract_sanitized_x_position(
                info,
                previous_x_pos=previous_stage_x,
                goal_line_x=stage_goal_x,
            )
            if current_ws is not None:
                stage_x_positions[current_ws] = stage_x_pos
                stage_max_x_positions[current_ws] = max(
                    stage_max_x_positions.get(current_ws, 0),
                    stage_x_pos,
                )
            episode_max_x = max(episode_max_x, stage_x_pos)
            if info.get("stage_clear", False):
                stage_clears += 1
            if stage_order_index(current_ws) > stage_order_index(furthest_ws):
                furthest_ws = current_ws
            done = bool(terminated or truncated)

        episode_flags.append(bool(info.get("flag_get", False)))

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        if preview_config.whole_game:
            progress_x = (
                stage_max_x_positions.get(furthest_ws, 0)
                if furthest_ws is not None
                else 0
            )
            progress_goal_x = goal_line_x_for_world_stage(furthest_ws, fallback=goal_x)
        else:
            progress_x = episode_max_x
            progress_goal_x = goal_x
        max_x_positions.append(int(progress_x))
        episode_progress_goal_xs.append(int(progress_goal_x))
        episode_stage_clears.append(stage_clears)
        episode_furthest_world_stages.append(furthest_ws)
        episode_final_world_stages.append(current_ws)

    video_path: Path | None = None
    if record_video and captured_frames:
        video_path = output_path / "preview.mp4"
        save_video(captured_frames, video_path, fps=fps)

    stage_metric_kwargs: dict[str, Any] = {}
    if preview_config.whole_game:
        stage_metric_kwargs = {
            "episode_stage_clears": episode_stage_clears,
            "episode_furthest_world_stages": episode_furthest_world_stages,
            "episode_final_world_stages": episode_final_world_stages,
        }

    summary = build_rollout_summary(
        episodes=episodes,
        deterministic=deterministic,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        episode_flags=episode_flags,
        max_x_positions=max_x_positions,
        video_path=str(video_path) if video_path else None,
        video_fps=fps if video_path and captured_frames else None,
        video_num_frames=len(captured_frames) if video_path and captured_frames else None,
        goal_line_x=effective_goal_line_x(preview_config),
        episode_goal_line_xs=episode_progress_goal_xs if preview_config.whole_game else None,
        **stage_metric_kwargs,
    )
    write_json(summary, output_path / "summary.json")
    env.close()
    return summary


class PreviewCallback(BaseCallback):
    """Periodically preview the current policy during training."""

    def __init__(
        self,
        *,
        preview_freq: int,
        output_dir: str | os.PathLike[str],
        preview_steps: int = 512,
        preview_episodes: int = 1,
        preview_fps: int = 15,
        deterministic: bool = True,
        render_human: bool = False,
        record_video: bool = True,
        seed: int = 123,
        start_timesteps: int = 0,
        config: EnvConfig | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.preview_freq = preview_freq
        self.output_dir = Path(output_dir)
        self.preview_steps = preview_steps
        self.preview_episodes = preview_episodes
        self.preview_fps = preview_fps
        self.deterministic = deterministic
        self.render_human = render_human
        self.record_video = record_video
        self.seed = seed
        self.config = config
        self._next_preview = (
            ((start_timesteps // preview_freq) + 1) * preview_freq if preview_freq > 0 else 0
        )

    def _on_step(self) -> bool:
        if self.preview_freq <= 0 or self.num_timesteps < self._next_preview:
            return True

        preview_dir = self.output_dir / f"step_{self.num_timesteps:07d}"
        summary = run_policy_preview(
            self.model,
            preview_dir,
            episodes=self.preview_episodes,
            max_steps=self.preview_steps,
            fps=self.preview_fps,
            deterministic=self.deterministic,
            render_human=self.render_human,
            record_video=self.record_video,
            seed=self.seed + self.num_timesteps,
            config=self.config,
        )

        if self.verbose > 0:
            video = summary.get("video") or "disabled"
            print(
                "preview_ok "
                f"step={self.num_timesteps} "
                f"furthest={summary.get('best_world_stage', ['?', '?'])} "
                f"best_remaining={summary.get('best_remaining_distance')} "
                f"best_max_x={summary.get('best_max_x')} "
                f"video={video}"
            )

        while self._next_preview <= self.num_timesteps:
            self._next_preview += self.preview_freq

        return True


def evaluation_score_tuple(summary: dict[str, Any]) -> tuple[float, ...]:
    if "best_world_stage_index" in summary:
        return (
            float(summary.get("best_world_stage_index", 0.0)),
            -float(summary.get("best_remaining_distance", float(LEVEL_1_1_GOAL_X))),
            -float(summary.get("average_remaining_distance", float(LEVEL_1_1_GOAL_X))),
            float(summary.get("best_max_x", 0.0)),
            float(summary.get("average_return", 0.0)),
        )
    return (
        float(summary.get("clear_rate", 0.0)),
        -float(summary.get("best_remaining_distance", float(LEVEL_1_1_GOAL_X))),
        float(summary.get("best_max_x", 0.0)),
        -float(summary.get("median_remaining_distance", float(LEVEL_1_1_GOAL_X))),
        -float(summary.get("average_remaining_distance", float(LEVEL_1_1_GOAL_X))),
        float(summary.get("average_return", 0.0)),
    )


class EvalCheckpointCallback(BaseCallback):
    """Run deterministic evaluation and save the best model by progress metrics."""

    def __init__(
        self,
        *,
        eval_freq: int,
        output_dir: str | os.PathLike[str],
        eval_steps: int = 4_000,
        eval_episodes: int = 5,
        deterministic: bool = True,
        seed: int = 123,
        start_timesteps: int = 0,
        config: EnvConfig | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_freq = eval_freq
        self.output_dir = Path(output_dir)
        self.eval_steps = eval_steps
        self.eval_episodes = eval_episodes
        self.deterministic = deterministic
        self.seed = seed
        self.config = config
        self.best_score = (
            -float("inf"),
            -float("inf"),
            -float("inf"),
            -float("inf"),
            -float("inf"),
            -float("inf"),
        )
        self._next_eval = ((start_timesteps // eval_freq) + 1) * eval_freq if eval_freq > 0 else 0

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps < self._next_eval:
            return True

        self.output_dir.mkdir(parents=True, exist_ok=True)
        eval_dir = self.output_dir / f"step_{self.num_timesteps:07d}"
        summary = run_policy_preview(
            self.model,
            eval_dir,
            episodes=self.eval_episodes,
            max_steps=self.eval_steps,
            deterministic=self.deterministic,
            seed=self.seed + self.num_timesteps,
            config=self.config,
            record_video=False,
        )
        summary["num_timesteps"] = self.num_timesteps

        score = evaluation_score_tuple(summary)
        is_best = score > self.best_score
        summary["selection_score"] = list(score)
        summary["is_best"] = is_best

        if is_best:
            self.best_score = score
            best_model_path = self.output_dir / "best_eval.zip"
            self.model.save(best_model_path)
            write_json(
                {
                    "num_timesteps": self.num_timesteps,
                    "best_model": str(best_model_path),
                    "selection_score": list(score),
                    "summary": summary,
                },
                self.output_dir / "best_eval_summary.json",
            )

        write_json(summary, eval_dir / "summary.json")

        if self.verbose > 0:
            print(
                "eval_ok "
                f"step={self.num_timesteps} "
                f"furthest={summary.get('best_world_stage', ['?', '?'])} "
                f"best_remaining={summary.get('best_remaining_distance')} "
                f"best_max_x={summary.get('best_max_x')} "
                f"best={is_best}"
            )

        while self._next_eval <= self.num_timesteps:
            self._next_eval += self.eval_freq

        return True
