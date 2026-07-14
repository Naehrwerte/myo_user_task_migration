from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import EventTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ..universal.universal_task_component import SequentialTaskLogic, TaskEntity

# Buttons are colored purely by their role in the sequence (per environment):
#   green  -> the button currently being pressed
#   red    -> still to be pressed (has an occurrence at/after the current phase)
#   blue   -> already done (or not part of this episode's sequence)
_RGBA_CURRENT = (0.0, 1.0, 0.0, 1.0)
_RGBA_TODO = (1.0, 0.0, 0.0, 0.5)
_RGBA_DONE = (0.0, 0.0, 1.0, 0.5)


class NumpadTaskLogic(SequentialTaskLogic):
    """Sequential task logic with a randomized press sequence over a fixed pool.

    The universal :class:`SequentialTaskLogic` treats the target list itself as
    the press sequence (``current_target_id`` runs ``0, 1, ... num_targets-1``).

    The numpad instead keeps *all* targets as an always-shown pool and presses a
    separately drawn sequence of ``sequence_length`` buttons. A per-environment
    ``phase_seq`` tensor maps each sequence position (phase) to a physical target
    index; it is resampled at every reset (with replacement by default, so digits
    may repeat and the sequence may be longer than the pool).

    Attribute contract with the universal observation/reward terms is preserved
    (``current_target_id``, ``completed_target_count``, ``inside_target``,
    ``distance_to_target``, ``target_size``, ``current_phase_completed``), so the
    inherited ``_inside_target`` and most universal terms keep working. Only the
    sequence-length/position dependent terms are overridden (see numpad component).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        asset: TaskEntity = self.asset
        self._device = env.device
        self.sample_with_replacement = asset.task_cfg.sample_with_replacement

        asset.seq_len = int(asset.task_cfg.sequence_length)
        if asset.seq_len < 1:
            raise ValueError(f"sequence_length must be >= 1, got {asset.seq_len}")
        if not self.sample_with_replacement and asset.seq_len > asset.num_targets:
            raise ValueError(
                "sequence_length cannot exceed the button pool size when sampling "
                f"without replacement ({asset.seq_len} > {asset.num_targets})"
            )

        asset.phase_seq = torch.zeros(
            (env.num_envs, asset.seq_len), dtype=torch.long, device=env.device
        )
        asset.current_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        asset.remaining_target_distances = torch.zeros(
            (env.num_envs, asset.seq_len), dtype=torch.float32, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | None) -> None:
        asset: TaskEntity = self.asset

        if env_ids is None:
            env_ids = asset.env_ids

        n = env_ids.shape[0]
        seq_len = asset.seq_len

        asset.current_phase[env_ids] = 0
        asset.completed_target_count[env_ids] = 0
        asset.steps_inside_target[env_ids] = 0
        asset.current_phase_completed[env_ids] = False

        if self.sample_with_replacement:
            seq = torch.randint(
                0, asset.num_targets, (n, seq_len), device=self._device
            )
        else:
            seq = torch.argsort(
                torch.rand(n, asset.num_targets, device=self._device), dim=1
            )[:, :seq_len]

        asset.phase_seq[env_ids] = seq
        asset.current_target_id[env_ids] = seq[:, 0]

        asset.target_pos[env_ids] = asset.data.body_com_pos_w[
            env_ids[:, None],
            asset.target_pos_ids,
        ]

        asset.remaining_target_distances[env_ids] = 0.0
        if seq_len <= 1:
            return

        seq_pos = asset.target_pos[env_ids[:, None], seq]
        segment_distances = torch.linalg.vector_norm(
            seq_pos[:, 1:] - seq_pos[:, :-1], dim=-1
        )
        asset.remaining_target_distances[env_ids, : seq_len - 1] = (
            segment_distances.flip(1).cumsum(1).flip(1)
        )

    def __call__(self, env: ManagerBasedRlEnv, env_ids: None, asset_cfg: SceneEntityCfg) -> None:
        asset: TaskEntity = self.asset

        completed = asset.current_phase_completed

        asset.completed_target_count += completed
        asset.completed_target_count.clamp_(max=asset.seq_len)

        asset.steps_inside_target.masked_fill_(completed, 0)

        asset.current_phase = asset.completed_target_count.clamp(max=asset.seq_len - 1)
        asset.current_target_id = asset.phase_seq.gather(
            1, asset.current_phase[:, None]
        ).squeeze(1)

        asset.current_target_dwell_steps = asset.target_dwell_steps[asset.current_target_id]

        (
            asset.inside_target,
            asset.distance_to_target,
            asset.target_size,
        ) = self._inside_target()

        asset.steps_inside_target += asset.inside_target

        if asset.task_cfg.reach.dwell_continuous:
            asset.steps_inside_target *= asset.inside_target

        asset.current_phase_completed = (
            asset.steps_inside_target >= asset.current_target_dwell_steps
        )


def np_phase_progress(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: TaskEntity = env.scene[asset_cfg.name]

    if asset.seq_len <= 1:
        return torch.zeros((env.num_envs, 1), device=env.device)

    phase_progress = asset.current_phase.float() / (asset.seq_len - 1)
    return -1.0 + 2.0 * phase_progress[:, None]


def np_sequential_distance_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    exponential_distance_reward,
    distance_metric,
) -> torch.Tensor:
    asset: TaskEntity = env.scene[asset_cfg.name]

    distance_to_target = asset.distance_to_target

    if asset.seq_len > 1:
        remaining_distance = asset.remaining_target_distances.gather(
            dim=1,
            index=asset.current_phase[:, None],
        ).squeeze(1)
        distances_total = distance_to_target + remaining_distance
    else:
        distances_total = distance_to_target

    if not exponential_distance_reward:
        return -distances_total

    outside_target = (~asset.inside_target).float()
    return outside_target * (torch.exp(-distances_total * distance_metric) - 1.0) / distance_metric


def np_sequence_completed(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    phase_id: int = 0,
) -> torch.Tensor:
    """True once sequence phase ``phase_id`` has been (or is being) completed."""
    asset: TaskEntity = env.scene[asset_cfg.name]

    currently_completing_phase = (
        asset.current_phase_completed
        & (asset.current_phase == phase_id)
    )

    return (asset.completed_target_count > phase_id) | currently_completing_phase


@requires_model_fields("geom_rgba")
def np_color_targets_by_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Color every pool button by its role in the sequence (per environment).

    Each physical button is painted green/red/blue depending on whether it is the
    current target, still to be pressed, or already done — its configured color is
    not used. A button may occur several times in the (with-replacement) sequence;
    it counts as "todo" while it still has an occurrence at or after the current
    phase, otherwise as "done". Buttons absent from the sequence are shown as done.

    Runs as a per-step event so the colors follow the randomized sequence; the
    ``requires_model_fields`` decorator makes mjlab expand ``geom_rgba`` to
    per-world memory so each environment can be colored independently.
    """
    asset: TaskEntity = env.scene[asset_cfg.name]

    device = asset.phase_seq.device
    num_envs = asset.phase_seq.shape[0]
    num_targets = asset.num_targets

    buttons = torch.arange(num_targets, device=device)          # [P]
    phase_idx = torch.arange(asset.seq_len, device=device)      # [S]

    # For every (env, button): does the button occur, and at which last phase?
    occ = asset.phase_seq[:, :, None] == buttons[None, None, :]  # [E, S, P]
    appears = occ.any(dim=1)                                     # [E, P]
    last_occ = torch.where(occ, phase_idx[None, :, None], torch.full_like(occ, -1, dtype=torch.long))
    last_occ = last_occ.max(dim=1).values                       # [E, P]

    current_phase = asset.current_phase[:, None]                # [E, 1]
    is_current = buttons[None, :] == asset.current_target_id[:, None]  # [E, P]
    is_todo = appears & (last_occ >= current_phase) & ~is_current

    current = torch.tensor(_RGBA_CURRENT, dtype=torch.float32, device=device)
    todo = torch.tensor(_RGBA_TODO, dtype=torch.float32, device=device)
    done = torch.tensor(_RGBA_DONE, dtype=torch.float32, device=device)

    colors = done.expand(num_envs, num_targets, 4).clone()      # default: done/inactive
    colors[is_todo] = todo
    colors[is_current] = current                                # highest priority

    env.sim.model.geom_rgba[:, asset.target_size_ids, :] = colors
