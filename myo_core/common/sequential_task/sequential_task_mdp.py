"""Observation / reward / termination terms shared by every sequential task.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .sequential_task_entity import TaskEntity


def target_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: TaskEntity = env.scene[asset_cfg.name]

    return asset.target_pos[asset.env_ids, asset.current_target_id]


def target_size(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: TaskEntity = env.scene[asset_cfg.name]

    target_size_ids = asset.target_size_ids[asset.current_target_id]

    return asset.data.model.geom_size[asset.env_ids, target_size_ids]


def phase_progress(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: TaskEntity = env.scene[asset_cfg.name]

    if asset.num_phases <= 1:
        return torch.zeros((env.num_envs, 1), device=env.device)

    progress = asset.current_phase.float() / (asset.num_phases - 1)

    return -1.0 + 2.0 * progress[:, None]


def dwell_fraction(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: TaskEntity = env.scene[asset_cfg.name]

    fraction = (
        (asset.current_target_dwell_steps > 0)
        * asset.steps_inside_target
        / asset.current_target_dwell_steps.clip(min=1)
    )

    return fraction[:, None]


def sequential_distance_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    exponential_distance_reward,
    distance_metric
) -> torch.Tensor:
    asset: TaskEntity = env.scene[asset_cfg.name]

    distance_to_target = asset.distance_to_target

    if asset.num_phases > 1:
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


def phase_bonus(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: TaskEntity = env.scene[asset_cfg.name]

    return asset.current_phase_completed.float()


def phase_successfully_completed(
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
