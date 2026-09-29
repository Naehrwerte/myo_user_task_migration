from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from myo_core.common.sequential_task import SequentialTaskLogic, TaskEntity


class VrRayTaskLogic(SequentialTaskLogic):
    """Sequential task logic where a target counts as hit while the controller
    ray points at it (instead of the end effector touching it).
    """

    def _resolve_site_ids(self, asset: TaskEntity, asset_cfg: SceneEntityCfg) -> None:
        super()._resolve_site_ids(asset, asset_cfg)
        asset.ray_end_site_id = asset_cfg.site_ids[1]

    def _inside_target(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        asset = self.asset

        current_target_id = asset.current_target_id
        target_size_id = asset.target_size_ids[current_target_id]

        ray_origin = asset.data.site_pos_w[asset.env_ids, asset.end_effector_site_id]
        ray_end = asset.data.site_pos_w[asset.env_ids, asset.ray_end_site_id]
        target_position = asset.target_pos[asset.env_ids, current_target_id]
        target_radius = asset.data.model.geom_size[asset.env_ids, target_size_id, 0]

        ray_dir = torch.nn.functional.normalize(ray_end - ray_origin, dim=-1, eps=1e-8)

        v = target_position - ray_origin
        proj = (v * ray_dir).sum(dim=-1)
        perp_vec = v - proj[:, None] * ray_dir
        perp_distance = torch.linalg.vector_norm(perp_vec, dim=-1)

        pointing_forward = proj > 0.0
        inside_target = (perp_distance < target_radius) & pointing_forward

        distance_to_target = torch.where(
            pointing_forward,
            perp_distance,
            torch.linalg.vector_norm(v, dim=-1),
        )

        return inside_target, distance_to_target, target_radius
