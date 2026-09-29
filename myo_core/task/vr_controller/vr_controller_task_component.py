from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from myo_core.common.sequential_task import PointingTargetConfig, SequentialTaskComponent
from ..task_registry import myo_register_task
from .vr_controller_task_config import VrControllerTaskConfig
from .vr_controller_task_logic import VrRayTaskLogic


def _ray_origin(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.site_pos_w[:, asset_cfg.site_ids[0]]


def _ray_dir_world(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ray_origin = asset.data.site_pos_w[:, asset_cfg.site_ids[0]]
    ray_end = asset.data.site_pos_w[:, asset_cfg.site_ids[1]]
    return torch.nn.functional.normalize(ray_end - ray_origin, dim=-1, eps=1e-8)


@myo_register_task("vr_controller")
class VrControllerTaskComponent(SequentialTaskComponent):
    """Pointing task: aim a hand-held VR controller's ray at a target sequence.

    Uses the shared sequential task pipeline with
    the ray-based completion criterion and the ray observations replacing the
    end-effector position.
    """

    entity_name = "vr_controller_robot"
    task_logic_cls = VrRayTaskLogic
    supported_target_types = (PointingTargetConfig,)

    cfg: VrControllerTaskConfig

    def __init__(self, cfg: VrControllerTaskConfig):
        super().__init__(cfg)

    def _site_names(self) -> list[str]:
        return [self.cfg.reach.end_effector_site, self.cfg.reach.ray_end_site]

    def _observation_terms(self, entity_cfg: SceneEntityCfg) -> dict[str, ObservationTermCfg]:
        terms = super()._observation_terms(entity_cfg)

        # The controller ray replaces the end effector position.
        terms.pop("ee_pos", None)
        terms["ray_origin"] = ObservationTermCfg(func=_ray_origin, params={"asset_cfg": entity_cfg})
        terms["ray_dir"] = ObservationTermCfg(func=_ray_dir_world, params={"asset_cfg": entity_cfg})

        return terms
