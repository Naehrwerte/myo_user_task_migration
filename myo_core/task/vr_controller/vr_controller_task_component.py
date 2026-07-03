from __future__ import annotations

import math
from dataclasses import dataclass, field

from mjlab.entity import EntityCfg
from mjlab.envs.mdp import Entity
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr, events as event_fns
from mjlab.managers import ManagerTermBase
from mjlab.envs.mdp import terminations as mdp_terminations
from mjlab.managers import EventTermCfg, MetricsTermCfg, RewardTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

import mujoco
import numpy as np
import torch

import myo_core.common as myo
from ..task_registry import myo_register_task
from .vr_controller_task_config import VrControllerTaskConfig
from ..universal.universal_task_config import PointingTargetConfig


_VR_ENTITY_NAME = "vr_controller_robot"


def _target_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: TaskEntity = env.scene[asset_cfg.name]

    target_pos = asset.target_pos[asset.env_ids, asset.current_target_id]

    return target_pos


def _target_size(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: TaskEntity = env.scene[asset_cfg.name]

    target_size_ids = asset.target_size_ids[asset.current_target_id]
    target_size = asset.data.model.geom_size[asset.env_ids, target_size_ids]

    return target_size


def _phase_progress(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: TaskEntity = env.scene[asset_cfg.name]

    if asset.num_targets <= 1:
        return torch.zeros((env.num_envs, 1), device=env.device)

    phase_progress = asset.current_target_id.float() / (asset.num_targets - 1)
    phase_progress = -1.0 + 2.0 * phase_progress[:, None]

    return phase_progress


def _dwell_fraction(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: TaskEntity = env.scene[asset_cfg.name]

    dwell_fraction = (
            (asset.current_target_dwell_steps > 0)
            * asset.steps_inside_target
            / asset.current_target_dwell_steps.clip(min=1)
    )

    return dwell_fraction[:, None]


def _sequential_distance_reward(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
        exponential_distance_reward,
        distance_metric
) -> torch.Tensor:
    asset: TaskEntity = env.scene[asset_cfg.name]

    distance_to_target = asset.distance_to_target

    if asset.num_targets > 1:
        remaining_distance = asset.remaining_target_distances.gather(
            dim=1,
            index=asset.current_target_id[:, None],
        ).squeeze(1)

        distances_total = distance_to_target + remaining_distance
    else:
        distances_total = distance_to_target

    if not exponential_distance_reward:
        return -distances_total

    outside_target = (~asset.inside_target).float()

    return outside_target * (torch.exp(-distances_total * distance_metric) - 1.0) / distance_metric


def _phase_bonus(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    phase_bonus = asset.current_phase_completed.float()

    return phase_bonus


def _phase_successfully_completed(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
        phase_id: int = 0,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    completed_count = asset.completed_target_count
    current_id = asset.current_target_id

    currently_completing_phase = (
            asset.current_phase_completed
            & (current_id == phase_id)
    )

    return (completed_count > phase_id) | currently_completing_phase


@dataclass
class TaskEntityCfg(EntityCfg):
    task_cfg: VrControllerTaskConfig | None = None

    def build(self) -> TaskEntity:
        """Build task entity instance from this config.
        """
        return TaskEntity(self)


class TaskEntity(Entity):
    num_targets: int
    task_cfg: VrControllerTaskConfig

    # index tensors
    ray_origin_site_id: int
    ray_end_site_id: int
    target_pos_ids: torch.Tensor  # [num_targets], long
    target_size_ids: torch.Tensor  # [num_targets], long
    env_ids: torch.Tensor  # [num_envs], long
    current_target_id: torch.Tensor  # [num_envs], long

    # int tensors
    completed_target_count: torch.Tensor  # [num_envs], int
    steps_inside_target: torch.Tensor  # [num_envs], int
    target_dwell_steps: torch.Tensor  # [num_targets], int
    current_target_dwell_steps: torch.Tensor  # [num_envs], int

    # bool tensors
    inside_target: torch.Tensor  # [num_envs], bool
    current_phase_completed: torch.Tensor  # [num_envs], bool

    # float tensors
    distance_to_target: torch.Tensor  # [num_envs], float
    target_size: torch.Tensor  # [num_envs], float
    remaining_target_distances: torch.Tensor  # [num_envs, num_targets], float
    target_pos: torch.Tensor  # [num_envs, num_targets, 3], float

    def __init__(self, cfg: TaskEntityCfg):
        super().__init__(cfg)

        self.task_cfg = self.cfg.task_cfg
        self.num_targets = len(self.task_cfg.targets)


class SequentialTaskLogic(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        asset: TaskEntity = env.scene[_VR_ENTITY_NAME]
        asset_cfg: SceneEntityCfg = cfg.params['asset_cfg']

        self.asset = asset

        asset.ray_origin_site_id = asset_cfg.site_ids[0]
        asset.ray_end_site_id = asset_cfg.site_ids[1]
        asset.target_pos_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.long, device=env.device)
        asset.target_size_ids = torch.tensor(asset_cfg.geom_ids, dtype=torch.long, device=env.device)
        asset.env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        asset.current_target_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        asset.completed_target_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        asset.steps_inside_target = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        asset.target_dwell_steps = torch.tensor(
            [math.ceil(t.dwell_duration / env.step_dt) for t in asset.task_cfg.targets],
            dtype=torch.int32,
            device=env.device
        )
        asset.current_target_dwell_steps = asset.target_dwell_steps[asset.current_target_id]

        asset.target_pos = asset.data.body_com_pos_w[asset.env_ids[:, None], asset.target_pos_ids]
        asset.inside_target, asset.distance_to_target, asset.target_size = self._inside_target()

        asset.remaining_target_distances = torch.zeros((env.num_envs, asset.num_targets), dtype=torch.float32,
                                                       device=env.device)

        asset.current_phase_completed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None) -> None:
        "Note: This is run once before first initial step and always after every reset event is done by mjlab"
        asset = self.asset

        if env_ids is None:
            env_ids = asset.env_ids

        asset.current_target_id[env_ids] = 0
        asset.completed_target_count[env_ids] = 0
        asset.steps_inside_target[env_ids] = 0
        asset.current_phase_completed[env_ids] = False

        asset.target_pos[env_ids] = asset.data.body_com_pos_w[
            env_ids[:, None],
            asset.target_pos_ids,
        ]  # [num_reset_envs, num_targets, 3]

        if asset.num_targets <= 1:
            return

        segment_distances = torch.linalg.vector_norm(
            asset.target_pos[env_ids, 1:] - asset.target_pos[env_ids, :-1],
            dim=-1,
        )  # [num_reset_envs, num_targets - 1]

        asset.remaining_target_distances[env_ids] = 0.0
        asset.remaining_target_distances[env_ids, : asset.num_targets - 1] = (
            segment_distances.flip(1).cumsum(1).flip(1)
        )

    def __call__(self, env: ManagerBasedRlEnv, env_ids: None, asset_cfg: SceneEntityCfg) -> None:
        asset = self.asset

        completed = asset.current_phase_completed

        asset.completed_target_count += completed
        asset.completed_target_count.clamp_(max=asset.num_targets)

        asset.steps_inside_target.masked_fill_(completed, 0)

        asset.current_target_id = asset.completed_target_count.clamp(
            max=asset.num_targets - 1
        )

        asset.current_target_dwell_steps = asset.target_dwell_steps[
            asset.current_target_id
        ]

        (
            asset.inside_target,
            asset.distance_to_target,
            asset.target_size
        ) = self._inside_target()

        asset.steps_inside_target += asset.inside_target

        if asset.task_cfg.reach.dwell_continuous:
            asset.steps_inside_target *= asset.inside_target

        asset.current_phase_completed = (
                asset.steps_inside_target >= asset.current_target_dwell_steps
        )

    def _inside_target(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        asset = self.asset

        current_target_id = asset.current_target_id
        target_size_id = asset.target_size_ids[current_target_id]

        ray_origin = asset.data.site_pos_w[asset.env_ids, asset.ray_origin_site_id]
        ray_end = asset.data.site_pos_w[asset.env_ids, asset.ray_end_site_id]
        target_position = asset.target_pos[asset.env_ids, current_target_id]
        target_radius = asset.data.model.geom_size[asset.env_ids, target_size_id, 0]

        ray_dir = torch.nn.functional.normalize(ray_end - ray_origin, dim=-1, eps=1e-8)

        v = target_position - ray_origin
        proj = (v * ray_dir).sum(dim=-1)
        perp_vec = v - proj[:, None] * ray_dir
        distance_to_target = torch.linalg.vector_norm(perp_vec, dim=-1)

        inside_target = (distance_to_target < target_radius) & (proj > 0.0)

        return inside_target, distance_to_target, target_radius



@dataclass
class TargetDomainRandomization:
    body_pos: dict[str, Ranges] = field(default_factory=dict)
    geom_size: dict[str, Ranges] = field(default_factory=dict)
    geom_rgb: dict[str, Ranges] = field(default_factory=dict)


def _find_body_in_spec(root_body, target_name: str):
    if root_body.name == target_name:
        return root_body
    child = root_body.first_body()
    while child is not None:
        result = _find_body_in_spec(child, target_name)
        if result is not None:
            return result
        child = child.next_body(root_body)
    return None



def _ray_origin(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.site_pos_w[:, asset_cfg.site_ids[0]]


def _ray_dir_world(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ray_origin = asset.data.site_pos_w[:, asset_cfg.site_ids[0]]
    ray_end = asset.data.site_pos_w[:, asset_cfg.site_ids[1]]
    return torch.nn.functional.normalize(ray_end - ray_origin, dim=-1, eps=1e-8)


@myo_register_task("vr_controller")
class VrControllerTaskComponent(myo.MyoComponent):
    def __init__(self, cfg: VrControllerTaskConfig):
        self.cfg = cfg

        if len(self.cfg.targets) == 0:
            raise ValueError("No targets defined")

    def modify_env_cfg(self, cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
        _, target_dr, model_names = self._create_model()

        cfg.scene.entities.update({
            _VR_ENTITY_NAME: TaskEntityCfg(
                spec_fn=lambda: self._create_model()[0],
                articulation=EntityArticulationInfoCfg(
                    actuators=(
                        XmlActuatorCfg(
                            target_names_expr=tuple(model_names.tendon_names),
                            transmission_type=TransmissionType.TENDON,
                        ),
                    )
                ),
                task_cfg=self.cfg,
            )
        })

        num_targets = len(self.cfg.targets)

        entity_cfg = SceneEntityCfg(
            _VR_ENTITY_NAME,
            joint_names=model_names.independent_joint_names,
            site_names=[self.cfg.reach.end_effector_site, self.cfg.reach.ray_end_site],
        )
        target_entity_cfg = SceneEntityCfg(
            _VR_ENTITY_NAME,
            body_names=[f"body_target_{i}" for i in range(num_targets)],
            geom_names=[f"geom_target_{i}" for i in range(num_targets)],
            site_names=[self.cfg.reach.end_effector_site, self.cfg.reach.ray_end_site],
        )

        _obs_terms_complete = {
            "time": ObservationTermCfg(func=myo.time),
            "qpos": ObservationTermCfg(func=myo.joint_qpos, params={"asset_cfg": entity_cfg}),
            "qvel": ObservationTermCfg(func=myo.joint_qvel, params={"asset_cfg": entity_cfg}),
            "qacc": ObservationTermCfg(func=myo.joint_qacc, params={"asset_cfg": entity_cfg}),
            "act": ObservationTermCfg(func=myo.act, params={"asset_cfg": entity_cfg}),
            "ray_origin": ObservationTermCfg(func=_ray_origin, params={"asset_cfg": entity_cfg}),
            "ray_dir": ObservationTermCfg(func=_ray_dir_world, params={"asset_cfg": entity_cfg}),
            "target_pos": ObservationTermCfg(func=_target_pos, params={"asset_cfg": entity_cfg}),
            "target_size": ObservationTermCfg(func=_target_size, params={"asset_cfg": entity_cfg}),
            "phase_progress": ObservationTermCfg(func=_phase_progress, params={"asset_cfg": entity_cfg}),
            "dwell_fraction": ObservationTermCfg(func=_dwell_fraction, params={"asset_cfg": entity_cfg}),
        }

        cfg.observations.update({
            "agent_state": ObservationGroupCfg(
                terms={k: v for k, v in _obs_terms_complete.items() if k in self.cfg.agent_state_keys}
            ),
            "task_state": ObservationGroupCfg(
                terms={k: v for k, v in _obs_terms_complete.items() if k in self.cfg.task_state_keys}
            ),
        })

        cfg.actions.update({
            "muscles": myo.MyoMuscleActivationActionCfg(
                entity_name=_VR_ENTITY_NAME,
                actuator_names=model_names.tendon_names,
            ),
        })

        cfg.rewards.update({
            "distance": RewardTermCfg(
                func=_sequential_distance_reward,
                params={
                    "asset_cfg": entity_cfg,
                    "exponential_distance_reward": self.cfg.reward.distance_exponential,
                    "distance_metric": self.cfg.reward.distance_metric,
                },
                weight=self.cfg.reward.weights.get("distance", 0.0),
            ),
            "neural_effort": RewardTermCfg(
                func=myo.neural_effort,
                params={"asset_cfg": entity_cfg},
                weight=self.cfg.reward.weights.get("neural_effort", 0.0),
            ),
            "jac_effort": RewardTermCfg(
                func=myo.jac_effort,
                params={"asset_cfg": entity_cfg},
                weight=self.cfg.reward.weights.get("jac_effort", 0.0),
            ),
            "phase_bonus": RewardTermCfg(
                func=_phase_bonus,
                params={"asset_cfg": entity_cfg},
                weight=self.cfg.reward.weights.get("phase_bonus", 0.0),
            ),
            "done": RewardTermCfg(
                func=_phase_successfully_completed,
                params={"asset_cfg": entity_cfg, "phase_id": num_targets - 1},
                weight=self.cfg.reward.weights.get("done", 0.0),
            ),
        })

        cfg.events.update({
            "reset_joints": EventTermCfg(
                func=event_fns.reset_joints_by_offset,
                params={"asset_cfg": entity_cfg, "position_range": (-0.1, 0.1), "velocity_range": (-0.1, 0.1)},
                mode="reset",
            ),
            "task_logic_update": EventTermCfg(
                func=SequentialTaskLogic,
                params={"asset_cfg": target_entity_cfg},
                mode="step",
            ),
        })

        if len(target_dr.body_pos) > 0:
            cfg.events["target_pos_dr"] = EventTermCfg(
                mode="reset",
                func=dr.body_pos,
                params={
                    "asset_cfg": SceneEntityCfg(_VR_ENTITY_NAME, body_names=tuple(target_dr.body_pos.keys())),
                    "ranges": target_dr.body_pos,
                    "operation": "abs",
                },
            )

        if len(target_dr.geom_size) > 0:
            cfg.events["target_size_dr"] = EventTermCfg(
                mode="reset",
                func=dr.geom_size,
                params={
                    "asset_cfg": SceneEntityCfg(_VR_ENTITY_NAME, geom_names=tuple(target_dr.geom_size.keys())),
                    "ranges": target_dr.geom_size,
                    "operation": "abs",
                },
            )

        if len(target_dr.geom_rgb) > 0:
            cfg.events["target_rgba_dr"] = EventTermCfg(
                mode="reset",
                func=dr.geom_rgba,
                params={
                    "asset_cfg": SceneEntityCfg(_VR_ENTITY_NAME, geom_names=tuple(target_dr.geom_rgb.keys())),
                    "ranges": target_dr.geom_rgb,
                    "axes": [0, 1, 2],
                    "operation": "abs",
                },
            )

        cfg.terminations.update({
            "time_out": TerminationTermCfg(
                func=mdp_terminations.time_out,
                time_out=True,
            ),
            "episode_success": TerminationTermCfg(
                func=_phase_successfully_completed,
                params={"asset_cfg": entity_cfg, "phase_id": num_targets - 1},
            ),
        })

        for i in range(num_targets):
            cfg.metrics[f"phase_{i}_success"] = MetricsTermCfg(
                func=_phase_successfully_completed,
                params={"asset_cfg": entity_cfg, "phase_id": i},
                reduce="last",
            )

        cfg.metrics.update({
            "inside_target": MetricsTermCfg(
                func=lambda env: env.scene[_VR_ENTITY_NAME].inside_target
            ),
            "total_initial_distance": MetricsTermCfg(
                func=lambda env: env.scene[_VR_ENTITY_NAME].remaining_target_distances[:, 0]
            ),
            "target_size": MetricsTermCfg(
                func=lambda env: env.scene[_VR_ENTITY_NAME].target_size
            ),
            "completed_target_count": MetricsTermCfg(
                func=lambda env: env.scene[_VR_ENTITY_NAME].completed_target_count,
                reduce="last",
            ),
        })

    def _create_model(self) -> tuple[mujoco.MjSpec, TargetDomainRandomization, myo.MyoModelNames]:
        spec = mujoco.MjSpec.from_file(self.cfg.model_path)

        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        model_names = myo.myo_get_model_names(model)

        reference_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.cfg.reach.reference_site)
        if reference_site < 0:
            raise ValueError(f"Unknown reference site: {self.cfg.reach.reference_site}")
        target_origin = data.site_xpos[reference_site] + np.array(self.cfg.reach.reference_offset)

        spec, dr_ = self._modify_spec(spec, target_origin)

        return spec, dr_, model_names

    def _modify_spec(
        self, spec: mujoco.MjSpec, target_origin: np.ndarray
    ) -> tuple[mujoco.MjSpec, TargetDomainRandomization]:
        dr_ = TargetDomainRandomization()

        for target_id, target in enumerate(self.cfg.targets):
            target_body_name = f"body_target_{target_id}"
            target_geom_name = f"geom_target_{target_id}"

            if not isinstance(target, PointingTargetConfig):
                raise ValueError("Currently only pointing targets are implemented")
            target: PointingTargetConfig = target

            body_pos = np.array(target_origin, copy=True)
            if target.position.value is not None:
                body_pos += np.array(target.position.value[:3])
            else:
                dr_.body_pos[target_body_name] = {
                    dim: (
                        target_origin[dim] + target.position.min[dim],
                        target_origin[dim] + target.position.max[dim],
                    )
                    for dim in range(3)
                }

            target_body = spec.worldbody.add_body(
                name=target_body_name,
                pos=body_pos,
            )

            geom_size = np.ones(3)
            if target.size.value is not None:
                geom_size *= target.size.value
            else:
                dr_.geom_size[target_geom_name] = {
                    dim: (target.size.min, target.size.max) for dim in range(3)
                }

            geom_rgba = np.ones(4)
            if target.rgb.value is not None:
                geom_rgba[:3] = np.array(target.rgb.value)
            else:
                dr_.geom_rgb[target_geom_name] = {
                    dim: (target.rgb.min[dim], target.rgb.max[dim]) for dim in range(3)
                }

            target_body.add_geom(
                name=target_geom_name,
                type=mujoco.mjtGeom.mjGEOM_SPHERE if target.shape == "sphere" else mujoco.mjtGeom.mjGEOM_BOX,
                pos=np.zeros(3),
                size=geom_size,
                rgba=geom_rgba,
            )
        return spec, dr_
