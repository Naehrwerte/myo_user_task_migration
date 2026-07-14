from __future__ import annotations

import math
from dataclasses import dataclass, field

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg, ManagerBasedRlEnv
from mjlab.envs.mdp import Entity
from mjlab.envs.mdp import terminations as mdp_terminations
from mjlab.envs.mdp import dr, events as event_fns
from mjlab.envs.mdp.dr._core import Ranges
from mjlab.managers import EventTermCfg, ManagerTermBase, MetricsTermCfg, RewardTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
import mujoco
import numpy as np
import torch

import myo_core.common as myo
from .universal_task_config import UniversalTaskConfig, PointingTargetConfig, ButtonTargetConfig
from ..task_registry import myo_register_task

_UNIVERSAL_ENTITY_NAME = "universal_robot"

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
    task_cfg: UniversalTaskConfig | None = None
  
    def build(self) -> TaskEntity:
        """Build task entity instance from this config.
        """
        return TaskEntity(self)

class TaskEntity(Entity):
    num_targets: int
    task_cfg: UniversalTaskConfig

    # index tensors
    end_effector_site_id: int
    target_pos_ids: torch.Tensor                 # [num_targets], long
    target_size_ids: torch.Tensor                # [num_targets], long
    env_ids: torch.Tensor                        # [num_envs], long
    current_target_id: torch.Tensor              # [num_envs], long

    # button targets (touch-based completion)
    target_is_button: torch.Tensor               # [num_targets], bool
    target_min_touch_force: torch.Tensor         # [num_targets], float
    target_sensor_adr: torch.Tensor              # [num_targets], long (-1 for non-buttons)

    # int tensors
    completed_target_count: torch.Tensor         # [num_envs], int
    steps_inside_target: torch.Tensor            # [num_envs], int
    target_dwell_steps: torch.Tensor             # [num_targets], int
    current_target_dwell_steps: torch.Tensor     # [num_envs], int

    # bool tensors
    inside_target: torch.Tensor                  # [num_envs], bool
    current_phase_completed: torch.Tensor        # [num_envs], bool

    # float tensors
    distance_to_target: torch.Tensor             # [num_envs], float
    target_size: torch.Tensor                    # [num_envs], float
    remaining_target_distances: torch.Tensor     # [num_envs, num_targets], float
    target_pos: torch.Tensor                     # [num_envs, num_targets, 3], float

    def __init__(self, cfg: TaskEntityCfg):
        super().__init__(cfg)

        self.task_cfg = self.cfg.task_cfg
        self.num_targets = len(self.task_cfg.targets)

class SequentialTaskLogic(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        asset: TaskEntity = env.scene[_UNIVERSAL_ENTITY_NAME]
        asset_cfg: SceneEntityCfg = cfg.params['asset_cfg']

        self.asset = asset

        asset.end_effector_site_id = asset_cfg.site_ids[0]
        asset.target_pos_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.long, device=env.device)
        asset.target_size_ids = torch.tensor(asset_cfg.geom_ids, dtype=torch.long, device=env.device)
        asset.env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        asset.current_target_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        asset.completed_target_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        asset.steps_inside_target = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

        is_button = [isinstance(t, ButtonTargetConfig) for t in asset.task_cfg.targets]
        asset.target_is_button = torch.tensor(is_button, dtype=torch.bool, device=env.device)
        asset.target_min_touch_force = torch.tensor(
            [getattr(t, "min_touch_force", 0.0) for t in asset.task_cfg.targets],
            dtype=torch.float32,
            device=env.device,
        )

        dwell_steps = []
        for t, button in zip(asset.task_cfg.targets, is_button):
            steps = math.ceil(t.dwell_duration / env.step_dt)
            dwell_steps.append(max(1, steps) if button else steps)
        asset.target_dwell_steps = torch.tensor(dwell_steps, dtype=torch.int32, device=env.device)
        asset.current_target_dwell_steps = asset.target_dwell_steps[asset.current_target_id]

        sensor_adr = []
        for target_id, button in enumerate(is_button):
            if not button:
                sensor_adr.append(-1)
                continue
            sensor = env.sim.mj_model.sensor(f"{_UNIVERSAL_ENTITY_NAME}/sensor_target_{target_id}")
            sensor_adr.append(int(sensor.adr[0]))
        asset.target_sensor_adr = torch.tensor(sensor_adr, dtype=torch.long, device=env.device)

        asset.target_pos = asset.data.body_com_pos_w[asset.env_ids[:, None], asset.target_pos_ids]
        asset.inside_target, asset.distance_to_target, asset.target_size = self._inside_target()
        asset.remaining_target_distances = torch.zeros((env.num_envs, asset.num_targets), dtype=torch.float32, device=env.device)

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

    def _inside_target(self) -> torch.Tensor:
        asset = self.asset

        current_target_id = asset.current_target_id
        target_size_id = asset.target_size_ids[current_target_id]

        ee_pos = asset.data.site_pos_w[asset.env_ids, asset.end_effector_site_id]
        target_pos = asset.target_pos[asset.env_ids, current_target_id]
        target_size = asset.data.model.geom_size[asset.env_ids, target_size_id, 0]

        distance_to_target = torch.linalg.vector_norm(ee_pos - target_pos, dim=-1, keepdim=True).reshape(-1)
        inside_target = distance_to_target < target_size

        is_button = asset.target_is_button[current_target_id]
        if is_button.any():
            sensor_adr = asset.target_sensor_adr[current_target_id].clamp(min=0)
            touch_force = asset.data.data.sensordata[asset.env_ids, sensor_adr]
            min_force = asset.target_min_touch_force[current_target_id]
            button_pressed = touch_force >= min_force
            inside_target = torch.where(is_button, button_pressed, inside_target)

        return inside_target, distance_to_target, target_size

@dataclass
class TargetDomainRandomization:
    body_pos: dict[str, Ranges] = field(default_factory=dict)
    geom_size: dict[str, Ranges] = field(default_factory=dict)
    geom_rgb: dict[str, Ranges] = field(default_factory=dict)

@myo_register_task("universal")
class UniversalTaskComponent(myo.MyoComponent):
    def __init__(self, cfg: UniversalTaskConfig):
        self.cfg = cfg

        if len(self.cfg.targets) == 0:
            raise ValueError("No tragets defined")

    def modify_env_cfg(self, cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
        _, target_dr, model_names = self._create_model()

        # Collidable button boxes form BOX<->MESH pairs with the hand meshes,
        # which mujoco_warp rejects while MULTICCD is enabled (the hand meshes
        # carry a non-zero margin). Disable MULTICCD when any button is present.
        if any(isinstance(t, ButtonTargetConfig) for t in self.cfg.targets):
            if "multiccd" not in cfg.sim.mujoco.disableflags:
                cfg.sim.mujoco.disableflags = (*cfg.sim.mujoco.disableflags, "multiccd")

        cfg.scene.entities.update({
            _UNIVERSAL_ENTITY_NAME: TaskEntityCfg(
                spec_fn=lambda: self._create_model()[0],
                articulation=EntityArticulationInfoCfg(
                    actuators=(
                        XmlActuatorCfg(
                            target_names_expr=tuple(model_names.tendon_names),
                            transmission_type=TransmissionType.TENDON,
                        ),
                    )
                ),
                task_cfg=self.cfg
            )
        })

        num_targets = len(self.cfg.targets)

        entity_cfg = SceneEntityCfg(
            _UNIVERSAL_ENTITY_NAME,
            joint_names=model_names.independent_joint_names,
            site_names=[self.cfg.reach.end_effector_site]
        )
        target_entity_cfg = SceneEntityCfg(
            _UNIVERSAL_ENTITY_NAME,
            body_names=[f"body_target_{i}" for i in range(num_targets)],
            geom_names=[f"geom_target_{i}" for i in range(num_targets)],
            site_names=[self.cfg.reach.end_effector_site]
        )

        # Expose the resolved scene-entity configs so subclasses (e.g. the numpad
        # task) can overwrite individual sequence-dependent terms after calling
        # super().modify_env_cfg() without re-deriving them. Purely informational,
        # does not change universal behaviour.
        self._entity_cfg = entity_cfg
        self._target_entity_cfg = target_entity_cfg
        self._num_targets = num_targets

        _obs_terms_complete = {
            "time": ObservationTermCfg(func=myo.time),
            "qpos": ObservationTermCfg(func=myo.joint_qpos, params={"asset_cfg": entity_cfg}),
            "qvel": ObservationTermCfg(func=myo.joint_qvel, params={"asset_cfg": entity_cfg}),
            "qacc": ObservationTermCfg(func=myo.joint_qacc, params={"asset_cfg": entity_cfg}),
            "act": ObservationTermCfg(func=myo.act, params={"asset_cfg": entity_cfg}),
            "ee_pos": ObservationTermCfg(func=myo.site_pos, params={"asset_cfg": entity_cfg}),
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
                entity_name=_UNIVERSAL_ENTITY_NAME,
                actuator_names=model_names.tendon_names
            ),
        })

        cfg.rewards.update({
            "distance": RewardTermCfg(
                func=_sequential_distance_reward,
                params={
                    "asset_cfg": entity_cfg,
                    "exponential_distance_reward": self.cfg.reward.distance_exponential,
                    "distance_metric": self.cfg.reward.distance_metric
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

            # Task logic update: check if current target is reached and update to the next target accordingly
            "task_logic_update": EventTermCfg(
                func=SequentialTaskLogic,
                params={"asset_cfg": target_entity_cfg},
                mode="step",
            )
        })

        if len(target_dr.body_pos) > 0:
            cfg.events["target_pos_dr"] = EventTermCfg(
                mode="reset",
                func=dr.body_pos,
                params={
                    "asset_cfg": SceneEntityCfg(_UNIVERSAL_ENTITY_NAME, body_names=tuple(target_dr.body_pos.keys())),
                    "ranges": target_dr.body_pos,
                    "operation": "abs",
                },
            )

        if len(target_dr.geom_size) > 0:
            cfg.events["target_size_dr"] = EventTermCfg(
                mode="reset",
                func=dr.geom_size,
                params={
                    "asset_cfg": SceneEntityCfg(_UNIVERSAL_ENTITY_NAME, geom_names=tuple(target_dr.geom_size.keys())),
                    "ranges": target_dr.geom_size,
                    "operation": "abs",
                },
            )

        if len(target_dr.geom_rgb) > 0:
            cfg.events["target_rgba_dr"] = EventTermCfg(
                mode="reset",
                func=dr.geom_rgba,
                params={
                    "asset_cfg": SceneEntityCfg(_UNIVERSAL_ENTITY_NAME, geom_names=tuple(target_dr.geom_rgb.keys())),
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
                reduce="last"
            )

        cfg.metrics.update({
            "inside_target": MetricsTermCfg(
                func=lambda env: env.scene[_UNIVERSAL_ENTITY_NAME].inside_target
            ),
            "total_initial_distance": MetricsTermCfg(
                func=lambda env: env.scene[_UNIVERSAL_ENTITY_NAME].remaining_target_distances[:, 0]
            ),
            "target_size": MetricsTermCfg(
                func=lambda env: env.scene[_UNIVERSAL_ENTITY_NAME].target_size
            ),
            "completed_target_count": MetricsTermCfg(
                func=lambda env: env.scene[_UNIVERSAL_ENTITY_NAME].completed_target_count,
                reduce="last"
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
        
        spec, dr = self._modify_spec(spec, target_origin)

        return spec, dr, model_names

    def _modify_spec(self, spec: mujoco.MjSpec, target_origin: np.ndarray) -> tuple[mujoco.MjSpec, TargetDomainRandomization]:
        dr = TargetDomainRandomization()

        for target_id, target in enumerate(self.cfg.targets):
            if isinstance(target, ButtonTargetConfig):
                self._add_button_target(spec, dr, target_id, target, target_origin)
            elif isinstance(target, PointingTargetConfig):
                self._add_pointing_target(spec, dr, target_id, target, target_origin)
            else:
                raise ValueError(f"Unsupported target type: {type(target).__name__}")

        return spec, dr

    def _resolve_body_pos(
        self,
        dr: TargetDomainRandomization,
        target_body_name: str,
        position,
        target_origin: np.ndarray,
    ) -> np.ndarray:
        body_pos = np.array(target_origin, copy=True)
        if position.value is not None:
            body_pos += np.array(position.value[:3])
        else:
            dr.body_pos[target_body_name] = {
                dim: (
                    target_origin[dim] + position.min[dim],
                    target_origin[dim] + position.max[dim],
                ) for dim in range(3)
            }
        return body_pos

    def _resolve_rgba(
        self,
        dr: TargetDomainRandomization,
        target_geom_name: str,
        rgb,
    ) -> np.ndarray:
        geom_rgba = np.ones(4)
        if rgb.value is not None:
            geom_rgba[:3] = np.array(rgb.value)
        else:
            dr.geom_rgb[target_geom_name] = {
                dim: (rgb.min[dim], rgb.max[dim]) for dim in range(3)
            }
        return geom_rgba

    def _add_pointing_target(
        self,
        spec: mujoco.MjSpec,
        dr: TargetDomainRandomization,
        target_id: int,
        target: PointingTargetConfig,
        target_origin: np.ndarray,
    ) -> None:
        target_body_name = f"body_target_{target_id}"
        target_geom_name = f"geom_target_{target_id}"

        body_pos = self._resolve_body_pos(dr, target_body_name, target.position, target_origin)
        target_body = spec.worldbody.add_body(name=target_body_name, pos=body_pos)

        geom_size = np.ones(3)
        if target.size.value is not None:
            geom_size *= target.size.value
        else:
            dr.geom_size[target_geom_name] = {
                dim: (target.size.min, target.size.max) for dim in range(3)
            }

        geom_rgba = self._resolve_rgba(dr, target_geom_name, target.rgb)

        target_body.add_geom(
            name=target_geom_name,
            type=mujoco.mjtGeom.mjGEOM_SPHERE if target.shape == 'sphere' else mujoco.mjtGeom.mjGEOM_BOX,
            pos=np.zeros(3),
            size=geom_size,
            rgba=geom_rgba,
        )

    def _add_button_target(
        self,
        spec: mujoco.MjSpec,
        dr: TargetDomainRandomization,
        target_id: int,
        target: ButtonTargetConfig,
        target_origin: np.ndarray,
    ) -> None:
        target_body_name = f"body_target_{target_id}"
        target_geom_name = f"geom_target_{target_id}"
        target_site_name = f"site_target_{target_id}"
        target_sensor_name = f"sensor_target_{target_id}"

        body_pos = self._resolve_body_pos(dr, target_body_name, target.position, target_origin)
        target_body = spec.worldbody.add_body(
            name=target_body_name,
            pos=body_pos,
            euler=np.array(target.euler),
        )

        geom_size = np.ones(3)
        if target.size.value is not None:
            geom_size = np.array(target.size.value[:3])
        else:
            dr.geom_size[target_geom_name] = {
                dim: (target.size.min[dim], target.size.max[dim]) for dim in range(3)
            }

        geom_rgba = self._resolve_rgba(dr, target_geom_name, target.rgb)

        # Collidable box the fingertip physically pushes against.
        target_body.add_geom(
            name=target_geom_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=np.zeros(3),
            size=geom_size,
            rgba=geom_rgba,
            margin=target.geom_margin,
            contype=1,
            conaffinity=1,
        )

        # Touch-sensing zone sitting on the button surface.
        site_pos = np.array(target.site_pos.value[:3]) if target.site_pos.value is not None else np.zeros(3)
        site_size = np.array(target.site_size.value[:3]) if target.site_size.value is not None else geom_size
        target_body.add_site(
            name=target_site_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=site_pos,
            size=site_size,
        )
        spec.add_sensor(
            name=target_sensor_name,
            type=mujoco.mjtSensor.mjSENS_TOUCH,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=target_site_name,
        )
