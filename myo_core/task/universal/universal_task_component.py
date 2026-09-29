from __future__ import annotations

import math
import mujoco
import numpy as np
import torch

from functools import partial
from dataclasses import dataclass, field

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import terminations as mdp_terminations
from mjlab.envs.mdp import events as event_fns
from mjlab.managers import EventTermCfg, MetricsTermCfg, RewardTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType

import myo_core.common as myo
from .universal_task_config import UniversalTaskConfig, ShapeTargetConfig, ChoiceReactionConfig, Vec3Range
from .universal_task_entity import (
    UniversalTaskEntityCfg,
    UniversalTaskLogic,
    target_pos,
    target_size,
    target_rgb,
    trial_progress,
    dwell_fraction,
    distance_reward,
    inside_target_reward,
    distractor_penalty,
    trial_bonus,
    trial_successfully_completed,
    episode_successfully_completed
)
from ..task_registry import myo_register_task

_UNIVERSAL_ENTITY_NAME = "universal_robot"


@dataclass
class TaskEntitySpec:
    spec: str
    init_state: EntityCfg.InitialStateCfg = field(default_factory=EntityCfg.InitialStateCfg)


@myo_register_task("universal")
class UniversalTaskComponent(myo.MyoComponent):
    def __init__(self, cfg: UniversalTaskConfig):
        self.cfg = cfg

        if len(self.cfg.targets) == 0:
            raise ValueError("No tragets defined")

        (
            self.spec,
            self.entity_specs,
            self.object_offset,
            self.model_names
        ) = self._create_model()

    def modify_env_cfg(self, cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
        cfg.scene.entities.update({
            _UNIVERSAL_ENTITY_NAME: UniversalTaskEntityCfg(
                spec_fn=lambda: myo.mjspec_from_string(self.spec),
                articulation=EntityArticulationInfoCfg(
                    actuators=(
                        XmlActuatorCfg(
                            target_names_expr=tuple(self.model_names.tendon_names),
                            transmission_type=TransmissionType.TENDON,
                        ),
                    )
                ),
                task_cfg=self.cfg
            ),
        })

        for name, entity in self.entity_specs.items():
            cfg.scene.entities[name] = EntityCfg(
                spec_fn=partial(myo.mjspec_from_string, entity.spec),
                init_state=entity.init_state
            )

        entity_cfg = SceneEntityCfg(
            _UNIVERSAL_ENTITY_NAME,
            joint_names=self.model_names.independent_joint_names,
            site_names=[self.cfg.reach.end_effector_site]
        )

        _avail_obs_terms = {
            "time": ObservationTermCfg(func=myo.time),
            "qpos": ObservationTermCfg(func=myo.joint_qpos, params={"asset_cfg": entity_cfg}),
            "qvel": ObservationTermCfg(func=myo.joint_qvel, params={"asset_cfg": entity_cfg}),
            "qacc": ObservationTermCfg(func=myo.joint_qacc, params={"asset_cfg": entity_cfg}),
            "act": ObservationTermCfg(func=myo.act, params={"asset_cfg": entity_cfg}),
            "ee_pos": ObservationTermCfg(func=myo.site_pos, params={"asset_cfg": entity_cfg}),
            "target_pos": ObservationTermCfg(func=target_pos, params={"asset_cfg": entity_cfg}),
            "target_color": ObservationTermCfg(func=target_rgb, params={"asset_cfg": entity_cfg}),
            "target_size": ObservationTermCfg(func=target_size, params={"asset_cfg": entity_cfg}),
            "trial_progress": ObservationTermCfg(func=trial_progress, params={"asset_cfg": entity_cfg}),
            "dwell_fraction": ObservationTermCfg(func=dwell_fraction, params={"asset_cfg": entity_cfg}),
        }

        def _select_obs_terms(keys: list[str]) -> dict[str, ObservationTermCfg]:
            missing = [key for key in keys if key not in _avail_obs_terms]
            if missing:
                raise KeyError(f"Unknown observation keys: {missing}")

            return {key: _avail_obs_terms[key] for key in keys}

        cfg.observations.update({
            "agent_state": ObservationGroupCfg(
                terms=_select_obs_terms(self.cfg.agent_state_keys),
            ),
            "task_state": ObservationGroupCfg(
                terms=_select_obs_terms(self.cfg.task_state_keys),
            ),
            "task_query": ObservationGroupCfg(
                terms=_select_obs_terms(self.cfg.task_query_keys),
            ),
        })

        cfg.actions.update({
            "muscles": myo.MyoMuscleActivationActionCfg(
                entity_name=_UNIVERSAL_ENTITY_NAME,
                actuator_names=self.model_names.tendon_names
            ),
        })

        cfg.rewards.update({
            "distance": RewardTermCfg(
                func=distance_reward,
                params={
                    "asset_cfg": entity_cfg,
                    "exponential_distance_reward": self.cfg.reward.distance_exponential,
                    "distance_metric": self.cfg.reward.distance_metric
                },
                weight=self.cfg.reward.weights.get("distance", 0.0),
            ),
            "dc_effort": RewardTermCfg(
                func=myo.dc_effort,
                params={"asset_cfg": entity_cfg},
                weight=self.cfg.reward.weights.get("dc_effort", 0.0),
            ),
            "jac_effort": RewardTermCfg(
                func=myo.jac_effort,
                params={"asset_cfg": entity_cfg},
                weight=self.cfg.reward.weights.get("jac_effort", 0.0),
            ),
            "inside_target_reward": RewardTermCfg(
                func=inside_target_reward,
                params={"asset_cfg": entity_cfg},
                weight=self.cfg.reward.weights.get("inside_target_reward", 0.0),
            ),
            "distractor_penalty": RewardTermCfg(
                func=distractor_penalty,
                params={"asset_cfg": entity_cfg},
                weight=self.cfg.reward.weights.get("distractor_penalty", 0.0),
            ),
            "trial_bonus": RewardTermCfg(
                func=trial_bonus,
                params={"asset_cfg": entity_cfg},
                weight=self.cfg.reward.weights.get("trial_bonus", 0.0),
            ),
            "done": RewardTermCfg(
                func=episode_successfully_completed, 
                params={"asset_cfg": entity_cfg},
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
                func=UniversalTaskLogic,
                params={"asset_name": _UNIVERSAL_ENTITY_NAME, "object_offset": self.object_offset},
                mode="step",
            )
        })

        cfg.terminations.update({
            "time_out": TerminationTermCfg(
                func=mdp_terminations.time_out,
                time_out=True,
            ),
            "episode_success": TerminationTermCfg(
                func=episode_successfully_completed,
                params={"asset_cfg": entity_cfg},
            ),
        })

        for i in range(self.cfg.max_trials):
            cfg.metrics[f"trial_{i}_success"] = MetricsTermCfg(
                func=trial_successfully_completed,
                params={"asset_cfg": entity_cfg, "trial_id": i},
                reduce="last"
            )

        cfg.metrics.update({
            "distance_target": MetricsTermCfg(
                func=lambda env: env.scene[_UNIVERSAL_ENTITY_NAME].distance_to_target,
            ),
            "inside_target": MetricsTermCfg(
                func=lambda env: env.scene[_UNIVERSAL_ENTITY_NAME].inside_target,
            ),
            "inside_distractor": MetricsTermCfg(
                func=lambda env: env.scene[_UNIVERSAL_ENTITY_NAME].inside_distractor,
            ),
            "completed_target_count": MetricsTermCfg(
                func=lambda env: env.scene[_UNIVERSAL_ENTITY_NAME].completed_target_count,
                reduce="last"
            ),
        })

    def _create_model(self) -> tuple[str, dict[str, TaskEntitySpec], torch.Tensor, myo.MyoModelNames]:
        spec = mujoco.MjSpec.from_file(self.cfg.model_path)

        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        model_names = myo.myo_get_model_names(model)

        reference_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.cfg.reach.reference_site)
        if reference_site < 0:
            raise ValueError(f"Unknown reference site: {self.cfg.reach.reference_site}")
        object_offset = data.site_xpos[reference_site] + np.array(self.cfg.reach.reference_offset)
        
        entity_specs = self._create_entity_specs(model, data, object_offset)
        spec_str = myo.mjspec_to_string(spec, self.cfg.model_path)

        return spec_str, entity_specs, object_offset, model_names

    def _create_entity_specs(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        object_offset: np.ndarray,
    ) -> dict[str, TaskEntitySpec]:
        entity_specs: dict[str, TaskEntitySpec] = {}

        if self.cfg.choice_reaction.enabled:
            entity = self._create_choice_reaction_screen(model, data, object_offset)
            entity_specs['choice_reaction'] = entity

        for prefix, configs in (
            ("target", self.cfg.targets),
            ("distractor", self.cfg.distractors),
        ):
            for idx, config in enumerate(configs):
                if not isinstance(config, ShapeTargetConfig):
                    raise ValueError(
                        f"Only ShapeTargetConfig is implemented; "
                        f"got {type(config).__name__} for {prefix}_{idx}"
                    )

                name = f"{prefix}_{idx}"
                entity = self._create_shape_target_spec(object_offset, config)

                entity_specs[name] = entity

        return entity_specs
    
    def _create_choice_reaction_screen(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        object_offset: np.ndarray,
    ) -> TaskEntitySpec:
        config = self.cfg.choice_reaction

        camera_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            "fixed-eye-cr",
        )
        if camera_id < 0:
            raise ValueError("Camera 'fixed-eye' was not found.")

        camera_pos = data.cam_xpos[camera_id].copy()
        camera_mat = data.cam_xmat[camera_id].reshape(3, 3).copy()

        camera_quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(camera_quat, camera_mat.ravel())

        distance = float(config.distance)
        sector_x = float(config.sector_x)
        sector_y = float(config.sector_y)

        # Camera-local coordinates:
        #   +X = rendered image right
        #   +Y = rendered image up
        #   -Z = in front of the camera
        target_local_positions = {
            "tl": np.array([-sector_x, +sector_y, -distance]),
            "tr": np.array([+sector_x, +sector_y, -distance]),
            "br": np.array([+sector_x, -sector_y, -distance]),
            "bl": np.array([-sector_x, -sector_y, -distance]),
        }

        offset_extent = np.array([0.025, 0.025, 0.0])
        world_offset_extent = np.abs(camera_mat) @ offset_extent

        for idx, (name, local_pos) in enumerate(target_local_positions.items()):
            centroid = (camera_pos + camera_mat @ local_pos) - object_offset
            self.cfg.targets[idx].position = Vec3Range(
                #value=centroid.tolist(),
                min=(centroid - world_offset_extent).tolist(),
                max=(centroid + world_offset_extent).tolist()
            )

        # The cue panel can have a separate XY offset, but should normally lie
        # on the same camera-depth plane as the targets.
        #
        # Example cue_position: (0.0, 0.15, -distance)
        screen_local_pos = np.asarray(config.position, dtype=np.float64)

        # Optional convenience: allow (x, y) in config and force the shared depth.
        if screen_local_pos.shape == (2,):
            screen_local_pos = np.array(
                [screen_local_pos[0], screen_local_pos[1], -distance],
                dtype=np.float64,
            )

        screen_world_pos = camera_pos + camera_mat @ screen_local_pos

        # This spec only contains the visual cue panel.
        spec = mujoco.MjSpec()

        screen_body = spec.worldbody.add_body(
            name="body",
            mocap=True
        )

        screen_body.add_geom(
            name="geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=np.zeros(3),
            size=np.asarray(config.size, dtype=np.float64),
            rgba=np.array([0.5, 0.5, 0.5, 1.0]),  # overwritten at reset
            contype=0,
            conaffinity=0,
        )

        return TaskEntitySpec(
            spec=myo.mjspec_to_string(spec, self.cfg.model_path),
            init_state=EntityCfg.InitialStateCfg(
                pos=screen_world_pos,
                rot=camera_quat,
            )
        )

    def _create_shape_target_spec(
        self,
        object_offset: np.ndarray,
        config: ShapeTargetConfig
    ) -> TaskEntitySpec:
        spec = mujoco.MjSpec()

        body = spec.worldbody.add_body(
            name="body",
            mocap=True
        )

        geom_type = mujoco.mjtGeom.mjGEOM_SPHERE
        if config.shape == 'capsule':
            geom_type = mujoco.mjtGeom.mjGEOM_CAPSULE
        elif config.shape == 'cylinder':
            geom_type = mujoco.mjtGeom.mjGEOM_CYLINDER
        elif config.shape == 'box':
            geom_type = mujoco.mjtGeom.mjGEOM_BOX

        body.add_geom(
            name="geom",
            type=geom_type,
            pos=np.zeros(3),
            size=np.asarray([config.size.average(), config.size.average() * 2, config.size.average() * 4]),
            rgba=np.append(config.rgb.average(), 1.0),
            contype=0,
            conaffinity=0,
        )

        return TaskEntitySpec(
            spec=myo.mjspec_to_string(spec, self.cfg.model_path),
            init_state=EntityCfg.InitialStateCfg(
                pos=(object_offset + config.position.average())
            )
        )
