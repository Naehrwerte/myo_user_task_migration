"""Base component for sequential (target-by-target) tasks.

Self-contained copy of the legacy universal task component: it builds the model with
the configured targets, wires up observations/rewards/events/terminations and
metrics.


"""

from __future__ import annotations

from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr, events as event_fns
from mjlab.envs.mdp import terminations as mdp_terminations
from mjlab.managers import EventTermCfg, MetricsTermCfg, RewardTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
import mujoco
import numpy as np

import myo_core.common as myo
from . import sequential_task_mdp as mdp
from .sequential_task_config import (
    ButtonTargetConfig,
    PointingTargetConfig,
    SequentialTaskConfig,
)
from .sequential_task_entity import TargetDomainRandomization, TaskEntityCfg
from .sequential_task_logic import SequentialTaskLogic


class SequentialTaskComponent(myo.MyoComponent):
    entity_name: str = "sequential_robot"
    task_logic_cls: type[SequentialTaskLogic] = SequentialTaskLogic
    supported_target_types: tuple[type, ...] = (PointingTargetConfig, ButtonTargetConfig)

    def __init__(self, cfg: SequentialTaskConfig):
        self.cfg = cfg

        if len(self.cfg.targets) == 0:
            raise ValueError("No targets defined")

    @property
    def _num_phases(self) -> int:
        """legacy: num phases derived from num targets. Should be overwritten depending on logic"""
        return len(self.cfg.targets)

    def _site_names(self) -> list[str]:
        """Sites the task logic needs, in the order it resolves them."""
        return [self.cfg.reach.end_effector_site]

    def _observation_terms(self, entity_cfg: SceneEntityCfg) -> dict[str, ObservationTermCfg]:
        return {
            "time": ObservationTermCfg(func=myo.time),
            "qpos": ObservationTermCfg(func=myo.joint_qpos, params={"asset_cfg": entity_cfg}),
            "qvel": ObservationTermCfg(func=myo.joint_qvel, params={"asset_cfg": entity_cfg}),
            "qacc": ObservationTermCfg(func=myo.joint_qacc, params={"asset_cfg": entity_cfg}),
            "act": ObservationTermCfg(func=myo.act, params={"asset_cfg": entity_cfg}),
            "ee_pos": ObservationTermCfg(func=myo.site_pos, params={"asset_cfg": entity_cfg}),
            "target_pos": ObservationTermCfg(func=mdp.target_pos, params={"asset_cfg": entity_cfg}),
            "target_size": ObservationTermCfg(func=mdp.target_size, params={"asset_cfg": entity_cfg}),
            "phase_progress": ObservationTermCfg(func=mdp.phase_progress, params={"asset_cfg": entity_cfg}),
            "dwell_fraction": ObservationTermCfg(func=mdp.dwell_fraction, params={"asset_cfg": entity_cfg}),
        }

    def modify_env_cfg(self, cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
        _, target_dr, model_names = self._create_model()

        entity_name = self.entity_name

        # Collidable button boxes form BOX<->MESH pairs with the hand meshes,
        # which mujoco_warp rejects while MULTICCD is enabled (the hand meshes
        # carry a non-zero margin). Disable MULTICCD when any button is present.
        if any(isinstance(t, ButtonTargetConfig) for t in self.cfg.targets):
            if "multiccd" not in cfg.sim.mujoco.disableflags:
                cfg.sim.mujoco.disableflags = (*cfg.sim.mujoco.disableflags, "multiccd")

        cfg.scene.entities.update({
            entity_name: TaskEntityCfg(
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
        num_phases = self._num_phases
        site_names = self._site_names()

        entity_cfg = SceneEntityCfg(
            entity_name,
            joint_names=model_names.independent_joint_names,
            site_names=site_names
        )
        target_entity_cfg = SceneEntityCfg(
            entity_name,
            body_names=[f"body_target_{i}" for i in range(num_targets)],
            geom_names=[f"geom_target_{i}" for i in range(num_targets)],
            site_names=site_names
        )

        # Expose the resolved scene-entity configs so subclasses can add or
        # overwrite terms after calling super().modify_env_cfg() without
        # re-deriving them.
        self._entity_cfg = entity_cfg
        self._target_entity_cfg = target_entity_cfg
        self._num_targets = num_targets

        _obs_terms_complete = self._observation_terms(entity_cfg)

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
                entity_name=entity_name,
                actuator_names=model_names.tendon_names
            ),
        })

        cfg.rewards.update({
            "distance": RewardTermCfg(
                func=mdp.sequential_distance_reward,
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
                func=mdp.phase_bonus,
                params={"asset_cfg": entity_cfg},
                weight=self.cfg.reward.weights.get("phase_bonus", 0.0),
            ),
            "done": RewardTermCfg(
                func=mdp.phase_successfully_completed,
                params={"asset_cfg": entity_cfg, "phase_id": num_phases - 1},
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
                func=self.task_logic_cls,
                params={"asset_cfg": target_entity_cfg},
                mode="step",
            )
        })

        if len(target_dr.body_pos) > 0:
            cfg.events["target_pos_dr"] = EventTermCfg(
                mode="reset",
                func=dr.body_pos,
                params={
                    "asset_cfg": SceneEntityCfg(entity_name, body_names=tuple(target_dr.body_pos.keys())),
                    "ranges": target_dr.body_pos,
                    "operation": "abs",
                },
            )

        if len(target_dr.geom_size) > 0:
            cfg.events["target_size_dr"] = EventTermCfg(
                mode="reset",
                func=dr.geom_size,
                params={
                    "asset_cfg": SceneEntityCfg(entity_name, geom_names=tuple(target_dr.geom_size.keys())),
                    "ranges": target_dr.geom_size,
                    "operation": "abs",
                },
            )

        if len(target_dr.geom_rgb) > 0:
            cfg.events["target_rgba_dr"] = EventTermCfg(
                mode="reset",
                func=dr.geom_rgba,
                params={
                    "asset_cfg": SceneEntityCfg(entity_name, geom_names=tuple(target_dr.geom_rgb.keys())),
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
                func=mdp.phase_successfully_completed,
                params={"asset_cfg": entity_cfg, "phase_id": num_phases - 1},
            ),
        })

        for i in range(num_phases):
            cfg.metrics[f"phase_{i}_success"] = MetricsTermCfg(
                func=mdp.phase_successfully_completed,
                params={"asset_cfg": entity_cfg, "phase_id": i},
                reduce="last"
            )

        cfg.metrics.update({
            "inside_target": MetricsTermCfg(
                func=lambda env: env.scene[entity_name].inside_target
            ),
            "total_initial_distance": MetricsTermCfg(
                func=lambda env: env.scene[entity_name].remaining_target_distances[:, 0]
            ),
            "target_size": MetricsTermCfg(
                func=lambda env: env.scene[entity_name].target_size
            ),
            "completed_target_count": MetricsTermCfg(
                func=lambda env: env.scene[entity_name].completed_target_count,
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
            if not isinstance(target, self.supported_target_types):
                raise ValueError(f"Unsupported target type: {type(target).__name__}")

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
