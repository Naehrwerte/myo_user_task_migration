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
    BUTTON_HOUSING_PARTS,
    ButtonTargetConfig,
    Color_mode,
    PointingTargetConfig,
    SequentialTaskConfig,
)
from .sequential_task_entity import TargetDomainRandomization, TaskEntityCfg
from .sequential_task_logic import SequentialTaskLogic
from .target_coloring import TargetStateOverlay, color_targets_by_state


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
            "button_press_fraction": ObservationTermCfg(func=mdp.button_press_fraction, params={"asset_cfg": entity_cfg}),
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
                spec_fn=lambda: self._create_model(play)[0],
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

        color_mode = self.cfg.target_state_color_mode
        if play and color_mode == Color_mode.RECOLOR:
            # Variant 1: recolor target geoms in the model (results in viewer lag because of reloading model).
            cfg.events["target_state_coloring"] = EventTermCfg(
                func=color_targets_by_state,
                params={"asset_cfg": entity_cfg},
                mode="step",
            )
        elif play and color_mode == Color_mode.OVERLAY:
            # Variant 2: overlay colored debug geometry (no model change, no lag).
            cfg.events["target_state_overlay"] = EventTermCfg(
                func=TargetStateOverlay,
                params={"asset_cfg": entity_cfg},
                mode="reset",
            )

    def _create_model(self, play: bool = False) -> tuple[mujoco.MjSpec, TargetDomainRandomization, myo.MyoModelNames]:
        spec = mujoco.MjSpec.from_file(self.cfg.model_path)

        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        model_names = myo.myo_get_model_names(model)

        reference_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.cfg.reach.reference_site)
        if reference_site < 0:
            raise ValueError(f"Unknown reference site: {self.cfg.reach.reference_site}")
        target_origin = data.site_xpos[reference_site] + np.array(self.cfg.reach.reference_offset.value)

        spec, dr = self._modify_spec(spec, target_origin)

        if play and self.cfg.show_button_labels:
            self._add_button_labels(spec)

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
            euler=np.array(target.euler.value),
        )

        geom_size = np.ones(3)
        if target.size.value is not None:
            geom_size = np.array(target.size.value[:3])
        else:
            dr.geom_size[target_geom_name] = {
                dim: (target.size.min[dim], target.size.max[dim]) for dim in range(3)
            }

        geom_rgba = self._resolve_rgba(dr, target_geom_name, target.rgb)

        # Touch-sensing zone sitting on the button surface.
        site_pos = np.array(target.site_pos.value[:3]) if target.site_pos.value is not None else np.zeros(3)
        site_size = np.array(target.site_size.value[:3]) if target.site_size.value is not None else geom_size

        if self.cfg.button_press.enabled:
            # geom_target_i stays as invisible, non-colliding reference box, so target
            # position (body COM) and target size observations are unchanged; the
            # visible, colliding housing is a frame around the cap.
            target_body.add_geom(
                name=target_geom_name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=geom_size,
                rgba=[*geom_rgba[:3], 0.0],
                contype=0,
                conaffinity=0,
                group=3,
            )
            self._add_button_housing(target_body, target_id, target, site_size, geom_rgba)
            self._add_button_cap(spec, target_body, target_id, target, site_size)
            return

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

    _HOUSING_PLATE_HALF_THICKNESS = 0.001

    @staticmethod
    def _housing_half(target: ButtonTargetConfig) -> list[float]:
        # With randomized sizes use the largest one so the cap always fits.
        size = target.size
        return list(size.value if size.value is not None else size.max)

    def _add_button_housing(
        self,
        target_body: mujoco.MjsBody,
        target_id: int,
        target: ButtonTargetConfig,
        cap_size: np.ndarray,
        rgba: np.ndarray,
    ) -> None:
        """Housing frame: back plate plus four walls around a hole the cap moves through."""
        hx, hy, hz = self._housing_half(target)
        ox = cap_size[0] + self.cfg.button_press.hole_clearance
        oy = cap_size[1] + self.cfg.button_press.hole_clearance
        p = self._HOUSING_PLATE_HALF_THICKNESS
        if ox >= hx or oy >= hy:
            raise ValueError(
                f"Button {target_id}: cap ({cap_size[:2]}) plus hole clearance does not fit "
                f"into the housing ({hx}, {hy})"
            )

        parts = [  # (pos, half extents) in the button frame
            ([0.0, 0.0, -hz + p], [hx, hy, p]),
            ([(hx + ox) / 2, 0.0, 0.0], [(hx - ox) / 2, hy, hz]),
            ([-(hx + ox) / 2, 0.0, 0.0], [(hx - ox) / 2, hy, hz]),
            ([0.0, (hy + oy) / 2, 0.0], [ox, (hy - oy) / 2, hz]),
            ([0.0, -(hy + oy) / 2, 0.0], [ox, (hy - oy) / 2, hz]),
        ]
        assert len(parts) == BUTTON_HOUSING_PARTS
        for k, (pos, half) in enumerate(parts):
            target_body.add_geom(
                name=f"geom_housing_{target_id}_{k}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=pos,
                size=half,
                rgba=rgba,
                margin=target.geom_margin,
                contype=1,
                conaffinity=1,
                density=0.0,  # the reference box carries the mass (keeps the body COM)
            )

    def _add_button_cap(
        self,
        spec: mujoco.MjSpec,
        target_body: mujoco.MjsBody,
        target_id: int,
        target: ButtonTargetConfig,
        cap_size: np.ndarray,
    ) -> None:
        """Movable cap on a spring-loaded slide joint along the button normal (local z).
        """
        press = self.cfg.button_press

        housing_top = self._housing_half(target)[2]
        cap_top_rest = housing_top + press.pressed_protrusion + press.max_press_depth
        cap_bottom_rest = cap_top_rest - 2 * press.cap_half_height
        cap_bottom_pressed = cap_bottom_rest - press.max_press_depth
        if cap_bottom_rest >= housing_top:
            raise ValueError(
                "button_press.cap_half_height too small: the cap has to reach into the "
                f"housing at rest (2 * cap_half_height > travel + pressed_protrusion)"
            )
        if cap_bottom_pressed <= -housing_top + 2 * self._HOUSING_PLATE_HALF_THICKNESS:
            raise ValueError(
                "button_press.cap_half_height too large: the pressed cap would reach "
                "through the back plate of the housing"
            )

        cap_body = target_body.add_body(
            name=f"body_button_cap_{target_id}",
            pos=[0.0, 0.0, cap_top_rest - press.cap_half_height],
        )
        cap_body.add_joint(
            name=f"joint_button_{target_id}",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0.0, 0.0, 1.0],
            range=[-press.max_press_depth, 0.0],
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            stiffness=press.stiffness,
            springref=press.preload,
            damping=press.damping,
            # Stiff end stops (default limits let the cap sink millimeters past them).
            solref_limit=[0.004, 1.0],
            solimp_limit=[0.95, 0.99, 0.001, 0.5, 2.0],
        )

        half_extents = [cap_size[0], cap_size[1], press.cap_half_height]
        cap_body.add_geom(
            name=f"geom_button_cap_{target_id}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half_extents,
            rgba=[*press.cap_rgb.value, 1.0],
            mass=press.cap_mass,
            margin=target.geom_margin,
            contype=1,
            conaffinity=1,
        )

        # Slightly larger than the cap so contacts on its surface lie inside the site.
        site_name = f"site_target_{target_id}"
        cap_body.add_site(
            name=site_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[e + 0.002 for e in half_extents],
            rgba=[0.0, 0.0, 0.0, 0.0],
            # viser draws every site opaque (fully transparent ones in grey), which would
            # cover the cap; group 3 is hidden by default in viser and MuJoCo.
            group=3,
        )
        spec.add_sensor(
            name=f"sensor_target_{target_id}",
            type=mujoco.mjtSensor.mjSENS_TOUCH,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=site_name,
        )

    # Seven-segment layout: segment -> (center_u, center_v, horizontal) in units of
    # (digit width, digit height), u pointing right and v up.
    _SEGMENTS = {
        "a": (0.0, 0.5, True), "b": (0.5, 0.25, False), "c": (0.5, -0.25, False),
        "d": (0.0, -0.5, True), "e": (-0.5, -0.25, False), "f": (-0.5, 0.25, False),
        "g": (0.0, 0.0, True),
    }
    _DIGIT_SEGMENTS = {
        "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg",
        "5": "acdfg", "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg",
    }

    def _add_button_labels(self, spec: mujoco.MjSpec) -> None:
        """Draw every button ``label`` as non-colliding seven-segment boxes on its surface.
        """
        press = self.cfg.button_press
        # spec.body() misses bodies added after the spec was compiled once (see _create_model).
        bodies = {b.name: b for b in spec.bodies}

        for target_id, target in enumerate(self.cfg.targets):
            if not isinstance(target, ButtonTargetConfig) or not target.label:
                continue

            size = target.size
            housing_half = size.value if size.value is not None else size.min
            # Cap and touch site both use site_size, see _add_button_target.
            surface_half = target.site_size.value or housing_half
            if press.enabled:
                body = bodies[f"body_button_cap_{target_id}"]
                center = [0.0, 0.0]
                surface_z = press.cap_half_height
            else:
                body = bodies[f"body_target_{target_id}"]
                site_pos = target.site_pos.value or [0.0, 0.0, 0.0]
                center = site_pos[:2]
                surface_z = site_pos[2] + surface_half[2]

            # Digit geometry relative to the (smaller) surface side.
            height = 1.2 * min(surface_half[0], surface_half[1])
            width = 0.55 * height
            stroke = 0.12 * height
            gap = 0.35 * width
            n = len(target.label)
            label_width = n * width + (n - 1) * gap
            # Thin raised legend, starting slightly inside the surface.
            raise_ = 0.0008
            half_thickness = (raise_ + 0.0005) / 2
            z = surface_z + raise_ - half_thickness

            for char_idx, char in enumerate(target.label):
                u0 = -label_width / 2 + width / 2 + char_idx * (width + gap)
                for seg in self._DIGIT_SEGMENTS[char]:
                    cu, cv, horizontal = self._SEGMENTS[seg]
                    u, v = u0 + cu * width, cv * height
                    half_u = (width + stroke) / 2 if horizontal else stroke / 2
                    half_v = stroke / 2 if horizontal else (height / 2 + stroke) / 2
                    # Surface frame: up = local +x, right = local -y.
                    body.add_geom(
                        name=f"geom_label_{target_id}_{char_idx}{seg}",
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        pos=[center[0] + v, center[1] - u, z],
                        size=[half_v, half_u, half_thickness],
                        rgba=[0.05, 0.05, 0.05, 1.0],
                        contype=0,
                        conaffinity=0,
                        density=0.0,
                    )
