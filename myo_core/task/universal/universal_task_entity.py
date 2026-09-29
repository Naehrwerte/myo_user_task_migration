from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import mujoco
import torch

from mjlab.entity import EntityCfg, Entity
from mjlab.envs import ManagerBasedRlEnv, mdp
from mjlab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from mjlab.managers.event_manager import requires_model_fields

from myo_core.common import BatchedDistributionSampler, BatchedTrajectorySampler
from .universal_task_config import (
    UniversalTaskConfig,
    ShapeTargetConfig,
    PointingTargetConfig,
    TrackingTargetConfig,
    TargetOrder
)

# MuJoCo geom type integer constants.
_GEOM_SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE.value
_GEOM_CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE.value
_GEOM_ELLIPSOID = mujoco.mjtGeom.mjGEOM_ELLIPSOID.value
_GEOM_CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER.value
_GEOM_BOX = mujoco.mjtGeom.mjGEOM_BOX.value

def target_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: UniversalTaskEntity = env.scene[asset_cfg.name]

    target_pos = asset.object_pos[asset.env_ids, asset.current_target_id]
    return target_pos

def target_size(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: UniversalTaskEntity = env.scene[asset_cfg.name]

    target_size = asset.object_size[asset.env_ids, asset.current_target_id]
    size_range = asset.target_size_range[asset.current_target_id]

    size_min = size_range[:, 0]
    size_max = size_range[:, 1]
    size_span = size_max - size_min

    normalized_size = torch.where(
        size_span > 0,
        (target_size - size_min) / size_span,
        torch.ones_like(target_size),
    )

    return normalized_size[:, None]

def target_rgb(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: UniversalTaskEntity = env.scene[asset_cfg.name]

    target_rgb = asset.object_rgb[asset.env_ids, asset.current_target_id]
    return target_rgb

def trial_progress(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: UniversalTaskEntity = env.scene[asset_cfg.name]

    if asset.max_trials <= 1:
        return torch.zeros((env.num_envs, 1), device=env.device)

    progress = (
        asset.completed_target_count.float()
        / float(asset.max_trials - 1)
    ).clamp_(0.0, 1.0)

    return -1.0 + 2.0 * progress[:, None]

def dwell_fraction(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: UniversalTaskEntity = env.scene[asset_cfg.name]

    dwell_fraction = (
        (asset.current_target_dwell_steps > 0)
        * asset.steps_inside_target
        / asset.current_target_dwell_steps.clip(min=1)
    )

    return dwell_fraction[:, None]

def inside_target_reward(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: UniversalTaskEntity = env.scene[asset_cfg.name]

    inside_target = asset.inside_target.to(torch.float32)

    return inside_target

def distance_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    exponential_distance_reward: bool,
    distance_metric: float
) -> torch.Tensor:
    asset: UniversalTaskEntity = env.scene[asset_cfg.name]

    distance = asset.distance_to_target.clamp_min(0.0)

    if not exponential_distance_reward:
        return -distance

    return torch.expm1(
        -distance_metric * distance
    ) / distance_metric

def distractor_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: UniversalTaskEntity = env.scene[asset_cfg.name]

    inside_distractor = asset.inside_distractor.to(torch.float32)

    return -inside_distractor

def trial_bonus(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: UniversalTaskEntity = env.scene[asset_cfg.name]

    if asset.max_trials <= 1:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    trial_bonus = asset.current_trial_completed.to(torch.float32)

    return trial_bonus

def trial_successfully_completed(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    trial_id: int = 0,
) -> torch.Tensor:
    asset: UniversalTaskEntity = env.scene[asset_cfg.name]

    if not 0 <= trial_id < asset.max_trials:
        raise ValueError(
            f"trial_id must be in [0, {asset.max_trials}), got {trial_id}."
        )

    completed_count = asset.completed_target_count

    currently_completing_trial = (
        (completed_count == trial_id) & asset.current_trial_completed
    )

    return (completed_count > trial_id) | currently_completing_trial

def episode_successfully_completed(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    asset: UniversalTaskEntity = env.scene[asset_cfg.name]
    return trial_successfully_completed(
        env,
        asset_cfg,
        asset.max_trials - 1
    )

@dataclass
class UniversalTaskEntityCfg(EntityCfg):
    task_cfg: UniversalTaskConfig | None = None

    def build(self) -> UniversalTaskEntity:
        return UniversalTaskEntity(self)

class UniversalTaskEntity(Entity):
    num_targets: int
    num_distractors: int
    num_objects: int
    max_trials: int

    task_cfg: UniversalTaskConfig

    # Static tensors
    end_effector_site_id: int
    env_ids: torch.Tensor                      # [E], long

    object_mocap_ids: torch.Tensor             # [O], long
    object_geom_ids: torch.Tensor              # [O], long

    target_size_range: torch.Tensor            # [T, 2], float
    target_dwell_steps: torch.Tensor           # [T], int

    tracking_object_ids: torch.Tensor          # [OT], long
    tracking_mocap_ids: torch.Tensor           # [OT], long

    pointing_object_ids: torch.Tensor          # [OP], long

    # Sequential-task state: target-local indexing only.
    step_count: torch.Tensor                   # [E], long
    current_target_id: torch.Tensor            # [E], long
    completed_target_count: torch.Tensor       # [E], int
    steps_inside_target: torch.Tensor          # [E], int
    current_target_dwell_steps: torch.Tensor   # [E], int
    current_trial_completed: torch.Tensor      # [E], bool
    target_order: torch.Tensor                 # [E, L], long

    # Physical-object state
    tracking_trj: torch.Tensor                 # [E, S, OT, 3], float
    object_pos: torch.Tensor                   # [E, O, 3], float
    object_size: torch.Tensor                  # [E, O], float
    object_rgb: torch.Tensor                   # [E, O, 3], float

    # Current interaction state
    distance_to_target: torch.Tensor           # [E], float
    inside_target: torch.Tensor                # [E], bool
    inside_distractor: torch.Tensor            # [E], bool
    inside_objects: torch.Tensor                # [E, O], bool

    # Choice reaction
    choice_reaction_screen_id: int | None

    def __init__(self, cfg: UniversalTaskEntityCfg):
        super().__init__(cfg)

        if cfg.task_cfg is None:
            raise ValueError("UniversalTaskEntityCfg.task_cfg must be provided.")

        self.task_cfg = cfg.task_cfg

        self.num_targets = len(self.task_cfg.targets)
        self.num_distractors = len(self.task_cfg.distractors)
        self.num_objects = self.num_targets + self.num_distractors
        self.max_trials = self.task_cfg.max_trials

        if self.num_targets <= 0:
            raise ValueError("UniversalTaskConfig requires at least one target.")

        if not isinstance(self.max_trials, int) or self.max_trials <= 0:
            raise ValueError(
                f"max_trials must be a positive int, got {self.max_trials!r}."
            )

        if self.task_cfg.choice_reaction.enabled and self.num_targets > 4:
            raise ValueError("ChoiceReaction only supports up to 4 targets")

@requires_model_fields("geom_size", "geom_rbound", "geom_aabb", "geom_rgba")
class UniversalTaskLogic(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset_name = cfg.params['asset_name']
        asset: UniversalTaskEntity = env.scene[asset_name]

        self.env = env
        self.asset = asset
        config = asset.task_cfg

        E = env.num_envs
        S = env.max_episode_length
        T = asset.num_targets
        D = asset.num_distractors
        O = asset.num_objects
        device = env.device

        # === Domain randomization ===

        targets: list[ShapeTargetConfig] = [*(config.targets)]
        objects: list[ShapeTargetConfig] = [
            *(config.targets),
            *(config.distractors)
        ]

        tracking_indices: list[int] = []
        pointing_indices: list[int] = []
        tracking_ranges = []
        pointing_ranges = []

        for object_id, obj in enumerate(objects):
            # Check Tracking first in case it inherits from PointingTargetConfig.
            if isinstance(obj, TrackingTargetConfig):
                tracking_indices.append(object_id)
                tracking_ranges.append(obj.position)

            elif isinstance(obj, PointingTargetConfig):
                pointing_indices.append(object_id)
                pointing_ranges.append(obj.position)

            else:
                raise TypeError(
                    f"Object {object_id} has unsupported config type "
                    f"{type(obj).__name__}. Expected TrackingTargetConfig or "
                    "PointingTargetConfig."
                )

        asset.tracking_object_ids = torch.tensor(tracking_indices, dtype=torch.long, device=device,)
        asset.pointing_object_ids = torch.tensor(pointing_indices, dtype=torch.long, device=device,)

        self.trj_sampler = BatchedTrajectorySampler.from_ranges(
            tracking_ranges,
            num_steps=S,
            max_frequency=5.0,
            device=device,
        ) if tracking_ranges else None

        self.pos_sampler = BatchedDistributionSampler.from_ranges(
            pointing_ranges,
            distribution=mdp.dr.uniform,
            device=device,
        ) if pointing_ranges else None

        self.size_sampler = BatchedDistributionSampler.from_ranges(
            [obj.size for obj in objects],
            distribution=mdp.dr.uniform,
            device=device
        )
        self.rgb_sampler = BatchedDistributionSampler.from_ranges(
            [obj.rgb for obj in objects],
            distribution=mdp.dr.uniform,
            device=device
        )
        self.object_offset = torch.tensor(cfg.params['object_offset'], dtype=torch.float32, device=device)

        self.choice_reaction_colors = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]
        ], dtype=torch.float32, device=device)

        # === Static scene data ===

        asset.end_effector_site_id = asset.find_sites(config.reach.end_effector_site)[0][0]
        asset.env_ids = torch.arange(E, dtype=torch.long, device=device)

        object_names = [
            *(f"target_{i}" for i in range(T)),
            *(f"distractor_{i}" for i in range(D)),
        ]

        def get_mujoco_id(type: Any, name: str):
            return mujoco.mj_name2id(env.sim.mj_model, type, name)

        body_ids = [get_mujoco_id(mujoco.mjtObj.mjOBJ_BODY, f'{obj}/body') for obj in object_names]
        geom_ids = [get_mujoco_id(mujoco.mjtObj.mjOBJ_GEOM, f'{obj}/geom') for obj in object_names]

        object_body_ids = torch.tensor(body_ids, dtype=torch.long, device=device)
        asset.object_mocap_ids = env.sim.model.body_mocapid[object_body_ids].to(dtype=torch.long, device=device)
        asset.object_geom_ids = torch.tensor(geom_ids, dtype=torch.long, device=device)

        asset.target_size_range = torch.tensor(
            [target.size.bounds() for target in targets],
            dtype=torch.float32,
            device=device,
        )

        asset.target_dwell_steps = torch.tensor(
            [
                math.ceil(t.dwell_duration / env.step_dt)
                if t.dwell_duration is not None else S
                for t in config.targets
            ],
            dtype=torch.int32,
            device=device
        )

        asset.tracking_mocap_ids = asset.object_mocap_ids[
            asset.tracking_object_ids
        ]

        # === Sequential-task state ===

        asset.step_count              = torch.zeros(E, dtype=torch.long,  device=device)
        asset.current_target_id       = torch.zeros(E, dtype=torch.long,  device=device)
        asset.completed_target_count  = torch.zeros(E, dtype=torch.int32, device=device)
        asset.steps_inside_target     = torch.zeros(E, dtype=torch.int32, device=device)
        asset.current_trial_completed = torch.zeros(E, dtype=torch.bool,  device=device)

        asset.current_target_dwell_steps = torch.full(
            (E,),
            fill_value=asset.target_dwell_steps[0].item(),
            dtype=torch.int32,
            device=device,
        )

        if config.sequence.order == TargetOrder.FIXED:
            order = torch.arange(asset.max_trials, dtype=torch.long, device=device) % asset.num_targets
            asset.target_order = order.unsqueeze(0).expand(E, -1)
        else:
            asset.target_order = torch.empty((E, asset.max_trials), dtype=torch.long, device=device)

        # === Episode-static object data ===

        OT = asset.tracking_object_ids.numel()

        asset.tracking_trj = torch.empty((E, S, OT, 3), dtype=torch.float32, device=device)
        asset.object_pos   = torch.empty((E, O, 3),     dtype=torch.float32, device=device)
        asset.object_size  = torch.empty((E, O),        dtype=torch.float32, device=device)
        asset.object_rgb   = torch.empty((E, O, 3),     dtype=torch.float32, device=device)

        # === Per-step physical interaction state ===

        asset.distance_to_target         = torch.empty(E,      dtype=torch.float32, device=device)
        asset.inside_target              = torch.empty(E,      dtype=torch.bool,    device=device)
        asset.inside_distractor          = torch.empty(E,      dtype=torch.bool,    device=device)
        asset.inside_objects              = torch.empty((E, O), dtype=torch.bool,    device=device)

        # === Choice reaction data ===

        asset.choice_reaction_screen_id = None

        if config.choice_reaction.enabled:
            asset.choice_reaction_screen_id = get_mujoco_id(
                mujoco.mjtObj.mjOBJ_GEOM,
                'choice_reaction/geom'
            )

    def reset(self, env_ids: torch.Tensor | None) -> None:
        "Note: This is run once before first initial step and always after every reset event is done by mjlab"
        asset = self.asset
        sim = self.env.sim

        if env_ids is None:
            env_ids = asset.env_ids

        # === Reset episodic state ===

        asset.step_count[env_ids] = 0
        asset.completed_target_count[env_ids] = 0
        asset.steps_inside_target[env_ids] = 0
        asset.current_trial_completed[env_ids] = False

        # === Domain randomization ===

        E = env_ids.shape[0]

        # == POS ===
        if self.trj_sampler is not None:
            tracking_trj = self.trj_sampler.sample(E) + self.object_offset # [B, S, K, 3]

            asset.tracking_trj[env_ids] = tracking_trj

            asset.object_pos[
                env_ids[:, None],
                asset.tracking_object_ids[None, :],
                :,
            ] = tracking_trj[:, 0]

        if self.pos_sampler is not None:
            pointing_pos = self.pos_sampler.sample(E) + self.object_offset # [B, P, 3]

            asset.object_pos[
                env_ids[:, None],
                asset.pointing_object_ids[None, :],
                :,
            ] = pointing_pos

        # One reset-time write for all objects.
        sim.data.mocap_pos[
            env_ids[:, None],
            asset.object_mocap_ids[None, :],
            :,
        ] = asset.object_pos[env_ids]

        # == SIZE ===
        dr_size = self.size_sampler.sample(E)

        sim.model.geom_size[
            env_ids[:, None],
            asset.object_geom_ids
        ] = dr_size[..., None].expand(-1, -1, 3)

        if not self.size_sampler.is_fully_fixed:
            self._recompute_geom_bounds(self.env, env_ids, asset.object_geom_ids)

        asset.object_size[env_ids] = dr_size

        # == RGB ===
        if asset.task_cfg.choice_reaction.enabled:
            indices = torch.rand(
                E,
                self.choice_reaction_colors.shape[0],
                device=self.env.device
            ).argsort(dim=1)[:, :asset.num_targets]

            dr_rgb = self.choice_reaction_colors[indices]
        else:
            dr_rgb = self.rgb_sampler.sample(E)

        sim.model.geom_rgba[
            env_ids[:, None],
            asset.object_geom_ids,
            :3
        ] = dr_rgb

        asset.object_rgb[env_ids] = dr_rgb

        # == Sequence ==
        order = asset.task_cfg.sequence.order

        if order == TargetOrder.SHUFFLED:
            num_cycles = math.ceil(asset.max_trials / asset.num_targets)

            # [B, cycles, T]: independent target permutations per cycle.
            shuffled_cycles = torch.rand(
                (E, num_cycles, asset.num_targets),
                device=asset.target_order.device,
            ).argsort(dim=-1)

            asset.target_order[env_ids] = shuffled_cycles.flatten(1)[
                :,
                :asset.max_trials,
            ]

        elif order == TargetOrder.RANDOM:
            # Sampling with replacement: targets may repeat immediately.
            asset.target_order[env_ids] = torch.randint(
                low=0,
                high=asset.num_targets,
                size=(E, asset.max_trials),
                dtype=torch.long,
                device=asset.target_order.device,
            )

        elif order != TargetOrder.FIXED:
            raise ValueError(f"Unsupported target order: {order!r}")

        asset.current_target_id[env_ids] = asset.target_order[env_ids, 0]

        asset.current_target_dwell_steps[env_ids] = asset.target_dwell_steps[
            asset.current_target_id[env_ids]
        ]

        # == Choice reaction ==

        if asset.task_cfg.choice_reaction.enabled:
            self.env.sim.model.geom_rgba[
                env_ids,
                asset.choice_reaction_screen_id,
                :3
            ] = asset.object_rgb[
                env_ids,
                asset.current_target_id[env_ids],
            ]


    def __call__(self, env: ManagerBasedRlEnv, *args, **kwargs) -> None:
        asset = self.asset

        completed = asset.current_trial_completed

        asset.completed_target_count.add_(
            completed.to(asset.completed_target_count.dtype)
        )
        asset.completed_target_count.clamp_(max=asset.max_trials)

        asset.steps_inside_target.masked_fill_(completed, 0)

        trial_id = asset.completed_target_count.clamp(
            max=asset.max_trials - 1
        )

        asset.current_target_id.copy_(
            asset.target_order[
                asset.env_ids,
                trial_id,
            ]
        )

        if asset.task_cfg.choice_reaction.enabled:
            self.env.sim.model.geom_rgba[
                :,
                asset.choice_reaction_screen_id,
                :3
            ] = asset.object_rgb[asset.env_ids, asset.current_target_id]

        asset.current_target_dwell_steps.copy_(
            asset.target_dwell_steps[asset.current_target_id]
        )

        self._update_object_contacts()

        inside_target_i = asset.inside_target.to(asset.steps_inside_target.dtype)
        asset.steps_inside_target.add_(inside_target_i)

        if asset.task_cfg.reach.dwell_continuous:
            asset.steps_inside_target.mul_(inside_target_i)

        asset.current_trial_completed.copy_(
            asset.steps_inside_target >= asset.current_target_dwell_steps
        )

        if self.trj_sampler is not None:
            step = asset.step_count.clamp_max(asset.tracking_trj.shape[1] - 1)
            pos = asset.tracking_trj[asset.env_ids, step]  # [E, K, 3]

            asset.object_pos[:, asset.tracking_object_ids, :] = pos
            env.sim.data.mocap_pos[:, asset.tracking_mocap_ids, :] = pos

        asset.step_count.add_(1)

    def _update_object_contacts(self) -> None:
        asset = self.asset

        env_ids = asset.env_ids
        target_ids = asset.current_target_id

        ee_pos = asset.data.site_pos_w[
            env_ids,
            asset.end_effector_site_id,
        ]  # [E, 3]

        distances = torch.linalg.vector_norm(
            ee_pos[:, None, :] - asset.object_pos,
            dim=-1,
        )  # [E, O]

        inside_objects = distances < asset.object_size  # [E, O]

        distance_to_target = distances[
            env_ids,
            target_ids,
        ]  # [E]

        inside_target = inside_objects[
            env_ids,
            target_ids,
        ]  # [E], bool

        # Counts all objects currently containing the end effector.
        #
        # Subtracting the current target's contribution answers:
        # "Am I inside at least one object other than the current target?"
        inside_non_current_object = (
            inside_objects.sum(dim=1)
            > inside_target.to(torch.int64)
        )  # [E], bool

        # Current target takes precedence over all other overlaps.
        inside_distractor = (
            ~inside_target
            & inside_non_current_object
        )

        asset.distance_to_target.copy_(distance_to_target)
        asset.inside_target.copy_(inside_target)
        asset.inside_distractor.copy_(inside_distractor)
        asset.inside_objects.copy_(inside_objects)

    def _recompute_geom_bounds(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        geom_ids: torch.Tensor,
    ) -> None:
        """Recompute ``geom_rbound`` and ``geom_aabb`` from current ``geom_size``.

        Only primitive types (sphere, capsule, ellipsoid, cylinder, box) are handled. Plane,
        hfield, mesh, and SDF geoms are left unchanged.
        """
        geom_ids = geom_ids.to(device=env.device, dtype=torch.long)

        env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
        size = env.sim.model.geom_size[env_grid, geom_grid]  # (n_envs, n_geoms, 3)
        s0, s1, s2 = size[..., 0], size[..., 1], size[..., 2]

        # geom_type is (ngeom,) int, not per-world.
        geom_type_all = torch.as_tensor(env.sim.model.geom_type, device=env.device)
        gtype = geom_type_all[geom_ids]  # (G,) int

        # Compute rbound per type via torch.where cascades.
        rbound = torch.zeros_like(s0)

        is_sphere = gtype[None, :] == _GEOM_SPHERE
        is_capsule = gtype[None, :] == _GEOM_CAPSULE
        is_ellipsoid = gtype[None, :] == _GEOM_ELLIPSOID
        is_cylinder = gtype[None, :] == _GEOM_CYLINDER
        is_box = gtype[None, :] == _GEOM_BOX

        rbound = torch.where(is_sphere, s0, rbound)
        rbound = torch.where(is_capsule, s0 + s1, rbound)
        rbound = torch.where(is_ellipsoid, torch.maximum(s0, torch.maximum(s1, s2)), rbound)
        rbound = torch.where(is_cylinder, torch.sqrt(s0 * s0 + s1 * s1), rbound)
        rbound = torch.where(is_box, torch.sqrt(s0 * s0 + s1 * s1 + s2 * s2), rbound)

        # Compute aabb half-sizes per type.
        aabb_half_x = torch.zeros_like(s0)
        aabb_half_y = torch.zeros_like(s0)
        aabb_half_z = torch.zeros_like(s0)

        # Sphere: (s0, s0, s0)
        aabb_half_x = torch.where(is_sphere, s0, aabb_half_x)
        aabb_half_y = torch.where(is_sphere, s0, aabb_half_y)
        aabb_half_z = torch.where(is_sphere, s0, aabb_half_z)
        # Capsule: (s0, s0, s0+s1)
        aabb_half_x = torch.where(is_capsule, s0, aabb_half_x)
        aabb_half_y = torch.where(is_capsule, s0, aabb_half_y)
        aabb_half_z = torch.where(is_capsule, s0 + s1, aabb_half_z)
        # Cylinder: (s0, s0, s1)
        aabb_half_x = torch.where(is_cylinder, s0, aabb_half_x)
        aabb_half_y = torch.where(is_cylinder, s0, aabb_half_y)
        aabb_half_z = torch.where(is_cylinder, s1, aabb_half_z)
        # Ellipsoid: (s0, s1, s2)
        aabb_half_x = torch.where(is_ellipsoid, s0, aabb_half_x)
        aabb_half_y = torch.where(is_ellipsoid, s1, aabb_half_y)
        aabb_half_z = torch.where(is_ellipsoid, s2, aabb_half_z)
        # Box: (s0, s1, s2)
        aabb_half_x = torch.where(is_box, s0, aabb_half_x)
        aabb_half_y = torch.where(is_box, s1, aabb_half_y)
        aabb_half_z = torch.where(is_box, s2, aabb_half_z)

        is_supported = is_sphere | is_capsule | is_ellipsoid | is_cylinder | is_box

        if not is_supported.all():
            unsupported = gtype[~is_supported[0]]
            names = [mujoco.mjtGeom(t.item()).name for t in unsupported]
            raise ValueError(
                f"dr.geom_size only supports primitive geom types (sphere, capsule, "
                f"ellipsoid, cylinder, box). Got unsupported types: {names}"
            )

        env.sim.model.geom_rbound[env_grid, geom_grid] = rbound

        # aabb shape is (*, ngeom, 2, 3) — [0] is center, [1] is half-size.
        # Center stays (0,0,0) for all primitives; only update half-size.
        aabb_half = torch.stack([aabb_half_x, aabb_half_y, aabb_half_z], dim=-1)
        env.sim.model.geom_aabb[env_grid, geom_grid, 1] = aabb_half
