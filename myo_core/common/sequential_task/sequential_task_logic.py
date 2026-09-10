"""Task logic shared by every sequential task.

Self-contained copy of the universal task logic. It is written in terms of
*phases* instead of raw target indices so that a subclass only has to say how
many phases there are and which target belongs to a phase (see
``_resolve_num_phases`` / ``_sample_phase_targets``).
"""

from __future__ import annotations

import math

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import EventTermCfg, ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .sequential_task_config import ButtonTargetConfig
from .sequential_task_entity import TaskEntity


class SequentialTaskLogic(ManagerTermBase):
    """Advances the target sequence: dwell/touch detection and phase bookkeeping."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)

        asset_cfg: SceneEntityCfg = cfg.params['asset_cfg']
        asset: TaskEntity = env.scene[asset_cfg.name]

        self.asset = asset
        self.entity_name = asset_cfg.name
        self._device = env.device

        self._resolve_site_ids(asset, asset_cfg)
        asset.target_pos_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.long, device=env.device)
        asset.target_size_ids = torch.tensor(asset_cfg.geom_ids, dtype=torch.long, device=env.device)
        asset.env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

        asset.num_phases = self._resolve_num_phases(asset)
        asset.phase_targets = torch.zeros(
            (env.num_envs, asset.num_phases), dtype=torch.long, device=env.device
        )
        asset.phase_targets[:] = self._sample_phase_targets(asset, asset.env_ids)
        asset.current_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        asset.current_target_id = asset.phase_targets[:, 0].clone()

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
            sensor = env.sim.mj_model.sensor(f"{self.entity_name}/sensor_target_{target_id}")
            sensor_adr.append(int(sensor.adr[0]))
        asset.target_sensor_adr = torch.tensor(sensor_adr, dtype=torch.long, device=env.device)

        asset.target_pos = asset.data.body_com_pos_w[asset.env_ids[:, None], asset.target_pos_ids]
        asset.inside_target, asset.distance_to_target, asset.target_size = self._inside_target()
        asset.remaining_target_distances = torch.zeros(
            (env.num_envs, asset.num_phases), dtype=torch.float32, device=env.device
        )

        asset.current_phase_completed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Sequence definition -----------------------------------------------------

    def _resolve_site_ids(self, asset: TaskEntity, asset_cfg: SceneEntityCfg) -> None:
        """Cache the site ids the completion criterion needs on the entity.

        Site ids follow the order of ``SequentialTaskComponent._site_names()``.
        """
        asset.end_effector_site_id = asset_cfg.site_ids[0]

    def _resolve_num_phases(self, asset: TaskEntity) -> int:
        """Number of targets that have to be completed per episode."""
        return asset.num_targets

    def _sample_phase_targets(self, asset: TaskEntity, env_ids: torch.Tensor) -> torch.Tensor:
        """Target index for every phase, as [len(env_ids), num_phases].

        The plain sequential task walks the target list in order, so the mapping
        is the identity and is the same for every environment.
        """
        phases = torch.arange(asset.num_phases, dtype=torch.long, device=self._device)
        return phases[None, :].expand(env_ids.shape[0], asset.num_phases)

    # Manager term ------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor | None) -> None:
        "Note: This is run once before first initial step and always after every reset event is done by mjlab"
        asset = self.asset

        if env_ids is None:
            env_ids = asset.env_ids

        asset.current_phase[env_ids] = 0
        asset.completed_target_count[env_ids] = 0
        asset.steps_inside_target[env_ids] = 0
        asset.current_phase_completed[env_ids] = False

        asset.phase_targets[env_ids] = self._sample_phase_targets(asset, env_ids)
        asset.current_target_id[env_ids] = asset.phase_targets[env_ids, 0]

        asset.target_pos[env_ids] = asset.data.body_com_pos_w[
            env_ids[:, None],
            asset.target_pos_ids,
        ]  # [num_reset_envs, num_targets, 3]

        asset.remaining_target_distances[env_ids] = 0.0

        if asset.num_phases <= 1:
            return

        phase_pos = asset.target_pos[
            env_ids[:, None],
            asset.phase_targets[env_ids],
        ]  # [num_reset_envs, num_phases, 3]

        segment_distances = torch.linalg.vector_norm(
            phase_pos[:, 1:] - phase_pos[:, :-1],
            dim=-1,
        )  # [num_reset_envs, num_phases - 1]

        asset.remaining_target_distances[env_ids, : asset.num_phases - 1] = (
            segment_distances.flip(1).cumsum(1).flip(1)
        )

    def __call__(self, env: ManagerBasedRlEnv, env_ids: None, asset_cfg: SceneEntityCfg) -> None:
        asset = self.asset

        completed = asset.current_phase_completed

        asset.completed_target_count += completed
        asset.completed_target_count.clamp_(max=asset.num_phases)

        asset.steps_inside_target.masked_fill_(completed, 0)

        asset.current_phase = asset.completed_target_count.clamp(
            max=asset.num_phases - 1
        )
        asset.current_target_id = asset.phase_targets.gather(
            1, asset.current_phase[:, None]
        ).squeeze(1)

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

    # Completion criterion ----------------------------------------------------

    def _inside_target(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
