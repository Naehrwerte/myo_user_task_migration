"""Scene entity shared by every sequential task.

Copy of the universal task entity, extended with the *phase* indirection: a
phase is a step of the sequence to complete, ``phase_targets`` maps it to the
physical target that has to be reached in that phase. For a plain sequential
task that mapping is the identity (phase i == target i); the numpad draws a
random mapping at every reset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mjlab.entity import EntityCfg
from mjlab.envs.mdp import Entity
from mjlab.envs.mdp.dr._core import Ranges

from .sequential_task_config import SequentialTaskConfig


@dataclass
class TaskEntityCfg(EntityCfg):
    task_cfg: SequentialTaskConfig | None = None

    def build(self) -> TaskEntity:
        """Build task entity instance from this config."""
        return TaskEntity(self)


class TaskEntity(Entity):
    num_targets: int                             # size of the target pool
    num_phases: int                              # sequence length
    task_cfg: SequentialTaskConfig

    # index tensors
    end_effector_site_id: int
    target_pos_ids: torch.Tensor                 # [num_targets], long
    target_size_ids: torch.Tensor                # [num_targets], long
    env_ids: torch.Tensor                        # [num_envs], long
    phase_targets: torch.Tensor                  # [num_envs, num_phases], long
    current_phase: torch.Tensor                  # [num_envs], long
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
    remaining_target_distances: torch.Tensor     # [num_envs, num_phases], float
    target_pos: torch.Tensor                     # [num_envs, num_targets, 3], float

    def __init__(self, cfg: TaskEntityCfg):
        super().__init__(cfg)

        self.task_cfg = self.cfg.task_cfg
        self.num_targets = len(self.task_cfg.targets)


@dataclass
class TargetDomainRandomization:
    body_pos: dict[str, Ranges] = field(default_factory=dict)
    geom_size: dict[str, Ranges] = field(default_factory=dict)
    geom_rgb: dict[str, Ranges] = field(default_factory=dict)
