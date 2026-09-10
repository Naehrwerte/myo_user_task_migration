from dataclasses import dataclass, field
from typing import List

from myo_core.common.sequential_task import ReachConfig as SequentialReachConfig
from myo_core.common.sequential_task import SequentialTaskConfig


@dataclass
class ReachConfig(SequentialReachConfig):
    # The "end effector" of this task is the ray origin on the controller.
    end_effector_site: str = "ray_origin"
    ray_end_site: str = "ray_end"


@dataclass
class VrControllerConfig:
    controller_pos: List[float] = field(default_factory=lambda: [0.01, -0.03, -0.015])
    controller_quat: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])

    ray_start_offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.02])
    ray_length: float = 1.5
    ray_axis: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    ray_radius: float = 0.003

    controller_mesh: str = "controller-right"


    hand_body_name: str = "hand"
    ray_origin_site_name: str = "ray_origin"
    ray_end_site_name: str = "ray_end"


@dataclass
class VrControllerTaskConfig(SequentialTaskConfig):
    model_path: str = "myo_user/envs/myo/assets/arm/mobl_arms_index_vr_myouser.xml"

    reach: ReachConfig = field(default_factory=ReachConfig)
    vr_controller: VrControllerConfig = field(default_factory=VrControllerConfig)

    agent_state_keys: list[str] = field(default_factory=lambda: [
        'qpos',
        'qvel',
        'qacc',
        'act',
        'ray_origin',
        "ray_dir"
    ])
