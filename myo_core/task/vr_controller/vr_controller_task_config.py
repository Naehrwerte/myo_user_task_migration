from dataclasses import dataclass, field
from typing import Literal, Any, List
from hydra.utils import instantiate

from myo_core.common.config import Vec3, ScalarRange, Vec3Range
from ..task_config import TaskConfig

ShapeType = Literal["sphere", "box"]


@dataclass
class ReachConfig:
    reference_site: str = "humphant"
    reference_offset: Vec3 = field(default_factory=lambda: (0.0, 0.0, 0.0))
    end_effector_site: str = "ray_origin"
    ray_end_site: str = "ray_end"
    dwell_continuous: bool = False



@dataclass
class DistractorConfig:
    count: int = 0
    similar_color_prob: float = 0.5
    similar_shape_prob: float = 0.5


@dataclass
class SequenceConfig:
    show_future_targets: bool = True
    color_code_progress: bool = True


@dataclass
class RewardConfig:
    weights: dict[str, float] = field(default_factory=lambda: {
        "distance": 1,
        "phase_bonus": 10,
        "done": 10
    })
    distance_exponential: bool = False
    distance_metric: float = 10.0


@dataclass
class VrControllerConfig:
    controller_pos: List[float] = field(default_factory=lambda: [0.01, -0.03, -0.015])
    controller_quat: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])

    ray_start_offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.02])
    ray_length: float = 1.5
    ray_axis: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    ray_radius: float = 0.003

    controller_mass: float = 0.129
    controller_mesh: str = "controller-right"


    hand_body_name: str = "hand"
    ray_origin_site_name: str = "ray_origin"
    ray_end_site_name: str = "ray_end"


@dataclass
class VrControllerTaskConfig(TaskConfig):
    model_path: str = "myo_user/envs/myo/assets/arm/mobl_arms_index_vr_myouser.xml"

    targets: list[Any] = field(default_factory=list)

    reach: ReachConfig = field(default_factory=lambda: ReachConfig(end_effector_site="ray_origin"))
    vr_controller: VrControllerConfig = field(default_factory=VrControllerConfig)
    distractor: DistractorConfig = field(default_factory=DistractorConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    agent_state_keys: list[str] = field(default_factory=lambda: [
        'qpos',
        'qvel',
        'qacc',
        'act',
        'ee_pos',
        "ray_dir"
    ])

    task_state_keys: list[str] = field(default_factory=lambda: [
        'target_pos',
        'target_size',
        'phase_progress',
        'dwell_fraction'
    ])

    def __post_init__(self) -> None:
        self.targets = instantiate(self.targets, _convert_="object")