from dataclasses import dataclass, field
from typing import Literal, Any
from enum import StrEnum
from hydra.utils import instantiate

from myo_core.common.config import ScalarRange, Vec3Range
from ..task_config import TaskConfig


ShapeType = Literal["sphere", "capsule", "cylinder", "box"]


class TargetOrder(StrEnum):
    FIXED = "fixed"
    SHUFFLED = "shuffled"
    RANDOM = "random"


@dataclass
class TargetConfig:
    position: Vec3Range = field(default_factory=lambda: Vec3Range(
        value=[0.3, 0.0, 0.0]
    ))
    rgb: Vec3Range = field(default_factory=lambda: Vec3Range(
        min=[0, 0, 0],
        max=[1, 1, 1]
    ))

    def __post_init__(self) -> None:
        self.position = Vec3Range.of(self.position)
        self.rgb = Vec3Range.of(self.rgb)


@dataclass
class ShapeTargetConfig(TargetConfig):
    shape: ShapeType = "sphere"
    size: ScalarRange = field(default_factory=lambda: ScalarRange(
        value=0.05
    ))

    def __post_init__(self) -> None:
        super().__post_init__()
        self.size = ScalarRange.of(self.size)


@dataclass
class PointingTargetConfig(ShapeTargetConfig):
    dwell_duration: float = 0.25


@dataclass
class TrackingTargetConfig(ShapeTargetConfig):
    dwell_duration: float | None = None


@dataclass
class ButtonTargetConfig(TargetConfig):
    size: Vec3Range = field(default_factory=lambda: Vec3Range(
        value=[0.025, 0.025, 0.01]
    ))
    site_pos: Vec3Range = field(default_factory=lambda: Vec3Range(
        value=[0.0, 0.0, 0.01]
    ))
    euler: list[float] = field(default_factory=lambda: [0, -0.79, 0])
    geom_margin: float = 0.001
    min_touch_force: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.size = Vec3Range.of(self.size)
        self.site_pos = Vec3Range.of(self.site_pos)



@dataclass
class ReachConfig:
    reference_site: str = "humphant"
    reference_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    end_effector_site: str = "fingertip"
    dwell_continuous: bool = False


@dataclass
class RewardConfig:
    weights: dict[str, float] = field(default_factory=lambda: {
        "distance": 1,
        "trial_bonus": 8,
        "done": 8
    })
    distance_exponential: bool = False
    distance_metric: float = 10.0


@dataclass
class SequenceConfig:
    max_trials: int | None = None
    order: TargetOrder = TargetOrder.FIXED


@dataclass
class ChoiceReactionConfig:
    enabled: bool = False

    distance: float = 0.5
    sector_x: float = 0.19
    sector_y: float = 0.19

    position: list[float] = field(default_factory=lambda: [0.0, 0.0, -0.75])
    size: list[float] = field(default_factory=lambda: [0.1, 0.1, 0.001])


@dataclass
class UniversalTaskConfig(TaskConfig):
    model_path: str = "myo_user/envs/myo/assets/arm/mobl_arms_index_universal_myouser.xml"

    targets: list[Any] = field(default_factory=list)
    distractors: list[Any] = field(default_factory=list)

    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    choice_reaction: ChoiceReactionConfig = field(default_factory=ChoiceReactionConfig)

    reach: ReachConfig = field(default_factory=ReachConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    agent_state_keys: list[str] = field(default_factory=lambda: [
        'qpos',
        'qvel',
        'qacc',
        'act',
        'ee_pos',
        'trial_progress',
    ])
    task_state_keys: list[str] = field(default_factory=lambda: [
        'target_pos',
        'dwell_fraction',
    ])
    task_query_keys: list[str] = field(default_factory=lambda: [
        'target_color',
        'target_size',
    ])

    @property
    def max_trials(self):
        trials = self.sequence.max_trials
        return trials if trials is not None and trials > 0 else len(self.targets)

    def __post_init__(self) -> None:
        self.targets = instantiate(self.targets, _convert_="object")
        self.distractors = instantiate(self.distractors, _convert_="object")
