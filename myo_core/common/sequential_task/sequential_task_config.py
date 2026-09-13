from dataclasses import dataclass, field
from typing import Any, Literal

from hydra.utils import instantiate

from myo_core.common.config import Vec3, ScalarRange, Vec3Range
from myo_core.task.task_config import TaskConfig


ShapeType = Literal["sphere", "box"]


@dataclass
class ReachConfig:
    reference_site: str = "humphant"
    reference_offset: Vec3 = field(default_factory=lambda: [0.0, 0.0, 0.0])
    end_effector_site: str = "fingertip"
    dwell_continuous: bool = False


@dataclass
class TargetConfig:
    position: Vec3Range = field(default_factory=lambda: Vec3Range(
        value=(0.3, 0.0, 0.0)
    ))
    rgb: Vec3Range = field(default_factory=lambda: Vec3Range(
        min=(0, 0, 0),
        max=(1, 1, 1)
    ))

    def __post_init__(self) -> None:
        self.position = Vec3Range.of(self.position)
        self.rgb = Vec3Range.of(self.rgb)


@dataclass
class PointingTargetConfig(TargetConfig):
    shape: ShapeType = "sphere"
    size: ScalarRange = field(default_factory=lambda: ScalarRange(
        value=0.05
    ))
    dwell_duration: float = 0.25

    def __post_init__(self) -> None:
        super().__post_init__()
        self.size = ScalarRange.of(self.size)


@dataclass
class ButtonTargetConfig(TargetConfig):
    size: Vec3Range = field(default_factory=lambda: Vec3Range(
        value=(0.025, 0.025, 0.01)
    ))
    site_size: Vec3Range = field(default_factory=lambda: Vec3Range(
        value=(0.02, 0.02, 0.01)
    ))
    site_pos: Vec3Range = field(default_factory=lambda: Vec3Range(
        value=(0.0, 0.0, 0.01)
    ))
    euler: Vec3 = field(default_factory=lambda: [0.0, -0.79, 0.0])

    geom_margin: float = 0.0
    min_touch_force: float = 1.0
    dwell_duration: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.size = Vec3Range.of(self.size)
        self.site_size = Vec3Range.of(self.site_size)
        self.site_pos = Vec3Range.of(self.site_pos)


@dataclass
class ButtonPressConfig:
    """Physically pressable buttons.

    When enabled, every button target gets a movable cap (its xy half-extents are
    taken from ``site_size``) on a spring-loaded slide joint along the button
    normal. The button activates once the cap is pushed in by at least
    ``activation_depth`` instead of on touch force. All lengths are in meters.
    """

    enabled: bool = False
    travel: float = 0.008              # maximum press depth (joint range)
    activation_depth: float = 0.006    # press depth at which the button activates
    cap_half_height: float = 0.004
    cap_mass: float = 0.01
    cap_rgb: Vec3 = field(default_factory=lambda: [0.6, 0.6, 0.6])
    stiffness: float = 150.0           # spring stiffness [N/m]
    preload: float = 0.002             # spring rest offset above the top stop, keeps the cap seated
    damping: float = 3.0               # [N s/m]

    # Independent of ``enabled`` (works for touch and press activation): a completed
    # button has to be released before it can activate again, e.g. when the same
    # button occurs twice in a row in the sequence.
    require_release: bool = False

    def __post_init__(self) -> None:
        if self.travel <= 0.0:
            raise ValueError(f"button_press.travel must be > 0, got {self.travel}")
        if not 0.0 < self.activation_depth <= self.travel:
            raise ValueError(
                "button_press.activation_depth must be in (0, travel], "
                f"got {self.activation_depth} (travel={self.travel})"
            )


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
class SequentialTaskConfig(TaskConfig):
    """Base config for tasks that complete a sequence of targets one by one.
    """

    targets: list[Any] = field(default_factory=list)

    reach: ReachConfig = field(default_factory=ReachConfig)
    button_press: ButtonPressConfig = field(default_factory=ButtonPressConfig)
    distractor: DistractorConfig = field(default_factory=DistractorConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    agent_state_keys: list[str] = field(default_factory=lambda: [
        'qpos',
        'qvel',
        'qacc',
        'act',
        'ee_pos'
    ])
    task_state_keys: list[str] = field(default_factory=lambda: [
        'target_pos',
        'target_size',
        'phase_progress',
        'dwell_fraction'
    ])

    def __post_init__(self) -> None:
        self.targets = instantiate(self.targets, _convert_="object")
