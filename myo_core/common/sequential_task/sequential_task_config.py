from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from hydra.utils import instantiate

from myo_core.common.config import Vec3, ScalarRange, Vec3Range
from myo_core.task.task_config import TaskConfig


ShapeType = Literal["sphere", "box"]


class Color_mode(Enum):
    OFF = 0
    RECOLOR = 1
    OVERLAY = 2

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

    # Digits drawn on the button surface as seven-segment display (visual only, play
    # mode only). Reading direction on the surface: up = local +x, right = local -y.
    label: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.label is not None:
            self.label = str(self.label)
            if not self.label.isdigit():
                raise ValueError(f"Button label must consist of digits, got {self.label!r}")
        self.size = Vec3Range.of(self.size)
        self.site_size = Vec3Range.of(self.site_size)
        self.site_pos = Vec3Range.of(self.site_pos)


# Geoms the housing of a pressable button consists of (back plate + 4 walls),
# named geom_housing_{target_id}_{k}.
BUTTON_HOUSING_PARTS = 5


@dataclass
class ButtonPressConfig:

    enabled: bool = False
    max_press_depth: float = 0.008
    activation_depth: float = 0.006    # press depth at which the button activates
    stiffness: float = 150.0  # spring stiffness [N/m]
    preload: float = 0.002  # spring rest offset above the top stop, keeps the cap seated
    damping: float = 3.0  # [N s/m] is needed to prevent "swinging" of the button after pressing it. Should be put in relation to stiffnes

    #general positional stuff
    cap_half_height: float = 0.008     # the cap reaches into the hole even at rest
    pressed_protrusion: float = 0.001  # cap top above the housing when fully pressed
    hole_clearance: float = 0.0005     # gap between cap and hole wall (per side)
    cap_mass: float = 0.01
    cap_rgb: Vec3 = field(default_factory=lambda: [0.6, 0.6, 0.6])

    require_release: bool = False

    def __post_init__(self) -> None:
        if self.max_press_depth <= 0.0:
            raise ValueError(f"button_press.travel must be > 0, got {self.max_press_depth}")
        if not 0.0 < self.activation_depth <= self.max_press_depth:
            raise ValueError(
                "button_press.activation_depth must be in (0, travel], "
                f"got {self.activation_depth} (travel={self.max_press_depth})"
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
    show_button_labels: bool = True
    target_state_color_mode: Color_mode = Color_mode.OFF
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
