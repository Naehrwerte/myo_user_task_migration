from dataclasses import dataclass, field

from ..universal.universal_task_config import UniversalTaskConfig, RewardConfig


@dataclass
class NumpadTaskConfig(UniversalTaskConfig):
    """Button-pressing task on the universal task framework.

    The ``targets`` list defines a fixed pool of buttons (the full numpad), all
    of which are always rendered. Unlike the universal task, the press sequence
    is *not* the target-list order: a sequence of ``sequence_length`` buttons is
    drawn randomly from the pool at every reset (independently per environment).

    Differs from :class:`UniversalTaskConfig` in the numpad-flavoured reward
    defaults (no dwell/phase bonus, joint-acceleration effort penalty) and the
    two sequence fields below.
    """

    sequence_length: int = 4
    sample_with_replacement: bool = True

    reward: RewardConfig = field(default_factory=lambda: RewardConfig(
        weights={
            "distance": 1,
            "phase_bonus": 0,
            "done": 10,
            "neural_effort": 0.0,
            "jac_effort": 1,
        }
    ))
