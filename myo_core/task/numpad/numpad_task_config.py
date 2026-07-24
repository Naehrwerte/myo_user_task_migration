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

    # Button state coloring (green=current, red=todo, blue=done), play mode only:
    #   0 = off (no visual change)
    #   1 = recolor button geoms via the model (accurate, but causes viewer lag
    #       because every color change forces viser to rebuild its mesh handles)
    #   2 = draw colored overlay boxes over the buttons as debug geometry
    #       (leaves the model untouched -> no viewer lag)
    target_state_color_mode: int = 0

    reward: RewardConfig = field(default_factory=lambda: RewardConfig(
        weights={
            "distance": 1,
            "phase_bonus": 0,
            "done": 10,
            "neural_effort": 0.0,
            "jac_effort": 1,
        }
    ))
