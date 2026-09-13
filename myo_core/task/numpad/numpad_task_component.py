from __future__ import annotations

from myo_core.common.sequential_task import SequentialTaskComponent
from ..task_registry import myo_register_task
from .numpad_task_config import NumpadTaskConfig
from .numpad_task_logic import NumpadTaskLogic


@myo_register_task("numpad")
class NumpadTaskComponent(SequentialTaskComponent):
    """Numpad button-pressing task with a randomized press sequence.

    numpad task logic:
    draw a random press sequence from the button pool at every reset. The phases of the
    episode are therefor not dependent on number of target lengths but a seperate sequence_length
    """

    entity_name = "numpad_robot"
    task_logic_cls = NumpadTaskLogic

    cfg: NumpadTaskConfig

    def __init__(self, cfg: NumpadTaskConfig):
        super().__init__(cfg)

    @property
    def _num_phases(self) -> int:
        return int(self.cfg.sequence_length)
