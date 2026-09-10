from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg

from myo_core.common.sequential_task import SequentialTaskComponent
from ..task_registry import myo_register_task
from .numpad_task_config import NumpadTaskConfig, Color_mode
from .numpad_task_logic import (
    NumpadTargetBoxOverlay,
    NumpadTaskLogic,
    np_color_targets_by_state,
)


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

    def modify_env_cfg(self, cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
        super().modify_env_cfg(cfg, play)

        entity_cfg = self._entity_cfg

        color_mode = self.cfg.target_state_color_mode
        if play and color_mode == Color_mode.RECOLOR:
            # Variant 1: recolor button geoms in the model (results in viewer lag because of reloading model).
            cfg.events["numpad_target_coloring"] = EventTermCfg(
                func=np_color_targets_by_state,
                params={"asset_cfg": entity_cfg},
                mode="step",
            )
        elif play and color_mode == Color_mode.OVERLAY:
            # Variant 2: overlay colored debug boxes (no model change, no lag).
            cfg.events["numpad_target_box_overlay"] = EventTermCfg(
                func=NumpadTargetBoxOverlay,
                params={"asset_cfg": entity_cfg},
                mode="reset",
            )
