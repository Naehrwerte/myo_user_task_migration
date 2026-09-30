from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RecorderTermCfg

from myo_core.common import MyoComponent
from myo_core.task.universal import UniversalRecorder
from myo_core.task.universal.universal_task_component import _UNIVERSAL_ENTITY_NAME
from .disabled_vision_config import DisabledVisionConfig
from ..vision_registry import myo_register_vision

@myo_register_vision("disabled")
class DisabledVisionComponent(MyoComponent):
    def __init__(self, cfg: DisabledVisionConfig):
       self.cfg = cfg

    def modify_env_cfg(self, cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
        # The recorder reads universal-task state; other tasks use their own entity.
        if play and _UNIVERSAL_ENTITY_NAME in cfg.scene.entities:
            cfg.recorders['universal'] = RecorderTermCfg(
                func=UniversalRecorder
            )

    def modify_rl_cfg(self, cfg: RslRlOnPolicyRunnerCfg) -> None:
        cfg.actor.class_name = "MLPModel"

        cfg.obs_groups = {
          "actor": ("agent_state", "task_state"),
          "critic": ("agent_state", "task_state")
        }
