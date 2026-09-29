from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RecorderTermCfg

from myo_core.common import MyoComponent
from myo_core.task.universal import UniversalRecorder
from .disabled_vision_config import DisabledVisionConfig
from ..vision_registry import myo_register_vision

@myo_register_vision("disabled")
class DisabledVisionComponent(MyoComponent):
    def __init__(self, cfg: DisabledVisionConfig):
       self.cfg = cfg

    def modify_env_cfg(self, cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
        if play:
            cfg.recorders['universal'] = RecorderTermCfg(
                func=UniversalRecorder
            )

    def modify_rl_cfg(self, cfg: RslRlOnPolicyRunnerCfg) -> None:
        cfg.actor.class_name = "MLPModel"

        cfg.obs_groups = {
          "actor": ("agent_state", "task_state"),
          "critic": ("agent_state", "task_state")
        }
