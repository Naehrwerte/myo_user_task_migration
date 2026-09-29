import re
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import RecorderTermCfg

from myo_core.task.universal import UniversalRecorder


class CnnRecorder(UniversalRecorder):
    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

    def _collect_state_tensors(
        self,
        env_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        state = super()._collect_state_tensors(env_ids)

        exclude = re.compile(r"vision_cnn\/.*")

        state = {
            key: value
            for key, value in state.items()
            if not exclude.search(key)
        }

        return state
