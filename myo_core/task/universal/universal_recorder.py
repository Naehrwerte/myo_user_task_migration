import re
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import RecorderTermCfg, SceneEntityCfg

from myo_core.common import H5EvalRecorder
from .universal_task_entity import (
    UniversalTaskEntity,
    episode_successfully_completed,
    trial_successfully_completed,
)
from .universal_task_component import _UNIVERSAL_ENTITY_NAME

class UniversalRecorder(H5EvalRecorder):
    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
  
        self._asset: UniversalTaskEntity = self._env.scene[_UNIVERSAL_ENTITY_NAME]
        self._asset_cfg = SceneEntityCfg(_UNIVERSAL_ENTITY_NAME)

    def _collect_state_tensors(
        self,
        env_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        state = super()._collect_state_tensors(env_ids)

        exclude = re.compile(r".*/(agent_state|task_state)/.*")

        state = {
            key: value
            for key, value in state.items()
            if not exclude.search(key)
        }

        state['task/distance_target']   = self._asset.distance_to_target[env_ids]
        state['task/inside_target']     = self._asset.inside_target[env_ids]
        state['task/inside_distractor'] = self._asset.inside_distractor[env_ids]
        state['task/inside_objects']    = self._asset.inside_objects[env_ids]

        state['task/current_target_id'] = self._asset.current_target_id[env_ids]
        state['task/target_order'] = self._asset.target_order[env_ids]
        state['task/completed_target_count'] = (
            self._asset.completed_target_count[env_ids]
        )
        state['task/current_trial_completed'] = (
            self._asset.current_trial_completed[env_ids]
        )
        state['task/steps_inside_target'] = self._asset.steps_inside_target[env_ids]
        state['task/episode_success'] = episode_successfully_completed(
            self._env,
            self._asset_cfg,
        )[env_ids]

        # These values fully identify the sampled task geometry at every step;
        # for tracking tasks object_pos also preserves the realized trajectory.
        state['task/object_pos']        = self._asset.object_pos[env_ids]
        state['task/object_size']       = self._asset.object_size[env_ids]
        state['task/object_rgb']        = self._asset.object_rgb[env_ids]

        for i in range(self._asset.max_trials):
            state[f'task/trial_{i}_success'] = trial_successfully_completed(
                self._env,
                self._asset_cfg,
                i
            )[env_ids]

        return state
