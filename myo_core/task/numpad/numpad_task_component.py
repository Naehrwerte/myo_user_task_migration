from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg, MetricsTermCfg, RewardTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from ..task_registry import myo_register_task
from ..universal.universal_task_component import UniversalTaskComponent
from .numpad_task_config import NumpadTaskConfig
from .numpad_task_logic import (
    NumpadTargetBoxOverlay,
    NumpadTaskLogic,
    np_color_targets_by_state,
    np_phase_progress,
    np_sequence_completed,
    np_sequential_distance_reward,
)


@myo_register_task("numpad")
class NumpadTaskComponent(UniversalTaskComponent):
    """Numpad button-pressing task with a randomized press sequence.

    Reuses the full universal task pipeline, then replaces only the terms that
    depend on the *press sequence* (rather than the target-list order): the task
    logic event, the distance reward, phase-progress observation, the done/
    termination condition and the per-phase metrics. Everything else (model,
    actions, effort rewards, button touch detection) is inherited unchanged.
    """

    def __init__(self, cfg: NumpadTaskConfig):
        super().__init__(cfg)

    def modify_env_cfg(self, cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
        super().modify_env_cfg(cfg, play)

        entity_cfg = self._entity_cfg
        target_entity_cfg = self._target_entity_cfg
        seq_len = int(self.cfg.sequence_length)
        last_phase = seq_len - 1

        cfg.events["task_logic_update"] = EventTermCfg(
            func=NumpadTaskLogic,
            params={"asset_cfg": target_entity_cfg},
            mode="step",
        )

        for group in (cfg.observations.get("agent_state"), cfg.observations.get("task_state")):
            if group is not None and "phase_progress" in group.terms:
                group.terms["phase_progress"] = ObservationTermCfg(
                    func=np_phase_progress, params={"asset_cfg": entity_cfg}
                )

        cfg.rewards["distance"] = RewardTermCfg(
            func=np_sequential_distance_reward,
            params={
                "asset_cfg": entity_cfg,
                "exponential_distance_reward": self.cfg.reward.distance_exponential,
                "distance_metric": self.cfg.reward.distance_metric,
            },
            weight=self.cfg.reward.weights.get("distance", 0.0),
        )

        cfg.rewards["done"] = RewardTermCfg(
            func=np_sequence_completed,
            params={"asset_cfg": entity_cfg, "phase_id": last_phase},
            weight=self.cfg.reward.weights.get("done", 0.0),
        )
        cfg.terminations["episode_success"] = TerminationTermCfg(
            func=np_sequence_completed,
            params={"asset_cfg": entity_cfg, "phase_id": last_phase},
        )

        for i in range(self._num_targets):
            cfg.metrics.pop(f"phase_{i}_success", None)
        for i in range(seq_len):
            cfg.metrics[f"phase_{i}_success"] = MetricsTermCfg(
                func=np_sequence_completed,
                params={"asset_cfg": entity_cfg, "phase_id": i},
                reduce="last",
            )

        color_mode = int(self.cfg.target_state_color_mode)
        if play and color_mode == 1:
            # Variant 1: recolor button geoms in the model (accurate, viewer lag).
            cfg.events["numpad_target_coloring"] = EventTermCfg(
                func=np_color_targets_by_state,
                params={"asset_cfg": entity_cfg},
                mode="step",
            )
        elif play and color_mode == 2:
            # Variant 2: overlay colored debug boxes (no model change, no lag).
            cfg.events["numpad_target_box_overlay"] = EventTermCfg(
                func=NumpadTargetBoxOverlay,
                params={"asset_cfg": entity_cfg},
                mode="reset",
            )