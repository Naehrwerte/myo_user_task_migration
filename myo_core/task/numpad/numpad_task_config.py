from dataclasses import dataclass, field

from myo_core.common.sequential_task import Color_mode, RewardConfig, SequentialTaskConfig

@dataclass
class NumpadTaskConfig(SequentialTaskConfig):
    """Button-pressing task on the shared sequential task framework.
    """

    model_path: str = "myo_user/envs/myo/assets/arm/mobl_arms_index_universal_myouser.xml"
    sequence_length: int = 4
    sample_with_replacement: bool = True
    target_state_color_mode: Color_mode = Color_mode.OVERLAY

    reward: RewardConfig = field(default_factory=lambda: RewardConfig(
        weights={
            "distance": 1,
            "phase_bonus": 0,
            "done": 10,
            "neural_effort": 0.0,
            "jac_effort": 1,
        }
    ))
