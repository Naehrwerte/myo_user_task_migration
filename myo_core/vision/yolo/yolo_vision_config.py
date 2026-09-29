from dataclasses import dataclass
from enum import Enum
from ..vision_config import MonoscopicDepthVisionConfig


class YoloColorMode(str, Enum):
    rgb = "rgb"
    hue = "hue"


class YoloDepthMode(str, Enum):
    average = "average"


@dataclass
class YoloVisionConfig(MonoscopicDepthVisionConfig):
    model: str = "yolo11n.pt"
    color_mode: YoloColorMode | None = None
    depth_mode: YoloDepthMode | None = None
    use_class: bool = True
    use_confidence: bool = True
    nms_top_k: int | None = 16

    perfect_detection: bool = False
    capture_geoms: str = "^(target|distractor|choice_reaction).*"

    transformer: bool = False
    history_length: int = 1
    task_query_pooling: bool = True
    task_query_head: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

        if not self.model:
            raise ValueError("model must not be empty")

