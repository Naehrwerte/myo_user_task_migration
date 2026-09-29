from __future__ import annotations

from dataclasses import dataclass, field
from mjlab.rl import RslRlModelCfg

@dataclass
class ObjectEncoderConfig:
    # Output size of the complete object branch.
    latent_dim: int = 32

    # Per-object encoder.
    object_embed_dim: int = 128
    object_encoder_hidden: tuple[int, ...] = (128,)

    # Optional per-frame object Transformer.
    use_transformer: bool = False
    transformer_layers: int = 1
    transformer_heads: int = 2
    transformer_ff_dim: int = 256

    # Temporal encoder.
    # Only used when history_dim > 1.
    frame_embed_dim: int = 128
    gru_hidden_dim: int = 128

    # Final object-branch prediction head.
    head_hidden: tuple[int, ...] = (128, 128)

    dropout: float = 0.0

    condition_pool_on_task: bool = True,
    concat_task_to_head: bool = False

@dataclass
class YoloModelConfig(RslRlModelCfg):
    object_encoder: ObjectEncoderConfig = field(default_factory=ObjectEncoderConfig)
    class_name: str = "myo_core.vision.yolo.yolo_model:YoloObjectModel"
