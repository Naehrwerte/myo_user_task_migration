from __future__ import annotations

from dataclasses import asdict
import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from tensordict import TensorDict

from .yolo_model_config import ObjectEncoderConfig


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        activation: type[nn.Module] = nn.ELU,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()

        dims = [in_dim, *hidden_dims, out_dim]
        layers: list[nn.Module] = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            is_last = i == len(dims) - 2
            if not is_last:
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TaskConditionedAttentionPool(nn.Module):
    """
    Permutation-invariant object pooling, optionally conditioned on a task query.

    Inputs:
        x:          [B*H, N, D]
        mask:       [B*H, N], True = valid object
        task_query: [B*H, T] or None

    Output:
        pooled:     [B*H, D]

    If task_query is provided, object relevance is computed relative to the task.
    If task_query is None, this falls back to learned unconditional attention pooling.
    """

    def __init__(
        self,
        object_dim: int,
        task_query_dim: int | None = None,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = object_dim

        self.object_dim = object_dim
        self.task_query_dim = task_query_dim
        self.attn_scale = object_dim**-0.5

        if task_query_dim is not None:
            self.object_key = nn.Sequential(
                nn.LayerNorm(object_dim),
                nn.Linear(object_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, object_dim),
            )

            self.task_query_proj = nn.Sequential(
                nn.LayerNorm(task_query_dim),
                nn.Linear(task_query_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, object_dim),
            )

            self.unconditional_score = None

        else:
            self.object_key = None
            self.task_query_proj = None

            self.unconditional_score = nn.Sequential(
                nn.LayerNorm(object_dim),
                nn.Linear(object_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        task_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.task_query_dim is not None:
            if task_query is None:
                raise ValueError(
                    "task_query must be provided when task_query_dim is not None."
                )

            keys = self.object_key(x)                  # [B*H, N, D]
            query = self.task_query_proj(task_query)   # [B*H, D]

            scores = (keys * query.unsqueeze(1)).sum(dim=-1)
            scores = scores * self.attn_scale

        else:
            scores = self.unconditional_score(x).squeeze(-1)

        if mask is not None:
            mask = mask.bool()

            empty_rows = ~mask.any(dim=1)

            # For empty rows, temporarily make object 0 valid so softmax
            # remains well-defined. The final pooled result is zeroed below.
            safe_mask = mask.clone()
            safe_mask[:, 0] |= empty_rows

            scores = scores.masked_fill(
                ~safe_mask,
                torch.finfo(scores.dtype).min,
            )

        weights = torch.softmax(scores, dim=1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)

        if mask is not None:
            pooled = pooled.masked_fill(empty_rows.unsqueeze(-1), 0.0)

        return pooled


class ObjectEncoderModel(nn.Module):
    """
    Slot-free object/history model with optional object Transformer and
    task-conditioned object attention pooling.

    Expected input:
        obs: [B, H, N, F]

    Optional:
        object_mask: [B, H, N], True = valid object
        task_query:  [B, T] or [B, H, T]

    Architecture:
        Per-object shared encoder
        Optional Transformer over objects per frame, no positional embeddings
        Task-conditioned permutation-invariant attention pooling
        GRU over frame embeddings
        Prediction head
    """

    def __init__(
        self,
        object_feat_dim: int,
        output_dim: int,
        history_dim: int,

        task_query_dim: int | None = None,

        object_embed_dim: int = 128,
        object_encoder_hidden: tuple[int, ...] = (128,),

        use_transformer: bool = False,
        transformer_layers: int = 1,
        transformer_heads: int = 2,
        transformer_ff_dim: int = 256,

        frame_embed_dim: int = 128,
        gru_hidden_dim: int = 128,
        head_hidden: tuple[int, ...] = (128, 128),

        dropout: float = 0.0,

        condition_pool_on_task: bool = True,
        concat_task_to_head: bool = False,
    ) -> None:
        super().__init__()

        if history_dim < 1:
            raise ValueError(
                f"history_dim must be >= 1, got {history_dim}."
            )

        if use_transformer and object_embed_dim % transformer_heads != 0:
            raise ValueError(
                f"object_embed_dim={object_embed_dim} must be divisible by "
                f"transformer_heads={transformer_heads}."
            )

        if concat_task_to_head and task_query_dim is None:
            raise ValueError(
                "concat_task_to_head=True requires task_query_dim to be set."
            )

        if condition_pool_on_task and task_query_dim is None:
            raise ValueError(
                "condition_pool_on_task=True requires task_query_dim to be set."
            )

        self.object_feat_dim = object_feat_dim
        self.output_dim = output_dim
        self.history_dim = history_dim
        self.task_query_dim = task_query_dim
        self.condition_pool_on_task = condition_pool_on_task
        self.concat_task_to_head = concat_task_to_head

        # --------------------------------------------------------------
        # Object encoder
        # --------------------------------------------------------------

        self.object_encoder = MLP(
            in_dim=object_feat_dim,
            hidden_dims=list(object_encoder_hidden),
            out_dim=object_embed_dim,
            activation=nn.ELU,
            layer_norm=True,
        )

        # --------------------------------------------------------------
        # Optional object Transformer
        # --------------------------------------------------------------

        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=object_embed_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

            self.object_transformer = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=transformer_layers,
            )
        else:
            self.object_transformer = None

        # --------------------------------------------------------------
        # Object pooling
        # --------------------------------------------------------------

        self.object_pool = TaskConditionedAttentionPool(
            object_dim=object_embed_dim,
            task_query_dim=(
                task_query_dim if condition_pool_on_task else None
            ),
            hidden_dim=object_embed_dim,
        )

        # --------------------------------------------------------------
        # Temporal encoder
        #
        # No temporal machinery is needed for H=1.
        # --------------------------------------------------------------

        if history_dim > 1:
            self.frame_proj = nn.Sequential(
                nn.LayerNorm(object_embed_dim),
                nn.Linear(object_embed_dim, frame_embed_dim),
                nn.ELU(),
            )

            self.temporal_encoder = nn.GRU(
                input_size=frame_embed_dim,
                hidden_size=gru_hidden_dim,
                num_layers=1,
                batch_first=True,
            )

            head_in_dim = gru_hidden_dim

        else:
            self.frame_proj = None
            self.temporal_encoder = None

            head_in_dim = object_embed_dim

        # --------------------------------------------------------------
        # Prediction head
        # --------------------------------------------------------------

        if concat_task_to_head:
            assert task_query_dim is not None
            head_in_dim += task_query_dim

        self.prediction_head = MLP(
            in_dim=head_in_dim,
            hidden_dims=list(head_hidden),
            out_dim=output_dim,
            activation=nn.ELU,
            layer_norm=True,
        )

    def _prepare_task_query(
        self,
        task_query: torch.Tensor | None,
        b: int,
        h: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Returns:
            flat_task_query:  [B*H, T] or None
            final_task_query: [B, T] or None
        """

        if self.task_query_dim is None:
            if task_query is not None:
                raise ValueError(
                    "task_query was provided, but task_query_dim=None "
                    "in constructor."
                )

            return None, None

        if task_query is None:
            raise ValueError(
                "task_query must be provided when task_query_dim is set."
            )

        t = self.task_query_dim

        if task_query.ndim == 2:
            if task_query.shape != (b, t):
                raise ValueError(
                    f"Expected task_query shape {(b, t)} or {(b, h, t)}, "
                    f"got {tuple(task_query.shape)}."
                )

            final_task_query = task_query

            flat_task_query = (
                task_query[:, None, :]
                .expand(-1, h, -1)
                .reshape(b * h, t)
            )

        elif task_query.ndim == 3:
            if task_query.shape != (b, h, t):
                raise ValueError(
                    f"Expected task_query shape {(b, h, t)}, "
                    f"got {tuple(task_query.shape)}."
                )

            flat_task_query = task_query.reshape(b * h, t)
            final_task_query = task_query[:, -1]

        else:
            raise ValueError(
                f"Expected task_query [B, T] or [B, H, T], "
                f"got {tuple(task_query.shape)}."
            )

        return flat_task_query, final_task_query

    def forward(
        self,
        obs: torch.Tensor,
        object_mask: torch.Tensor | None = None,
        task_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if obs.ndim != 4:
            raise ValueError(
                f"Expected obs shape [B, H, N, F], got {tuple(obs.shape)}."
            )

        b, h, n, f = obs.shape

        if h != self.history_dim:
            raise ValueError(
                f"Expected history dimension {self.history_dim}, got {h}."
            )

        if f != self.object_feat_dim:
            raise ValueError(
                f"Expected object feature dim {self.object_feat_dim}, got {f}."
            )

        if n == 0:
            raise ValueError("Expected at least one object slot (N > 0).")

        # --------------------------------------------------------------
        # Object mask
        # --------------------------------------------------------------

        flat_mask = None

        if object_mask is not None:
            if object_mask.shape != (b, h, n):
                raise ValueError(
                    f"Expected object_mask shape {(b, h, n)}, "
                    f"got {tuple(object_mask.shape)}."
                )

            flat_mask = object_mask.bool().reshape(b * h, n)

        # --------------------------------------------------------------
        # Task query
        # --------------------------------------------------------------

        flat_task_query, final_task_query = self._prepare_task_query(
            task_query,
            b,
            h,
        )

        # --------------------------------------------------------------
        # Encode objects
        #
        # [B, H, N, F]
        # -> [B*H, N, F]
        # -> [B*H, N, D]
        # --------------------------------------------------------------

        x = obs.reshape(b * h, n, f)
        x = self.object_encoder(x)

        # --------------------------------------------------------------
        # Optional object interaction
        # --------------------------------------------------------------

        if self.object_transformer is not None:
            padding_mask = None

            if flat_mask is not None:
                # Transformer convention:
                # True = ignored / padding.
                padding_mask = ~flat_mask

                # Avoid fully masked Transformer rows.
                # The original flat_mask remains unchanged, so object_pool
                # will still map these rows to exactly zero.
                has_objects = flat_mask.any(dim=1)
                padding_mask[:, 0] &= has_objects

            x = self.object_transformer(
                x,
                src_key_padding_mask=padding_mask,
            )

        # --------------------------------------------------------------
        # Object pooling
        #
        # [B*H, N, D] -> [B*H, D]
        # --------------------------------------------------------------

        frame = self.object_pool(
            x=x,
            mask=flat_mask,
            task_query=flat_task_query,
        )

        # --------------------------------------------------------------
        # History aggregation
        # --------------------------------------------------------------

        if self.history_dim > 1:
            assert self.frame_proj is not None
            assert self.temporal_encoder is not None

            # [B*H, D] -> [B*H, E]
            frame = self.frame_proj(frame)

            # [B*H, E] -> [B, H, E]
            frame = frame.reshape(b, h, -1)

            # [B, H, E] -> [B, G]
            _, hidden = self.temporal_encoder(frame)
            history_embedding = hidden[-1]

        else:
            # H == 1:
            #
            # [B*1, D] is already [B, D], so no frame projection or
            # temporal model is necessary.
            history_embedding = frame

        # --------------------------------------------------------------
        # Optional task concat
        # --------------------------------------------------------------

        if self.concat_task_to_head:
            assert final_task_query is not None

            history_embedding = torch.cat(
                (history_embedding, final_task_query),
                dim=-1,
            )

        return self.prediction_head(history_embedding)


class YoloObjectModel(MLPModel):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims=(256, 256, 256),
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
        object_encoder: dict | None = None,
        **kwargs,
    ) -> None:
        # --------------------------------------------------------------
        # Observation keys
        # --------------------------------------------------------------

        self.object_key = "yolo_objects"
        self.object_mask_key = "yolo_object_mask"
        self.task_key = "task_query"

        # --------------------------------------------------------------
        # Object encoder configuration
        #
        # Depending on how RSL-RL serializes the config, this may arrive
        # either as an ObjectEncoderConfig or as a plain dict.
        # --------------------------------------------------------------

        if object_encoder is None:
            object_encoder = ObjectEncoderConfig()
        elif isinstance(object_encoder, dict):
            object_encoder = ObjectEncoderConfig(**object_encoder)

        # --------------------------------------------------------------
        # Infer dimensions from observations
        #
        # yolo_objects: [B, H, N, F]
        # --------------------------------------------------------------

        if self.object_key not in obs:
            raise ValueError(
                f"Missing required observation '{self.object_key}'."
            )

        object_shape = obs[self.object_key].shape

        if len(object_shape) != 4:
            raise ValueError(
                f"{self.object_key} must be [B, H, N, F], "
                f"got {tuple(object_shape)}."
            )

        batch_dim, history_dim, _, object_feat_dim = object_shape

        if history_dim < 1:
            raise ValueError(
                f"{self.object_key} must have H >= 1, "
                f"got H={history_dim}."
            )

        # --------------------------------------------------------------
        # Infer task-query dimension
        # --------------------------------------------------------------

        if self.task_key in obs:
            task_shape = obs[self.task_key].shape

            if len(task_shape) != 2:
                raise ValueError(
                    f"{self.task_key} must be [B, T], "
                    f"got {tuple(task_shape)}."
                )

            if task_shape[0] != batch_dim:
                raise ValueError(
                    f"Batch mismatch: {self.object_key} has "
                    f"B={batch_dim}, but {self.task_key} has "
                    f"B={task_shape[0]}."
                )

            self.use_task_query = True
            task_query_dim = task_shape[-1]
        else:
            self.use_task_query = False
            task_query_dim = None

        # --------------------------------------------------------------
        # Object branch latent size
        #
        # Must be set before super().__init__() because MLPModel may call
        # the overridden _get_latent_dim() during construction.
        # --------------------------------------------------------------

        self.object_latent_dim = object_encoder.latent_dim

        # --------------------------------------------------------------
        # Initialize base MLP branch / nn.Module
        # --------------------------------------------------------------

        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
            **kwargs,
        )

        # --------------------------------------------------------------
        # Construct object branch
        #
        # Remove latent_dim because ObjectEncoderModel calls this
        # parameter output_dim.
        # --------------------------------------------------------------

        object_cfg = asdict(object_encoder)
        object_cfg.pop("latent_dim")

        self.object_model = ObjectEncoderModel(
            object_feat_dim=object_feat_dim,
            history_dim=history_dim,
            task_query_dim=task_query_dim,
            output_dim=self.object_latent_dim,
            **object_cfg,
        )

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        latent_1d = super().get_latent(obs)

        latent_obj = self.object_model(
            obs=obs[self.object_key],
            object_mask=obs.get(self.object_mask_key, None),
            task_query=(
                obs[self.task_key]
                if self.use_task_query
                else None
            ),
        )

        return torch.cat(
            (latent_1d, latent_obj),
            dim=-1,
        )

    def _get_obs_dim(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[list[str], int]:
        active = obs_groups[obs_set]

        excluded_keys = {
            self.object_key,
            self.object_mask_key,
            self.task_key,
        }

        obs_groups_1d = []
        obs_dim_1d = 0

        for key in active:
            if key in excluded_keys:
                continue

            shape = obs[key].shape

            if len(shape) != 2:
                raise ValueError(
                    f"Expected 1D observation '{key}' to have shape "
                    f"[B, C], got {tuple(shape)}."
                )

            obs_dim_1d += shape[-1]
            obs_groups_1d.append(key)

        return obs_groups_1d, obs_dim_1d

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.object_latent_dim
