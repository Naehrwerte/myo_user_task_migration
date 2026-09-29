from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

from mjlab.sensor import CameraSensorCfg, CameraSensor, CameraSensorData

import mujoco
import torch
import torch.cuda.nvtx as nvtx
import torch.nn.functional as F
from torchvision.ops import batched_nms, roi_pool
import mujoco_warp as mjwarp
from ultralytics import YOLO

VALID_DEPTH_MODES = ("average", None)
VALID_COLOR_MODES = ("rgb", "hue", None)

@dataclass
class YoloSensorCfg(CameraSensorCfg):
    model: str = "yolo11n.pt"

    use_class: bool = True
    use_confidence: bool = True

    color: Literal["rgb", "hue"] | None = None

    depth: Literal["average"] | None = None
    depth_min: float = 0.1
    depth_cutoff: float = 1.0

    half: bool = True
    play: bool = False

    nms_top_k: int | None = 16
    nms_pre_top_k: int | None = 64
    nms_conf_threshold: float = 0.0
    nms_iou_threshold: float = 0.7
    nms_class_agnostic: bool = True

    perfect_detection: bool = False

    def __post_init__(self) -> None:
        assert self.depth in VALID_DEPTH_MODES, (
            f"Invalid depth mode: {self.depth} "
            f"(allowed: {VALID_DEPTH_MODES})"
        )
        assert self.color in VALID_COLOR_MODES, (
            f"Invalid color mode: {self.color} "
            f"(allowed: {VALID_COLOR_MODES})"
        )
        assert self.depth_cutoff > self.depth_min, (
            f"Invalid depth cutoff"
        )
        super().__post_init__()

    def build(self) -> YoloSensor:
        return YoloSensor(self)


@dataclass
class YoloSensorData(CameraSensorData):
    objects: torch.Tensor | None = None
    object_mask: torch.Tensor | None = None

class YoloSensor(CameraSensor):
    requires_sensor_context = True

    def __init__(self, cfg: YoloSensorCfg):
        super().__init__(cfg)
        self.cfg = cfg

        self._device: str | None = None
        self._dtype: torch.dtype = torch.float16 if cfg.half else torch.float32
        self._infer_module: torch.nn.Module = YOLO(cfg.model, "detect").model

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        super().initialize(mj_model, model, data, device)
        self._device = device

        if device == 'cpu':
            self._dtype = torch.float32

        self._infer_module = (
            self._infer_module
            .eval()
            .requires_grad_(False)
            .to(
                device=self._device,
                dtype=self._dtype,
                memory_format=torch.channels_last
            )
        )

        self._infer_module = torch.compile(
            self._infer_module,
            fullgraph=False,
            options={
                "triton.cudagraphs": False,
            },
        )

    @torch.inference_mode()
    def _compute_data(self) -> YoloSensorData:
        camera_data = super()._compute_data()

        rgb_u8 = camera_data.rgb       # [B, H, W, 3]
        depth_raw = camera_data.depth  # [B, H, W, 1]

        assert rgb_u8 is not None, "YoloSensor requires RGB data"

        if self.cfg.perfect_detection:
            return YoloSensorData(
                rgb=rgb_u8,
                depth=depth_raw,
            )

        nvtx.range_push("yolo.preprocess")
        x = rgb_u8.permute(0, 3, 1, 2)
        x = x.to(device=self._device, dtype=self._dtype, non_blocking=True)
        x = x.contiguous(memory_format=torch.channels_last)
        x = x.div_(255.0)

        if self.cfg.color is not None:
            pixels = rgb_u8.to(device=self._device, dtype=self._dtype, non_blocking=True)
            pixels = pixels.div_(255.0)
        else:
            pixels = None

        assert depth_raw is not None, "YoloSensor requires depth data"
        depth = depth_raw.to(device=self._device, dtype=self._dtype, non_blocking=True)
        depth = depth.clamp_(min=self.cfg.depth_min, max=self.cfg.depth_cutoff)
        depth = depth.sub_(self.cfg.depth_min)
        depth = depth.div_(self.cfg.depth_cutoff - self.cfg.depth_min)
        depth = depth.clamp_(0.0, 1.0)
        nvtx.range_pop()

        nvtx.range_push("yolo.infer")
        raw = self._infer_module(x)[0] # [B, C, N]
        nvtx.range_pop()

        nvtx.range_push("yolo.nms")
        raw, mask = self._prefilter_raw_boxes_nms_batched(raw)
        nvtx.range_pop()

        nvtx.range_push("yolo.decode")
        objs = self._decode_and_augment_batched(
            raw=raw,
            pixels=pixels,
            depth=depth,
            draw_out=rgb_u8 if self.cfg.play else None,
        )
        nvtx.range_pop()

        return YoloSensorData(
            objects=objs,
            object_mask=mask,
            rgb=rgb_u8,
            depth=depth_raw,
        )

    def _prefilter_raw_boxes_nms_batched(
        self,
        raw: torch.Tensor   # [B, C, N]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Loop-free batched NMS prefilter.

        Returns:
            raw_filtered: [B, C, K]
            valid_mask:   [B, K], True = valid box

        Invalid padded slots are zeroed.
        """
        b, ctot, n = raw.shape

        top_k = self.cfg.nms_top_k
        if top_k is None or top_k <= 0:
            return raw, torch.ones((b, n), dtype=torch.bool, device=raw.device)

        top_k = min(int(top_k), n)

        num_classes = ctot - 4

        # ------------------------------------------------------------
        # Optional per-env confidence prefilter before NMS
        # [B, C, N] -> [B, C, pre_nms_topk]
        # ------------------------------------------------------------

        cls_scores = raw[:, 4:, :]              # [B, C_cls, N]
        conf, _ = torch.max(cls_scores, dim=1)  # [B, N]

        if self.cfg.nms_pre_top_k is not None and (0 < self.cfg.nms_pre_top_k < n):
            pre_idx = torch.topk(
                conf,
                k=self.cfg.nms_pre_top_k,
                dim=1,
                largest=True,
                sorted=True,
            ).indices  # [B, P]

            gather_idx = pre_idx[:, None, :].expand(-1, ctot, -1)
            raw = torch.gather(raw, dim=2, index=gather_idx)  # [B, C, P]

            _, _, n = raw.shape

        # Recompute after optional prefilter.
        boxes_cxcywh = raw[:, :4, :]            # [B, 4, N]
        cls_scores = raw[:, 4:, :]              # [B, C_cls, N]
        conf, cls_ids = torch.max(cls_scores, dim=1)  # [B, N], [B, N]

        cx = boxes_cxcywh[:, 0, :]
        cy = boxes_cxcywh[:, 1, :]
        bw = boxes_cxcywh[:, 2, :]
        bh = boxes_cxcywh[:, 3, :]

        x1 = (cx - 0.5 * bw).clamp(0.0, float(self.cfg.width))
        y1 = (cy - 0.5 * bh).clamp(0.0, float(self.cfg.height))
        x2 = (cx + 0.5 * bw).clamp(0.0, float(self.cfg.width))
        y2 = (cy + 0.5 * bh).clamp(0.0, float(self.cfg.height))

        # [B, N, 4]
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        valid = (
            torch.isfinite(conf)
            & torch.isfinite(boxes).all(dim=-1)
            & (conf >= self.cfg.nms_conf_threshold)
            & (x2 > x1)
            & (y2 > y1)
        )  # [B, N]

        # ------------------------------------------------------------
        # Flatten all envs into one NMS call
        # ------------------------------------------------------------

        flat_boxes = boxes.reshape(b * n, 4)
        flat_scores = conf.reshape(b * n)
        flat_cls_ids = cls_ids.reshape(b * n)
        flat_valid = valid.reshape(b * n)

        flat_valid_idx = torch.nonzero(flat_valid, as_tuple=False).squeeze(1)

        selected_idx = torch.zeros((b, top_k), dtype=torch.long, device=raw.device)
        selected_mask = torch.zeros((b, top_k), dtype=torch.bool, device=raw.device)

        if flat_valid_idx.numel() > 0:
            valid_boxes = flat_boxes[flat_valid_idx]
            valid_scores = flat_scores[flat_valid_idx]
            valid_cls_ids = flat_cls_ids[flat_valid_idx]

            # batch id for each flattened box.
            # flat index = batch_id * n + object_id
            valid_batch_ids = flat_valid_idx // n

            if self.cfg.nms_class_agnostic:
                # Suppress only within the same batch item.
                nms_groups = valid_batch_ids
            else:
                # Suppress only within same batch item AND same predicted class.
                # This prevents boxes from different envs suppressing each other.
                nms_groups = valid_batch_ids * num_classes + valid_cls_ids

            keep_local = batched_nms(
                boxes=valid_boxes,
                scores=valid_scores,
                idxs=nms_groups,
                iou_threshold=self.cfg.nms_iou_threshold,
            )

            keep_flat_idx = flat_valid_idx[keep_local]  # indices into [B*N]

            keep_mask_flat = torch.zeros(
                b * n,
                dtype=torch.bool,
                device=raw.device,
            )
            keep_mask_flat[keep_flat_idx] = True

            keep_mask = keep_mask_flat.reshape(b, n)  # [B, N]

            # Keep top-K NMS survivors per batch by confidence.
            neg_inf = torch.full_like(conf, -torch.inf)
            kept_scores = torch.where(keep_mask, conf, neg_inf)  # [B, N]

            topk_scores, selected_idx = torch.topk(
                kept_scores,
                k=top_k,
                dim=1,
                largest=True,
                sorted=True,
            )  # [B, K]

            selected_mask = torch.isfinite(topk_scores)

        # ------------------------------------------------------------
        # Gather raw boxes to fixed [B, C, K]
        # ------------------------------------------------------------

        gather_idx = selected_idx[:, None, :].expand(-1, ctot, -1)
        raw_filtered = torch.gather(raw, dim=2, index=gather_idx)  # [B, C, K]

        raw_filtered = raw_filtered * selected_mask[:, None, :].to(raw_filtered.dtype)

        return raw_filtered, selected_mask

    def _decode_and_augment_batched(
        self,
        raw: torch.Tensor,                      # [B, C, N]
        pixels: torch.Tensor | None,            # [B, H, W, 3], float16/32 in [0,1]
        depth: torch.Tensor | None,             # [B, H, W, 1], float16/32 in [0,1]
        draw_out: torch.Tensor | None = None,   # [B, H, W, 3], uint8
    ) -> torch.Tensor:
        _, ctot, _ = raw.shape

        num_classes = ctot - 4
        if num_classes <= 0:
            raise ValueError(f"Unexpected output channels C={ctot}; expected >= 4 + classes")

        nvtx.range_push("decode.box_geometry")

        boxes_cxcywh = raw[:, :4, :]   # [B, 4, N]
        cls_scores = raw[:, 4:, :]     # [B, K, N]

        conf, cls_ids = torch.max(cls_scores, dim=1)  # [B, N], [B, N]

        cls_ids = cls_ids.to(torch.float32)

        if num_classes > 1:
            class_norm = cls_ids / float(num_classes - 1)
        else:
            class_norm = torch.zeros_like(cls_ids)

        denom_x = float(self.cfg.width)
        denom_y = float(self.cfg.height)

        cx = boxes_cxcywh[:, 0, :]   # [B, N]
        cy = boxes_cxcywh[:, 1, :]
        bw = boxes_cxcywh[:, 2, :]
        bh = boxes_cxcywh[:, 3, :]

        ncx = (cx / denom_x).clamp(0.0, 1.0)
        ncy = (cy / denom_y).clamp(0.0, 1.0)
        nbw = (bw / denom_x).clamp(0.0, 1.0)
        nbh = (bh / denom_y).clamp(0.0, 1.0)

        x1 = cx - 0.5 * bw
        y1 = cy - 0.5 * bh
        x2 = cx + 0.5 * bw
        y2 = cy + 0.5 * bh

        x1_raw = torch.floor(x1).to(torch.int64).clamp(0, self.cfg.width)
        y1_raw = torch.floor(y1).to(torch.int64).clamp(0, self.cfg.height)
        x2_raw = torch.ceil(x2).to(torch.int64).clamp(0, self.cfg.width)
        y2_raw = torch.ceil(y2).to(torch.int64).clamp(0, self.cfg.height)

        xa = torch.minimum(x1_raw, x2_raw)
        xb = torch.maximum(x1_raw, x2_raw)
        ya = torch.minimum(y1_raw, y2_raw)
        yb = torch.maximum(y1_raw, y2_raw)

        conf_out = torch.where(
            torch.isfinite(conf), conf, torch.zeros_like(conf)
        ).to(torch.float32)

        if draw_out is not None:
            nvtx.range_push("decode.draw")
            self._draw_topk_boxes_from_coords_inplace(
                rgb_out=draw_out,
                conf=conf_out,
                x1=x1_raw,
                y1=y1_raw,
                x2=x2_raw,
                y2=y2_raw,
                topk=self.cfg.nms_top_k,
            )
            nvtx.range_pop()

        cols = [
            ncx[..., None],         # [B, N, 1]
            ncy[..., None],
            nbw[..., None],
            nbh[..., None],
        ]

        if self.cfg.use_class:
            cols.append(class_norm[..., None])

        if self.cfg.use_confidence:
            cols.append(conf_out[..., None])

        nvtx.range_pop()

        if self.cfg.depth is not None:
            nvtx.range_push("decode.depth")
            depth_feats = self._extract_depth_batched(depth, xa, ya, xb, yb)  # [B, N, 3]
            nvtx.range_pop()
            cols.append(depth_feats)

        if self.cfg.color is not None:
            nvtx.range_push("decode.color")
            mc = self._extract_color_batched(pixels, depth, xa, ya, xb, yb)  # [B, N, Cc]
            nvtx.range_pop()
            cols.append(mc)

        return torch.cat(cols, dim=-1).to(torch.float32)  # [B, N, F]

    def _integral_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., H, W]
        returns SAT: [..., H+1, W+1]
        """
        sat = x.cumsum(dim=-2).cumsum(dim=-1)
        sat = F.pad(sat, (1, 0, 1, 0), mode="constant", value=0.0)
        return sat

    def _rect_sum_from_sat_batched(
        self,
        sat: torch.Tensor,   # [B, H+1, W+1]
        x1: torch.Tensor,    # [B, N]
        y1: torch.Tensor,    # [B, N]
        x2: torch.Tensor,    # [B, N]
        y2: torch.Tensor,    # [B, N]
    ) -> torch.Tensor:
        # Use cached batch indices instead of allocating torch.arange every call.
        batch_idx = torch.arange(sat.shape[0], device=self._device)[:, None]

        return (
            sat[batch_idx, y2, x2]
            - sat[batch_idx, y1, x2]
            - sat[batch_idx, y2, x1]
            + sat[batch_idx, y1, x1]
        )

    def _extract_depth_batched(
        self,
        depth: torch.Tensor,   # [B, H, W, 1], already float32 in [0,1]
        x1i: torch.Tensor,     # [B, N]
        y1i: torch.Tensor,     # [B, N]
        x2i: torch.Tensor,     # [B, N]
        y2i: torch.Tensor,     # [B, N]
        grid_size: int = 4,
        zero_eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        Approximate per-box depth statistics using a fixed sampling grid.

        Samples with depth <= zero_eps are treated as invalid background.

        Returns:
            [B, N, 3] = [min_depth, max_depth, mean_depth]

        Boxes without valid depth samples return zeros.
        """
        d2 = depth[..., 0].to(torch.float32)  # [B, H, W]

        batch_size, height, width = d2.shape
        _, num_boxes = x1i.shape
        device = d2.device

        # Sample bin centers. x2/y2 are exclusive.
        t = (
            torch.arange(grid_size, device=device, dtype=torch.float32) + 0.5
        ) / grid_size

        x1f = x1i.to(torch.float32)
        y1f = y1i.to(torch.float32)
        box_width = (x2i - x1i).to(torch.float32)
        box_height = (y2i - y1i).to(torch.float32)

        xs = x1f[..., None] + box_width[..., None] * t
        ys = y1f[..., None] + box_height[..., None] * t

        xg = xs[..., None, :].expand(-1, -1, grid_size, -1)
        yg = ys[..., :, None].expand(-1, -1, -1, grid_size)

        xi = xg.floor().to(torch.int64).clamp(0, width - 1)
        yi = yg.floor().to(torch.int64).clamp(0, height - 1)

        batch_idx = torch.arange(
            batch_size,
            device=device,
        ).view(batch_size, 1, 1, 1)

        samples = d2[batch_idx, yi, xi].reshape(
            batch_size,
            num_boxes,
            grid_size * grid_size,
        )

        valid_box = (x2i > x1i) & (y2i > y1i)

        valid = (
            torch.isfinite(samples)
            & (samples > zero_eps)
            & valid_box[..., None]
        )

        count = valid.sum(dim=-1)
        has_valid = count > 0

        min_depth = torch.where(
            valid,
            samples,
            torch.inf,
        ).amin(dim=-1)

        max_depth = torch.where(
            valid,
            samples,
            -torch.inf,
        ).amax(dim=-1)

        depth_sum = torch.where(
            valid,
            samples,
            0.0,
        ).sum(dim=-1)

        mean_depth = depth_sum / count.clamp_min(1).to(depth_sum.dtype)

        min_depth = torch.where(has_valid, min_depth, 0.0)
        max_depth = torch.where(has_valid, max_depth, 0.0)
        mean_depth = torch.where(has_valid, mean_depth, 0.0)

        return torch.stack(
            [min_depth, max_depth, mean_depth],
            dim=-1,
        )

    def _extract_color_batched(
        self,
        pixels: torch.Tensor,  # [B, H, W, 3]
        depth: torch.Tensor,   # [B, H, W, 1]
        x1i: torch.Tensor,     # [B, N]
        y1i: torch.Tensor,
        x2i: torch.Tensor,
        y2i: torch.Tensor,
        zero_eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        Compute the exact mean color over pixels with valid depth.

        RGB does not determine validity. A pixel is valid only when its depth
        is finite and greater than zero_eps.

        Boxes containing no valid pixels return zero.
        """
        rgb = pixels.to(torch.float32)
        d2 = depth[..., 0].to(torch.float32)

        valid = torch.isfinite(d2) & (d2 > zero_eps)
        valid_float = valid.to(rgb.dtype)

        # Prevent NaN/Inf RGB values from contaminating the integral image.
        rgb = torch.where(
            torch.isfinite(rgb),
            rgb,
            torch.zeros_like(rgb),
        )

        masked_rgb = rgb * valid_float[..., None]

        # [B, H, W, 3] -> [B, 3, H, W]
        sat_rgb = self._integral_image(
            masked_rgb.permute(0, 3, 1, 2)
        )  # [B, 3, H+1, W+1]

        sat_count = self._integral_image(
            valid_float
        )  # [B, H+1, W+1]

        batch_idx = torch.arange(
            rgb.shape[0],
            device=rgb.device,
        )[:, None]

        # Due to advanced indexing, these have shape [B, N, 3].
        rgb_sum = (
            sat_rgb[batch_idx, :, y2i, x2i]
            - sat_rgb[batch_idx, :, y1i, x2i]
            - sat_rgb[batch_idx, :, y2i, x1i]
            + sat_rgb[batch_idx, :, y1i, x1i]
        )  # [B, N, 3]

        valid_count = self._rect_sum_from_sat_batched(
            sat_count,
            x1i,
            y1i,
            x2i,
            y2i,
        )  # [B, N]

        mean_rgb = (
            rgb_sum
            / valid_count.clamp_min(1.0)[..., None]
        )  # [B, N, 3]

        mean_rgb = torch.where(
            (valid_count > 0.0)[..., None],
            mean_rgb,
            torch.zeros_like(mean_rgb),
        )

        if self.cfg.color != "hue":
            return mean_rgb

        r, g, b = mean_rgb.unbind(dim=-1)

        cmax = mean_rgb.amax(dim=-1)
        cmin = mean_rgb.amin(dim=-1)
        delta = cmax - cmin
        delta_safe = delta.clamp_min(1e-6)

        h_r = torch.remainder((g - b) / delta_safe, 6.0)
        h_g = ((b - r) / delta_safe) + 2.0
        h_b = ((r - g) / delta_safe) + 4.0

        h6 = torch.where(
            delta <= 1e-6,
            torch.zeros_like(delta),
            torch.where(
                cmax == r,
                h_r,
                torch.where(cmax == g, h_g, h_b),
            ),
        )

        hue = (h6 / 6.0).clamp(0.0, 1.0)

        hue = torch.where(
            valid_count > 0.0,
            hue,
            torch.zeros_like(hue),
        )

        return hue[..., None]

    def _draw_topk_boxes_from_coords_inplace(
        self,
        rgb_out: torch.Tensor,   # [B, H, W, 3], usually uint8
        conf: torch.Tensor,      # [B, N]
        x1: torch.Tensor,        # [B, N]
        y1: torch.Tensor,        # [B, N]
        x2: torch.Tensor,        # [B, N]
        y2: torch.Tensor,        # [B, N]
        topk: int = 16,
    ) -> None:
        """
        Draw top-k boxes directly into rgb_out in-place.

        Reuses already computed confidence and pixel-space corners from decode,
        so no second box decode pass is needed.
        """
        if rgb_out.ndim != 4 or rgb_out.shape[-1] != 3:
            raise ValueError(f"Expected rgb_out shape [B,H,W,3], got {tuple(rgb_out.shape)}")

        b, h, w, _ = rgb_out.shape
        k = min(topk, conf.shape[1])
        if k <= 0:
            return

        topk_idx = torch.topk(conf, k=k, dim=1).indices  # [B, K]
        batch_idx = torch.arange(b, device=self._device)[:, None]

        x1_top = x1[batch_idx, topk_idx]
        y1_top = y1[batch_idx, topk_idx]
        x2_top = x2[batch_idx, topk_idx]
        y2_top = y2[batch_idx, topk_idx]

        if rgb_out.dtype == torch.uint8:
            color = torch.tensor([0, 255, 255], dtype=rgb_out.dtype, device=rgb_out.device)
        else:
            color = torch.tensor([0.0, 1.0, 1.0], dtype=rgb_out.dtype, device=rgb_out.device)

        # Debug/play path: small Python loops are acceptable.
        for bi in range(b):
            for ki in range(k):
                xa = int(x1_top[bi, ki].item())
                ya = int(y1_top[bi, ki].item())
                xb = int(x2_top[bi, ki].item())
                yb = int(y2_top[bi, ki].item())

                xa = max(0, min(xa, w - 1))
                xb = max(0, min(xb, w - 1))
                ya = max(0, min(ya, h - 1))
                yb = max(0, min(yb, h - 1))

                if xb < xa or yb < ya:
                    continue

                rgb_out[bi, ya, xa:xb + 1, :] = color
                rgb_out[bi, yb, xa:xb + 1, :] = color
                rgb_out[bi, ya:yb + 1, xa, :] = color
                rgb_out[bi, ya:yb + 1, xb, :] = color
