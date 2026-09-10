from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import EventTermCfg, ManagerTermBase
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from myo_core.common.sequential_task import SequentialTaskLogic, TaskEntity

from mjlab.viewer.debug_visualizer import DebugVisualizer

_RGBA_CURRENT = (0.0, 1.0, 0.0, 1.0)
_RGBA_TODO = (1.0, 0.0, 0.0, 0.5)
_RGBA_DONE = (0.0, 0.0, 1.0, 0.5)
_OVERLAY_SCALE = 1.02


class NumpadTaskLogic(SequentialTaskLogic):
    """Sequential task logic with a randomized press sequence over a fixed pool.
    """

    def _resolve_num_phases(self, asset: TaskEntity) -> int:
        """mostly validates the input depending on context"""
        seq_len = int(asset.task_cfg.sequence_length)

        if seq_len < 1:
            raise ValueError(f"sequence_length must be >= 1, got {seq_len}")
        if not asset.task_cfg.sample_with_replacement and seq_len > asset.num_targets:
            raise ValueError(
                "sequence_length cannot exceed the button pool size when sampling "
                f"without replacement ({seq_len} > {asset.num_targets})"
            )

        return seq_len

    def _sample_phase_targets(self, asset: TaskEntity, env_ids: torch.Tensor) -> torch.Tensor:
        n = env_ids.shape[0]

        if asset.task_cfg.sample_with_replacement:
            return torch.randint(
                0, asset.num_targets, (n, asset.num_phases), device=self._device
            )

        return torch.argsort(
            torch.rand(n, asset.num_targets, device=self._device), dim=1
        )[:, : asset.num_phases]


def _target_state_colors(asset: TaskEntity) -> torch.Tensor:
    """
    Each pool button is green/red/blue depending on whether it is the current
    target, still to be pressed, or already done — its configured color is not
    used. A button may occur several times in the (with-replacement) sequence; it
    counts as "todo" while it still has an occurrence at or after the current
    phase, otherwise as "done". Buttons absent from the sequence are shown as done.
    """
    device = asset.phase_targets.device
    num_envs = asset.phase_targets.shape[0]
    num_targets = asset.num_targets

    buttons = torch.arange(num_targets, device=device)          # [P]
    phase_idx = torch.arange(asset.num_phases, device=device)   # [S]

    # For every (env, button): does the button occur, and at which last phase?
    occ = asset.phase_targets[:, :, None] == buttons[None, None, :]  # [E, S, P]
    appears = occ.any(dim=1)                                     # [E, P]
    last_occ = torch.where(occ, phase_idx[None, :, None], torch.full_like(occ, -1, dtype=torch.long))
    last_occ = last_occ.max(dim=1).values                       # [E, P]

    current_phase = asset.current_phase[:, None]                # [E, 1]
    is_current = buttons[None, :] == asset.current_target_id[:, None]  # [E, P]
    is_todo = appears & (last_occ >= current_phase) & ~is_current

    current = torch.tensor(_RGBA_CURRENT, dtype=torch.float32, device=device)
    todo = torch.tensor(_RGBA_TODO, dtype=torch.float32, device=device)
    done = torch.tensor(_RGBA_DONE, dtype=torch.float32, device=device)

    colors = done.expand(num_envs, num_targets, 4).clone()      # default: done/inactive
    colors[is_todo] = todo
    colors[is_current] = current                                # highest priority
    return colors


@requires_model_fields("geom_rgba")
def np_color_targets_by_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Coloring variant 1: recolor the button geoms in the model (per environment).

    Downside: writing ``geom_rgba`` changes the viewer's appearance fingerprint,
    which forces viser to rebuild its mesh handles every time a color changes ->
    noticeable viewer lag. Use :class:`NumpadTargetBoxOverlay` (variant 2) to
    avoid touching the model.
    """
    asset: TaskEntity = env.scene[asset_cfg.name]
    env.sim.model.geom_rgba[:, asset.target_size_ids, :] = _target_state_colors(asset)


class NumpadTargetBoxOverlay(ManagerTermBase):
    """Coloring variant 2: draw a colored box over each button as debug geometry.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: TaskEntity = env.scene[asset_cfg.name]

    def __call__(self, env: ManagerBasedRlEnv, env_ids: None, asset_cfg: SceneEntityCfg) -> None:
        del env, env_ids, asset_cfg

    def generate_new_boxes_as_overlay(self, visualizer: "DebugVisualizer") -> None:
        asset = self.asset
        geom_ids = asset.target_size_ids                        # [P]
        colors = _target_state_colors(asset)                    # [E, P, 4]

        raw = asset.data.data                                   # batched sim data
        geom_size = asset.data.model.geom_size                  # [E, ngeom, 3]

        for env_idx in visualizer.get_env_indices(asset.phase_targets.shape[0]):
            centers = raw.geom_xpos[env_idx][geom_ids]          # [P, 3]
            mats = raw.geom_xmat[env_idx][geom_ids]             # [P, 3, 3] / [P, 9]
            sizes = geom_size[env_idx][geom_ids] * _OVERLAY_SCALE  # half-extents
            env_colors = colors[env_idx]                        # [P, 4]

            for i in range(geom_ids.shape[0]):
                r, g, b = env_colors[i, :3].tolist()
                visualizer.add_box(
                    center=centers[i],
                    size=sizes[i],
                    mat=mats[i],
                    color=(r, g, b, 1.0),
                )
