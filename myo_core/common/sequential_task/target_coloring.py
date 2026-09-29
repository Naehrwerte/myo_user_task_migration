from __future__ import annotations

import mujoco
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import EventTermCfg, ManagerTermBase
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.viewer.debug_visualizer import DebugVisualizer

from .sequential_task_entity import TaskEntity

_RGBA_CURRENT = (0.0, 1.0, 0.0, 1.0)
_RGBA_TODO = (1.0, 0.0, 0.0, 0.5)
_RGBA_DONE = (0.0, 0.0, 1.0, 0.5)
_OVERLAY_SCALE = 1.02


def target_state_colors(asset: TaskEntity) -> torch.Tensor:
    """RGBA per target, [num_envs, num_targets, 4].

    Each target is green/red/blue depending on whether it is the current target, still
    to be completed, or already done — its configured color is not used. A target may
    occur several times in the (with-replacement) sequence; it counts as "todo" while it
    still has an occurrence at or after the current phase, otherwise as "done". Targets
    absent from the sequence are shown as done.
    """
    device = asset.phase_targets.device
    num_envs = asset.phase_targets.shape[0]
    num_targets = asset.num_targets

    targets = torch.arange(num_targets, device=device)          # [P]
    phase_idx = torch.arange(asset.num_phases, device=device)   # [S]

    # For every (env, target): does the target occur, and at which last phase?
    occ = asset.phase_targets[:, :, None] == targets[None, None, :]  # [E, S, P]
    appears = occ.any(dim=1)                                     # [E, P]
    last_occ = torch.where(occ, phase_idx[None, :, None], torch.full_like(occ, -1, dtype=torch.long))
    last_occ = last_occ.max(dim=1).values                       # [E, P]

    current_phase = asset.current_phase[:, None]                # [E, 1]
    is_current = targets[None, :] == asset.current_target_id[:, None]  # [E, P]
    is_todo = appears & (last_occ >= current_phase) & ~is_current

    current = torch.tensor(_RGBA_CURRENT, dtype=torch.float32, device=device)
    todo = torch.tensor(_RGBA_TODO, dtype=torch.float32, device=device)
    done = torch.tensor(_RGBA_DONE, dtype=torch.float32, device=device)

    colors = done.expand(num_envs, num_targets, 4).clone()      # default: done/inactive
    colors[is_todo] = todo
    colors[is_current] = current                                # highest priority
    return colors


@requires_model_fields("geom_rgba")
def color_targets_by_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Coloring variant 1: recolor the visible target geoms in the model (per environment).

    Downside: writing ``geom_rgba`` changes the viewer's appearance fingerprint,
    which forces viser to rebuild its mesh handles every time a color changes ->
    noticeable viewer lag. Use :class:`TargetStateOverlay` (variant 2) to
    avoid touching the model.
    """
    asset: TaskEntity = env.scene[asset_cfg.name]
    colors = target_state_colors(asset)                         # [E, P, 4]
    env.sim.model.geom_rgba[:, asset.target_visual_geom_ids, :] = colors[:, asset.target_visual_geom_target]


class TargetStateOverlay(ManagerTermBase):
    """Coloring variant 2: draw a colored box/sphere over every visible target geom as
    debug geometry (no model change, no lag).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: TaskEntity = env.scene[asset_cfg.name]

        geom_types = env.sim.mj_model.geom_type[self.asset.target_visual_geom_ids.cpu().numpy()]
        supported = (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_SPHERE)
        if any(t not in supported for t in geom_types):
            raise ValueError("TargetStateOverlay only supports box and sphere targets")
        self._is_sphere = [t == mujoco.mjtGeom.mjGEOM_SPHERE for t in geom_types]

    def __call__(self, env: ManagerBasedRlEnv, env_ids: None, asset_cfg: SceneEntityCfg) -> None:
        del env, env_ids, asset_cfg

    def debug_vis(self, visualizer: "DebugVisualizer") -> None:
        """Called by mjlab's EventManager, which looks up class-based event terms by
        the method name ``debug_vis`` -- do not rename even if its a sub-optimal description of its purpose.
        """
        asset = self.asset
        # Visible target geoms (one per target, or the frame parts of pressable buttons).
        geom_ids = asset.target_visual_geom_ids                 # [G]
        colors = target_state_colors(asset)[:, asset.target_visual_geom_target]  # [E, G, 4]

        raw = asset.data.data                                   # batched sim data
        geom_size = asset.data.model.geom_size                  # [E, ngeom, 3]

        for env_idx in visualizer.get_env_indices(asset.phase_targets.shape[0]):
            centers = raw.geom_xpos[env_idx][geom_ids]          # [G, 3]
            mats = raw.geom_xmat[env_idx][geom_ids]             # [G, 3, 3] / [G, 9]
            sizes = geom_size[env_idx][geom_ids] * _OVERLAY_SCALE  # half-extents / radius
            env_colors = colors[env_idx]                        # [G, 4]

            for i in range(geom_ids.shape[0]):
                r, g, b = env_colors[i, :3].tolist()
                if self._is_sphere[i]:
                    visualizer.add_sphere(
                        center=centers[i],
                        radius=float(sizes[i, 0]),
                        color=(r, g, b, 1.0),
                    )
                else:
                    visualizer.add_box(
                        center=centers[i],
                        size=sizes[i],
                        mat=mats[i],
                        color=(r, g, b, 1.0),
                    )
