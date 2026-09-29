from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import re
import mujoco
import torch

import torch.nn.functional as F
from torchvision.ops import roi_pool

from mjlab.envs import ManagerBasedRlEnv
from mjlab.entity import EntityCfg, Entity
from mjlab.sensor import CameraSensor, CameraSensorData
from mjlab.managers import ManagerTermBase, ManagerTermBaseCfg


def perfect_detection_select(
    env: ManagerBasedRlEnv,
    asset_name: str,
    field: Literal["objects", "object_mask", "shuffled_objects", "shuffled_object_mask"] = "objects"
) -> torch.Tensor:
    asset: PerfectDetectorEntity = env.scene[asset_name]
    return getattr(asset, field, None)


@dataclass
class PerfectDetectorEntityCfg(EntityCfg):
    def build(self) -> PerfectDetectorEntity:
        return PerfectDetectorEntity(self)


class PerfectDetectorEntity(Entity):
    captured_geoms: dict[str, int]

    objects: torch.Tensor                # [E, O, 10], float32
    object_mask: torch.Tensor            # [E, O], bool

    shuffled_objects: torch.Tensor       # [E, O, 10], float32
    shuffled_object_mask: torch.Tensor   # [E, O], bool

    def __init__(self, cfg: PerfectDetectorEntityCfg):
        super().__init__(cfg)


class PerfectDetector(ManagerTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)

        asset: PerfectDetectorEntity = env.scene[cfg.params['asset_name']]
        self._asset = asset

        self._model = env.sim.model
        self._data = env.sim.data
        self._sensor: CameraSensor = self._env.scene[cfg.params['sensor_name']]
        self._play = bool(cfg.params['play'])

        self._captured_geoms: dict[str, int] = {}

        regex = re.compile(cfg.params['capture_geoms'])

        for geom_id in range(env.sim.mj_model.ngeom):
            name = mujoco.mj_id2name(
                env.sim.mj_model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_id,
            )
            if name is not None and regex.fullmatch(name):
                self._captured_geoms[name] = geom_id

        self._geom_ids = torch.tensor(
            tuple(self._captured_geoms.values()),
            device=env.device,
            dtype=torch.long,
        )

        E = env.num_envs
        G = self._geom_ids.numel()
        device = env.device

        asset.captured_geoms = self._captured_geoms

        asset.objects     = torch.empty((E, G, 10), dtype=torch.float32, device=device)
        asset.object_mask = torch.empty((E, G), dtype=torch.bool, device=device)

        asset.shuffled_objects     = torch.empty((E, G, 10), dtype=torch.float32, device=device)
        asset.shuffled_object_mask = torch.empty((E, G), dtype=torch.bool, device=device)

        self._empty_tensor = torch.zeros((E, 1), dtype=torch.float32, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        asset = self._asset

        asset.objects[env_ids] = 0
        asset.object_mask[env_ids] = 0

        asset.shuffled_objects[env_ids] = 0
        asset.shuffled_object_mask[env_ids] = 0

    def __call__(self, env: ManagerBasedRlEnv, *args, **kwargs) -> torch.Tensor:
        objects, object_mask = self._capture_geom_feature()

        self._asset.objects = objects
        self._asset.object_mask = object_mask

        batch_size, _, _ = objects.shape
        num_geoms = self._geom_ids.numel()

        permutation = torch.rand(
            batch_size,
            num_geoms,
            device=env.device,
        ).argsort(dim=1)

        self._asset.shuffled_objects = objects.gather(
            dim=1,
            index=permutation[..., None].expand(-1, -1, objects.shape[-1]),
        )

        self._asset.shuffled_object_mask = object_mask.gather(
            dim=1,
            index=permutation,
        )

        if self._play:
            self._render_geom_boxes(objects, object_mask)

        return self._empty_tensor

    def _capture_geom_feature(
        self,
        near: float = 1e-4,
        min_valid_depth: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ground-truth object features for selected environments.

        Supported primitives:
            - sphere: analytic perspective silhouette
            - capsule: analytic bounds from its two endpoint spheres
            - cylinder: analytic bounds from its two circular end disks
            - box: exact projection of its eight vertices

        Returns:
            objects:
                Shape [E, K, 10], where K is the number of captured geoms.

            object_mask:
                Shape [E, K]. True where the projected geom is valid, intersects
                the image, and its bounding box contains valid depth pixels.

        Feature layout:
            0:2    bbox_centroid_xy
            2:4    bbox_size_wh
            4:7    depth_min_max_avg
            7:10   render_rgb_avg
        """
        if not self._captured_geoms:
            raise RuntimeError("No capture geometries are configured.")

        sensor_data: CameraSensorData = self._sensor.data

        if sensor_data.depth is None:
            raise RuntimeError("CameraSensorData.depth is not available.")
        if sensor_data.rgb is None:
            raise RuntimeError("CameraSensorData.rgb is not available.")

        depth_source = sensor_data.depth
        device = depth_source.device
        dtype = depth_source.dtype
        eps = torch.finfo(dtype).eps

        depth = (
            depth_source
            .permute(0, 3, 1, 2)
            .squeeze(1)
        )

        # Camera RGB is uint8 [0, 255]. Convert to normalized float RGB so the
        # extracted color feature remains in [0, 1].
        rgb = (
            sensor_data.rgb
            .to(device=device, dtype=dtype)
            .permute(0, 3, 1, 2)
            / 255.0
        )

        batch_size, height, width = depth.shape
        num_geoms = self._geom_ids.numel()
        camera_id = int(self._sensor.camera_idx)

        # ------------------------------------------------------------------
        # Geometry and camera state
        # ------------------------------------------------------------------

        geom_pos = self._data.geom_xpos[:][:, self._geom_ids].to(
            device=device,
            dtype=dtype,
        )

        geom_xmat = (
            self._data.geom_xmat[:][:, self._geom_ids]
            .reshape(batch_size, num_geoms, 3, 3)
            .to(device=device, dtype=dtype)
        )

        geom_size = self._model.geom_size[:][:, self._geom_ids].to(
            device=device,
            dtype=dtype,
        )

        geom_type = self._model.geom_type[self._geom_ids].to(
            device=device,
            dtype=torch.long,
        )

        cam_pos = self._data.cam_xpos[:, camera_id].to(
            device=device,
            dtype=dtype,
        )

        cam_xmat = (
            self._data.cam_xmat[:, camera_id]
            .reshape(batch_size, 3, 3)
            .to(device=device, dtype=dtype)
        )

        world_to_camera = cam_xmat.transpose(-1, -2)

        fovy = torch.deg2rad(
            self._model.cam_fovy[:, camera_id].to(
                device=device,
                dtype=dtype,
            )
        )

        # MuJoCo cam_fovy is the vertical field of view. This assumes square
        # pixels, so the horizontal and vertical focal lengths match.
        focal = (
            0.5 * float(height)
            / torch.tan(0.5 * fovy)
        )

        principal_x = 0.5 * float(width)
        principal_y = 0.5 * float(height)

        # ------------------------------------------------------------------
        # Primitive type masks
        # ------------------------------------------------------------------

        is_sphere = (
            geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE)
        )
        is_capsule = (
            geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
        )
        is_cylinder = (
            geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
        )
        is_box = (
            geom_type == int(mujoco.mjtGeom.mjGEOM_BOX)
        )

        supported = (
            is_sphere
            | is_capsule
            | is_cylinder
            | is_box
        )

        # ------------------------------------------------------------------
        # Transform geom centers and orientations into camera space.
        #
        # MuJoCo cameras look along local -Z. Negating camera-space Z produces
        # a coordinate system where positive Z points forward.
        # ------------------------------------------------------------------

        center_camera_mj = torch.einsum(
            "eij,ekj->eki",
            world_to_camera,
            geom_pos - cam_pos[:, None, :],
        )

        center_camera = torch.stack(
            (
                center_camera_mj[..., 0],
                center_camera_mj[..., 1],
                -center_camera_mj[..., 2],
            ),
            dim=-1,
        )

        rotation_camera_mj = torch.einsum(
            "eij,ekjl->ekil",
            world_to_camera,
            geom_xmat,
        )

        rotation_camera = torch.stack(
            (
                rotation_camera_mj[..., 0, :],
                rotation_camera_mj[..., 1, :],
                -rotation_camera_mj[..., 2, :],
            ),
            dim=-2,
        )

        # Local primitive axes expressed in camera coordinates.
        axis_x = rotation_camera[..., :, 0]
        axis_y = rotation_camera[..., :, 1]
        axis_z = rotation_camera[..., :, 2]

        radius = geom_size[..., 0]
        half_length = geom_size[..., 1]

        # ==================================================================
        # Sphere projection
        # ==================================================================

        def project_spheres(
            centers: torch.Tensor,
            radii: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Project spheres with arbitrary leading dimensions."""
            center_x = centers[..., 0]
            center_y = centers[..., 1]
            center_z = centers[..., 2]

            radius_sq = radii.square()
            depth_sq = center_z.square()

            denominator = depth_sq - radius_sq

            valid = (
                (radii > 0.0)
                & ((center_z - radii) > near)
                & (denominator > eps)
            )

            safe_denominator = torch.where(
                valid,
                denominator,
                torch.ones_like(denominator),
            )

            horizontal_root = torch.sqrt(
                (
                    center_x.square()
                    + depth_sq
                    - radius_sq
                ).clamp_min(0.0)
            )

            vertical_root = torch.sqrt(
                (
                    center_y.square()
                    + depth_sq
                    - radius_sq
                ).clamp_min(0.0)
            )

            slope_x_min = (
                center_x * center_z
                - radii * horizontal_root
            ) / safe_denominator

            slope_x_max = (
                center_x * center_z
                + radii * horizontal_root
            ) / safe_denominator

            slope_y_min = (
                center_y * center_z
                - radii * vertical_root
            ) / safe_denominator

            slope_y_max = (
                center_y * center_z
                + radii * vertical_root
            ) / safe_denominator

            focal_view = focal.reshape(
                batch_size,
                *([1] * (center_x.ndim - 1)),
            )

            x1 = principal_x + focal_view * slope_x_min
            x2 = principal_x + focal_view * slope_x_max

            # Image-space Y points downward.
            y1 = principal_y - focal_view * slope_y_max
            y2 = principal_y - focal_view * slope_y_min

            bounds = torch.stack(
                (x1, y1, x2, y2),
                dim=-1,
            )

            return bounds, valid

        sphere_bounds, sphere_valid = project_spheres(
            centers=center_camera,
            radii=radius,
        )

        # ==================================================================
        # Capsule projection
        #
        # A capsule is the convex hull of two equal spheres centered at the
        # endpoints of its central segment.
        # ==================================================================

        endpoint_offsets = torch.stack(
            (-half_length, half_length),
            dim=-1,
        )

        endpoint_centers = (
            center_camera[:, :, None, :]
            + endpoint_offsets[..., None]
            * axis_z[:, :, None, :]
        )

        endpoint_radii = radius[:, :, None].expand(
            batch_size,
            num_geoms,
            2,
        )

        capsule_endpoint_bounds, capsule_endpoint_valid = project_spheres(
            centers=endpoint_centers,
            radii=endpoint_radii,
        )

        capsule_bounds = torch.stack(
            (
                capsule_endpoint_bounds[..., 0].amin(dim=-1),
                capsule_endpoint_bounds[..., 1].amin(dim=-1),
                capsule_endpoint_bounds[..., 2].amax(dim=-1),
                capsule_endpoint_bounds[..., 3].amax(dim=-1),
            ),
            dim=-1,
        )

        capsule_valid = capsule_endpoint_valid.all(dim=-1)

        # ==================================================================
        # Oriented circular-disk projection
        # ==================================================================

        def ratio_bounds_on_disks(
            coordinate: torch.Tensor,
            depth_value: torch.Tensor,
            basis_u_coordinate: torch.Tensor,
            basis_v_coordinate: torch.Tensor,
            basis_u_depth: torch.Tensor,
            basis_v_depth: torch.Tensor,
            radii: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Calculate exact extrema of coordinate / depth over each disk."""
            radius_sq = radii.square()

            coordinate_norm_sq = (
                basis_u_coordinate.square()
                + basis_v_coordinate.square()
            )

            depth_norm_sq = (
                basis_u_depth.square()
                + basis_v_depth.square()
            )

            coordinate_depth_dot = (
                basis_u_coordinate * basis_u_depth
                + basis_v_coordinate * basis_v_depth
            )

            quadratic_a = (
                depth_value.square()
                - radius_sq * depth_norm_sq
            )

            quadratic_b = (
                -2.0 * coordinate * depth_value
                + 2.0 * radius_sq * coordinate_depth_dot
            )

            quadratic_c = (
                coordinate.square()
                - radius_sq * coordinate_norm_sq
            )

            discriminant = (
                quadratic_b.square()
                - 4.0 * quadratic_a * quadratic_c
            )

            valid = quadratic_a > eps

            safe_a = torch.where(
                valid,
                quadratic_a,
                torch.ones_like(quadratic_a),
            )

            sqrt_discriminant = torch.sqrt(
                discriminant.clamp_min(0.0)
            )

            root_1 = (
                -quadratic_b - sqrt_discriminant
            ) / (2.0 * safe_a)

            root_2 = (
                -quadratic_b + sqrt_discriminant
            ) / (2.0 * safe_a)

            return (
                torch.minimum(root_1, root_2),
                torch.maximum(root_1, root_2),
                valid,
            )

        def project_disks(
            centers: torch.Tensor,
            basis_u: torch.Tensor,
            basis_v: torch.Tensor,
            radii: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Project oriented circular disks."""
            center_x = centers[..., 0]
            center_y = centers[..., 1]
            center_z = centers[..., 2]

            u_x = basis_u[..., 0]
            u_y = basis_u[..., 1]
            u_z = basis_u[..., 2]

            v_x = basis_v[..., 0]
            v_y = basis_v[..., 1]
            v_z = basis_v[..., 2]

            slope_x_min, slope_x_max, valid_x = ratio_bounds_on_disks(
                coordinate=center_x,
                depth_value=center_z,
                basis_u_coordinate=u_x,
                basis_v_coordinate=v_x,
                basis_u_depth=u_z,
                basis_v_depth=v_z,
                radii=radii,
            )

            slope_y_min, slope_y_max, valid_y = ratio_bounds_on_disks(
                coordinate=center_y,
                depth_value=center_z,
                basis_u_coordinate=u_y,
                basis_v_coordinate=v_y,
                basis_u_depth=u_z,
                basis_v_depth=v_z,
                radii=radii,
            )

            # Smallest possible depth over the oriented disk.
            depth_radius = radii * torch.sqrt(
                u_z.square() + v_z.square()
            )

            valid_depth = (
                center_z - depth_radius
            ) > near

            valid = (
                (radii > 0.0)
                & valid_x
                & valid_y
                & valid_depth
            )

            focal_view = focal[:, None, None]

            x1 = principal_x + focal_view * slope_x_min
            x2 = principal_x + focal_view * slope_x_max

            y1 = principal_y - focal_view * slope_y_max
            y2 = principal_y - focal_view * slope_y_min

            bounds = torch.stack(
                (x1, y1, x2, y2),
                dim=-1,
            )

            return bounds, valid

        # ==================================================================
        # Cylinder projection
        #
        # A cylinder is the convex hull of its two circular end disks.
        # ==================================================================

        cylinder_basis_u = axis_x[:, :, None, :].expand(
            batch_size,
            num_geoms,
            2,
            3,
        )

        cylinder_basis_v = axis_y[:, :, None, :].expand(
            batch_size,
            num_geoms,
            2,
            3,
        )

        cylinder_radii = radius[:, :, None].expand(
            batch_size,
            num_geoms,
            2,
        )

        cylinder_disk_bounds, cylinder_disk_valid = project_disks(
            centers=endpoint_centers,
            basis_u=cylinder_basis_u,
            basis_v=cylinder_basis_v,
            radii=cylinder_radii,
        )

        cylinder_bounds = torch.stack(
            (
                cylinder_disk_bounds[..., 0].amin(dim=-1),
                cylinder_disk_bounds[..., 1].amin(dim=-1),
                cylinder_disk_bounds[..., 2].amax(dim=-1),
                cylinder_disk_bounds[..., 3].amax(dim=-1),
            ),
            dim=-1,
        )

        cylinder_valid = cylinder_disk_valid.all(dim=-1)

        # ==================================================================
        # Box projection
        # ==================================================================

        corner_signs = geom_size.new_tensor(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0,  1.0],
                [-1.0,  1.0, -1.0],
                [-1.0,  1.0,  1.0],
                [ 1.0, -1.0, -1.0],
                [ 1.0, -1.0,  1.0],
                [ 1.0,  1.0, -1.0],
                [ 1.0,  1.0,  1.0],
            ]
        )

        local_box_corners = (
            geom_size[:, :, None, :]
            * corner_signs[None, None, :, :]
        )

        box_corners = (
            center_camera[:, :, None, :]
            + torch.einsum(
                "ekij,ekcj->ekci",
                rotation_camera,
                local_box_corners,
            )
        )

        box_corner_z = box_corners[..., 2]

        # Simplified near-plane handling. Boxes crossing the near plane are
        # considered invalid rather than clipping their edges.
        box_valid = (
            box_corner_z > near
        ).all(dim=-1)

        box_corner_z_safe = box_corner_z.clamp_min(near)

        box_u = (
            focal[:, None, None]
            * box_corners[..., 0]
            / box_corner_z_safe
            + principal_x
        )

        box_v = (
            principal_y
            - focal[:, None, None]
            * box_corners[..., 1]
            / box_corner_z_safe
        )

        box_bounds = torch.stack(
            (
                box_u.amin(dim=-1),
                box_v.amin(dim=-1),
                box_u.amax(dim=-1),
                box_v.amax(dim=-1),
            ),
            dim=-1,
        )

        # ==================================================================
        # Select primitive-specific bounds
        # ==================================================================

        raw_boxes = box_bounds
        projection_valid = box_valid

        raw_boxes = torch.where(
            is_sphere[None, :, None],
            sphere_bounds,
            raw_boxes,
        )

        projection_valid = torch.where(
            is_sphere[None, :],
            sphere_valid,
            projection_valid,
        )

        raw_boxes = torch.where(
            is_capsule[None, :, None],
            capsule_bounds,
            raw_boxes,
        )

        projection_valid = torch.where(
            is_capsule[None, :],
            capsule_valid,
            projection_valid,
        )

        raw_boxes = torch.where(
            is_cylinder[None, :, None],
            cylinder_bounds,
            raw_boxes,
        )

        projection_valid = torch.where(
            is_cylinder[None, :],
            cylinder_valid,
            projection_valid,
        )

        raw_x1 = raw_boxes[..., 0]
        raw_y1 = raw_boxes[..., 1]
        raw_x2 = raw_boxes[..., 2]
        raw_y2 = raw_boxes[..., 3]

        finite = torch.isfinite(raw_boxes).all(dim=-1)

        nonempty = (
            (raw_x2 > raw_x1)
            & (raw_y2 > raw_y1)
        )

        # Check image intersection before clamping. Otherwise an entirely
        # off-screen primitive could collapse onto the image boundary.
        intersects_image = (
            (raw_x2 > 0.0)
            & (raw_y2 > 0.0)
            & (raw_x1 < float(width))
            & (raw_y1 < float(height))
        )

        projection_mask = (
            supported[None, :]
            & projection_valid
            & finite
            & nonempty
            & intersects_image
        )

        x1 = raw_x1.clamp(0.0, float(width))
        y1 = raw_y1.clamp(0.0, float(height))
        x2 = raw_x2.clamp(0.0, float(width))
        y2 = raw_y2.clamp(0.0, float(height))

        bbox_w = x2 - x1
        bbox_h = y2 - y1

        projection_mask = (
            projection_mask
            & (bbox_w > 0.0)
            & (bbox_h > 0.0)
        )

        # ------------------------------------------------------------------
        # Normalized screen-space box features
        # ------------------------------------------------------------------

        centroid = torch.stack(
            (
                (x1 + x2) * (0.5 / float(width)),
                (y1 + y2) * (0.5 / float(height)),
            ),
            dim=-1,
        )

        screen_size = torch.stack(
            (
                bbox_w / float(width),
                bbox_h / float(height),
            ),
            dim=-1,
        )

        # ------------------------------------------------------------------
        # Convert continuous bounds to inclusive pixel bounds
        # ------------------------------------------------------------------

        x_lo = (
            x1.floor()
            .long()
            .clamp(0, width - 1)
        )

        y_lo = (
            y1.floor()
            .long()
            .clamp(0, height - 1)
        )

        x_hi = (
            x2.ceil().long() - 1
        ).clamp(0, width - 1)

        y_hi = (
            y2.ceil().long() - 1
        ).clamp(0, height - 1)

        roi_nonempty = (
            (x_hi >= x_lo)
            & (y_hi >= y_lo)
        )

        projection_mask = projection_mask & roi_nonempty

        # ==================================================================
        # Depth average using integral images
        # ==================================================================

        valid_depth = depth > min_valid_depth

        depth_valid = torch.where(
            valid_depth,
            depth,
            torch.zeros_like(depth),
        )

        depth_sum = (
            F.pad(depth_valid, (1, 0, 1, 0))
            .cumsum(dim=1)
            .cumsum(dim=2)
        )

        depth_count = (
            F.pad(valid_depth.to(dtype), (1, 0, 1, 0))
            .cumsum(dim=1)
            .cumsum(dim=2)
        )

        batch_ids = torch.arange(
            batch_size,
            device=device,
        )[:, None]

        sum_depth = (
            depth_sum[batch_ids, y_hi + 1, x_hi + 1]
            - depth_sum[batch_ids, y_lo, x_hi + 1]
            - depth_sum[batch_ids, y_hi + 1, x_lo]
            + depth_sum[batch_ids, y_lo, x_lo]
        )

        valid_pixel_count = (
            depth_count[batch_ids, y_hi + 1, x_hi + 1]
            - depth_count[batch_ids, y_lo, x_hi + 1]
            - depth_count[batch_ids, y_hi + 1, x_lo]
            + depth_count[batch_ids, y_lo, x_lo]
        )

        has_valid_depth = valid_pixel_count > 0.0

        avg_depth = (
            sum_depth
            / valid_pixel_count.clamp_min(1.0)
        )

        # ==================================================================
        # Rendered RGB average using the same valid-depth pixels
        # ==================================================================

        # Pixels whose depth is close to zero are excluded from the RGB
        # average as well. This keeps color and average depth based on the
        # exact same set of image pixels.
        rgb_valid = torch.where(
            valid_depth[:, None, :, :],
            rgb,
            torch.zeros_like(rgb),
        )

        rgb_sum = (
            F.pad(rgb_valid, (1, 0, 1, 0))
            .cumsum(dim=2)
            .cumsum(dim=3)
            .permute(0, 2, 3, 1)
        )

        sum_rgb = (
            rgb_sum[batch_ids, y_hi + 1, x_hi + 1]
            - rgb_sum[batch_ids, y_lo, x_hi + 1]
            - rgb_sum[batch_ids, y_hi + 1, x_lo]
            + rgb_sum[batch_ids, y_lo, x_lo]
        )

        avg_rgb = (
            sum_rgb
            / valid_pixel_count.clamp_min(1.0)[..., None]
        )

        # ==================================================================
        # Depth minimum and maximum using ROI pooling
        # ==================================================================

        roi_batch_ids = (
            torch.arange(
                batch_size,
                device=device,
                dtype=dtype,
            )
            .unsqueeze(1)
            .expand(batch_size, num_geoms)
        )

        rois = torch.stack(
            (
                roi_batch_ids,
                x_lo.to(dtype),
                y_lo.to(dtype),
                x_hi.to(dtype),
                y_hi.to(dtype),
            ),
            dim=-1,
        ).reshape(-1, 5)

        neg_inf = torch.full_like(
            depth,
            -torch.inf,
        )

        max_depth_input = torch.where(
            valid_depth,
            depth,
            neg_inf,
        )

        min_depth_input = torch.where(
            valid_depth,
            -depth,
            neg_inf,
        )

        depth_for_pooling = torch.stack(
            (
                max_depth_input,
                min_depth_input,
            ),
            dim=1,
        )

        pooled_depth = roi_pool(
            depth_for_pooling,
            rois,
            output_size=(1, 1),
            spatial_scale=1.0,
        ).reshape(
            batch_size,
            num_geoms,
            2,
        )

        max_depth = pooled_depth[..., 0]
        min_depth = -pooled_depth[..., 1]

        # A feature is valid only when the geom projection is valid and the
        # resulting image-space ROI contains at least one valid depth pixel.
        object_mask = (
            projection_mask
            & has_valid_depth
        )

        depth_features = torch.stack(
            (
                min_depth,
                max_depth,
                avg_depth,
            ),
            dim=-1,
        )

        objects = torch.cat(
            (
                centroid,
                screen_size,
                depth_features,
                avg_rgb,
            ),
            dim=-1,
        )

        objects = torch.where(
            object_mask[..., None],
            objects,
            torch.zeros_like(objects),
        )

        return objects, object_mask

    def _render_geom_boxes(
        self,
        objects: torch.Tensor,
        object_mask: torch.Tensor,
        alpha: float = 0.6,
    ) -> None:
        """Render one-pixel bounding-box borders into CameraSensorData.rgb.

        Overlapping borders have no defined ordering; one object's color wins.
        """
        sensor_data: CameraSensorData = self._sensor.data
        if sensor_data.rgb is None:
            raise RuntimeError("CameraSensorData.rgb is not available.")

        rgb = sensor_data.rgb
        num_envs, height, width, _ = rgb.shape
        device = rgb.device

        objects = objects.to(device=device)
        object_mask = object_mask.to(device=device)

        # Normalized center-size boxes -> inclusive pixel bounds.
        center = objects[..., 0:2]
        size = objects[..., 2:4]

        x1 = ((center[..., 0] - 0.5 * size[..., 0]) * width).floor().long()
        y1 = ((center[..., 1] - 0.5 * size[..., 1]) * height).floor().long()
        x2 = ((center[..., 0] + 0.5 * size[..., 0]) * width).ceil().long() - 1
        y2 = ((center[..., 1] + 0.5 * size[..., 1]) * height).ceil().long() - 1

        x1.clamp_(0, width - 1)
        x2.clamp_(0, width - 1)
        y1.clamp_(0, height - 1)
        y2.clamp_(0, height - 1)

        object_mask &= (x2 >= x1) & (y2 >= y1)

        # ------------------------------------------------------------------
        # Horizontal edges: top and bottom
        # ------------------------------------------------------------------

        x_range = torch.arange(width, device=device).view(1, 1, width)
        horizontal_valid = (
            object_mask[..., None]
            & (x_range >= x1[..., None])
            & (x_range <= x2[..., None])
        )

        horizontal_x = torch.cat((x_range, x_range), dim=-1)
        horizontal_x = horizontal_x.expand(num_envs, objects.shape[1], -1)

        horizontal_y = torch.cat(
            (
                y1[..., None].expand(-1, -1, width),
                y2[..., None].expand(-1, -1, width),
            ),
            dim=-1,
        )

        horizontal_valid = torch.cat(
            (horizontal_valid, horizontal_valid),
            dim=-1,
        )

        # ------------------------------------------------------------------
        # Vertical edges: left and right
        # ------------------------------------------------------------------

        y_range = torch.arange(height, device=device).view(1, 1, height)
        vertical_valid = (
            object_mask[..., None]
            & (y_range >= y1[..., None])
            & (y_range <= y2[..., None])
        )

        vertical_y = torch.cat((y_range, y_range), dim=-1)
        vertical_y = vertical_y.expand(num_envs, objects.shape[1], -1)

        vertical_x = torch.cat(
            (
                x1[..., None].expand(-1, -1, height),
                x2[..., None].expand(-1, -1, height),
            ),
            dim=-1,
        )

        vertical_valid = torch.cat(
            (vertical_valid, vertical_valid),
            dim=-1,
        )

        # ------------------------------------------------------------------
        # Combine edge coordinates
        # ------------------------------------------------------------------

        pixel_x = torch.cat((horizontal_x, vertical_x), dim=-1)
        pixel_y = torch.cat((horizontal_y, vertical_y), dim=-1)
        valid = torch.cat((horizontal_valid, vertical_valid), dim=-1)

        # Convert [environment, y, x] to a single flattened pixel index.
        env_offset = (
            torch.arange(num_envs, device=device)[:, None, None]
            * height
            * width
        )

        pixel_index = (
            env_offset + pixel_y * width + pixel_x
        )[valid]

        # Bounding-box visualization intentionally uses the configured geom
        # color rather than the observed/averaged RGB feature.
        colors = (
            self._model.geom_rgba[:][:, self._geom_ids, :3]
            .to(device=device, dtype=torch.float32)
            .clamp(0.0, 1.0)
            .mul(255.0)
        )

        pixel_colors = colors[..., None, :].expand(
            -1,
            -1,
            pixel_x.shape[-1],
            -1,
        )[valid]

        # ------------------------------------------------------------------
        # Blend directly into the flattened RGB tensor
        # ------------------------------------------------------------------

        flat_rgb = rgb.view(-1, 3)
        current = flat_rgb[pixel_index].to(pixel_colors.dtype)

        flat_rgb[pixel_index] = torch.lerp(
            current,
            pixel_colors,
            alpha,
        ).round().clamp_(0.0, 255.0).to(torch.uint8)
