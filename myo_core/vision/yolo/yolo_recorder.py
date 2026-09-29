import re
import h5py
import torch
import numpy as np

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import RecorderTermCfg

from myo_core.task.universal import UniversalRecorder
from .perfect_detector import PerfectDetectorEntity
from .yolo_sensor import YoloSensor


class YoloRecorder(UniversalRecorder):
    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self._sensor: YoloSensor = self._env.scene[cfg.params['sensor_name']]
        self._perfect_detector: PerfectDetectorEntity = self._env.scene[cfg.params['asset_name']]

        self._register_yolo_metadata()
        self._register_ground_truth_metadata()
        self._register_sensor_metadata()

    def _register_sensor_metadata(self) -> None:
        """Persist detector settings needed to reproduce offline metrics."""
        cfg = self._sensor.cfg
        group = self._file.require_group("metadata/yolo")

        values = {
            "model": cfg.model,
            "width": cfg.width,
            "height": cfg.height,
            "use_class": cfg.use_class,
            "use_confidence": cfg.use_confidence,
            "color_mode": str(cfg.color) if cfg.color is not None else "none",
            "depth_mode": str(cfg.depth) if cfg.depth is not None else "none",
            "depth_min": cfg.depth_min,
            "depth_cutoff": cfg.depth_cutoff,
            "nms_top_k": cfg.nms_top_k if cfg.nms_top_k is not None else -1,
            "nms_pre_top_k": (
                cfg.nms_pre_top_k if cfg.nms_pre_top_k is not None else -1
            ),
            "nms_conf_threshold": cfg.nms_conf_threshold,
            "nms_iou_threshold": cfg.nms_iou_threshold,
            "nms_class_agnostic": cfg.nms_class_agnostic,
            "perfect_detection": cfg.perfect_detection,
        }
        for name, value in values.items():
            group.attrs[name] = value

    def _collect_state_tensors(
        self,
        env_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        state = super()._collect_state_tensors(env_ids)

        exclude = re.compile(r".*\/__internal__")

        state = {
            key: value
            for key, value in state.items()
            if not exclude.search(key)
        }

        return state

    def _register_yolo_metadata(self) -> None:
        yolo_objects_group = self._file.require_group(
            "states/observation/yolo_objects"
        )

        string_dtype = h5py.string_dtype(encoding="utf-8")

        feature_names = [
            "bbox_center_x",
            "bbox_center_y",
            "bbox_width",
            "bbox_height",
            "class_id_normalized",
            "confidence",
        ]

        cfg = self._sensor.cfg

        if cfg.depth is not None:
            feature_names.extend(
                (
                    "depth_min",
                    "depth_max",
                    "depth_mean",
                )
            )

        if cfg.color == "hue":
            feature_names.append("box_mean_rgb_hue")
        elif cfg.color is not None:
            feature_names.extend(
                (
                    "box_mean_red",
                    "box_mean_green",
                    "box_mean_blue",
                )
            )

        yolo_objects_group.attrs.create(
            "feature_names",
            np.asarray(feature_names, dtype=string_dtype,),
        )

        yolo_objects_group.attrs.create(
            "class_names",
            np.asarray([
                'person',
                'bicycle',
                'car',
                'motorcycle',
                'airplane',
                'bus',
                'train',
                'truck',
                'boat',
                'traffic light',
                'fire hydrant',
                'stop sign',
                'parking meter',
                'bench',
                'bird',
                'cat',
                'dog',
                'horse',
                'sheep',
                'cow',
                'elephant',
                'bear',
                'zebra',
                'giraffe',
                'backpack',
                'umbrella',
                'handbag',
                'tie',
                'suitcase',
                'frisbee',
                'skis',
                'snowboard',
                'sports ball',
                'kite',
                'baseball bat',
                'baseball glove',
                'skateboard',
                'surfboard',
                'tennis racket',
                'bottle',
                'wine glass',
                'cup',
                'fork',
                'knife',
                'spoon',
                'bowl',
                'banana',
                'apple',
                'sandwich',
                'orange',
                'broccoli',
                'carrot',
                'hot dog',
                'pizza',
                'donut',
                'cake',
                'chair',
                'couch',
                'potted plant',
                'bed',
                'dining table',
                'toilet',
                'tv',
                'laptop',
                'mouse',
                'remote',
                'keyboard',
                'cell phone',
                'microwave',
                'oven',
                'toaster',
                'sink',
                'refrigerator',
                'book',
                'clock',
                'vase',
                'scissors',
                'teddy bear',
                'hair drier',
                'toothbrush'
            ], dtype=string_dtype)
        )

    def _register_ground_truth_metadata(self) -> None:
        ground_truth_group = self._file.require_group(
            "states/observation/perfect_detector"
        )

        string_dtype = h5py.string_dtype(encoding="utf-8")

        ground_truth_group.attrs.create(
            "object_names",
            np.asarray(
                tuple(self._perfect_detector.captured_geoms.keys()),
                dtype=string_dtype,
            ),
        )

        ground_truth_group.attrs.create(
            "feature_names",
            np.asarray(
                (
                    "bbox_center_x",
                    "bbox_center_y",
                    "bbox_width",
                    "bbox_height",
                    "depth_min",
                    "depth_max",
                    "depth_mean",
                    "geom_red",
                    "geom_green",
                    "geom_blue",
                ),
                dtype=string_dtype,
            ),
        )
