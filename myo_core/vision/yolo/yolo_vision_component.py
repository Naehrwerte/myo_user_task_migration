import torch
import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg, ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.managers import ObservationTermCfg, ObservationGroupCfg, RecorderTermCfg, EventTermCfg

from myo_core.common import MyoComponent, dataclass_as_derived
from .yolo_vision_config import YoloVisionConfig
from .yolo_sensor import YoloSensorCfg, YoloSensorData, YoloSensor
from .yolo_model_config import YoloModelConfig
from .yolo_recorder import YoloRecorder
from .perfect_detector import perfect_detection_select, PerfectDetectorEntityCfg, PerfectDetector
from ..vision_registry import myo_register_vision

def yolo_objects(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: YoloSensor = env.scene[sensor_name]
  data: YoloSensorData = sensor.data

  objects = data.objects  # (B, N, F)
  assert objects is not None, f"YOLO '{sensor_name}' has no objects data"
  return objects

def yolo_object_mask(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: YoloSensor = env.scene[sensor_name]
  data: YoloSensorData = sensor.data

  object_mask = data.object_mask  # (B, N, F)
  assert object_mask is not None, f"YOLO '{sensor_name}' has no object_mask data"
  return object_mask

@myo_register_vision("yolo")
class YoloVisionComponent(MyoComponent):
    def __init__(self, cfg: YoloVisionConfig):
       self.cfg = cfg

    def modify_env_cfg(self, cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
        cfg.scene.sensors += (
            YoloSensorCfg(
                name="yolo_sensor",
                camera_name=self.cfg.camera_name,
                width=self.cfg.width,
                height=self.cfg.height,
                data_types=("rgb", "depth"),
                use_textures=self.cfg.use_textures,
                use_shadows=self.cfg.use_shadows,
                enabled_geom_groups=self.cfg.enabled_geom_groups,

                model=self.cfg.model,
                use_class=self.cfg.use_class,
                use_confidence=self.cfg.use_confidence,
                color=self.cfg.color_mode,
                depth=self.cfg.depth_mode,
                depth_min=self.cfg.depth_min,
                depth_cutoff=self.cfg.depth_cutoff,
                half=True,
                play=play,
                nms_top_k=self.cfg.nms_top_k,

                perfect_detection=self.cfg.perfect_detection
            ),
        )

        if play or self.cfg.perfect_detection:
            cfg.scene.entities.update({
                "perfect_detector": PerfectDetectorEntityCfg(
                    spec_fn=self._create_detector_spec,
                )
            })
            cfg.observations.update({
                "perfect_detector": ObservationGroupCfg(
                    terms={
                        "__internal__": ObservationTermCfg(
                            func=PerfectDetector,
                            params={
                                "asset_name": "perfect_detector",
                                "sensor_name": "yolo_sensor",
                                "capture_geoms": self.cfg.capture_geoms,
                                "play": play
                            },
                        )
                    },
                    concatenate_terms=not play
                )
            })

        if self.cfg.perfect_detection:
            cfg.observations.update({
                "yolo_objects": ObservationGroupCfg(
                    terms={
                        "objects": ObservationTermCfg(
                            func=perfect_detection_select,
                            params={"asset_name": "perfect_detector", "field": "shuffled_objects"},
                            history_length=self.cfg.history_length,
                            flatten_history_dim=False,
                        )
                    }
                ),
                "yolo_object_mask": ObservationGroupCfg(
                    terms={
                        "object_mask": ObservationTermCfg(
                            func=perfect_detection_select,
                            params={"asset_name": "perfect_detector", "field": "shuffled_object_mask"},
                            history_length=self.cfg.history_length,
                            flatten_history_dim=False,
                        )
                    }
                )
            })
        else:
            cfg.observations.update({
                "yolo_objects": ObservationGroupCfg(
                    terms={
                        "objects": ObservationTermCfg(
                            func=yolo_objects,
                            params={"sensor_name": "yolo_sensor"},
                            history_length=self.cfg.history_length,
                            flatten_history_dim=False,
                        )
                    }
                ),
                "yolo_object_mask": ObservationGroupCfg(
                    terms={
                        "object_mask": ObservationTermCfg(
                            func=yolo_object_mask,
                            params={"sensor_name": "yolo_sensor"},
                            history_length=self.cfg.history_length,
                            flatten_history_dim=False,
                        )
                    }
                )
            })
        
        if play:
            cfg.observations['perfect_detector'].terms.update({
                "objects": ObservationTermCfg(
                    func=perfect_detection_select,
                    params={"asset_name": "perfect_detector", "field": "objects"}
                ),
                "object_mask": ObservationTermCfg(
                    func=perfect_detection_select,
                    params={"asset_name": "perfect_detector", "field": "object_mask"}
                )
            })
            cfg.recorders.update({
                "yolo": RecorderTermCfg(
                    func=YoloRecorder,
                    params={
                        "sensor_name": "yolo_sensor",
                        "asset_name": "perfect_detector"
                    }
                )
            })

    def modify_rl_cfg(self, cfg: RslRlOnPolicyRunnerCfg) -> None:
        cfg.actor = dataclass_as_derived(cfg.actor, YoloModelConfig)

        cfg.actor.class_name = "myo_core.vision.yolo.yolo_model:YoloObjectModel"
        cfg.actor.object_encoder.use_transformer = self.cfg.transformer
        cfg.actor.object_encoder.condition_pool_on_task = self.cfg.task_query_pooling
        cfg.actor.object_encoder.concat_task_to_head = self.cfg.task_query_head

        mask = (("yolo_object_mask",) if (self.cfg.nms_top_k is not None or self.cfg.perfect_detection) else ())

        cfg.obs_groups = {
          "actor": ("agent_state", "task_query", "yolo_objects") + mask,
          "critic": ("agent_state", "task_state")
        }

    def _create_detector_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()

        site = spec.worldbody.add_site()
        site.type = mujoco.mjtGeom.mjGEOM_SPHERE
        site.size[:] = [0.01, 0.0, 0.0]
        site.rgba[:] = [1.0, 0.0, 0.0, 1.0]

        return spec
