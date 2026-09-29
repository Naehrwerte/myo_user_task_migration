# myouser/tasks/target_resolver.py
from __future__ import annotations

from omegaconf import OmegaConf


_TARGETS: dict[str, str] = {
    "pointing": "myo_core.task.universal.universal_task_config.PointingTargetConfig",
    "tracking": "myo_core.task.universal.universal_task_config.TrackingTargetConfig",
    "button": "myo_core.task.universal.universal_task_config.ButtonTargetConfig",
}


def _resolve_target(name: str) -> str:
    try:
        return _TARGETS[name]
    except KeyError:
        known = ", ".join(sorted(_TARGETS))
        raise ValueError(
            f"Unknown target type {name!r}. Known target types: {known}"
        ) from None


def register_target_resolver() -> None:
    if OmegaConf.has_resolver("target"):
        return

    OmegaConf.register_new_resolver(
        "target",
        _resolve_target,
        use_cache=True,
    )

register_target_resolver()
