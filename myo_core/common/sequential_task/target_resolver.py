"""OmegaConf resolver for the shared target configs.
"""

from __future__ import annotations

from omegaconf import OmegaConf


_RESOLVER_NAME = "task_target"

_TARGETS: dict[str, str] = {
    "pointing": "myo_core.common.sequential_task.sequential_task_config.PointingTargetConfig",
    "button": "myo_core.common.sequential_task.sequential_task_config.ButtonTargetConfig",
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
    if OmegaConf.has_resolver(_RESOLVER_NAME):
        return

    OmegaConf.register_new_resolver(
        _RESOLVER_NAME,
        _resolve_target,
        use_cache=True,
    )
