from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from typing import Any, ClassVar

from omegaconf import OmegaConf

@dataclass
class RangeCfg:
    """Base class for a fixed value or a min/max sampling range."""

    DIM: ClassVar[int] = 0

    value: Any | None = None
    min: Any | None = None
    max: Any | None = None

    def __post_init__(self) -> None:
        has_value = self.value is not None
        has_range = self.min is not None or self.max is not None

        if has_value == has_range:
            raise ValueError(
                f"{type(self).__name__} requires either value or min/max."
            )

        self._validate_value("value", self.value)
        self._validate_value("min", self.min)
        self._validate_value("max", self.max)

        if self.min is not None:
            assert self.max is not None

            for lo, hi in zip(self._as_list(self.min), self._as_list(self.max)):
                if hi < lo:
                    raise ValueError(
                        f"{type(self).__name__}.max must be >= min."
                    )

    @classmethod
    def of(cls, value: Any) -> "RangeCfg":
        if isinstance(value, cls):
            return value

        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)

        if isinstance(value, dict):
            return cls(**value)

        raise TypeError(
            f"Expected {cls.__name__} or dict, got {type(value).__name__}."
        )

    def bounds(self) -> tuple[Any, Any]:
        if self.value is not None:
            return self.value, self.value

        assert self.min is not None
        assert self.max is not None
        return self.min, self.max
    
    def average(self) -> np.ndarray:
        lower, upper = self.bounds()
        return (np.asarray(lower) + np.asarray(upper)) / 2

    def _validate_value(self, name: str, value: Any | None) -> None:
        if value is None:
            return

        if self.DIM == 1:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"{type(self).__name__}.{name} must be a scalar."
                )
            return

        if not isinstance(value, list):
            raise TypeError(
                f"{type(self).__name__}.{name} must be a list of length {self.DIM}."
            )

        if len(value) != self.DIM:
            raise ValueError(
                f"{type(self).__name__}.{name} must have length {self.DIM}, "
                f"got {len(value)}."
            )

        if any(not isinstance(x, (int, float)) or isinstance(x, bool) for x in value):
            raise TypeError(
                f"{type(self).__name__}.{name} must contain only numbers."
            )

    def _as_list(self, value: Any) -> list[float]:
        if self.DIM == 1:
            assert isinstance(value, (int, float))
            return [float(value)]

        assert isinstance(value, list)
        return [float(x) for x in value]

class ScalarRange(RangeCfg):
    DIM = 1

class Vec3Range(RangeCfg):
    DIM = 3
