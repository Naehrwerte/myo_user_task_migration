from __future__ import annotations

import torch
from math import prod
from typing import Any, Sequence
from mjlab.envs.mdp.dr import Distribution

from .config import RangeCfg

class BatchedDistributionSampler:
    """Sample batched values from cached per-entry bounds.

    Stored bounds:
        [N, *event_shape]

    sample(B) returns:
        [B, N, *event_shape]

    Fixed entries are represented by lower == upper.
    If every entry is fixed, no random numbers are generated.
    Otherwise, all entries are sampled in one distribution call.
    """

    def __init__(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
        distribution: Distribution,
    ) -> None:
        if lower.shape != upper.shape:
            raise ValueError(
                "Lower and upper bounds must have equal shapes, got "
                f"{tuple(lower.shape)} and {tuple(upper.shape)}."
            )

        if lower.ndim < 1:
            raise ValueError(
                "Bounds must have a leading entry dimension: [N, ...]."
            )

        if torch.any(upper < lower):
            raise ValueError(
                "Each upper bound must be greater than or equal to its lower bound."
            )

        self.lower = lower
        self.upper = upper
        self.distribution = distribution
        self.is_fully_fixed = bool(torch.equal(lower, upper))

    @classmethod
    def from_ranges(
        cls,
        ranges: Sequence[RangeCfg],
        *,
        distribution: Distribution,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> "BatchedDistributionSampler":
        if not ranges:
            raise ValueError("At least one range is required.")

        lower_values: list[torch.Tensor] = []
        upper_values: list[torch.Tensor] = []

        for value in ranges:
            cfg = RangeCfg.of(value)
            lower, upper = cfg.bounds()

            lower_values.append(
                torch.tensor(lower, device=device, dtype=dtype)
            )
            upper_values.append(
                torch.tensor(upper, device=device, dtype=dtype)
            )

        try:
            lower_tensor = torch.stack(lower_values, dim=0)
            upper_tensor = torch.stack(upper_values, dim=0)
        except RuntimeError as error:
            raise ValueError(
                "All ranges in one sampler must have the same value shape."
            ) from error

        return cls(
            lower=lower_tensor,
            upper=upper_tensor,
            distribution=distribution,
        )

    def sample(
        self,
        batch_shape: int | tuple[int, ...] = (),
    ) -> torch.Tensor:
        """Sample values with shape [*batch_shape, N, *event_shape]."""
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        else:
            batch_shape = tuple(batch_shape)

        if any(dim < 0 for dim in batch_shape):
            raise ValueError(f"Invalid batch shape: {batch_shape}.")

        output_shape = (*batch_shape, *self.lower.shape)

        if self.is_fully_fixed:
            values = self.lower.expand(output_shape)
            return values.clone()

        return self.distribution.sample(
            self.lower,
            self.upper,
            output_shape,
            str(self.lower.device),
        )


class BatchedTrajectorySampler:
    """Sample smooth bounded trajectories from cached per-entry bounds.

    Stored bounds:
        [N, *event_shape]

    sample(B) returns:
        [B, N, *event_shape, num_steps]

    Each scalar bound entry receives one independent 1D trajectory.
    """

    def __init__(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
        *,
        num_steps: int,
        min_frequency: float | torch.Tensor = 0.0,
        max_frequency: float | torch.Tensor = 0.5,
        num_components: int = 5,
        min_amplitude: float = 1.0,
        max_amplitude: float = 5.0,
        workspace_fraction: float = 1.0,
    ) -> None:
        if lower.shape != upper.shape:
            raise ValueError(
                "Lower and upper bounds must have equal shapes, got "
                f"{tuple(lower.shape)} and {tuple(upper.shape)}."
            )

        if lower.ndim < 1:
            raise ValueError(
                "Bounds must have a leading entry dimension: [N, ...]."
            )

        if not lower.is_floating_point() or not upper.is_floating_point():
            raise ValueError("Trajectory bounds must use a floating-point dtype.")

        if torch.any(upper < lower):
            raise ValueError(
                "Each upper bound must be greater than or equal to its lower bound."
            )

        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")

        if num_components < 1:
            raise ValueError("num_components must be at least 1.")

        if min_amplitude < 0.0 or max_amplitude < min_amplitude:
            raise ValueError(
                "Amplitudes must satisfy 0 <= min_amplitude <= max_amplitude."
            )

        if not 0.0 < workspace_fraction <= 1.0:
            raise ValueError("workspace_fraction must be in (0, 1].")

        def expand_frequency(
            value: float | torch.Tensor,
            name: str,
        ) -> torch.Tensor:
            value = torch.as_tensor(
                value,
                device=lower.device,
                dtype=lower.dtype,
            )

            try:
                return torch.broadcast_to(value, lower.shape).reshape(-1)
            except RuntimeError as error:
                raise ValueError(
                    f"{name} must be scalar or broadcastable to bounds shape "
                    f"{tuple(lower.shape)}, got {tuple(value.shape)}."
                ) from error

        min_frequency = expand_frequency(min_frequency, "min_frequency")
        max_frequency = expand_frequency(max_frequency, "max_frequency")

        if torch.any(min_frequency < 0.0):
            raise ValueError("min_frequency must be non-negative.")

        if torch.any(max_frequency < min_frequency):
            raise ValueError(
                "Each max_frequency value must be >= its min_frequency value."
            )

        self.lower = lower
        self.upper = upper
        self.num_steps = num_steps
        self.num_components = num_components
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
        self.workspace_fraction = workspace_fraction

        # Stored as [N * prod(event_shape)].
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

    @classmethod
    def from_ranges(
        cls,
        ranges: Sequence[RangeCfg],
        *,
        num_steps: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        min_frequency: float | torch.Tensor = 0.0,
        max_frequency: float | torch.Tensor = 0.5,
        num_components: int = 5,
        min_amplitude: float = 1.0,
        max_amplitude: float = 5.0,
        workspace_fraction: float = 1.0,
    ) -> "BatchedTrajectorySampler":
        if not ranges:
            raise ValueError("At least one range is required.")

        lower_values: list[torch.Tensor] = []
        upper_values: list[torch.Tensor] = []

        for value in ranges:
            cfg = RangeCfg.of(value)
            lower, upper = cfg.bounds()

            lower_values.append(
                torch.as_tensor(lower, device=device, dtype=dtype)
            )
            upper_values.append(
                torch.as_tensor(upper, device=device, dtype=dtype)
            )

        try:
            lower_tensor = torch.stack(lower_values, dim=0)
            upper_tensor = torch.stack(upper_values, dim=0)
        except RuntimeError as error:
            raise ValueError(
                "All ranges in one sampler must have the same value shape."
            ) from error

        return cls(
            lower=lower_tensor,
            upper=upper_tensor,
            num_steps=num_steps,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            num_components=num_components,
            min_amplitude=min_amplitude,
            max_amplitude=max_amplitude,
            workspace_fraction=workspace_fraction,
        )

    def sample(
        self,
        batch_shape: int | tuple[int, ...] = (),
    ) -> torch.Tensor:
        """Sample trajectories with shape [*batch_shape, T, N, *event_shape]."""

        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        else:
            batch_shape = tuple(batch_shape)

        if any(dim < 0 for dim in batch_shape):
            raise ValueError(f"Invalid batch shape: {batch_shape}.")

        num_batches = prod(batch_shape) if batch_shape else 1
        output_shape = (*batch_shape, self.num_steps, *self.lower.shape)

        if num_batches == 0:
            return self.lower.new_empty(output_shape)

        flat_lower = self.lower.reshape(-1)
        flat_upper = self.upper.reshape(-1)

        limits = torch.stack((flat_lower, flat_upper), dim=-1)

        # [B * num_values, T]
        trajectories = generate_bounded_sine_trajectories(
            limits=limits.repeat(num_batches, 1),
            num_steps=self.num_steps,
            min_frequency=self.min_frequency.repeat(num_batches),
            max_frequency=self.max_frequency.repeat(num_batches),
            num_components=self.num_components,
            min_amplitude=self.min_amplitude,
            max_amplitude=self.max_amplitude,
            workspace_fraction=self.workspace_fraction,
        )

        # [B, num_values, T] -> [B, T, num_values]
        trajectories = trajectories.view(
            num_batches,
            flat_lower.numel(),
            self.num_steps,
        ).transpose(1, 2)

        # [B, T, num_values] -> [*batch_shape, T, N, *event_shape]
        return trajectories.reshape(output_shape)


def generate_bounded_sine_trajectories(
    limits: torch.Tensor,              # [K, 2]
    num_steps: int,
    *,
    min_frequency: float | torch.Tensor = 0.0,
    max_frequency: float | torch.Tensor = 0.5,
    num_components: int = 5,
    min_amplitude: float = 1.0,
    max_amplitude: float = 5.0,
    workspace_fraction: float = 1.0,
) -> torch.Tensor:
    """Generate K independent bounded 1D trajectories with shape [K, num_steps]."""

    if limits.ndim != 2 or limits.shape[1] != 2:
        raise ValueError("limits must have shape [num_trajectories, 2].")
    if num_steps < 1 or num_components < 1:
        raise ValueError("num_steps and num_components must be at least 1.")
    if not 0.0 < workspace_fraction <= 1.0:
        raise ValueError("workspace_fraction must be in (0, 1].")

    limits = limits.float()
    low, high = limits.unbind(dim=-1)

    if torch.any(high <= low):
        raise ValueError("Each limit row must satisfy low < high.")

    k = limits.shape[0]
    device, dtype = limits.device, limits.dtype

    min_frequency = torch.as_tensor(
        min_frequency, device=device, dtype=dtype
    ).expand(k)

    max_frequency = torch.as_tensor(
        max_frequency, device=device, dtype=dtype
    ).expand(k)

    if torch.any(min_frequency < 0.0) or torch.any(max_frequency < min_frequency):
        raise ValueError("Frequencies must satisfy 0 <= min_frequency <= max_frequency.")

    t = torch.linspace(0.0, 1.0, num_steps, device=device, dtype=dtype)[None, None, :]

    amplitudes = min_amplitude + (
        max_amplitude - min_amplitude
    ) * torch.rand(k, num_components, 1, device=device, dtype=dtype)

    frequencies = min_frequency[:, None, None] + (
        max_frequency - min_frequency
    )[:, None, None] * torch.rand(
        k, num_components, 1, device=device, dtype=dtype
    )

    phases = 2.0 * torch.pi * torch.rand(
        k, num_components, 1, device=device, dtype=dtype
    )

    signal = (
        amplitudes * torch.sin(2.0 * torch.pi * frequencies * t + phases)
    ).sum(dim=1)

    normalized = signal / amplitudes.sum(dim=1).clamp_min_(1e-8)

    center = (low + high) * 0.5
    half_range = (high - low) * 0.5 * workspace_fraction

    return center[:, None] + half_range[:, None] * normalized