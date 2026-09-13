from __future__ import annotations

import torch

from myo_core.common.sequential_task import SequentialTaskLogic, TaskEntity


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

