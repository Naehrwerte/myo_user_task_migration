from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import RecorderTermCfg, RecorderTerm

_PATH = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class H5EvalRecorder(RecorderTerm):
    """Streams evaluation states and transitions to an HDF5 file.

    File layout:
      states/<name>:
          One row per recorded policy state.

      transitions/<name>:
          One row per environment step. `source_state_row` points into
          `states`; `next_state_row` points to the post-action state, including
          for terminal transitions.

    Override `_collect_state_tensors()` to record the exact tensors used by
    the attention module rather than a flattened actor observation.
    """

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        path = Path(f"record/{_PATH}.h5")
        path.parent.mkdir(parents=True, exist_ok=True)

        self._file = h5py.File(path, "w")
        self._file.attrs["format"] = "mjlab_eval_hdf5_v1"
        self._file.attrs["created_at"] = datetime.now().astimezone().isoformat()
        self._file.attrs["num_envs"] = int(env.num_envs)

        # Keep timing information next to episode_step so analyses do not have
        # to reconstruct seconds from the launch configuration.
        step_dt = getattr(env, "step_dt", None)
        if step_dt is not None:
            self._file.attrs["step_dt_seconds"] = float(step_dt)

        max_episode_length = getattr(env, "max_episode_length", None)
        if max_episode_length is not None:
            self._file.attrs["max_episode_length_steps"] = int(max_episode_length)

        # Components may attach stable run provenance without coupling this
        # generic recorder to Hydra, W&B, or a particular launcher.
        metadata = cfg.params.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("Recorder metadata must be a dictionary.")
        metadata_group = self._file.require_group("metadata")
        metadata_group.attrs["launcher_argv_json"] = json.dumps(sys.argv)

        env_cfg = getattr(env, "cfg", None)
        evaluation_seed = getattr(env_cfg, "seed", None)
        if evaluation_seed is not None:
            metadata_group.attrs["environment_seed"] = int(evaluation_seed)

        for name, value in metadata.items():
            if value is not None:
                metadata_group.attrs[name] = value

        self._datasets: dict[tuple[str, str], h5py.Dataset] = {}
        self._row_counts = {
            "states": 0,
            "transitions": 0,
        }

        # Larger chunks reduce metadata and write overhead. It does not retain
        # this amount in RAM; it only determines on-disk HDF5 chunk layout.
        self._chunk_rows = int(cfg.params.get("chunk_rows", 4096))
        self._compression = cfg.params.get("compression", None)
        self._flush_every = int(cfg.params.get("flush_every", 64))
        self._writes_since_flush = 0

        device = env.device
        num_envs = env.num_envs

        # Per-environment bookkeeping remains on GPU and is tiny.
        self._episode_id = torch.full(
            (num_envs,),
            -1,
            device=device,
            dtype=torch.int64,
        )
        self._episode_step = torch.zeros(
            num_envs,
            device=device,
            dtype=torch.int32,
        )
        self._last_state_row = torch.full(
            (num_envs,),
            -1,
            device=device,
            dtype=torch.int64,
        )

    def _collect_state_tensors(
        self,
        env_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return individual observation terms for the selected environments."""
        observation_manager = self._env.observation_manager
        reward_manager = self._env.reward_manager
        termination_manager = self._env.termination_manager

        num_envs = env_ids.numel()

        state: dict[str, torch.Tensor] = {}

        for group_name, group_data in self._env.obs_buf.items():
            term_names = observation_manager.active_terms[group_name]
            term_shapes = observation_manager.group_obs_term_dim[group_name]
            term_sizes = [math.prod(shape) for shape in term_shapes]

            if isinstance(group_data, torch.Tensor):
                group_flat = group_data[env_ids].reshape(num_envs, -1)

                if group_flat.shape[1] != sum(term_sizes):
                    raise ValueError(
                        f"Observation group {group_name!r} has shape "
                        f"{tuple(group_flat.shape)}, expected "
                        f"{sum(term_sizes)} values per environment."
                    )

                term_tensors = torch.split(group_flat, term_sizes, dim=1)

            else:
                term_tensors = (
                    group_data[term_name][env_ids]
                    for term_name in term_names
                )

            for term_name, term_shape, term_tensor in zip(
                term_names,
                term_shapes,
                term_tensors,
                strict=True,
            ):
                state[f"observation/{group_name}/{term_name}"] = term_tensor.reshape(
                    num_envs,
                    *term_shape,
                )

        for reward_index, reward_name in enumerate(reward_manager.active_terms):
            weight = reward_manager._term_cfgs[reward_index].weight
            if weight != 0:
                reward_tensor = reward_manager._step_reward[env_ids, reward_index]
                state[f"reward/{reward_name}"] = reward_tensor

        for termination_name in termination_manager.active_terms:
            termination_tensor = termination_manager.get_term(termination_name)
            state[f"termination/{termination_name}"] = termination_tensor[env_ids]

        return state

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        tensor = tensor.detach()

        # NumPy/HDF5 do not support bfloat16 directly.
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float16)

        return tensor.to(device="cpu").contiguous().numpy()

    def _append(
        self,
        group_name: str,
        tensors: dict[str, torch.Tensor],
    ) -> tuple[int, int]:
        """Append a batch along axis 0 and return [start, end) row IDs."""
        if not tensors:
            raise ValueError("Cannot append an empty tensor batch.")

        arrays: dict[str, np.ndarray] = {}
        batch_size: int | None = None

        for name, tensor in tensors.items():
            if tensor.ndim == 0:
                raise ValueError(
                    f"{group_name}/{name} must have a leading batch dimension."
                )

            if batch_size is None:
                batch_size = tensor.shape[0]
            elif tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Batch-size mismatch in {group_name}: "
                    f"{name} has {tensor.shape[0]}, expected {batch_size}."
                )

            arrays[name] = self._to_numpy(tensor)

        assert batch_size is not None

        start = self._row_counts[group_name]
        end = start + batch_size
        group = self._file.require_group(group_name)

        for name, array in arrays.items():
            key = (group_name, name)
            dataset = self._datasets.get(key)

            if dataset is None:
                chunk_rows = min(max(batch_size, 1), self._chunk_rows)

                create_kwargs: dict[str, object] = {
                    "shape": (0, *array.shape[1:]),
                    "maxshape": (None, *array.shape[1:]),
                    "chunks": (chunk_rows, *array.shape[1:]),
                    "dtype": array.dtype,
                }

                # Leave this as None for detector features initially. Random
                # float activations usually compress poorly and compression can
                # become the bottleneck.
                if self._compression is not None:
                    create_kwargs["compression"] = self._compression

                dataset = group.create_dataset(name, **create_kwargs)
                self._datasets[key] = dataset

            if dataset.shape[1:] != array.shape[1:]:
                raise ValueError(
                    f"Shape changed for {group_name}/{name}: "
                    f"expected {dataset.shape[1:]}, got {array.shape[1:]}."
                )

            dataset.resize((end, *dataset.shape[1:]))
            dataset[start:end] = array

        self._row_counts[group_name] = end
        self._writes_since_flush += 1

        if self._writes_since_flush >= self._flush_every:
            self._file.flush()
            self._writes_since_flush = 0

        return start, end

    def _append_state(
        self,
        env_ids: torch.Tensor,
        *,
        episode_step: torch.Tensor,
        is_initial: bool,
        is_terminal: bool = False,
    ) -> torch.Tensor:
        state = self._collect_state_tensors(env_ids)
        num_rows = env_ids.numel()

        state.update(
            env_id=env_ids.to(torch.int32),
            episode_id=self._episode_id[env_ids],
            episode_step=episode_step,
            is_initial=torch.full(
                (num_rows,),
                is_initial,
                device=env_ids.device,
                dtype=torch.bool,
            ),
            is_terminal=torch.full(
                (num_rows,),
                is_terminal,
                device=env_ids.device,
                dtype=torch.bool,
            ),
        )

        start, end = self._append("states", state)

        row_ids = torch.arange(
            start,
            end,
            device=env_ids.device,
            dtype=torch.int64,
        )
        self._last_state_row[env_ids] = row_ids

        return row_ids

    def _append_transition(
        self,
        env_ids: torch.Tensor,
        *,
        source_state_rows: torch.Tensor,
        next_state_rows: torch.Tensor,
        done: bool,
    ) -> None:
        num_rows = env_ids.numel()

        if done:
            terminated = self._env.reset_terminated[env_ids]
            timed_out = self._env.reset_time_outs[env_ids]
        else:
            terminated = torch.zeros(
                num_rows,
                device=env_ids.device,
                dtype=torch.bool,
            )
            timed_out = torch.zeros(
                num_rows,
                device=env_ids.device,
                dtype=torch.bool,
            )

        self._append(
            "transitions",
            {
                "source_state_row": source_state_rows,
                "next_state_row": next_state_rows,
                "env_id": env_ids.to(torch.int32),
                "episode_id": self._episode_id[env_ids],
                "episode_step": self._episode_step[env_ids],
                "action": self._env.action_manager.action[env_ids],
                "reward": self._env.reward_buf[env_ids],
                "done": torch.full(
                    (num_rows,),
                    done,
                    device=env_ids.device,
                    dtype=torch.bool,
                ),
                "terminated": terminated,
                "timed_out": timed_out,
            },
        )

    def record_post_reset(self, env_ids: torch.Tensor) -> None:
        """Write the initial state of every newly created episode."""
        if env_ids.numel() == 0:
            return

        self._episode_id[env_ids] += 1
        self._episode_step[env_ids] = 0

        self._append_state(
            env_ids,
            episode_step=self._episode_step[env_ids],
            is_initial=True,
        )

    def record_pre_reset(self, env_ids: torch.Tensor) -> None:
        """Write terminal post-action states and transitions before reset."""
        if env_ids.numel() == 0:
            return

        source_rows = self._last_state_row[env_ids]

        if torch.any(source_rows < 0):
            raise RuntimeError(
                "A terminal transition has no recorded source state. "
                "record_post_reset() must run before the first step."
            )

        terminal_steps = self._episode_step[env_ids] + 1
        terminal_rows = self._append_state(
            env_ids,
            episode_step=terminal_steps,
            is_initial=False,
            is_terminal=True,
        )

        self._append_transition(
            env_ids,
            source_state_rows=source_rows,
            next_state_rows=terminal_rows,
            done=True,
        )

        self._episode_step[env_ids] = terminal_steps

    def record_post_step(self) -> None:
        """Write non-terminal next states and their preceding transitions."""
        live_env_ids = torch.nonzero(
            ~self._env.reset_buf,
            as_tuple=False,
        ).squeeze(-1)

        if live_env_ids.numel() == 0:
            return

        source_rows = self._last_state_row[live_env_ids].clone()
        next_episode_steps = self._episode_step[live_env_ids] + 1

        next_rows = self._append_state(
            live_env_ids,
            episode_step=next_episode_steps,
            is_initial=False,
        )

        self._append_transition(
            live_env_ids,
            source_state_rows=source_rows,
            next_state_rows=next_rows,
            done=False,
        )

        self._episode_step[live_env_ids] = next_episode_steps

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
