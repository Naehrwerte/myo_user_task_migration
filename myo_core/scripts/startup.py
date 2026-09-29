from typing import Sequence

import sys
import shlex
from collections.abc import Callable
from pathlib import Path
from argparse import ArgumentParser

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from myo_core import MyoConfig, myo_mjlab_register
from .wandb_helper import patch_wandb


# Only uncomment for debug (20% slow-down)
# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"


def split_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """
    Split command-line arguments into Hydra args and MJLab args.

    Arguments before `--` are treated as Hydra overrides.
    Arguments after `--` are forwarded to MJLab/tyro.

    Example:
        task=universal vision=yolo -- MyoUser --max-iterations 500
    """
    argv = list(argv)

    if "--" not in argv:
        return argv, []

    sep = argv.index("--")
    return argv[:sep], argv[sep + 1:]


def parse_compose_args(
    argv: Sequence[str],
    *,
    default_config_dir: Path,
) -> tuple[Path, str, list[str]]:
    parser = ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--config-path", "-cp")
    parser.add_argument("--config-name", "-cn", default="base")

    # Known flags are consumed; all Hydra overrides remain here.
    args, overrides = parser.parse_known_args(argv)

    config_dir = (
        Path(args.config_path).expanduser()
        if args.config_path is not None
        else default_config_dir
    )

    # Define relative --config-path relative to the current working directory.
    if not config_dir.is_absolute():
        config_dir = Path.cwd() / config_dir

    config_dir = config_dir.resolve()

    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory does not exist: {config_dir}")

    return config_dir, args.config_name, overrides


def startup(mjlab_main: Callable[[], int | None]) -> None:
    """
    Initialize Hydra, register Myo components, patch W&B, and run MJLab.

    Hydra receives the arguments before `--`. MJLab receives the arguments after
    `--`, with `MyoUser` inserted automatically when missing.
    """
    sys.argv[0] = sys.argv[0].removesuffix(".exe")

    entrypoint = Path(sys.argv[0]).name
    hydra_args, mjlab_args = split_args(sys.argv[1:])

    root = Path(__file__).resolve().parents[2]

    config_dir, config_name, overrides = parse_compose_args(
        hydra_args,
        default_config_dir=root / "myo_config",
    )

    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir),
        job_name="myo_startup",
    ):
        cfg = compose(
            config_name=config_name,
            overrides=overrides,
            return_hydra_config=True,
        )

    HydraConfig.instance().set_config(cfg)

    task_cfg = cfg.copy()
    with open_dict(task_cfg):
        task_cfg.pop("hydra", None)

    myo_cfg: MyoConfig = OmegaConf.to_object(task_cfg)
    myo_mjlab_register(myo_cfg)

    if "MyoUser" not in mjlab_args:
        mjlab_args = ["MyoUser", *mjlab_args]

    full_argv = [sys.argv[0], *hydra_args, "--", *mjlab_args]
    launch_command = shlex.join(["uv", "run", entrypoint, *hydra_args, "--", *mjlab_args])

    patch_wandb(
        full_argv=full_argv,
        launch_command=launch_command,
        hydra_cfg=task_cfg,
        myo_cfg=myo_cfg,
    )

    sys.argv = [sys.argv[0], *mjlab_args]
    exit_code = mjlab_main()

    sys.exit(0 if exit_code is None else exit_code)
