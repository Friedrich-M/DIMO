"""YAML configuration merged with ``key=value`` command-line overrides."""

import argparse
import os
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml")


def load_config(argv: Optional[List[str]] = None, description: str = "",
                require_object: bool = True) -> DictConfig:
    """Load the YAML config and apply ``key=value`` overrides.

    ``require_object`` is False for the utilities that do not work on one object's working directory
    (``check_llm.py``, ``caption_dataset.py``), so they do not demand an unused ``object.name``.
    """
    parser = argparse.ArgumentParser(description=description, epilog="Any other `key=value` (or `section.key=value`) argument overrides the config.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to the YAML config file")
    args, overrides = parser.parse_known_args(argv)
    base = OmegaConf.load(args.config)
    # Struct mode makes a misspelled override an error instead of a silently added key. Whole GPU
    # stages hinge on `clips.num_frames` and `sv4d.model`, and `clips.num_frame=21` used to be
    # accepted while changing nothing.
    OmegaConf.set_struct(base, True)
    try:
        cfg = OmegaConf.merge(base, OmegaConf.from_cli(overrides))
    except Exception as e:
        key = getattr(e, "full_key", None) or str(e)
        raise SystemExit(f"unknown config key {key!r}. Check the spelling against {args.config}, "
                         f"which lists every available key.")
    OmegaConf.set_struct(cfg, False)
    if require_object and not cfg.object.name:
        raise SystemExit("`object.name` must be set (e.g. object.name=trump)")
    return cfg


def object_dir(cfg: DictConfig) -> str:
    """Working directory holding every intermediate result of one object."""
    return os.path.join(cfg.workdir, cfg.object.name)


def multiview_dir(cfg: DictConfig) -> str:
    """Multi-view renders of one object, kept per model so layouts cannot be mixed up."""
    return os.path.join(object_dir(cfg), "multiview", cfg.sv4d.model)
