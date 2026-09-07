"""Configuration: a YAML file merged with ``key=value`` command-line overrides (OmegaConf)."""

import argparse
import os
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml")


def load_config(argv: Optional[List[str]] = None, description: str = "") -> DictConfig:
    parser = argparse.ArgumentParser(description=description, epilog="Any other `key=value` argument overrides the config.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to the YAML config file")
    args, overrides = parser.parse_known_args(argv)
    base = OmegaConf.load(args.config)
    # Struct mode turns a misspelled override into an error rather than a silently added key, so
    # `iters_s2=30000` cannot quietly leave the real `iters_s2` at its default for a whole run.
    OmegaConf.set_struct(base, True)
    try:
        cfg = OmegaConf.merge(base, OmegaConf.from_cli(overrides))
    except Exception as e:
        key = getattr(e, "full_key", None) or str(e)
        raise SystemExit(f"unknown config key {key!r}. Check the spelling against {args.config}, "
                         f"which lists every available key.")
    OmegaConf.set_struct(cfg, False)
    return cfg


def save_config(cfg: DictConfig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        OmegaConf.save(cfg, f)
