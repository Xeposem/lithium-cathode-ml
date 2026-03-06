"""YAML config loader, seed setter, and path resolver."""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from dotenv import load_dotenv


def load_config(config_path: str = "configs/data.yaml") -> dict:
    """Load YAML configuration file.

    Loads environment variables from .env file first, then reads
    the YAML config and returns it as a dict.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config_path does not exist.
    """
    load_dotenv()
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def set_seeds(config: dict) -> None:
    """Set random seeds for reproducibility.

    Reads seed values from config["random_seeds"] and sets
    Python random and numpy random seeds.

    Args:
        config: Configuration dictionary with random_seeds section.
    """
    seeds = config["random_seeds"]
    random.seed(seeds["python"])
    np.random.seed(seeds["numpy"])


def get_project_root() -> Path:
    """Return the project root directory.

    Returns:
        Path to the project root (parent of cathode_ml package).
    """
    return Path(__file__).parent.parent
