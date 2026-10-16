"""Load YAML config and resolve paths relative to the project root."""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str = "data.yaml") -> dict:
    with open(PROJECT_ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path