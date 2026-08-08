from __future__ import annotations

import yaml


def load_config(path: str = "configs/default.yaml") -> dict:
    """Load a YAML configuration file and return it as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)
