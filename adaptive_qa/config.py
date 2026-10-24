"""Configuration loading: one JSON tree, sections per component.

The brief requires separate settings for answering model, retrieval model,
embedding model, dataset and decision classifier: the sections
``answering``, ``retrieval`` (with the optional ``embedding_model`` section),
``dataset`` and ``decision`` provide exactly that separation.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load a config file and deep-merge it over ``configs/default.json``.

    Any section the user config omits stays the documented default, so runs
    are reproducible with minimal config files.
    """
    defaults = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    if path is None:
        return defaults
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    user_cfg = json.loads(p.read_text(encoding="utf-8"))
    merged = _deep_merge(defaults, user_cfg)
    return merged


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_path(path_value: str) -> Path:
    """Resolve a possibly-relative path in the config against the project root."""
    p = Path(path_value)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def data_root() -> Path:
    return PROJECT_ROOT / "data"
