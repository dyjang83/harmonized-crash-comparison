"""Loader for config.yaml.

config.yaml documented `extraction.canonical_model` from the start, but nothing
ever read it -- build_features.load() picked whichever extraction row happened to
appear first in the JSONL instead (see load(model=...)). This module makes the
declared value load-bearing so the canonical model is stated in exactly one place.

Lookups are dotted paths with an explicit default:

    canonical_model()                       -> "claude-sonnet-4-6"
    get("lift_test.n_boot", 5000)           -> 5000
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

# Repo root = two levels up from src/utils/config.py
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.environ.get("AV_CRASH_CONFIG", os.path.join(_ROOT, "config.yaml"))


@lru_cache(maxsize=None)
def load(path: str = CONFIG_PATH) -> dict:
    """Parse config.yaml. Returns {} if it is absent or pyyaml is not installed.

    A missing config is not an error: every caller passes an explicit default, so
    the pipeline stays runnable from a bare checkout. What must never happen is a
    *silent wrong* value, which is why get() takes the default at the call site
    rather than burying fallbacks here.
    """
    try:
        import yaml
    except ImportError:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get(dotted: str, default: Any = None, path: str = CONFIG_PATH) -> Any:
    node: Any = load(path)
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node is None else node


def canonical_model(default: str = "claude-sonnet-4-6") -> str:
    """The extraction model whose output builds T for the lift test."""
    return str(get("extraction.canonical_model", default))
