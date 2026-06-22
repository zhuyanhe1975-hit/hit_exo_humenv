from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "latent_z.json"


@lru_cache(maxsize=1)
def latent_z_config() -> dict[str, Any]:
    with DEFAULT_CONFIG_PATH.open() as f:
        return json.load(f)


def cfg_path(*keys: str) -> Any:
    value: Any = latent_z_config()
    for key in keys:
        value = value[key]
    return value


def cfg_tuple(*keys: str) -> tuple[Any, ...]:
    return tuple(cfg_path(*keys))
