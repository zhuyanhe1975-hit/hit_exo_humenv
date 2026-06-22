#!/usr/bin/env python
from __future__ import annotations

import hit_exo_humenv.mjlab  # noqa: F401
from hit_exo_humenv.mjlab.walking_env_cfg import TASK_ID
from mjlab.scripts.train import launch_training


def main() -> None:
    launch_training(TASK_ID)


if __name__ == "__main__":
    main()
