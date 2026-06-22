"""mjlab task registration for hit-exo-humenv."""

from hit_exo_humenv.mjlab.walking_env_cfg import (
    TASK_ID,
    humenv_knee_exo_walking_env_cfg,
    humenv_knee_exo_walking_ppo_cfg,
)

try:
    from mjlab.tasks.registry import register_mjlab_task

    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=humenv_knee_exo_walking_env_cfg(play=False),
        play_env_cfg=humenv_knee_exo_walking_env_cfg(play=True),
        rl_cfg=humenv_knee_exo_walking_ppo_cfg(),
        runner_cls=None,
    )
except Exception as exc:  # pragma: no cover - keeps imports usable outside mjlab envs.
    MJLAB_REGISTRATION_ERROR = exc
else:
    MJLAB_REGISTRATION_ERROR = None

__all__ = [
    "TASK_ID",
    "MJLAB_REGISTRATION_ERROR",
    "humenv_knee_exo_walking_env_cfg",
    "humenv_knee_exo_walking_ppo_cfg",
]
