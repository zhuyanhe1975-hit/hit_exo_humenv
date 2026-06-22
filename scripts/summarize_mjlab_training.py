from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from hit_exo_humenv.latent_z_config import cfg_path

DEFAULT_LOG_ROOT = Path(cfg_path("paths", "checkpoint_root"))
DEFAULT_TAGS = (
    "Train/mean_episode_length",
    "Train/mean_reward",
    "Episode_Reward/knee_pd_torque",
    "Episode_Termination/fallen",
    "Episode_Termination/time_out",
    "Train/mean_std",
    "Perf/collection_time",
    "Perf/total_fps",
)


def _last_scalar(accumulator: EventAccumulator, tag: str) -> dict[str, float] | None:
    if tag not in accumulator.Tags().get("scalars", []):
        return None
    values = accumulator.Scalars(tag)
    if not values:
        return None
    value = values[-1]
    return {"step": value.step, "wall_time": value.wall_time, "value": value.value}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize mjlab TensorBoard training logs.")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--output-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []

    for event_file in sorted(args.log_root.glob("*/events.out.tfevents*")):
        run_dir = event_file.parent
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        scalars = {
            tag: scalar
            for tag in DEFAULT_TAGS
            if (scalar := _last_scalar(accumulator, tag)) is not None
        }
        checkpoints = sorted(run_dir.glob("model_*.pt"), key=lambda path: path.stat().st_mtime)
        summaries.append(
            {
                "run": run_dir.name,
                "event_file": str(event_file),
                "latest_checkpoint": str(checkpoints[-1]) if checkpoints else None,
                "scalars": scalars,
            }
        )

    payload = {"log_root": str(args.log_root), "runs": summaries}
    text = json.dumps(payload, indent=2)
    print(text)

    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(text + "\n")


if __name__ == "__main__":
    main()
