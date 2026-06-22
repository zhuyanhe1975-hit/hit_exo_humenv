from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

from hit_exo_humenv.latent_z_config import cfg_path


@dataclass(frozen=True)
class Candidate:
    name: str
    power_weight: float
    smoothness_weight: float
    entropy_coef: float
    init_std: float
    action_repeat: int
    num_steps_per_env: int


@dataclass(frozen=True)
class Target:
    min_saved_percent: float
    min_efficiency: float
    max_net_input_change_percent: float
    max_total_fallen: int


def default_candidates(preset: str) -> list[Candidate]:
    smoke = [
        Candidate(
            name="smoke_balanced",
            power_weight=-0.001,
            smoothness_weight=-0.02,
            entropy_coef=0.005,
            init_std=0.8,
            action_repeat=1,
            num_steps_per_env=16,
        )
    ]
    if preset == "smoke":
        return smoke
    return [
        Candidate(
            name="balanced",
            power_weight=-0.001,
            smoothness_weight=-0.02,
            entropy_coef=0.005,
            init_std=0.8,
            action_repeat=1,
            num_steps_per_env=16,
        ),
        Candidate(
            name="power_focus",
            power_weight=-0.002,
            smoothness_weight=-0.01,
            entropy_coef=0.003,
            init_std=0.6,
            action_repeat=1,
            num_steps_per_env=32,
        ),
        Candidate(
            name="smooth_light",
            power_weight=-0.0015,
            smoothness_weight=-0.005,
            entropy_coef=0.003,
            init_std=0.6,
            action_repeat=1,
            num_steps_per_env=32,
        ),
        Candidate(
            name="s1_repeat2",
            power_weight=-0.0015,
            smoothness_weight=-0.01,
            entropy_coef=0.003,
            init_std=0.6,
            action_repeat=2,
            num_steps_per_env=32,
        ),
    ]


def target_passed(row: dict[str, object], target: Target) -> bool:
    saved = _float_or_nan(row.get("human_abs_power_saved_percent"))
    efficiency = _float_or_nan(row.get("assist_efficiency_human_saved_per_exo_abs_power"))
    net_change = _float_or_nan(row.get("net_system_input_power_change_percent"))
    total_fallen = int(float(row.get("total_fallen", 0) or 0))
    return (
        not math.isnan(saved)
        and not math.isnan(efficiency)
        and not math.isnan(net_change)
        and saved >= target.min_saved_percent
        and efficiency >= target.min_efficiency
        and net_change <= target.max_net_input_change_percent
        and total_fallen <= target.max_total_fallen
    )


def rank_rows(rows: Iterable[dict[str, object]], target: Target) -> list[dict[str, object]]:
    ranked = list(rows)

    def key(row: dict[str, object]) -> tuple[float, float, float, float, float]:
        saved = _float_or_nan(row.get("human_abs_power_saved_percent"))
        efficiency = _float_or_nan(row.get("assist_efficiency_human_saved_per_exo_abs_power"))
        net_change = _float_or_nan(row.get("net_system_input_power_change_percent"))
        fallen = _float_or_nan(row.get("total_fallen"))
        return (
            1.0 if target_passed(row, target) else 0.0,
            _nan_to_low(saved),
            _nan_to_low(efficiency),
            -_nan_to_high(net_change),
            -_nan_to_high(fallen),
        )

    ranked.sort(key=key, reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
        row["passed_target"] = target_passed(row, target)
    return ranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train short latent-z assist candidates, evaluate power metrics, and rank by convergence outcome."
    )
    parser.add_argument("--preset", choices=("default", "smoke"), default="default")
    parser.add_argument("--output-root", type=Path, default=Path("logs/eval/train_eval_sweep"))
    parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl_sweep"))
    parser.add_argument("--num-envs", type=int, default=cfg_path("train", "num_envs"))
    parser.add_argument("--max-iterations", type=int, default=150)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--eval-num-envs", type=int, default=cfg_path("eval", "num_envs"))
    parser.add_argument("--eval-steps", type=int, default=cfg_path("eval", "steps"))
    parser.add_argument("--seed", type=int, default=cfg_path("eval", "seed"))
    parser.add_argument("--gpu-ids", default="0", help="'0', '0,1', 'all', or 'cpu'.")
    parser.add_argument("--dry-run", action="store_true", help="Only write the plan; do not train or evaluate.")
    parser.add_argument("--skip-baseline", action="store_true", help="Reuse --baseline-csv instead of running zero-exo baseline.")
    parser.add_argument("--baseline-csv", type=Path, default=None)
    parser.add_argument("--min-saved-percent", type=float, default=20.0)
    parser.add_argument("--min-efficiency", type=float, default=1.0)
    parser.add_argument("--max-net-input-change-percent", type=float, default=0.0)
    parser.add_argument("--max-total-fallen", type=int, default=0)
    parser.add_argument("--keep-going", action="store_true", help="Continue training/eval after the first passing candidate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = Target(
        min_saved_percent=args.min_saved_percent,
        min_efficiency=args.min_efficiency,
        max_net_input_change_percent=args.max_net_input_change_percent,
        max_total_fallen=args.max_total_fallen,
    )
    candidates = default_candidates(args.preset)
    run_dir = args.output_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    plan_path = run_dir / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "target": asdict(target),
                "train": {
                    "num_envs": args.num_envs,
                    "max_iterations": args.max_iterations,
                    "save_interval": args.save_interval,
                    "log_root": str(args.log_root),
                    "gpu_ids": args.gpu_ids,
                },
                "eval": {
                    "num_envs": args.eval_num_envs,
                    "steps": args.eval_steps,
                    "seed": args.seed,
                },
                "candidates": [asdict(candidate) for candidate in candidates],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"[INFO] sweep plan: {plan_path}")
    if args.dry_run:
        print("[INFO] dry-run enabled; no training/evaluation launched.")
        return

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        print(f"[INFO] training candidate: {candidate.name}")
        train_log_dir = _train_candidate(candidate, args)
        checkpoint = _latest_checkpoint(train_log_dir)
        candidate_dir = run_dir / candidate.name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        assisted_csv = candidate_dir / "assisted_trained.csv"
        analysis_json = candidate_dir / "assist_power.json"
        report_md = candidate_dir / "report.md"

        _run_eval(
            agent="trained",
            output_csv=assisted_csv,
            args=args,
            checkpoint_file=checkpoint,
            action_repeat=candidate.action_repeat,
        )
        baseline_csv = _baseline_csv(args, run_dir, candidate.action_repeat)
        _run_analyze(baseline_csv, assisted_csv, analysis_json)
        _run_report(analysis_json, report_md)
        row = _summary_row(candidate, train_log_dir, checkpoint, assisted_csv, analysis_json)
        rows.append(row)
        ranked = rank_rows(rows, target)
        _write_summary(run_dir, ranked, target)
        if target_passed(row, target) and not args.keep_going:
            print(f"[INFO] target reached by {candidate.name}; stopping early.")
            break

    ranked = rank_rows(rows, target)
    _write_summary(run_dir, ranked, target)
    if ranked:
        best = ranked[0]
        print(
            "[INFO] best candidate: "
            f"{best['candidate']} saved={best['human_abs_power_saved_percent']:.2f}% "
            f"eff={best['assist_efficiency_human_saved_per_exo_abs_power']:.3f} "
            f"net={best['net_system_input_power_change_percent']:.2f}%"
        )
    print(f"[INFO] sweep report: {run_dir / 'report.md'}")


def _baseline_csv(args: argparse.Namespace, run_dir: Path, action_repeat: int) -> Path:
    if args.skip_baseline:
        if args.baseline_csv is None:
            raise ValueError("--skip-baseline requires --baseline-csv")
        return args.baseline_csv
    if args.baseline_csv is not None:
        return args.baseline_csv
    output_csv = run_dir / f"baseline_zero_s1_repeat{action_repeat}.csv"
    if output_csv.exists():
        return output_csv
    _run_eval(
        agent="zero",
        output_csv=output_csv,
        args=args,
        checkpoint_file=None,
        action_repeat=action_repeat,
    )
    return output_csv


def _train_candidate(candidate: Candidate, args: argparse.Namespace) -> Path:
    import hit_exo_humenv.mjlab  # noqa: F401
    from mjlab.scripts.train import TrainConfig, launch_training

    cfg = TrainConfig.from_task(cfg_path("task_id"))
    cfg = replace(cfg, log_root=str(args.log_root), gpu_ids=_gpu_ids(args.gpu_ids))
    cfg.env.scene.num_envs = args.num_envs
    cfg.env.actions["human_s1"].action_repeat = candidate.action_repeat
    cfg.env.rewards["lower_limb_joint_power_cost"].weight = candidate.power_weight
    cfg.env.rewards["lower_limb_joint_velocity_delta_l2"].weight = candidate.smoothness_weight
    cfg.agent.run_name = candidate.name
    cfg.agent.max_iterations = args.max_iterations
    cfg.agent.save_interval = args.save_interval
    cfg.agent.num_steps_per_env = candidate.num_steps_per_env
    cfg.agent.logger = "tensorboard"
    cfg.agent.algorithm.entropy_coef = candidate.entropy_coef
    cfg.agent.actor.distribution_cfg["init_std"] = candidate.init_std
    launch_training(cfg_path("task_id"), cfg)
    return _latest_run_dir(args.log_root, cfg.agent.experiment_name, candidate.name)


def _run_eval(
    *,
    agent: str,
    output_csv: Path,
    args: argparse.Namespace,
    checkpoint_file: Path | None,
    action_repeat: int,
) -> None:
    cmd = [
        sys.executable,
        "scripts/eval_latent_z_power.py",
        "--agent",
        agent,
        "--output-csv",
        str(output_csv),
        "--num-envs",
        str(args.eval_num_envs),
        "--steps",
        str(args.eval_steps),
        "--seed",
        str(args.seed),
        "--human-action-repeat",
        str(action_repeat),
    ]
    if checkpoint_file is not None:
        cmd.extend(["--checkpoint-file", str(checkpoint_file)])
    _run(cmd)


def _run_analyze(baseline_csv: Path, assisted_csv: Path, output_json: Path) -> None:
    _run(
        [
            sys.executable,
            "scripts/analyze_assist_power.py",
            str(assisted_csv),
            "--before",
            str(baseline_csv),
            "--output",
            str(output_json),
            "--drop-fallen",
        ]
    )


def _run_report(analysis_json: Path, report_md: Path) -> None:
    _run(
        [
            sys.executable,
            "scripts/write_latent_z_power_report.py",
            str(analysis_json),
            "--output",
            str(report_md),
        ]
    )


def _summary_row(
    candidate: Candidate,
    train_log_dir: Path,
    checkpoint: Path,
    assisted_csv: Path,
    analysis_json: Path,
) -> dict[str, object]:
    data = json.loads(analysis_json.read_text())
    comparison = data["comparison"]
    after = data["after"]
    summary_path = assisted_csv.with_suffix(".summary.json")
    rollout_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return {
        "candidate": candidate.name,
        "power_weight": candidate.power_weight,
        "smoothness_weight": candidate.smoothness_weight,
        "entropy_coef": candidate.entropy_coef,
        "init_std": candidate.init_std,
        "action_repeat": candidate.action_repeat,
        "num_steps_per_env": candidate.num_steps_per_env,
        "human_abs_power_saved_percent": comparison["human_abs_power_saved_percent"],
        "human_abs_power_saved_w": comparison["human_abs_power_saved_w"],
        "assist_efficiency_human_saved_per_exo_abs_power": comparison[
            "assist_efficiency_human_saved_per_exo_abs_power"
        ],
        "net_system_input_power_change_percent": comparison["net_system_input_power_change_percent"],
        "assisted_exo_abs_power_w": after["exo_knee_abs_power_mean_w"],
        "total_fallen": rollout_summary.get("total_fallen", 0),
        "train_log_dir": str(train_log_dir),
        "checkpoint": str(checkpoint),
        "analysis_json": str(analysis_json),
    }


def _write_summary(run_dir: Path, ranked: list[dict[str, object]], target: Target) -> None:
    summary_json = run_dir / "summary.json"
    summary_csv = run_dir / "summary.csv"
    report_md = run_dir / "report.md"
    summary_json.write_text(json.dumps(ranked, indent=2, ensure_ascii=False) + "\n")
    if ranked:
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(ranked[0]))
            writer.writeheader()
            writer.writerows(ranked)
    lines = [
        "# 训练-评估扫参结果",
        "",
        f"目标：节省 >= {target.min_saved_percent:.1f}%，助力效率 >= {target.min_efficiency:.2f} W/W，"
        f"人机总输入变化 <= {target.max_net_input_change_percent:.1f}%，跌倒数 <= {target.max_total_fallen}。",
        "",
        "| 排名 | 候选 | 达标 | 节省比例 | 助力效率 | 人机总输入变化 | 跌倒数 | checkpoint |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for row in ranked:
        lines.append(
            "| {rank} | {candidate} | {passed} | {saved:.2f}% | {eff:.3f} | {net:.2f}% | {fallen} | `{checkpoint}` |".format(
                rank=row["rank"],
                candidate=row["candidate"],
                passed="是" if row["passed_target"] else "否",
                saved=_float_or_nan(row["human_abs_power_saved_percent"]),
                eff=_float_or_nan(row["assist_efficiency_human_saved_per_exo_abs_power"]),
                net=_float_or_nan(row["net_system_input_power_change_percent"]),
                fallen=int(float(row.get("total_fallen", 0) or 0)),
                checkpoint=row["checkpoint"],
            )
        )
    report_md.write_text("\n".join(lines) + "\n")


def _latest_run_dir(log_root: Path, experiment_name: str, run_name: str) -> Path:
    candidates = sorted(
        (log_root / experiment_name).glob(f"*_{run_name}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No run directory found for {run_name} under {log_root / experiment_name}")
    return candidates[0]


def _latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoint found in {run_dir}")
    return checkpoints[0]


def _gpu_ids(value: str):
    if value == "all":
        return "all"
    if value == "cpu":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _run(cmd: list[str]) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _float_or_nan(value: object) -> float:
    if value is None:
        return math.nan
    return float(value)


def _nan_to_low(value: float) -> float:
    return -1.0e30 if math.isnan(value) else value


def _nan_to_high(value: float) -> float:
    return 1.0e30 if math.isnan(value) else value


if __name__ == "__main__":
    main()
