# hit-exo-humenv

Minimal HumEnv + mjlab learning scaffold for walking with an abstract exoskeleton.

Initial scope:

- activity mode: walking only
- exoskeleton model: no rigid exo body, only extra torques at `L_Knee_x` and `R_Knee_x`
- baseline environment: Gymnasium/HumEnv wrapper
- mjlab integration: task registration/config scaffold for manager-based training

## Environment

Use the existing mjlab conda environment:

```bash
conda activate mjwarp_env
pip install -e /home/yhzhu/AI/humenv
pip install -e /home/yhzhu/mjlab
pip install -e /home/yhzhu/myWorks_vips/hit_exo_humenv
```

The current tested environment uses editable `mjlab==1.3.0` from
`/home/yhzhu/mjlab`, with `mujoco==3.8.1`, `mujoco-warp==3.8.1`, and
`rsl-rl-lib==5.2.0`.

`config/latent_z.json` is the source of truth for latent-z task defaults: task
id, log roots, speed commands, S-1 sampling, exo torque limit, simulation step,
and the metabolic power coefficients used by reward/eval scripts.

The mjlab training config defaults to the values in `config/latent_z.json`
(`4096` parallel environments and a 30 Hz control step implemented as 3 MuJoCo
substeps at 90 Hz). Override these from the CLI if you need a finer simulation:

```bash
./train_latent_z.sh --env.scene.num-envs 64 --env.sim.mujoco.timestep 0.0022222222222222222 --env.decimation 15
```

## Smoke Test

```bash
./run_latent_z.sh
```

`run_latent_z.sh` loads the newest local RSL-RL checkpoint from
`logs/rsl_rl/humenv_knee_exo_walking` and starts native MuJoCo viewer play.
It also writes the actual left/right knee PD torque to
`logs/run/knee_pd_torque/<timestamp>...csv` while the viewer is running.

## mjlab Task

The package registers:

```text
Mjlab-HumEnv-KneeExo-Walking
```

List tasks after editable install:

```bash
python -m mjlab.scripts.list_envs knee
```

Train through mjlab/RSL-RL:

```bash
./train_latent_z.sh
```

Train the exoskeleton policy with frozen S-1 human control and mocap gait
tracking rewards:

```bash
./train_mocap_track.sh
```

Run the newest mocap-track checkpoint in the viewer:

```bash
./run_mocap_track.sh
```

This keeps MetaMotivo S-1 fixed. The mocap reference is used to infer an S-1
tracking latent sequence, while the learned part is still just the
two-dimensional exoskeleton action. Override the reference motion with
`MOCAP_MOTION=/path/to/walk.hdf5 MOCAP_EPISODE=ep_0 ./train_mocap_track.sh`.
Unlike `train_latent_z.sh`, the mocap-track entrypoint removes speed-command observations.
The mocap file is treated as one fixed reference
clip. Frozen S-1 tracks the mocap `observation` sequence with MetaMotivo's
tracking inference; the exoskeleton policy still learns only the residual
knee-assist action.
The default exoskeleton limit is intentionally conservative (`25Nm`) so the
policy learns residual assistance without overpowering the frozen S-1 gait.
Use the same `MOCAP_MOTION` and `MOCAP_EPISODE` values with
`./run_mocap_track.sh` to replay the matching fixed-reference setup.

`train_latent_z.sh` samples the human walking command at episode reset and keeps
that command fixed until the next timeout or fall reset. The command currently
samples forward speeds from `0.5, 0.75, 1.0, 1.25, 1.5` m/s and only forward
direction `0` degrees, then selects the corresponding S-1 human task latent,
e.g. `move-ego-0-1.25`; it is not a velocity-tracking reward for the
exoskeleton policy.

Summarize TensorBoard scalars from all local runs:

```bash
conda run --no-capture-output -n mjwarp_env python scripts/summarize_mjlab_training.py --output-file logs/eval/training_summary.json
```

Evaluate the newest checkpoint against a zero-assist baseline without a viewer.
This writes the baseline and assisted rollout CSVs, an `assist_power.json`, and
a concise Chinese `report.md` under
`logs/eval/latent_z_power/<timestamp>_headless_compare`:

```bash
./eval.sh --num-envs 64 --steps 300
```

`run_latent_z.sh` uses the same random forward-speed command distribution as
training by default. For a fixed play command, use
`RANDOM_WALK_SPEED=0 WALK_SPEED=1.25 RANDOM_WALK_DIRECTION=0 WALK_DIRECTION=0 ./run_latent_z.sh`.
Play mode lays out parallel environments on a square-ish grid with
`ENV_SPACING=3.0` meters by default. Override it with
`ENV_SPACING=... ./run_latent_z.sh` or pass `./run_latent_z.sh --env-spacing ...`.
`run_latent_z.sh` keeps S-1 unmodified by default: `S1_LATENT_SPEED_SCALE=1.0`,
`HUMAN_ACTION_REPEAT=1`, `HUMAN_ACTION_SMOOTHING=0.0`, and
`HUMAN_ROOT_HEIGHT=0.94`.

The mjlab task is intentionally thin at this stage: it keeps the same walking-only,
knee-torque abstraction while making the config visible to mjlab/RSL-RL.

## Control Contract

The exoskeleton policy owns only the two knee-assist actions. The human controller
is a frozen base controller: either zero torque for smoke tests, or MetaMotivo S-1
for walking rollouts. S-1 is installed from `/home/yhzhu/AI/metamotivo` and loaded
with `local_files_only=True` from the local HuggingFace cache. The assist torque is
applied as feed-forward generalized force on `L_Knee_x` and `R_Knee_x`. The
training reward minimizes a lower-limb hip+knee+ankle metabolic proxy after
canceling XML passive joint forces used for simulation stability:
`positive_joint_power / 0.25 + negative_joint_power_magnitude / 1.20`, plus
joint-velocity fluctuation and fall penalties. Eval reports the same passive-adjusted
lower-limb metabolic proxy and also logs total actuator power for diagnostics, while
older knee-only logs are treated as legacy fallback input.
