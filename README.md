# hit-exo-humenv

面向外骨骼机器人虚拟助力研究的 HumEnv + mjlab 实验工程。当前工程已经实现了：

- **平地多速度前向行走**：速度覆盖 `0.5、0.75、1.0、1.25、1.5 m/s`。
- **冻结 MetaMotivo S-1 人体控制器**：人体动作由 S-1 产生，外骨骼策略只学习助力。
- **抽象双膝外骨骼助力**：在 `L_Knee_x` 和 `R_Knee_x` 两个自由度施加额外力矩。
- **mjlab/RSL-RL 训练闭环**：支持并行环境训练、查看器回放、无头评估和报告生成。
- **下肢代谢功率 proxy 评估**：统计髋、膝、踝关节功率，抵消仿真稳定用被动力后计算代谢 proxy。
- **mocap 跟踪与楼梯探索**：已有 AMASS/mocap 跟踪、上下楼地形、S-1 楼梯补偿等实验脚本和调试结果。

## 当前结论

`./train_latent_z.sh` 已经跑通 **平地不同行走速度下的双膝助力训练**。本地已有无头对比评估显示，训练后的膝助力策略相对零助力基线可以降低人体下肢代谢功率 proxy。

| 评估时间 | 人体下肢代谢 proxy 降低 | 人体下肢机械绝对功率降低 | 外骨骼双膝绝对功率 | 助力效率 |
| --- | ---: | ---: | ---: | ---: |
| 2026-05-19 21:22 | 24.35% | 24.54%（人机合计下肢绝对功率） | 71.44 W | 5.631 W/W |
| 2026-05-19 22:08 | 28.76% | 26.48% | 68.13 W | 6.953 W/W |
| 2026-05-20 12:45 | 25.73% | 21.25% | 58.24 W | 2.608 W/W |
| 2026-05-20 13:02 | 27.39% | 23.17% | 71.29 W | 3.208 W/W |

说明：

- 这里的“人体下肢代谢 proxy”不是肌骨模型代谢值，而是关节机械功率近似：`正功率 / 0.25 + 负功率幅值 / 1.20`。
- 统计范围是髋、膝、踝下肢关节。
- 当前外骨骼不是刚体模型，只是双膝额外广义力矩。
- 评估报告由 `./eval.sh` 生成，原始日志和模型 checkpoint 没有提交到 Git 仓库。

## 当前能力边界

已经比较稳定的部分：

- 平地前向多速度行走助力训练。
- 冻结 S-1 人体控制 + 2 维膝助力策略。
- 零助力基线与训练助力策略的无头对比评估。
- 下肢功率、外骨骼功率、助力效率报告。
- AMASS walking 数据处理说明和 mocap tracking 训练入口。

正在探索的部分：

- 上下楼 mocap 跟踪。
- S-1 楼梯补偿策略。
- 膝助力叠加楼梯补偿策略。
- Kimodo/G1/SMPLX 伪 mocap 与脚部支撑约束生成。

尚未完成的部分：

- 跑步、转弯、多方向行走的系统训练。
- 斜坡、崎岖地形的正式训练和评估。
- 真实外骨骼刚体、质量、绑缚、人机接触模型。
- OpenSim/AnyBody 级别的肌肉力、肌肉激活和真实代谢分析。

## 快速使用

使用现有 `mjwarp_env` 环境：

```bash
conda activate mjwarp_env
pip install -e /home/yhzhu/AI/humenv
pip install -e /home/yhzhu/mjlab
pip install -e /home/yhzhu/myWorks_vips/hit_exo_humenv
```

当前测试过的环境包括：

- `/home/yhzhu/mjlab` 的可编辑安装版 `mjlab==1.3.0`
- `mujoco==3.8.1`
- `mujoco-warp==3.8.1`
- `rsl-rl-lib==5.2.0`

## 训练平地多速度助力

```bash
./train_latent_z.sh
```

该脚本训练任务：

```text
Mjlab-HumEnv-KneeExo-Walking
```

训练逻辑：

- 每个 episode 重置时采样一个前向速度。
- 速度来自 `config/latent_z.json` 中的 `walking_command.speed_choices`。
- 目前方向只有 `0.0 deg`，即前向行走。
- S-1 根据速度选择对应 latent，例如 `move-ego-0-1.25`。
- 默认外骨骼策略输出左右膝两个助力动作。
- 可以用 `EXO_JOINT_GROUP` 扩展助力关节组。

默认训练配置：

- 并行环境数：`4096`
- 控制频率：约 `30 Hz`
- MuJoCo 子步：`3`
- 最大膝助力力矩：`25 Nm`

助力关节组：

| `EXO_JOINT_GROUP` | 动作维度 | 助力关节 |
| --- | ---: | --- |
| `knee` | 2 | `L_Knee_x`, `R_Knee_x` |
| `hip` | 2 | `L_Hip_x`, `R_Hip_x` |
| `ankle` | 2 | `L_Ankle_x`, `R_Ankle_x` |
| `hip_knee` | 4 | 双侧髋 + 双侧膝 |
| `knee_ankle` | 4 | 双侧膝 + 双侧踝 |
| `lower_limb` | 6 | 双侧髋 + 双侧膝 + 双侧踝 |

训练髋/膝/踝下肢助力：

```bash
EXO_JOINT_GROUP=lower_limb ./train_latent_z.sh
```

小规模调试训练示例：

```bash
./train_latent_z.sh --env.scene.num-envs 64 --agent.max-iterations 10
```

## 回放最新策略

```bash
./run_latent_z.sh
```

固定速度回放示例：

```bash
RANDOM_WALK_SPEED=0 WALK_SPEED=1.25 RANDOM_WALK_DIRECTION=0 WALK_DIRECTION=0 ./run_latent_z.sh
```

## 无头评估并生成报告

```bash
./eval.sh --num-envs 64 --steps 300
```

该命令会依次运行：

- 零助力基线 rollout。
- 最新训练 checkpoint 的助力 rollout。
- 助力功率分析。
- 中文 Markdown 报告生成。

输出目录：

```text
logs/eval/latent_z_power/<时间戳>_headless_compare/
```

## 训练-评估扫参

如果目标是用尽量少的总时间得到满足要求的网络，不建议只盯每步仿真速度或最终训练 reward。当前工程提供了一个小规模闭环扫参入口：

```bash
./train_eval_sweep.sh --preset smoke --max-iterations 50
```

它会自动执行：

- 为候选配置做短训练。
- 对零助力 baseline 和训练 checkpoint 做同条件无头评估。
- 计算人体下肢代谢功率 proxy 节省比例、助力效率、人机总输入变化和跌倒数。
- 生成 `summary.csv`、`summary.json` 和中文 `report.md`。
- 某个候选达到目标后默认提前停止。

先查看计划而不启动训练：

```bash
./train_eval_sweep.sh --dry-run
```

对比只膝、只髋、只踝、髋膝踝联合助力：

```bash
./train_eval_sweep.sh --preset assist-groups --max-iterations 150
```

默认达标条件：

- 人体下肢代谢功率 proxy 节省比例 `>= 20%`
- 助力效率 `>= 1.0 W/W`
- 人机分开计总输入不增加
- 评估 rollout 中无跌倒

输出目录：

```text
logs/eval/train_eval_sweep/<时间戳>/
```

## mocap 跟踪训练

```bash
./train_mocap_track.sh
```

该模式使用固定 mocap 参考片段推断 S-1 tracking latent，外骨骼策略仍然只学习左右膝助力。可以通过环境变量替换参考动作：

```bash
MOCAP_MOTION=/path/to/walk.hdf5 MOCAP_EPISODE=ep_0 ./train_mocap_track.sh
```

回放 mocap tracking checkpoint：

```bash
./run_mocap_track.sh
```

## 上下楼探索脚本

当前仓库保留了上下楼方向的实验入口：

```bash
./train_mocap_track_updown_stairs.sh
./train_stair_compensation_updown_stairs.sh
./train_knee_exo_updown_stairs_on_compensation.sh
./show_mocap_track_updown_stairs.sh
```

这些脚本已经用于本地探索楼梯地形、脚部落点跟踪和 S-1 残差补偿，但还不应视为已经完成的稳定上下楼助力方案。

## AMASS walking 数据准备

详细说明见 [docs_amass_walking_data.md](docs_amass_walking_data.md)。

推荐先下载：

- `KIT`
- `CMU`
- `BMLmovi`
- `BMLrub`
- `MPI_HDM05`
- `Transitions`

处理入口：

```bash
conda run --no-capture-output -n mjwarp_env python scripts/amass_walking_pipeline.py check
conda run --no-capture-output -n mjwarp_env python scripts/amass_walking_pipeline.py extract
conda run --no-capture-output -n mjwarp_env python scripts/amass_walking_pipeline.py select
conda run --no-capture-output -n mjwarp_env python scripts/amass_walking_pipeline.py process --num-workers 0
conda run --no-capture-output -n mjwarp_env python scripts/amass_walking_pipeline.py validate
```

## 核心文件

| 文件 | 作用 |
| --- | --- |
| `config/latent_z.json` | 平地 latent-z 任务、训练、评估、奖励和仿真参数 |
| `hit_exo_humenv/envs/humenv_knee_exo.py` | Gymnasium/HumEnv 抽象膝助力环境 |
| `hit_exo_humenv/mjlab/walking_env_cfg.py` | mjlab 任务和 PPO 配置 |
| `hit_exo_humenv/mjlab/actions.py` | S-1 人体动作、膝助力动作、残差补偿动作 |
| `hit_exo_humenv/mjlab/mdp.py` | 奖励、终止、下肢功率和 mocap tracking 指标 |
| `hit_exo_humenv/s1_policy.py` | MetaMotivo S-1 包装和 latent 缓存 |
| `scripts/eval_latent_z_power.py` | 无头 rollout 和功率日志 |
| `scripts/analyze_assist_power.py` | 助力前后对比分析 |
| `scripts/write_latent_z_power_report.py` | 中文评估报告生成 |
| `scripts/train_eval_sweep.py` | 短训练、功率评估、达标判定和候选排序 |

## 测试

```bash
conda run --no-capture-output -n mjwarp_env pytest -q tests
```

当前提交前验证结果：

```text
36 passed
```

## 版本控制说明

仓库只提交源码、配置、脚本、文档和测试。以下内容默认不进入 Git：

- AMASS 原始压缩包。
- HDF5/NPZ/PT checkpoint 等大文件。
- `logs/` 训练和评估日志。
- `.omx/` 本地实验状态、生成地形和调试产物。
- `.cache/` 本地缓存。
