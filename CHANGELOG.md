# VSCMG DRL 控制系统迭代日志

## [v0.5.12] - 训练过程日志增强
**日期：2026-04-23**

### Changed (增强)
- `train.py`：
  - 新增 `EpisodeSummary` 输出：每次 episode 结束都打印一行摘要，包含 `run / step / ep_reward / best_so_far / saved / path`
  - 新增 `SaveSummary` 输出：在保存 `best / checkpoint / final` 时打印统一格式摘要
  - 新增训练结束 `TrainSummary` 输出
  - 保留原有 heartbeat 与 checkpoint 日志，不改变训练逻辑
  - 修复策略 1/策略 2 同时触发导致同一 episode 打印两次的 bug（改为 `if ... elif ... else` 互斥）

### Verified (验证)
- 短 smoke run 下：
  - 能看到 `EpisodeSummary`，`best` 保存时 `saved=best`，未保存时 `saved=none`
  - `SaveSummary` 在 `best / checkpoint / final` 保存时正常打印
  - `TrainSummary` 在训练结束时打印
  - 同一 episode 无重复打印

### Notes (说明)
- 本版本主要提升训练过程可观测性
- 不涉及 reward 结构改动
- 不代表控制性能验收完成

## [v0.5.11] - Reward 拆项重构与实验目录独立化
**日期：2026-04-23**

### Changed (重构)
- `envs/vscmg_env.py`：
  - RewardConfig 从单一 `w_act` 拆分为：
    - `w_gimbal_act`（框架动作抑制，前 4 维）
    - `w_wheel_act`（飞轮动作抑制，后 4 维）
  - 默认权重更新为：
    - `w_att = 1.0`
    - `w_omega = 0.2`
    - `w_wheel_bias = 0.05`
    - `w_gimbal_act = 0.01`
    - `w_wheel_act = 0.02`
  - `step()` 中新增：
    - `gimbal_action_sq = np.sum(action[:4] ** 2)`
    - `wheel_action_sq = np.sum(action[4:] ** 2)`
    - `action_sq = gimbal_action_sq + wheel_action_sq`（兼容总指标）
  - `info` 中新增：
    - `reward_gimbal_act_penalty`
    - `reward_wheel_act_penalty`
    - `gimbal_action_sq`
    - `wheel_action_sq`

- `train.py`：
  - 版本号更新到 `v0.5.11`
  - 新增 `run_name` 自动生成（格式：`v0.5.11_<timestamp>_envs<N>_seed<S>_<reward_summary>`）
  - 新增独立实验目录：`models/<run_name>/`
  - 模型保存命名改为：
    - `best_episode_reward.pth`（替代 `best_model_parallel.pth`）
    - `checkpoint_step_<N>.pth`
    - `final_step_<N>.pth`
  - 新增 `run_config.json` 记录实验配置（训练调度、agent 参数、reward 权重）
  - 控制台输出补充 `run_name` 与 `model_dir`

- `eval_policy_viewer.py`：
  - 新增记录：`gimbal_action_sq`、`wheel_action_sq`
  - 执行器图第 4 张子图新增拆项曲线
  - `CSV / summary.txt / 控制台` 新增拆项统计

- `.gitignore`：
  - 新增 `eval_outputs/`
  - 新增 `*.png`

### Verified (验证)
- 环境与 reward 验证通过：
  - 零动作时 `gimbal_action_sq = 0`、`wheel_action_sq = 0`、`action_sq = 0`
  - 非零动作 `[0.5,0.5,0.5,0.5,0.3,0.3,0.3,0.3]` 时：
    - `gimbal_action_sq = 1.0`（预期 0.5²×4）
    - `wheel_action_sq = 0.36`（预期 0.3²×4）
    - `action_sq = 1.36`（预期 1.0+0.36）
- 训练 smoke run 通过：
  - `python train.py --num_envs 4 --batch_size 64 --update_every 50 --update_times 2 --start_steps 200 --device cpu --seed 42 --max_steps 2000`
  - 成功生成：
    - `models/<run_name>/final_step_2000.pth`
    - `models/<run_name>/run_config.json`
- eval 验证通过：
  - 新目录下模型可被 `eval_policy_viewer.py` 正常加载
  - `rollout.csv` 与 `summary.txt` 中包含：
    - `gimbal_action_sq`
    - `wheel_action_sq`

### Notes (范围说明)
- 本版本是 **reward 结构与实验管理重构版**
- **不代表 v1.0 控制性能验收完成**
- `best_episode_reward.pth` 基于 **单 episode reward** 触发保存
- 若短训步数不足以完成 episode，可能不会生成 best 文件，这是预期现象
- 训练链路验证仅确认功能正确性，未进行性能验收

## [v0.5.10] - v1.0 Reward 重构与工程文案统一
**日期：2026-04-22**

### Changed (重构)
- `envs/vscmg_env.py`：将 reward 从旧版三项硬编码结构重构为 v1.0 分项叠加结构：
  - `R_att`：姿态误差项，基于 `||sigma_err||²`
  - `R_omega`：角速度抑制项，基于 `||omega_B||²`
  - `R_wheel_bias`：飞轮偏置保持项，基于 `||Omega_w_tilde||²`
  - `R_act`：执行器抑制项，基于 `||action||²`
- `envs/vscmg_env.py`：新增 `RewardConfig`，将 `w_att / w_omega / w_wheel_bias / w_act` 集中管理，便于后续单项开关与调参。
- `envs/vscmg_env.py`：`step()` 返回 `info` reward breakdown，包含各 penalty 与 `sigma_err_sq / omega_sq / wheel_bias_sq / action_sq`，便于后续训练观察与调参。
- `train.py`：统一更新运行时版本标识、banner、CLI description 与完成文案到 `v0.5.10`。
- `configs/env_config.py`：修正文档字符串中的过期版本描述，使其与当前 v1.0 默认语义一致。

### Verified (验证)
- 环境级验证通过：
  - `env = VSCMGEnv(); obs, info = env.reset(seed=42)` 正常
  - `env.step(np.zeros(8, dtype=np.float32))` 正常返回
  - `reward` 为有限数，`info` 中 reward breakdown 各分项均为有限数
  - 返回结构符合 Gymnasium 规范，22 维观测接口保持不变
- 训练 smoke test 通过：
  - `python train.py --num_envs 4 --batch_size 256 --update_every 50 --device cpu --max_steps 20000`
  - 训练可正常启动并越过首次 `envs.reset(...)`
  - heartbeat、episode、checkpoint 正常
  - 无 NaN / Inf / traceback，正常退出
- 工程稳定性排查结论：
  - 在 `num_envs=1, CPU` / `num_envs=4, CPU` / `num_envs=16, CUDA` 三组 20k step smoke test 条件下，未复现 AsyncVectorEnv / BrokenPipeError / WinError 232 / worker 异常退出问题。

### Notes (范围说明)
- 本版本完成的是 v1.0 reward 结构重构、最小训练链路验证与工程文案统一，不代表 v1.0 验收已经完成。
- 当前 reward 默认权重为保守初始值，后续仍需通过 200k~500k 步短训观察 `reward_total` 与各分项趋势，再决定是否继续调参。
- `w_act` 当前基于归一化动作量而非物理缩放动作量；若后续观察到执行器饱和或飞轮偏置漂移，再决定是否进一步调整。

## [v0.5.9] - 启动 Bug 修复与 TD3 平滑噪声配置接通
**日期：2026-04-22**

### Fixed (修复)
- 修复 `envs/vscmg_env.py::_build_episode_cfg()` 中 `ep / episode_cfg` 局部变量混用导致的 `UnboundLocalError`：
  - 函数内部第 213 行调用 `ep.apply_randomization(...)`，但 `ep` 未定义
  - 第 209 行实际创建的是 `episode_cfg`，导致首次 `envs.reset(seed=...)` 时 AsyncVectorEnv 在 worker 内崩溃
  - 统一使用 `episode_cfg` 作为局部变量名，消除所有 `ep` 殷留
- 修复 `train.py` 启动阻塞问题：
  - 16 核并行环境 `train.py --num_envs 16 --batch_size 2048 --update_every 200 --device cuda` 现可正常启动
  - 配置快照正确打印，训练主循环已开始执行
- 接通 `policy_noise / noise_clip` 到完整 TD3 训练链路：
  - `agents/td3_agent.py`：`TD3.__init__()` 新增 `policy_noise` / `noise_clip` 参数
  - `agents/td3_agent.py`：`update()` 方法中目标策略平滑逻辑改用 `self.policy_noise` / `self.noise_clip`，移除硬编码 `0.5`
  - `configs/agent_config.py`：`policy_noise` 默认值从 `0.2` 改为 `0.5`，`noise_clip` 保持 `0.5`，与原硬编码行为一致
  - `configs/agent_config.py`：字段注释与 `make_default_agent_config()` docstring 更新为"已接入 TD3 训练链路"
  - `train.py`：`TD3(...)` 构造时传入 `policy_noise=agent_cfg.policy_noise` / `noise_clip=agent_cfg.noise_clip`
  - `train.py`：修正 `print_config_snapshot()` 中 `policy_noise / noise_clip` 的过时显示文案（移除"预留未接入"）
  - `train.py`：顶部用法示例修正为 `envs.single_observation_space.shape[0]` / `envs.single_action_space.shape[0]`

### Verified (验证)
- 最小验证：`env = VSCMGEnv(); env.reset(seed=42)` 正常通过
- 最小验证：`env.reset(seed=43, options={"j_sc": np.diag([10.0, 10.0, 10.0])})` 正常通过
- 训练启动验证：`train.py --num_envs 16 --batch_size 2048 --update_every 200 --device cuda` 成功越过首次 `envs.reset(seed=seed_value)`
  - 配置快照正常打印（`state_dim=22`, `action_dim=8`, `policy_noise=0.5`, `noise_clip=0.5`）
  - 训练主循环已启动，心跳打印正常：`[Heartbeat] Training progress: 2000 steps processed...`
  - Episode 完成并触发模型保存：`New best reward: -111.0863 -> Model saved.`

### Notes (范围说明 — 明确"还没做完什么")
- **本版本只完成启动 bug 修复与 TD3 平滑噪声配置接通**，不是 v1.0 完成版。
- reward 仍为 v0.5 旧版（sigma_err² + 0.1·omega² + 0.01·action²），尚未重构为分项系数叠加结构。
- 正式训练、收敛验证、v1.0 验收尚未开始。
- `eval_frequency` 在 `configs/train_config.py` 中是预留字段，当前未在训练循环中使用。
- 后续 `BrokenPipeError: [WinError 232] 管道正在被关闭` 是 AsyncVectorEnv 的 Worker 崩溃，与 `_build_episode_cfg()` 的 `ep/episode_cfg` 殷留问题无关，待后续排查。

---

## [v0.5.8] - 参数集中化第二阶段（训练侧 + Agent 侧）
**日期：2026-04-21**

### Added (新增)
- **训练配置集中入口** `configs/train_config.py`：`TrainConfig` dataclass 建立训练调度参数统一管理，包括并行数、步数、batch、更新频率、ReplayBuffer、日志/检查点路径、随机种子等。
- **Agent 配置集中入口** `configs/agent_config.py`：`AgentConfig` dataclass 建立 TD3 / 网络超参数统一管理，包括网络结构、TD3 核心参数、学习率、探索噪声等。

### Changed (重构)
- `train.py` 重构为配置驱动的训练主循环：
  - 新增 `parse_args()` 函数，CLI 参数默认值设为 `None`，实现"CLI 参数 > config 默认值"的优先级规则。
  - 新增 `_apply_cli_overrides()` 函数，处理 CLI 覆盖逻辑。
  - 新增 `set_global_seed()` 函数，统一设置 numpy、torch、random、cuda 的随机种子。
  - 新增 `print_config_snapshot()` 函数，打印训练和 Agent 配置快照。
  - `state_dim` / `action_dim` 改为运行时从 `envs.single_observation_space.shape[0]` / `envs.single_action_space.shape[0]` 自动获取，修正了向量环境维度获取错误。
  - `update_every` / `update_times` 独立控制，CLI 可分别覆盖。
  - `seed` 已接入训练链路，默认 `seed=0`（不固定随机种子，保持原行为）。
  - `tb_flush_secs` 已接入 `SummaryWriter`，默认 30 秒。
  - `checkpoint_frequency` 统一管理，默认 10 万步。
  - **22 维观测语义不变**：sigma_err(3) + omega_B(3) + sin(delta)(4) + cos(delta)(4) + delta_dot(4) + Omega_w_tilde(4)。
  - **8 维动作语义不变**：前 4 维 gimbal 指令 max 1 rad/s，后 4 维 wheel 指令 max 50 rad/s²。
  - **reward 结构未改**，仍保持 v0.5 旧版。

### Verified (验证)
- `num_envs=1` (SyncVectorEnv)：
  - `state_dim=22`、`action_dim=8` 正确获取
  - ReplayBuffer 创建成功（capacity=100000）
  - TD3 Agent 创建成功（Actor 22→8，Critic 30→1）
  - hidden_dim=256、batch_size=256、replay_capacity=100000、update_every=50、update_times=50、gamma=0.99、tau=0.005、seed=0
- `num_envs=4` (AsyncVectorEnv)：
  - `envs.observation_space.shape = (4, 22)`、`envs.single_observation_space.shape = (22,)`
  - `envs.action_space.shape = (4, 8)`、`envs.single_action_space.shape = (8,)`
  - `state_dim=22`、`action_dim=8` 正确获取
  - ReplayBuffer 创建成功（capacity=100000）
  - TD3 Agent 创建成功（Actor 22→8，Critic 30→1）
  - Windows 下 AsyncVectorEnv 的 timeout 发生在 `envs.close()` 收尾阶段，不影响"维度与构造链路验证"结论

### Notes (范围说明 — 明确"还没做完什么")
- **本版本只完成参数集中化第二阶段（训练侧 + Agent 侧）**，不是 v1.0 完成版。
- reward 仍为 v0.5 旧版（sigma_err² + 0.1·omega² + 0.01·action²），尚未重构为分项系数叠加结构。
- 正式训练、收敛验证、v1.0 验收尚未开始。
- `policy_noise` / `noise_clip` 在 `configs/agent_config.py` 中是预留字段，当前未真正接入 `td3_agent.py`（agent 内部硬编码为 0.5）。
- `eval_frequency` 在 `configs/train_config.py` 中是预留字段，当前未在训练循环中使用。
- 旧的 `v0.5.x` checkpoint 与当前接口不兼容，需重新训练。

---

## [v0.5.7] - 环境侧参数集中化接口第一阶段
**日期：2026-04-20**

### Added (新增)
- **环境配置集中入口** `configs/env_config.py`：`VSCMGEnvConfig` dataclass 建立 nominal / current / randomization 三层结构，统一管理所有环境物理参数。
- **动力学惯量更新接口** `envs/dynamics.py`：`SpacecraftDynamics` 新增 `update_inertia()` 方法和 `compute_angular_acceleration(j_sc=...)` 可选参数，解除"一次性求逆锁死"，支持 episode 级 J 随机化。

### Changed (重构)
- `envs/vscmg_env.py` 重构为配置驱动的 `VSCMGEnv`：
  - `__init__(config=)` 接受 `VSCMGEnvConfig` 实例（默认走 `make_default_config()`，即 v1.0 行为）。
  - 新增 `self.cfg`（永久基线配置）和 `self.episode_cfg`（每 episode 工作副本）。
  - 新增 `_sync_dynamics()` / `_build_episode_cfg()` 内部方法。
  - `reset(options=...)` 重构为"克隆 → randomization → options 覆盖"链路，options 仅作用于本 episode，不泄漏到后续 episode。
  - `step()` 改为读取 `self.episode_cfg` 所有物理参数。
  - **22 维观测语义不变**：sigma_err(3) + omega_B(3) + sin(delta)(4) + cos(delta)(4) + delta_dot(4) + Omega_w_tilde(4)。
  - **8 维动作语义不变**：前 4 维 gimbal 指令 max 1 rad/s，后 4 维 wheel 指令 max 50 rad/s²。
  - **reward 结构未改**，仍保持 v0.5 旧版。

### Verified (验证)
- 最小验证：连续两次 `reset`，第一次传入 `options={"j_sc": ...}` 覆盖惯量，第二次不带 options，确认 `episode_cfg.current_j_sc` 与 `dynamics.j_sc` 均回到默认 `[100, 100, 100]`，options 不泄漏。
- 默认配置下 `reset()` 返回 obs shape `(22,)`，`step()` 正常执行，无维度错误或数值炸裂。

### Notes (范围说明 — 明确"还没做完什么")
- **本版本只完成环境侧参数集中化接口第一阶段**，不是训练完成版，不是全项目参数集中化完成版。
- `train.py` 仍未接入统一配置链路，所有训练超参数仍为硬编码。
- `agents/td3_agent.py` 参数仍未集中化。
- `configs/train_config.py` / `configs/agent_config.py` 尚未建立。
- reward 仍为 v0.5 旧版（sigma_err² + 0.1·omega² + 0.01·action²），尚未重构为分项系数叠加结构。
- 正式训练、收敛验证、v1.0 验收尚未开始。
- 旧的 `v0.5.x` checkpoint 与当前接口不兼容，需重新训练。

---

## [v0.5.6] - 训练入口 22 维接口对齐
**日期：2026-04-19**

### Changed (训练入口对齐)
- `train.py` 的训练入口对齐到 22 维环境接口。
- `state_dim` 从 14 改为 22，移除旧的 14 维假设。
- 确认 `agents/td3_agent.py` 无需修改（Agent 通过参数动态接收输入维度）。

### Verified (最小训练 Smoke Test)
- 使用一次性 inline 命令完成最小训练 smoke test，验证链路完整性：
  * 环境创建与 reset 返回 22 维观测
  * Agent 以 `state_dim=22` 正常初始化
  * `reset -> action -> step -> buffer.push -> buffer.sample` 全流程通过
  * 采样维度验证：状态 batch 维度为 `[32, 22]`（符合预期）
- 训练入口 22 维接口已完全接通，可正常启动训练而不因维度错误崩溃。

### Notes (范围说明)
- 本版本只完成训练入口 22 维接口接通。
- 未修改 `tests/` 目录。
- 未修改 reward 最终设计（仍为 v0.5 旧版）。
- 未开始正式训练与收敛验证。
- 旧的 `v0.5.x` checkpoint 与当前 22 维接口不兼容，需重新训练。

---

## [v0.5.5] - 环境侧 v1.0 前置条件对齐
**日期：2026-04-19**

### Changed (姿态基础设施)
- 环境内部姿态表示改为 **scalar-first quaternion `[w, x, y, z]`**，替代原有的纯 MRP 内部姿态积分。
- 引入统一误差链路：`q_err = q_target ⊗ q_current*` → 双覆盖规避（标量部 `q[0] < 0` 取反）→ quaternion → MRP → shadow set。
- 观测接口切换为 22 维固定语义：`sigma_err`(3) + `omega_B`(3) + `sin(delta)`(4) + `cos(delta)`(4) + `delta_dot`(4) + `Omega_w_tilde`(4)。
- 飞轮内部状态改为显式的 `omega_w + I_w -> h_w` 三层结构，`Omega_w_tilde` 直接基于 `omega_w` 计算。

### Fixed (环境前置条件对齐)
- reset 初始化对齐 v1.0 前置要求：姿态误差限制在 ±5°（直接约束旋转角），本体角速度归零，飞轮初始化到 3000 rpm 偏置，外扰力矩关闭。
- 动作缩放对齐 v1.0 前置要求：gimbal `max 1 rad/s`，wheel `max 50 rad/s²`。
- 修复 `self.np_random` 使用 `randn()` 的 numpy 兼容性问题，改为 `standard_normal()`。

### Notes (范围说明)
- 本版本只完成环境侧前置对齐，尚未修改 `train.py`、`tests` 的 `state_dim=22` 对齐。
- reward 仍为 v0.5 旧版，待后续手动设计最终 reward 权重。
- 训练收敛验证未开始。
- 旧的 `v0.5.x` checkpoint 与当前环境接口不再兼容。

---

## [v0.5.4] - 架构规约统一：22维状态语义冻结
**日期：2026-04-18**

### Changed (规约修订)
- 统一了 `v1.0~v4.0` 的 22 维状态物理语义定义，明确冻结状态语义与动作语义，区分"语义冻结"与"前端编码器演化"的边界。
- 完善了 Dual-State Mapping 的数学表述，采用严谨的误差四元数流程（形成 q_err → 双覆盖规避 → 转换 σ_err → 影子集切换）。
- 具体化了 v4.0 的 Token 化打包方案（本体 Token + 4 个 VSCMG Token），明确前端重构不改变状态语义。
- 统一了各版本规约中的姿态误差约定表述，确保环境、奖励、测试、可视化遵循统一约定。

### Docs (文档)
- 详细定义了 22 维状态的 6 个分量：`sigma_err`（3维）、`omega_B`（3维）、`sin(delta)`（4维）、`cos(delta)`（4维）、`delta_dot`（4维）、`Omega_w_tilde`（4维）。

### Fixed (修复)
- 修复了 `docs/v1.0_design_spec.md` 中的章节编号重复问题。
- 修复了文档中会导致 PyCharm/Markdown 误解析的伪 Python 代码展示写法，统一变量名维度标注格式。

---


## [v0.5.2] - 战略规约档案入库
**日期：2026-04-12**

### Added (文档)
- 初始化全版本工程战略路线图（v1.0 - v4.0）与详细架构设计规约，统一归档于 `docs/` 目录，作为后续开发的绝对准则。

---

## [v0.5.1] - 极简监控大盘重构
**日期: 2026-04-12**

### Changed (优化)
- 彻底移除了 `train.py` 中 `_log_and_checkpoint` 函数对单体环境（Env_x）的冗余监控写入。
- 利用 TensorBoard 字母排序机制，将全局平均分命名空间重构为 `Global/Mean_Reward`，实现核心大盘置顶。
- 将 PyCharm 运行配置（16核并行环境及 TensorBoard 启动项）持久化至 `.run/` 目录并归档，确保多端环境下的启动参数一致性。

### Fixed (修复)
-  清除了 `train.py` 中关于设备分配的硬编码日志，使其能够自适应动态获取的 `device` 状态。

---

## [v0.5.0] - 架构级重构与监控系统升级
**日期: 2026-04-12**

### Added (新增)
- **API 架构统一**: 引入 `SyncVectorEnv` 实现单/多环境接口向下兼容，无论 `--num_envs` 为 1 或 N，`env.step()` 返回结构完全统一，消灭了所有 if/else 分支判断。
- **环境寿命限制**: 在 `VSCMGEnv.step()` 中安装 `max_episode_steps=1000` 截断逻辑，`truncated` 信号正式生效，彻底修复了飞船"长生不老"导致 Episode 战报永不打印的顽疾。
- **16 轨独立曲线监控**: TensorBoard 每条并行环境独立使用 `Reward/Env_{idx}` 标签写入，实现 16 条彩色独立折线分离展示。
- **均值基准线聚合**: 每批次完赛的得分统一收集并计算均值，写入 `Reward/Mean_Reward` 基准线，为策略收敛趋势提供稳定参照。
- **动态时间戳目录**: `SummaryWriter` log_dir 强制拼接时间戳（`{num_envs}envs_{mmdd_HHMMSS}`），彻底规避旧缓存覆盖问题。
- **Loss 强制刷盘**: `agent.update()` 返回三个 Loss 值，每次更新后立即执行 `writer.flush()`，确保 TensorBoard 实时可读。
- **心跳前置探针**: 训练启动时写入 `System/Ignition=1`（step=0），用于自证数据管道连通性。

### Fixed (修复)
- **作用域修复**: 修正 `best_reward` 全局变量遮蔽问题，将 `nonlocal` 改为 `global`，确保 `_log_and_checkpoint` 内赋值可穿透模块级作用域。
- **重置时序优化**: 引入 `_reset_envs` 延迟重置集合，将 `episode_rewards/lengths` 清零时机推迟至两个解析策略均读取完成后，解决策略 2 读到的永远是 0 的数据竞争 Bug。
- **Actor 更新帧返回值**: 修复 `update()` 方法在 Critic 仅更新帧返回 `None` 而非 `0.0` 的问题，统一 Loss 数据类型。

### Changed (变更)
- `agent.update()` 方法签名增加返回值 `(actor_loss, c1_loss, c2_loss)`，与调用方解耦，避免直接操作内部张量。
- `.gitignore` 新增 `models/` 目录与 `*.pth` 权重文件过滤规则。

---

## [v0.4.0] - 核心训练管线 (Training Pipeline) 竣工
**日期: 2026-04-11**

### Added (新增)
- **训练主程序**: 新增项目级入口文件 `train.py`，串联了 `VSCMGEnv` 环境与 `TD3` 算法大脑。
- **训练监控**: 集成了 `tensorboard` 日志记录，支持实时追踪 `EpisodeReward` 与 `EpisodeLength`。
- **权重检查点**: 实现基于最高得分 (Best Reward) 的动态模型权重自动保存机制 (`checkpoints/best_model.pth`)。

---
## [v0.3.1] - 代码规范与 IDE 警告消除 (Zero Warnings)
**日期: 2026-04-11**

### Fixed (修复)
- **代码规范**: 深度清理了 `tests/test_td3.py`, `geometry/pyramid_config.py` 与 `agents/td3_agent.py` 中的类型推导、命名规范及异常处理警告。
- **类型兼容**: 修复了 `zip` 迭代器在 PyCharm 静态检查中的类型推导报错，达成全工程 Zero Warnings 状态，且回归测试（Sanity Check）通过。

---
## [v0.3.0] - 核心算法大脑 (TD3 Agent) 竣工
**日期: 2026-04-10**

### Added (新增)
- **TD3 算法核心**: 新增 `agents/td3_agent.py`，从零实现了包含双 Q 网络、延迟策略更新机制的 TD3 强化学习算法。
- **工业级重构**: 引入独立的 `ReplayBuffer` 类进行经验回放管理，取代了原有的松散字典传参。
- **连通性测试**: 新增 `tests/test_td3.py` 脚本，成功验证了前向传播与反向梯度的无缝流转，并作为标准的单元测试进行工程化归档。

### Fixed (修复)
- 修复了原算法中计算 TD 目标值时缺少终止状态 (Done) 掩码的数学错误。
- 补充了原算法中缺失的目标策略平滑 (Target Policy Smoothing) 噪声截断机制。
- 修复了硬编码的动作截断边界，改为动态适配环境的 `action_bound`。

---

## [v0.2.1] - 环境接口合规性与代码规范修复
**日期: 2026-04-10**

### Fixed (修复)
- **Gymnasium 标准化**: 修复了 `reset` 方法的随机种子 (seed) 隔离问题，确保了环境输出的严格可确定性 (Determinism)。
- **空间边界安全**: 将观测空间的无穷大 (`np.inf`) 边界替换为安全的有限极大值 (`1e30`)，防止下游算法归一化时溢出报错。
- **代码规范**: 修正了局部变量的 PEP 8 命名规范及部分注释语病，达到 IDE 零警告状态。

---

## [v0.2.0] - 强化学习环境 (Gymnasium) 封装完成
**日期: 2026-04-10**

### Added (新增)
- **标准化环境**: 新增 `envs/vscmg_env.py`，完整实现了 Gymnasium 接口规范。
- **姿态表示**: 采用修正罗德里格斯参数 (MRPs) 作为姿态反馈，内置影子集 (Shadow Set) 映射，彻底消除万���节死锁与数值跳变。
- **状态与动作空间**: 确立 14 维连续观测空间与 8 维连续动作空间。
- **高频仿真**: 仿真步长设定为 0.01s (100Hz)，确保 CMG 框架高频动态的数值积分稳定性。

---

## [v0.1.2] - 修复测试块残余报错
**日期: 2026-04-10**

### Fixed (修复)
- **遗留代码清理**: 移除了 `envs/dynamics.py` 测试块中因重构遗留的 `omega_dot` 未解析引用报错，确保模块可独立安全运行。

---

## [v0.1.1] - 代码质量重构与规范化
**日期: 2026-04-10**

### Fixed (修复与优化)
- **命名规范优化**: 按照 PEP 8 标准重构了所有物理矩阵的变量名（���写转小写），消除了 PyCharm 的命名警告。
- **作用域隔离**: 修复了测试块中的变量名冲突（Shadows name from outer scope），提升了代码稳健性。
- **编码修复**: 解决了 Windows 环境下希腊字母打印导致的 Unicode 编码报错。
- **接口公开**: 将 `_cross_product` 提升为标准叉乘接口，优化了动力学求解器的调用逻辑。

---

## [v0.1.0] - 物理引擎与动力学底座竣工
**日期: 2026-04-09**

### Added (新增)
- **底层物理接口 (`geometry/base_config.py`)**: 实现了 VSCMG 阵列的抽象基类 `BaseVSCMGArray`，确立了控制力矩的底层计算法则。
- **金字塔构型 (`geometry/pyramid_config.py`)**: 实现了标准的四飞轮金字塔构型，精确推导了自旋轴与横向轴的雅可比矩阵，并通过了静态零点测试。
- **航天器动力学 (`envs/dynamics.py`)**: 基于牛顿-欧拉方程，实现了刚体姿态动力学求解器，完美解耦了 VSCMG 的牵连耦合力矩。

---

## [v0.0.0] - 工程地基初始化
**日期: 2026-04-09**

### Added (新增)
- 建立 7 大核心文件夹骨架 (`agents`, `configs`, `envs`, `geometry`, `classical_control`, `utils`, `checkpoints`)。
- 部署 `.gitignore` 防御结界，排除运行缓存、IDE 配置和大型权重文件。
- 确立面向对象的构型基类思想。
- 绑定 GitHub 远程仓库并完成首次纯净工程骨架的推送与 v0.0.0 里程碑打标。
