# VSCMG DRL 控制系统迭代日志

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
