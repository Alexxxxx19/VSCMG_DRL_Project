# VSCMG 端到端控制工程 - 全版本战略路线图 (Master Roadmap)



**【核心准则】**

从 v1.0 开始，TD3 神经网络的 22 维状态语义彻底冻结。动作语义分层推进：full_8d（8 维）是最终目标，gimbal_only（4 维）是当前 debug / v1.0 前置阶段的中间验证接口。版本演进的核心逻辑已从"普通 TD3 逐步加难度"升级为"BC baseline + residual actor + critic calibration + safety gate 逐步加难度"。

**冻结承诺：**
- v1.0~v5.0 的状态语义（22 维）保持不变
- full_8d 动作语义（8 维）是最终目标，v1.0 阶段先用 gimbal_only（4 维）验证闭环
- v2.0 改变初始化难度，v3.0 注入扰动与延迟，v4.0 重构前端编码器，v5.0 进入实机安全自适应
- "语义冻结"不等于"MLP 结构冻结"，v4.0 可引入 Token 化前端

---

**【主线架构原则】**

### 1. 普通 TD3 actor 不再作为主线

- 不再假设 actor 可以直接输出完整动作并长期稳定 fine-tune；
- 当前实验已证明普通 TD3 / 弱保护 TD3 会破坏 BC policy；
- 后续所有阶段都应避免回到 `a = π_TD3(s)` 直接完整输出动作的主线。

### 2. Residual actor 贯穿 v1.0~v5.0

| 阶段 | 网络形式 |
|------|---------|
| v1.0 | `a = π_BC(s) + α Δπ(s)` |
| v2.0 | `a_8d = [π_BC_gimbal(s), 0_4d] + α Δπ_8d(s)` |
| v3.0 | `a = robust_base(s) + α Δπ(s)` |
| v4.0 | `a = base(s) + α AttentionResidual(s)`，可选 null-space residual |
| v5.0 | `a_real = safety_filter(base(s) + α_real Δπ(s))` |

### 3. Critic 修正主线

- replay prefill；
- critic warmup；
- recovery data / DAgger-style 数据闭环；
- critic calibration gate；
- rollout-based model selection；
- 后续引入 critic ensemble / uncertainty gate。

### 4. 安全主线

- hard safety shield 与 reward 分开；
- 动作限幅、状态边界、fallback、rollback 必须逐步加入；
- 实机阶段不允许 actor 自由探索完整动作空间。

---

**【当前调试结论：v0.5.18-debug-bc-init】**

> debug tag: `v0.5.18-debug-bc-init`

1. **BC actor 有效**：在 `env.reset(options={"init_attitude_deg": angle})` 下，BC actor 的姿态恢复精度：
   - 10° → ~0.29°，20° → ~1.12°，30° → ~2.18°，45° → ~4.30°
2. **`--actor_init_path` 链路已验证**：BC actor 权重能正确加载到 `agent.actor` 和 `agent.target_actor`。
3. **普通 TD3 fine-tune 失败**：checkpoint_10k 已严重退化，动作从 BC 的小动作退化为高饱和大动作。
4. **`best_episode_reward.pth` 不是可靠的选模标准**：需改为 rollout-based model selection。
5. **下一阶段路线 → critic repair + residual actor**：
   - replay prefill（用 BC 数据预填充 replay buffer）
   - critic warmup（先训 critic，再放开 actor）
   - actor freeze（限制 actor 更新步数）
   - BC regularization（actor loss 中加入 BC 行为克隆正则项）
   - low actor_lr（降低 actor 学习率）
   - critic audit / critic calibration gate（审计 critic 是否正确区分好/坏动作）
   - recovery data（在 replay buffer 中补充偏离状态下的 recovery action）
   - rollout-based model selection（以实际 rollout 表现选模型，而非 episode reward）
   - residual actor（冻结 BC baseline，只训练小幅度 residual）

---

**【版本演进大纲】**



* **v1.0：温室学步期 (Greenhouse Foundation)**

    * **定位**：在 gimbal_only 模式下，先固化 BC baseline，再通过 protected TD3 / BC-assisted TD3 保护 BC，最后验证 residual actor 不劣于 BC。

    * **环境**：静止初始状态，零扰动，飞轮严格锁定 3000 rpm 偏置。初始误差从 5° 开始，逐步扩展到 10° / 20° / 30°，45° 作为测试上限。

    * **战术**：
      - v1.0-a：BC baseline 固化，确认 BC actor 是安全 baseline；
      - v1.0-b：critic repair / protected TD3，防止普通 TD3 fine-tune 破坏 BC policy；
      - v1.0-c：residual actor 验证，`a = clip(π_BC(s) + α Δπ(s), -1, 1)`，π_BC 冻结，只训练 Δπ。

    * **里程碑**：gimbal_only 4 维动作下，residual actor 不劣于 BC baseline。v1.0 前置阶段不引入 full_8d；full_8d 放到 v2.0 单独恢复和验证。



* **v2.0：全域机动期 (Full-Domain Maneuver)**

    * **前置依赖**：v1.0 residual actor 在 gimbal_only 下稳定；critic calibration gate 通过；rollout-based model selection 已替代 best_episode_reward。

    * **定位**：从 gimbal_only 扩展到 full_8d，领悟真正的 VSCMG 战术与奇异面规避。

    * **环境**：全空域随机初始状态（±180°），带随机初始翻滚角速度。

    * **网络形式**：`a_8d = [π_BC_gimbal(s), 0_4d] + α Δπ_8d(s)`，或在 full_8d BC/oracle 可用后：`a_8d = π_BC_8d(s) + α Δπ_8d(s)`。

    * **战术**：逼迫网络在长距离机动中，主动对飞轮加减速（逼近 5000 rpm 极限）以跨越框架奇异点，机动后恢复动量。residual actor 输出 8D residual，前 4 维为 gimbal residual，后 4 维为 wheel acceleration residual。

    * **安全**：引入 gimbal action safety、wheel speed / wheel acceleration safety、momentum health。不允许飞轮乱拉高转速却没有姿态收益。

    * **critic 机制**：critic ensemble / uncertainty gate 作为强推荐或默认机制，尤其监控 wheel_accel 维度的 Q uncertainty。



* **v3.0：抗扰压榨期 / Robust Residual Sim-to-Real Preparation**

    * **前置依赖**：至少完成 v2.0 稳定控制器后再进入 v3.0。不要过早引入扰动、延迟或参数随机化，否则会掩盖训练不稳定问题。

    * **定位**：跨越 Sim-to-Real 鸿沟，达到工业级部署标准。网络形式：`a = robust_base(s) + α Δπθ(s)`，其中 robust_base 来自 v2.0 稳定 baseline，Δπθ 学习模型误差、延迟、扰动补偿。

    * **环境**：注入转动惯量偏差（±15%）、一阶伺服电机延迟、以及空间周期性外部扰动力矩。

    * **战术**：
      - system identification + domain randomization，参数随机化范围基于真实不确定性；
      - 非对称 Actor-Critic 架构，critic 使用 privileged information（精确惯量、延迟参数、扰动力矩、无噪声状态），actor 仍只使用可测 22D observation；
      - critic ensemble 从 v3.0 起作为核心机制。

    * **验收**：critic ranking / uncertainty audit、domain randomized rollout matrix、不出现极限环、不因 residual actor 进入 bad state distribution、safety shield 不应频繁触发。



* **v4.0：学术巅峰期 (Academic Peak)**

    * **前置依赖**：v3.0 稳定。当前瓶颈是训练机制（critic calibration、residual actor、safety gate），不是 MLP 表达能力，因此不优先更换网络架构。

    * **定位**：打破 DRL 黑盒，实现多执行器控制的可解释性。网络形式为 Token/Attention residual actor：`tokens = tokenize_22d_state(s)`，`Δπθ = AttentionResidualActor(tokens)`，`a = base(s) + α Δπθ`。

    * **环境**：维持 v3.0 的全量地狱级难度。

    * **战术**：前端重构。将**同一套冻结的 22 维物理语义**重新打包为实体 Token（本体 Token + 4 个 VSCMG Token），引入物理引导的自注意力机制（Self-Attention），输出可解释的电机分配热力图。**v4.0 改变的是前端编码方式，不改变状态语义。** Attention 是表达能力升级，不替代 residual / critic calibration / safety gate。

    * **可选长期升级**：null-space residual actor：`a = a_base(s) + α N(s) zθ(tokens)`。前置条件：VSCMG control allocation matrix B(s) 已审计、null-space projector N(s) 已验证。不准确的 N(s) 会把"零运动"变成真实扰动，不应在 v1~v3 过早引入。

    * **critic**：critic ensemble 仍保留，Attention 改善表达能力，ensemble / uncertainty gate 负责可信度，两者不是替代关系。



* **v5.0：实机安全自适应期 (Real-Device Safe Residual Adaptation)**

    * **前置依赖**：v3.0 robust residual controller 稳定后才进入。如果使用 v4.0 Token/Attention，则必须先通过仿真和 HIL 验证。实机不允许普通 TD3 actor 直接输出完整动作并自由探索。

    * **定位**：将仿真中训练好的 robust baseline 部署到真实设备，通过小 residual online adaptation 补偿仿真-现实差距，保持硬件安全、可回滚、可审计。

    * **网络形式**：`a_real = safety_filter(a_base(s) + α_real Δπθ(s))`，其中 a_base(s) 为 v3/v4 得到的 robust baseline，Δπθ(s) 为实机上允许继续学习的小 residual，α_real 非常小（例如 0.01 / 0.02 / 0.05）。

    * **训练流程**：
      - Phase 5-a：只运行 baseline，不更新 actor，收集真实数据；
      - Phase 5-b：用真实数据 warmup / calibration critic；
      - Phase 5-c：critic audit 通过后，只允许 residual actor 小步更新；
      - Phase 5-d：每次更新后通过 rollout gate / safety gate 判断是否保留，性能变差则 rollback，触发安全边界则 fallback 到 baseline。

    * **安全机制**：action limit、gimbal rate limit、wheel speed / wheel acceleration limit、姿态误差边界、角速度边界、电流 / 温度 / 振动等硬件安全指标、emergency fallback、rollback checkpoint。

    * **critic 用途**：实机 ensemble critic 主要是 safety gate，不是探索发动机。critic disagreement 大时，不允许 actor 更新。只有 Q 稳定、uncertainty 低、rollout 不劣于 baseline，才允许保留更新。

    * **验收**：实机 residual adaptation 不劣于 baseline、safety shield 不频繁触发、没有动作饱和、没有高频震荡、不出现不可恢复姿态偏差、可回滚、所有训练/更新过程可审计。
