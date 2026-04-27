# VSCMG 端到端控制工程 - 全版本战略路线图 (Master Roadmap)



**【核心准则】**

从 v1.0 开始，TD3 神经网络的 22 维状态语义彻底冻结。动作语义分层推进：full_8d（8 维）是最终目标，gimbal_only（4 维）是当前 debug / v1.0 前置阶段的中间验证接口。版本演进的核心逻辑是：维持状态空间物理语义不变，在 gimbal_only 模式下先建立稳定控制闭环，再用 protected TD3 / BC-assisted TD3 逐步扩展到 full_8d 全动作空间，最终解锁更高级的 VSCMG 控制战术。

**冻结承诺：**
- v1.0~v4.0 的状态语义（22 维）保持不变
- full_8d 动作语义（8 维）是最终目标，v1.0 阶段先用 gimbal_only（4 维）验证闭环
- v2.0 改变初始化难度，v3.0 注入扰动与延迟，v4.0 重构前端编码器
- "语义冻结"不等于"MLP 结构冻结"，v4.0 可引入 Token 化前端

---

**【当前调试结论：v0.5.18-debug-bc-init】**

> debug tag: `v0.5.18-debug-bc-init`

1. **BC actor 有效**：在 `env.reset(options={"init_attitude_deg": angle})` 下，BC actor 的姿态恢复精度：
   - 10° → ~0.29°，20° → ~1.12°，30° → ~2.18°，45° → ~4.30°
2. **`--actor_init_path` 链路已验证**：BC actor 权重能正确加载到 `agent.actor` 和 `agent.target_actor`。
3. **普通 TD3 fine-tune 失败**：checkpoint_10k 已严重退化，动作从 BC 的小动作退化为高饱和大动作。
4. **`best_episode_reward.pth` 不是可靠的选模标准**：需改为 rollout-based 选模。
5. **下一阶段路线 → protected TD3 / BC-assisted TD3**：
   - actor freeze / actor update gate（限制 actor 更新频率或幅度）
   - critic warmup（先训 critic，再放开 actor）
   - replay prefill（用 BC 数据预填充 replay buffer）
   - low actor_lr（降低 actor 学习率）
   - BC regularization（actor loss 中加入 BC 行为克隆正则项）
   - rollout-based 选模（以实际 rollout 表现选模型，而非 episode reward）

---

**【版本演进大纲】**



* **v1.0：温室学步期 (Greenhouse Foundation)**

    * **定位**：在 gimbal_only 模式下，用 protected TD3 / BC-assisted TD3 建立稳定控制闭环。

    * **环境**：静止初始状态，零扰动，飞轮严格锁定 3000 rpm 偏置。初始误差从 5° 开始，逐步扩展到 10° / 20° / 30°，45° 作为测试上限。

    * **战术**：以 SGCMG 模式运行，轻微转动框架完成微调。使用 BC actor 初始化 + protected TD3 更新机制保护已习得的策略。

    * **里程碑**：gimbal_only 4 维动作下，30° 以内姿态误差收敛到 BC 水平或更好。v1.0 前置阶段不引入 full_8d；full_8d 放到 v1.0 后半段或 v1.x 阶段单独恢复和验证。



* **v2.0：全域机动期 (Full-Domain Maneuver)**

    * **前置依赖**：v1.0 protected TD3 在 gimbal_only 模式下稳定收敛。

    * **定位**：从 gimbal_only 扩展到 full_8d，领悟真正的 VSCMG 战术与奇异面规避。

    * **环境**：全空域随机初始状态（±180°），带随机初始翻滚角速度。

    * **战术**：逼迫网络在长距离机动中，主动对飞轮加减速（逼近 5000 rpm 极限）以跨越框架奇异点，机动后恢复动量。



* **v3.0：抗扰压榨期 (Robustness & Reality Gap)**

    * **前置依赖**：至少完成 v1.x/full_8d 稳定控制器，理想情况下完成 v2.0 后再进入 v3.0。不要过早引入扰动、延迟或参数随机化，否则会掩盖训练不稳定问题。

    * **定位**：跨越 Sim-to-Real 鸿沟，达到工业级部署标准。

    * **环境**：注入转动惯量偏差（±15%）、一阶伺服电机延迟、以及空间周期性外部扰动力矩。

    * **战术**：利用非对称 Actor-Critic 架构稳住阵脚，在恶劣硬件与外界干扰下消除极限环震荡。



* **v4.0：学术巅峰期 (Academic Peak)**

    * **前置依赖**：v3.0 稳定。当前瓶颈是 TD3 更新机制（普通 fine-tune 导致策略退化），不是 MLP 表达能力，因此不优先更换网络架构。

    * **定位**：打破 DRL 黑盒，实现多执行器控制的可解释性。

    * **环境**：维持 v3.0 的全量地狱级难度。

    * **战术**：前端重构。将**同一套冻结的 22 维物理语义**重新打包为实体 Token（本体 Token + 4 个 VSCMG Token），引入物理引导的自注意力机制（Self-Attention），输出可解释的电机分配热力图。**v4.0 改变的是前端编码方式，不改变状态语义。**
