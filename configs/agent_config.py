"""
VSCMG Agent / TD3 网络参数集中配置
===================================

用途：所有 Agent 相关超参数（网络结构、TD3 核心参数、学习率等）统一在此定义。
以后修改网络大小 / 学习率 / gamma / tau 等，只需改这里。

运行时覆盖说明：
    state_dim 和 action_dim 依赖环境，但可通过参数传入覆盖。
    config 文件中的默认值主要用于文档和 fallback。

优先级规则：
    train.py 调用 TD3 时传入的参数 > 本文件默认值

用法：
    from configs.agent_config import AgentConfig, make_default_agent_config

    # 运行时从向量环境自动获取维度（single space）
    cfg = make_default_agent_config()
    cfg.state_dim = envs.single_observation_space.shape[0]
    cfg.action_dim = envs.single_action_space.shape[0]
    agent = TD3(state_dim=cfg.state_dim, action_dim=cfg.action_dim,
                hidden_dim=cfg.hidden_dim, ...)
"""

from dataclasses import dataclass


@dataclass
class AgentConfig:
    """
    TD3 Agent 网络与算法参数配置

    包含：维度（运行时覆盖）、网络结构、TD3 核心参数、学习率等。
    """

    # --- 网络维度（运行时由环境覆盖，config 仅作文档默认值） ---
    state_dim:    int    = 22       # 状态维度（22维 VSCMG 观测空间）
    action_dim:   int    = 8        # 动作维度（8维 VSCMG 控制空间）
    action_bound: float  = 1.0      # 动作输出上界（归一化空间）

    # --- 网络结构 ---
    hidden_dim:   int    = 256      # 隐藏层维度

    # --- TD3 核心参数 ---
    gamma:        float  = 0.99      # 折扣因子
    tau:          float  = 0.005    # 软更新系数
    policy_delay: int    = 2        # Actor 更新延迟步数

    # --- 探索噪声（在线策略） ---
    sigma:        float  = 0.1      # 在线探索噪声标准差

    # --- 学习率 ---
    actor_lr:     float  = 3e-4     # Actor 学习率
    critic_lr:    float  = 3e-4     # Critic 学习率

    # --- 目标策略平滑（TD3 更新内部参数，已接入训练链路） ---
    policy_noise: float  = 0.2      # 目标策略平滑噪声标准差（v1.0 第一阶段保守起点）
    noise_clip:   float  = 0.2      # 目标策略平滑噪声截断范围（v1.0 第一阶段保守起点）

    # --- Protected TD3：Actor 冻结步数 ---
    actor_freeze_steps: int = 0     # 前 N 次 update 只更新 critic，不更新 actor（0=关闭）

    # --- Protected TD3：BC Regularization ---
    bc_reg_weight: float = 0.0     # BC regularization 权重（0=关闭，保持原 TD3 行为）
    bc_reg_steps: int = 0          # BC reg 只在前 N 次 actor update 内生效（0=全期生效）

    # --- 设备 ---
    device:       str    = "cpu"    # 计算设备 (cpu/cuda)，可被 train.py CLI 覆盖


def make_default_agent_config() -> AgentConfig:
    """
    v1.0 第一阶段默认 Agent 配置（面向第一轮正式训练的保守起点）

    注意：
    - state_dim / action_dim 由 train.py 运行时从 env 自动覆盖，
      此处仅作为文档和 fallback。
    - policy_noise / noise_clip：已接入 TD3 训练链路，
      默认值 0.2 / 0.2 是 v1.0 第一阶段保守起点，后续可通过 CLI 或
      config 覆盖。
    """
    return AgentConfig(
        state_dim=22,
        action_dim=8,
        action_bound=1.0,
        hidden_dim=256,
        gamma=0.99,
        tau=0.005,
        policy_delay=2,
        sigma=0.1,
        actor_lr=3e-4,
        critic_lr=3e-4,
        policy_noise=0.2,
        noise_clip=0.2,
        device="cpu",
        actor_freeze_steps=0,
        bc_reg_weight=0.0,
        bc_reg_steps=0,
    )
