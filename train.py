"""
VSCMG 姿态控制强化学习训练脚本
TD3 算法 — ��环境异步并行训练主循环 

v1.0 Reward 重构与工程稳定性验证
========================================

三层配置入口：
  - configs/env_config.py   : 环境物理参数
  - configs/train_config.py : 训练调度参数
  - configs/agent_config.py : TD3 / 网络超参数

[快速启动指南]
========================================

默认模式 (CPU单核):
    python train.py

调试模式 (CPU多核):
    python train.py --num_envs 4

极致压榨模式 (RTX 5070 + Ultra 7 专属):
    python train.py --num_envs 16 --batch_size 2048 --update_every 200 --device cuda

[优先级规则]
========================================
CLI 参数 > train_config.py 默认值
state_dim/action_dim 由运行时 env 自动覆盖（config 仅作文档/fallback）

========================================
"""

import os
# 底层防御：强制单线程模式，切断线程风暴防止 Windows 多进程死锁
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import datetime
import random
import time as _time_module
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from envs.vscmg_env import VSCMGEnv, RewardConfig, RewardNormalizationConfig
from agents.td3_agent import TD3, ReplayBuffer
from configs.train_config import TrainConfig, make_default_train_config
from configs.agent_config import AgentConfig, make_default_agent_config
from utils.version import get_run_version_label, get_git_version, get_git_commit, is_git_dirty
from configs.env_config import VSCMGEnvConfig, make_default_config as _make_default_env_config
import json


def generate_run_name(train_cfg: TrainConfig, agent_cfg: AgentConfig, reward_cfg: RewardConfig) -> str:
    """
    生成唯一实验标识符

    包含：版本号 + 时间戳 + 并行数 + seed + gimbal_rate + gamma + reward 权重摘要
    """
    version = get_run_version_label()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _env_cfg = _env_config_override if _env_config_override is not None else VSCMGEnvConfig()
    gr_str = f"gr{_env_cfg.max_gimbal_rate}"
    gamma_str = f"g{agent_cfg.gamma}"
    action_mode_str = "gimbal_only" if _env_cfg.action_mode == "gimbal_only" else "full8d"
    reward_summary = f"att{reward_cfg.w_att}_om{reward_cfg.w_omega}_wb{reward_cfg.w_wheel_bias}_ga{reward_cfg.w_gimbal_act}_wa{reward_cfg.w_wheel_act}"
    return f"{version}_{timestamp}_envs{train_cfg.num_envs}_seed{train_cfg.seed}_{gr_str}_{gamma_str}_{action_mode_str}_{reward_summary}"


def save_run_config(model_dir: str, run_name: str, train_cfg: TrainConfig,
                    agent_cfg: AgentConfig, reward_cfg: RewardConfig,
                    reward_norm_cfg,
                    actor_init_path: str | None = None,
                    critic_init_path: str | None = None):
    """保存实验配置到 JSON"""
    version = get_run_version_label()
    # 使用实际 env config（已被 CLI 覆盖）
    _env_cfg = _env_config_override if _env_config_override is not None else VSCMGEnvConfig()

    config = {
        "run_name": run_name,
        "version": version,
        "git_version": get_git_version(),
        "git_commit": get_git_commit(),
        "git_dirty": is_git_dirty(),
        "initial_delta_deg": _env_cfg.initial_delta_deg.tolist(),
        "max_gimbal_rate": _env_cfg.max_gimbal_rate,
        "max_wheel_accel": _env_cfg.max_wheel_accel,
        "action_mode": _env_cfg.action_mode,
        "actor_init_path": actor_init_path,
        "critic_init_path": critic_init_path,
        "init_attitude_range_deg": [
            _env_cfg.randomization.init_attitude_range.low,
            _env_cfg.randomization.init_attitude_range.high,
        ],
        "timestamp": datetime.datetime.now().isoformat(),
        "train_config": {
            "num_envs": train_cfg.num_envs,
            "device": train_cfg.device,
            "total_steps": train_cfg.total_steps,
            "start_steps": train_cfg.start_steps,
            "update_every": train_cfg.update_every,
            "update_times": train_cfg.update_times,
            "batch_size": train_cfg.batch_size,
            "replay_capacity": train_cfg.replay_capacity,
            "replay_prefill_path": train_cfg.replay_prefill_path,
            "critic_warmup_steps": train_cfg.critic_warmup_steps,
            "checkpoint_frequency": train_cfg.checkpoint_frequency,
            "seed": train_cfg.seed,
        },
        "agent_config": {
            "state_dim": agent_cfg.state_dim,
            "action_dim": agent_cfg.action_dim,
            "hidden_dim": agent_cfg.hidden_dim,
            "gamma": agent_cfg.gamma,
            "tau": agent_cfg.tau,
            "policy_delay": agent_cfg.policy_delay,
            "sigma": agent_cfg.sigma,
            "actor_lr": agent_cfg.actor_lr,
            "critic_lr": agent_cfg.critic_lr,
            "policy_noise": agent_cfg.policy_noise,
            "noise_clip": agent_cfg.noise_clip,
            "actor_freeze_steps": agent_cfg.actor_freeze_steps,
            "bc_reg_weight": agent_cfg.bc_reg_weight,
            "bc_reg_steps": agent_cfg.bc_reg_steps,
        },
        "reward_config": {
            "w_att": reward_cfg.w_att,
            "w_omega": reward_cfg.w_omega,
            "w_wheel_bias": reward_cfg.w_wheel_bias,
            "w_gimbal_act": reward_cfg.w_gimbal_act,
            "w_wheel_act": reward_cfg.w_wheel_act,
            "reward_scale": reward_cfg.reward_scale,
        },
        "reward_normalization_config": {
            "sigma_ref": reward_norm_cfg.sigma_ref,
            "omega_ref": reward_norm_cfg.omega_ref,
            "wheel_bias_ref": reward_norm_cfg.wheel_bias_ref,
            "gimbal_action_scale": reward_norm_cfg.gimbal_action_scale,
            "wheel_action_scale": reward_norm_cfg.wheel_action_scale,
        },
    }

    config_path = os.path.join(model_dir, "run_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def make_env():
    """
    顶层环境工厂函数（可序列化）
    用于 AsyncVectorEnv 实例化
    """
    return VSCMGEnv(config=_env_config_override)


# 模块级环境配置覆盖（由 main 中 CLI 参数设置）
_env_config_override = None


def load_actor_weights(agent, actor_init_path, expected_action_dim, device):
    """
    从 checkpoint 严格加载 actor 权重到 TD3 agent。

    支持格式：
    - {'actor': state_dict}
    - {'actor_state_dict': state_dict}

    要求：
    - fc3.weight.shape[0] 必须等于 expected_action_dim
    - action_dim 不匹配直接 raise RuntimeError
    - 不允许跳过输出层，不允许 partial load
    - strict=True 加载到 actor 和 actor_target
    """
    checkpoint = torch.load(actor_init_path, map_location=device)

    if 'actor' in checkpoint:
        actor_sd = checkpoint['actor']
    elif 'actor_state_dict' in checkpoint:
        actor_sd = checkpoint['actor_state_dict']
    else:
        raise ValueError(
            f"[load_actor_weights] 无法识别的 checkpoint 格式，可用键: {list(checkpoint.keys())}. "
            f"需要 'actor' 或 'actor_state_dict'。"
        )

    # 严格检查输出层维度
    if 'fc3.weight' in actor_sd and actor_sd['fc3.weight'].shape[0] != expected_action_dim:
        raise RuntimeError(
            f"[load_actor_weights] Shape mismatch: checkpoint fc3.weight 输出 dim = {actor_sd['fc3.weight'].shape[0]}, "
            f"expected action_dim = {expected_action_dim}. "
            f"DO NOT silently skip output layer. Use a compatible checkpoint."
        )

    # 加载到 actor 和 actor_target（strict=True）
    agent.actor.load_state_dict(actor_sd, strict=True)
    agent.target_actor.load_state_dict(actor_sd, strict=True)


def load_critic_weights_from_calibrated(agent, calibrated_ckpt_path, device):
    """
    Load critic weights from an offline calibrated critic checkpoint.

    This loads only:
    - critic_1
    - critic_2
    - target_critic_1
    - target_critic_2

    It does not load actor weights and does not load optimizer states.
    """
    ck = torch.load(calibrated_ckpt_path, map_location=device)

    required_keys = [
        "critic_1_state_dict",
        "critic_2_state_dict",
        "target_critic_1_state_dict",
        "target_critic_2_state_dict",
    ]
    for k in required_keys:
        if k not in ck:
            raise KeyError(f"Missing required key in calibrated critic checkpoint: {k}")

    agent.critic_1.load_state_dict(ck["critic_1_state_dict"], strict=True)
    agent.critic_2.load_state_dict(ck["critic_2_state_dict"], strict=True)
    agent.target_critic_1.load_state_dict(ck["target_critic_1_state_dict"], strict=True)
    agent.target_critic_2.load_state_dict(ck["target_critic_2_state_dict"], strict=True)

    return ck


def parse_args():
    """
    解析命令行参数（CLI 优先级高于 train_config 默认值）

    所有这些参数在 train_config.py 中都有对应的默认值，
    此处 CLI 值会覆盖默认值。
    """
    parser = argparse.ArgumentParser(description="VSCMG TD3 并行训练脚本")

    # --- 并行与设备 ---
    parser.add_argument("--num_envs", type=int, default=None,
                        help="并行环境数量（覆盖 train_config 默认值）")
    parser.add_argument("--device", type=str, default=None,
                        help="计算设备 cpu/cuda（覆盖 train_config 默认值）")

    # --- 训练步数 ---
    parser.add_argument("--max_steps", type=int, default=None,
                        help="最大训练步数（覆盖 train_config 默认值）")
    parser.add_argument("--start_steps", type=int, default=None,
                        help="随机探索步数（覆盖 train_config 默认值）")

    # --- 更新策略（update_every 和 update_times 独立控制） ---
    parser.add_argument("--update_every", type=int, default=None,
                        help="每N步触发一次网络更新（覆盖 train_config 默认值）")
    parser.add_argument("--update_times", type=int, default=None,
                        help="每次触发更新N轮（覆盖 train_config 默认值）")

    # --- Batch 与 Replay ---
    parser.add_argument("--batch_size", type=int, default=None,
                        help="批次大小（覆盖 train_config 默认值）")
    parser.add_argument("--replay_capacity", type=int, default=None,
                        help="经验回放池容量（覆盖 train_config 默认值）")
    parser.add_argument("--checkpoint_frequency", type=int, default=None,
                        help="每N步保存一次 checkpoint（覆盖 train_config 默认值）")

    # --- 随机种子 ---
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（覆盖 train_config 默认值）；0 或负值表示不固定随机种子")

    # --- TD3 噪声参数 ---
    parser.add_argument("--exploration_noise", type=float, default=None,
                        help="在线探索噪声 sigma（覆盖 agent_config 默认值）")
    parser.add_argument("--policy_noise", type=float, default=None,
                        help="目标策略平滑噪声标准差（覆盖 agent_config 默认值）")
    parser.add_argument("--noise_clip", type=float, default=None,
                        help="目标策略平滑噪声截断范围（覆盖 agent_config 默认值）")

    # --- 环境物理参数 ---
    parser.add_argument("--max_gimbal_rate", type=float, default=None,
                        help="框架角速度物理上限 rad/s（覆盖 env_config 默认值）")
    parser.add_argument("--init_attitude_min_deg", type=float, default=None,
                        help="初始等效旋转角下限（度）（覆盖 env_config，默认 0.0）")
    parser.add_argument("--init_attitude_max_deg", type=float, default=None,
                        help="初始等效旋转角上限（度）（覆盖 env_config，默认 5.0）")
    parser.add_argument("--action_mode", type=str, default=None,
                        choices=["full_8d", "gimbal_only"],
                        help="动作空间模式：full_8d=8维 gimbal+wheel（默认），gimbal_only=4维 仅 gimbal")

    # --- Agent 核心参数 ---
    parser.add_argument("--gamma", type=float, default=None,
                        help="TD3 折扣因子（覆盖 agent_config 默认值）")
    parser.add_argument("--actor_lr", type=float, default=None,
                        help="Actor 学习率（覆盖 agent_config 默认值）")
    parser.add_argument("--critic_lr", type=float, default=None,
                        help="Critic 学习率（覆盖 agent_config 默认值）")
    parser.add_argument("--actor_freeze_steps", type=int, default=None,
                        help="前 N 次 update 只更新 critic，不更新 actor（覆盖 agent_config 默认值）")
    parser.add_argument("--bc_reg_weight", type=float, default=None,
                        help="BC regularization 权重（覆盖 agent_config 默认值，0=关闭）")
    parser.add_argument("--bc_reg_steps", type=int, default=None,
                        help="BC reg 只在前 N 次 actor update 内生效（覆盖 agent_config 默认值，0=全期）")

    # --- Reward 权重参数 ---
    parser.add_argument("--w_gimbal_act", type=float, default=None,
                        help="gimbal action penalty 权重（覆盖 reward_config 默认值）")
    parser.add_argument("--w_wheel_act", type=float, default=None,
                        help="wheel action penalty 权重（覆盖 reward_config 默认值）")

    # --- Actor 初始化路径（BC-init TD3） ---
    parser.add_argument("--actor_init_path", type=str, default=None,
                        help="从指定路径加载 actor 权重初始化 TD3 actor（BC-init 用途）")

    # --- Critic 初始化路径（离线校准 critic） ---
    parser.add_argument(
        "--critic_init_path",
        type=str,
        default=None,
        help="Path to calibrated critic checkpoint for initializing critic_1/2 and target_critic_1/2 only.",
    )

    # --- Replay Prefill ---
    parser.add_argument("--replay_prefill_path", type=str, default=None,
                        help="npz 文件路径，训练开始前预填充 replay buffer")

    # --- Critic Warmup ---
    parser.add_argument("--critic_warmup_steps", type=int, default=None,
                        help="训练主循环前使用 replay prefill 数据进行 critic-only warmup 的 update 次数；0=关闭")

    return parser.parse_args()


def _apply_cli_overrides(cfg: TrainConfig, args) -> TrainConfig:
    """
    将 CLI 参数（已设置 default=None）覆盖到 TrainConfig 实例上。
    仅当 args.XXX is not None 时覆盖，保持 None 表示"使用 config 默认值"。
    """
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs
    if args.device is not None:
        cfg.device = args.device
    if args.max_steps is not None:
        cfg.total_steps = args.max_steps
    if args.start_steps is not None:
        cfg.start_steps = args.start_steps
    if args.update_every is not None:
        cfg.update_every = args.update_every
    if args.update_times is not None:
        cfg.update_times = args.update_times
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.replay_capacity is not None:
        cfg.replay_capacity = args.replay_capacity
    if args.checkpoint_frequency is not None:
        cfg.checkpoint_frequency = args.checkpoint_frequency
    if args.seed is not None:
        cfg.seed = args.seed
    if args.replay_prefill_path is not None:
        cfg.replay_prefill_path = args.replay_prefill_path
    if args.critic_warmup_steps is not None:
        cfg.critic_warmup_steps = args.critic_warmup_steps
    return cfg


def _apply_cli_agent_overrides(cfg: AgentConfig, args) -> AgentConfig:
    """
    将 CLI 噪声参数覆盖到 AgentConfig 实例上。
    仅当 args.XXX is not None 时覆盖。
    """
    if args.exploration_noise is not None:
        cfg.sigma = args.exploration_noise
    if args.policy_noise is not None:
        cfg.policy_noise = args.policy_noise
    if args.noise_clip is not None:
        cfg.noise_clip = args.noise_clip
    if args.gamma is not None:
        cfg.gamma = args.gamma
    if args.actor_lr is not None:
        cfg.actor_lr = args.actor_lr
    if args.critic_lr is not None:
        cfg.critic_lr = args.critic_lr
    if args.actor_freeze_steps is not None:
        cfg.actor_freeze_steps = args.actor_freeze_steps
    if args.bc_reg_weight is not None:
        cfg.bc_reg_weight = args.bc_reg_weight
    if args.bc_reg_steps is not None:
        cfg.bc_reg_steps = args.bc_reg_steps
    return cfg


def set_global_seed(seed: int) -> None:
    """
    设置全局随机种子（numpy + torch + random + cuda）。

    向量环境的 seed 通过首次 envs.reset(seed=seed) 设置，不在这里调用 reset。

    Args:
        seed: 随机种子（0 或负值 = 不设置，保持随机）
    """
    if seed <= 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_config_snapshot(train_cfg: TrainConfig, agent_cfg: AgentConfig,
                          state_dim: int, action_dim: int):
    """打印配置快照，方便人工核对"""
    print("=" * 60)
    print("【配置快照】")
    print("-" * 40)
    print("  [训练调度]")
    print(f"    num_envs        = {train_cfg.num_envs}")
    print(f"    device          = {train_cfg.device}")
    print(f"    total_steps     = {train_cfg.total_steps:,}")
    print(f"    start_steps     = {train_cfg.start_steps:,}")
    print(f"    update_every    = {train_cfg.update_every}")
    print(f"    update_times    = {train_cfg.update_times}")
    print(f"    batch_size      = {train_cfg.batch_size}")
    print(f"    replay_capacity = {train_cfg.replay_capacity:,}")
    print(f"    checkpoint_freq = {train_cfg.checkpoint_frequency:,}")
    print(f"    seed            = {train_cfg.seed}")
    print(f"    replay_prefill_path = {train_cfg.replay_prefill_path}")
    print(f"    critic_warmup_steps = {train_cfg.critic_warmup_steps}")
    print("-" * 40)
    print("  [Agent / TD3]")
    print(f"    state_dim       = {state_dim}")
    print(f"    action_dim      = {action_dim}")
    print(f"    hidden_dim      = {agent_cfg.hidden_dim}")
    print(f"    actor_lr        = {agent_cfg.actor_lr}")
    print(f"    critic_lr       = {agent_cfg.critic_lr}")
    print(f"    gamma           = {agent_cfg.gamma}")
    print(f"    tau             = {agent_cfg.tau}")
    print(f"    policy_delay    = {agent_cfg.policy_delay}")
    print(f"    sigma           = {agent_cfg.sigma}")
    print(f"    policy_noise    = {agent_cfg.policy_noise}")
    print(f"    noise_clip      = {agent_cfg.noise_clip}")
    print("-" * 40)
    print("  [Env / Physics]")
    _ecfg = _env_config_override if _env_config_override is not None else VSCMGEnvConfig()
    print(f"    max_gimbal_rate = {_ecfg.max_gimbal_rate} rad/s")
    print(f"    max_wheel_accel = {_ecfg.max_wheel_accel} rad/s^2")
    print(f"    initial_delta_deg = {_ecfg.initial_delta_deg.tolist()}")
    print(f"    init_attitude_range_deg = [{_ecfg.randomization.init_attitude_range.low}, {_ecfg.randomization.init_attitude_range.high}]")
    print("=" * 60)


# =============================================================================
# Replay prefill helper
# =============================================================================
def _prefill_replay_buffer(replay_buffer, prefill_path: str, state_dim: int, action_dim: int):
    """
    从 npz 文件加载 transition 数据并预填充到 ReplayBuffer。
    """
    import numpy as np

    print(f"\n[ReplayPrefill] Loading from: {prefill_path}")
    data = np.load(prefill_path, allow_pickle=False)

    # Required fields check
    required_fields = ["obs", "actions", "rewards", "next_obs", "dones"]
    for field in required_fields:
        if field not in data.keys():
            raise ValueError(f"Missing required field '{field}' in {prefill_path}")

    obs = data["obs"]
    actions = data["actions"]
    rewards = data["rewards"]
    next_obs = data["next_obs"]
    dones = data["dones"]

    # Shape checks
    if obs.ndim != 2:
        raise ValueError(f"obs.ndim should be 2, got {obs.ndim}")
    if actions.ndim != 2:
        raise ValueError(f"actions.ndim should be 2, got {actions.ndim}")
    if next_obs.ndim != 2:
        raise ValueError(f"next_obs.ndim should be 2, got {next_obs.ndim}")
    if rewards.ndim != 1:
        raise ValueError(f"rewards.ndim should be 1, got {rewards.ndim}")
    if dones.ndim != 1:
        raise ValueError(f"dones.ndim should be 1, got {dones.ndim}")

    n = obs.shape[0]
    if not (actions.shape[0] == n and rewards.shape[0] == n and next_obs.shape[0] == n and dones.shape[0] == n):
        raise ValueError(f"Shape mismatch among transition fields")

    # Dimension checks
    if obs.shape[1] != state_dim:
        raise ValueError(f"obs.shape[1]={obs.shape[1]} != state_dim={state_dim}")
    if next_obs.shape[1] != state_dim:
        raise ValueError(f"next_obs.shape[1]={next_obs.shape[1]} != state_dim={state_dim}")
    if actions.shape[1] != action_dim:
        raise ValueError(f"actions.shape[1]={actions.shape[1]} != action_dim={action_dim}")

    # Finite check
    if not np.isfinite(obs).all():
        raise ValueError("obs contains non-finite values")
    if not np.isfinite(actions).all():
        raise ValueError("actions contains non-finite values")
    if not np.isfinite(rewards).all():
        raise ValueError("rewards contains non-finite values")
    if not np.isfinite(next_obs).all():
        raise ValueError("next_obs contains non-finite values")

    # Push to replay buffer
    for i in range(n):
        replay_buffer.push(
            obs[i],
            actions[i],
            float(rewards[i]),
            next_obs[i],
            bool(dones[i])
        )

    # Statistics
    print(f"[ReplayPrefill] Loaded {n} transitions")
    print(f"[ReplayPrefill] obs shape: {obs.shape}, actions shape: {actions.shape}")
    print(f"[ReplayPrefill] rewards: min={float(rewards.min()):.6f}, max={float(rewards.max()):.6f}, mean={float(rewards.mean()):.6f}, std={float(rewards.std()):.6f}")
    print(f"[ReplayPrefill] action_abs_mean: {float(np.mean(np.abs(actions))):.6f}")
    print(f"[ReplayPrefill] action_abs_max: {float(np.max(np.abs(actions))):.6f}")
    print(f"[ReplayPrefill] action_sat_rate>0.95: {float(np.mean(np.abs(actions) > 0.95)):.6f}")
    print(f"[ReplayPrefill] done_true_ratio: {float(np.mean(dones.astype(bool))):.4f}")
    print(f"[ReplayPrefill] replay_buffer current length: {len(replay_buffer)}")

    # init_attitude_deg distribution (if available)
    if "init_attitude_deg" in data:
        iad = data["init_attitude_deg"]
        unique_angles = np.unique(iad[~np.isnan(iad)])
        print(f"[ReplayPrefill] init_attitude_deg unique values: {unique_angles.tolist()}")
        for angle in unique_angles:
            count = int(np.sum(iad == angle))
            print(f"  angle={float(angle):.1f} deg: {count} transitions")

    data.close()


# =============================================================================
# Critic warmup helper
# =============================================================================
def _run_critic_warmup(agent, replay_buffer, warmup_steps: int, batch_size: int):
    """
    在正式训练前使用 replay prefill 数据进行 critic-only warmup。

    工作原理：
    - warmup 前设置 agent.total_count = -warmup_steps
    - warmup 阶段 total_count 从 -warmup_steps 增加到 0
    - 由于 total_count <= 0，actor 不会更新（actor_should_update = total_count > actor_freeze_steps）
    - warmup 后 total_count 恰好为 0，不消耗 actor_freeze_steps 配额
    """
    if warmup_steps <= 0:
        return

    if len(replay_buffer) < batch_size:
        raise ValueError(
            f"critic_warmup_steps={warmup_steps} requires replay_buffer length >= batch_size={batch_size}, "
            f"but got {len(replay_buffer)}"
        )

    actor_update_count_before = agent.actor_update_count

    # 设置 total_count 为负数，warmup 后会恰好回到 0
    agent.total_count = -warmup_steps

    critic1_losses = []
    critic2_losses = []

    for i in range(warmup_steps):
        actor_loss, c1_loss, c2_loss, actor_q_loss, bc_reg_loss, bc_mse_loss, bc_reg_active = agent.update(
            replay_buffer,
            batch_size,
        )

        # warmup 阶段 actor 不应该更新
        if actor_loss is not None:
            raise RuntimeError(
                f"critic warmup unexpectedly updated actor at step {i + 1}. "
                f"actor_loss={actor_loss}, actor_update_count={agent.actor_update_count}"
            )

        critic1_losses.append(float(c1_loss))
        critic2_losses.append(float(c2_loss))

    # warmup 后 total_count 应该恰好为 0
    if agent.total_count != 0:
        raise RuntimeError(
            f"after warmup, total_count={agent.total_count}, expected 0"
        )

    # warmup 后 actor_update_count 不应该增加
    if agent.actor_update_count != actor_update_count_before:
        raise RuntimeError(
            f"after warmup, actor_update_count increased from {actor_update_count_before} to {agent.actor_update_count}"
        )

    print(f"\n[CriticWarmup] warmup_steps={warmup_steps}")
    print(f"[CriticWarmup] final total_count={agent.total_count}")
    print(f"[CriticWarmup] actor_update_count: {actor_update_count_before} -> {agent.actor_update_count} (unchanged)")
    print(f"[CriticWarmup] critic_1_loss: first={critic1_losses[0]:.6f}, last={critic1_losses[-1]:.6f}, mean={sum(critic1_losses) / len(critic1_losses):.6f}")
    print(f"[CriticWarmup] critic_2_loss: first={critic2_losses[0]:.6f}, last={critic2_losses[-1]:.6f}, mean={sum(critic2_losses) / len(critic2_losses):.6f}")


if __name__ == "__main__":
    # ============================================================================
    # 第一步：配置加载（config 默认值 + CLI 覆盖）
    # ============================================================================
    args = parse_args()

    # 从默认配置开始
    train_cfg = make_default_train_config()
    agent_cfg = make_default_agent_config()

    # CLI 覆盖（CLI 存在时优先于 config 默认值）
    train_cfg = _apply_cli_overrides(train_cfg, args)
    agent_cfg = _apply_cli_agent_overrides(agent_cfg, args)
    # 设备同步到 agent_cfg
    agent_cfg.device = train_cfg.device

    # 设置全局随机种子（numpy + torch + random + cuda）
    set_global_seed(train_cfg.seed)

    # ============================================================================
    # 第一步 b：环境配置（CLI 覆盖 max_gimbal_rate 等）
    # ============================================================================
    import sys
    _this = sys.modules[__name__]
    _this._env_config_override = _make_default_env_config()
    if args.max_gimbal_rate is not None:
        _this._env_config_override.max_gimbal_rate = args.max_gimbal_rate
    if args.action_mode is not None:
        _this._env_config_override.action_mode = args.action_mode
    if args.init_attitude_min_deg is not None or args.init_attitude_max_deg is not None:
        min_deg = args.init_attitude_min_deg if args.init_attitude_min_deg is not None else 0.0
        max_deg = args.init_attitude_max_deg if args.init_attitude_max_deg is not None else 5.0
        _this._env_config_override.randomization.init_attitude_enabled = True
        from configs.env_config import UniformRange
        _this._env_config_override.randomization.init_attitude_range = UniformRange(low=min_deg, high=max_deg)

    # ============================================================================
    # 第二步：环境创建（state_dim / action_dim 运行时自动覆盖）
    # ============================================================================
    # 创建向量环境
    env_fns = [make_env for _ in range(train_cfg.num_envs)]
    if train_cfg.num_envs == 1:
        print("[Tracer] 准备唤醒 1 个同步物理引擎...")
        envs = SyncVectorEnv(env_fns)
        print("[Tracer] 同步引擎唤醒成功！")
    else:
        print(f"[Tracer] 准备唤醒 {train_cfg.num_envs} 个并行物理引擎 (可能需等待10-30秒，请耐心)...")
        envs = AsyncVectorEnv(env_fns)
        print("[Tracer] 并行引擎唤醒成功！")

    # 运行时自动覆盖 agent 维度（从 env 的 space 自动读取）
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    agent_cfg.state_dim = state_dim
    agent_cfg.action_dim = action_dim

    # ============================================================================
    # 第三步：智能提示（检测是否使用最低配）
    # ============================================================================
    if train_cfg.num_envs == 1 and train_cfg.device == "cpu":
        print("\n" + "*" * 80)
        print("【智能提示】")
        print("检测到当前使用基础单核配置运行。")
        print()
        print("如果您的机器配置较高（例如拥有独立显卡和多核 CPU），")
        print("强烈建议在运行配置或终端中添加参数以激活并行引擎，例如：")
        print()
        print("  python train.py --num_envs 16 --device cuda --batch_size 2048")
        print()
        print("(详见代码顶部注释 [快速启动指南])")
        print("*" * 80 + "\n")

    # ============================================================================
    # 第四步：Agent + ReplayBuffer 创建
    # ============================================================================
    replay_buffer = ReplayBuffer(capacity=train_cfg.replay_capacity)

    agent = TD3(
        state_dim=agent_cfg.state_dim,
        action_dim=agent_cfg.action_dim,
        hidden_dim=agent_cfg.hidden_dim,
        action_bound=agent_cfg.action_bound,
        sigma=agent_cfg.sigma,
        tau=agent_cfg.tau,
        gamma=agent_cfg.gamma,
        critic_lr=agent_cfg.critic_lr,
        actor_lr=agent_cfg.actor_lr,
        delay=agent_cfg.policy_delay,
        policy_noise=agent_cfg.policy_noise,
        noise_clip=agent_cfg.noise_clip,
        device=agent_cfg.device,
        actor_freeze_steps=agent_cfg.actor_freeze_steps,
        bc_reg_weight=agent_cfg.bc_reg_weight,
        bc_reg_steps=agent_cfg.bc_reg_steps,
    )
    device_torch = torch.device(agent_cfg.device)
    print(f"[Tracer] TD3 神经网络已建立并载入 {'GPU' if device_torch.type == 'cuda' else 'CPU'}！")

    # ============================================================================
    # 第四步 b：BC-init TD3 — 从 checkpoint 加载 actor 权重
    # ============================================================================
    if args.actor_init_path is not None:
        load_actor_weights(
            agent=agent,
            actor_init_path=args.actor_init_path,
            expected_action_dim=agent_cfg.action_dim,
            device=agent_cfg.device,
        )
        print(f"[ActorInit] actor initialized from: {args.actor_init_path}")

        # BC regularization：从已加载的 actor 创建 frozen BC reference
        if agent_cfg.bc_reg_weight > 0:
            agent.set_bc_reference_from_current_actor()
            print(f"[ActorInit] BC reference actor frozen (bc_reg_weight={agent_cfg.bc_reg_weight}, bc_reg_steps={agent_cfg.bc_reg_steps})")
    elif agent_cfg.bc_reg_weight > 0:
        raise ValueError(
            "bc_reg_weight > 0 requires --actor_init_path so a frozen BC reference actor can be created."
        )

    # ============================================================================
    # 第四步 b.2：CriticInit — 从���线校准 critic 加载权重
    # ============================================================================
    if args.critic_init_path is not None:
        _critic_ckpt = load_critic_weights_from_calibrated(
            agent, args.critic_init_path, agent_cfg.device
        )
        print(f"[CriticInit] critic_1/2 and target_critic_1/2 initialized from: {args.critic_init_path}")
        print(
            "[CriticInit] ranking_mode={}, regression_mode={}, steps={}, n_groups={}".format(
                _critic_ckpt.get("ranking_mode"),
                _critic_ckpt.get("regression_mode"),
                _critic_ckpt.get("steps"),
                _critic_ckpt.get("n_groups"),
            )
        )

    # ============================================================================
    # 第四步 c：ReplayBuffer Prefill
    # ============================================================================
    if train_cfg.replay_prefill_path is not None:
        _prefill_replay_buffer(
            replay_buffer=replay_buffer,
            prefill_path=train_cfg.replay_prefill_path,
            state_dim=state_dim,
            action_dim=action_dim,
        )

    # ============================================================================
    # 第四步 d：Critic Warmup
    # ============================================================================
    if train_cfg.critic_warmup_steps > 0:
        if train_cfg.replay_prefill_path is None:
            raise ValueError("critic_warmup_steps > 0 requires --replay_prefill_path")
        _run_critic_warmup(
            agent=agent,
            replay_buffer=replay_buffer,
            warmup_steps=train_cfg.critic_warmup_steps,
            batch_size=train_cfg.batch_size,
        )

    # ============================================================================
    # 第五步：打印配置快照
    # ============================================================================
    print_config_snapshot(train_cfg, agent_cfg, state_dim, action_dim)

    # ============================================================================
    # 第六步：生成 run_name 和模型目录
    # ============================================================================
    reward_cfg = RewardConfig()  # 读取当前默认 reward 配置
    # CLI 覆盖 reward 权重
    if args.w_gimbal_act is not None:
        reward_cfg.w_gimbal_act = args.w_gimbal_act
    if args.w_wheel_act is not None:
        reward_cfg.w_wheel_act = args.w_wheel_act
    reward_norm_cfg = RewardNormalizationConfig()
    run_name = generate_run_name(train_cfg, agent_cfg, reward_cfg)
    model_dir = os.path.join(train_cfg.checkpoint_dir, run_name)
    os.makedirs(model_dir, exist_ok=True)

    # 保存实验配置
    save_run_config(
        model_dir, run_name, train_cfg, agent_cfg,
        reward_cfg, reward_norm_cfg,
        actor_init_path=args.actor_init_path,
        critic_init_path=args.critic_init_path,
    )

    # ============================================================================
    # 第七步：TensorBoard 日志
    # ============================================================================
    ts = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_dir = f"{train_cfg.log_dir_base}/vscmg_parallel_{train_cfg.num_envs}envs_{ts}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=train_cfg.tb_flush_secs)
    print(f"\n[Storage] run_name: {run_name}")
    print(f"[Storage] 模型保存目录: {os.path.abspath(model_dir)}")
    print(f"[Storage] TensorBoard 日志正实时写入绝对路径: {os.path.abspath(writer.log_dir)}\n")

    # ============================================================================
    # 训练状态跟踪
    # ============================================================================
    best_reward = -1e6
    best_step = None
    _latest_det_sat_rate = 0.0
    _latest_det_action_std = 1.0
    train_start_time = _time_module.time()
    episode_rewards = np.zeros(train_cfg.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(train_cfg.num_envs, dtype=np.int32)
    first_episode_done = False  # 心跳静默开关

    def _episode_summary(ep_reward: float, ep_step: int, saved: str, path: str | None):
        """每次 episode 结束时打印统一风格的 [EpisodeSummary]"""
        elapsed = _time_module.time() - train_start_time
        elapsed_hms = str(datetime.timedelta(seconds=int(elapsed)))
        best_str = f"{best_reward:.4f}@{best_step}" if best_step is not None else "None"
        print(f"[EpisodeSummary] run={run_name}"
              f" | step={ep_step}"
              f" | ep_reward={ep_reward:.4f}"
              f" | best_so_far={best_str}"
              f" | saved={saved}"
              f" | path={path}"
              f" | elapsed={elapsed_hms}")

    def _log_and_checkpoint(ep_reward: float, ep_step: int):
        """全局最佳模型保存（基于单 episode 得分触发，不写 TensorBoard）"""
        global best_reward, best_step, first_episode_done, _latest_det_sat_rate, _latest_det_action_std
        first_episode_done = True
        saved = "none"
        path = None
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_step = ep_step
            best_path = os.path.join(model_dir, "best_episode_reward.pth")
            agent.save_model(best_path)
            print(f"  >> [Checkpoint] New best reward: {best_reward:.4f} -> {best_path}")
            saved = "best"
            path = best_path
            # Policy 健康度警告
            if _latest_det_sat_rate > 0.5:
                print(f"  >> [PolicyHealth WARNING] sat_rate={_latest_det_sat_rate:.3f} > 0.5 — actor may be over-saturated!")
            if _latest_det_action_std < 0.05:
                print(f"  >> [PolicyHealth WARNING] action_std={_latest_det_action_std:.4f} < 0.05 — actor may be collapsed!")
        _episode_summary(ep_reward, ep_step, saved, path)

    # ============================================================================
    # 训练主循环（全局步数驱动）
    # ============================================================================
    print("=" * 60)
    print(f"VSCMG TD3 异步并行训练已启动 ({get_run_version_label()})")
    print("=" * 60)

    # 初始化环境（首次 reset 时设置环境 seed）
    seed_value = train_cfg.seed if train_cfg.seed > 0 else None
    states, infos = envs.reset(seed=seed_value)
    print("[Tracer] 环境初态重置完毕，正在冲击主循环！")

    # 全局步数驱动主循环
    _reset_envs: set = set()  # 延迟重置集合（完赛→先读后清）
    for global_step in range(0, train_cfg.total_steps, train_cfg.num_envs):
        # --- 心跳监测（仅在首个 Episode 落地前静默打印） ---
        if not first_episode_done and global_step > 0 and global_step % 2000 == 0:
            print(f"[Heartbeat] Training progress: {global_step} steps processed...")

        # --- 定期保存检查点（每 checkpoint_frequency 步）---
        if global_step > 0 and global_step % train_cfg.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(model_dir, f"checkpoint_step_{global_step}.pth")
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_1_state_dict': agent.critic_1.state_dict(),
                'critic_2_state_dict': agent.critic_2.state_dict(),
                'target_actor_state_dict': agent.target_actor.state_dict(),
                'target_critic_1_state_dict': agent.target_critic_1.state_dict(),
                'target_critic_2_state_dict': agent.target_critic_2.state_dict(),
                'global_step': global_step,
            }, checkpoint_path)
            print(f"[Checkpoint] 检查点已保存至: {checkpoint_path}")
            elapsed = _time_module.time() - train_start_time
            elapsed_hms = str(datetime.timedelta(seconds=int(elapsed)))
            best_str = f"{best_reward:.4f}@{best_step}" if best_step is not None else "None"
            print(f"[SaveSummary] type=checkpoint | run={run_name} | step={global_step} | best_so_far={best_str} | elapsed={elapsed_hms} | path={checkpoint_path}")

        # --- 批量动作选择 ---
        if global_step < train_cfg.start_steps:
            # 纯随机探索阶段
            actions = envs.action_space.sample()
            det_actions = actions  # 随机阶段也记录 deterministic action
        else:
            # 批量策略推理
            states_tensor = torch.FloatTensor(states).to(device_torch)
            with torch.no_grad():
                det_actions = agent.actor(states_tensor).cpu().numpy()

            # 添加探索噪声
            noise = agent.sigma * np.random.randn(train_cfg.num_envs, agent.action_dim)
            actions = det_actions + noise
            actions = np.clip(actions, -agent.action_bound, agent.action_bound)

        # --- Deterministic actor 诊断（不影响训练） ---
        with torch.no_grad():
            det_action_abs_mean = float(np.mean(np.abs(det_actions)))
            det_action_sat_rate = float(np.mean(np.abs(det_actions) >= 0.95))
            det_action_sq = float(np.mean(np.sum(det_actions ** 2, axis=1)))
            det_action_std = float(np.std(det_actions))
            det_action_min = float(np.min(det_actions))
            det_action_max = float(np.max(det_actions))
        writer.add_scalar("ActorDet/action_abs_mean", det_action_abs_mean, global_step)
        writer.add_scalar("ActorDet/action_sat_rate", det_action_sat_rate, global_step)
        writer.add_scalar("ActorDet/action_sq", det_action_sq, global_step)
        writer.add_scalar("ActorDet/action_std", det_action_std, global_step)
        writer.add_scalar("ActorDet/action_min", det_action_min, global_step)
        writer.add_scalar("ActorDet/action_max", det_action_max, global_step)

        # --- Policy 健康度检测 ---
        is_saturated = 1.0 if det_action_sat_rate > 0.5 else 0.0
        is_collapsed = 1.0 if det_action_std < 0.05 else 0.0
        _latest_det_sat_rate = det_action_sat_rate
        _latest_det_action_std = det_action_std
        writer.add_scalar("PolicyHealth/is_actor_saturated", is_saturated, global_step)
        writer.add_scalar("PolicyHealth/is_actor_collapsed", is_collapsed, global_step)

        # --- 异步并行环境交互 ---
        next_states, rewards, dones, truncateds, infos = envs.step(actions)

        # --- 动作统计（每步记录）---
        action_abs_mean = float(np.mean(np.abs(actions)))
        action_sat_rate = float(np.mean(np.abs(actions) >= 0.95))
        writer.add_scalar("Diag/action_abs_mean", action_abs_mean, global_step)
        writer.add_scalar("Diag/action_sat_rate", action_sat_rate, global_step)

        # --- reward breakdown 统计（每步从 infos 直接读取，Gymnasium 向量环境把 breakdown 放在顶层）---
        for key in ["sigma_err_sq", "omega_sq", "wheel_bias_sq", "action_sq"]:
            if key in infos:
                vals = infos[key]
                # vals 是 numpy 数组，取所有 env 的均值
                if isinstance(vals, np.ndarray):
                    mean_val = float(np.mean(vals))
                    writer.add_scalar(f"Diag/{key}", mean_val, global_step)

        # --- reward 分项 tag（批量写入）---
        # RewardCost 组
        cost_keys = {
            "reward_attitude_cost": "RewardCost/attitude",
            "reward_omega_cost": "RewardCost/omega",
            "reward_wheel_bias_cost": "RewardCost/wheel_bias",
            "reward_gimbal_action_cost": "RewardCost/gimbal_action",
            "reward_wheel_action_cost": "RewardCost/wheel_action",
        }
        penalty_keys = {
            "reward_att_penalty": "RewardPenalty/attitude",
            "reward_omega_penalty": "RewardPenalty/omega",
            "reward_wheel_bias_penalty": "RewardPenalty/wheel_bias",
            "reward_gimbal_act_penalty": "RewardPenalty/gimbal_action",
            "reward_wheel_act_penalty": "RewardPenalty/wheel_action",
            "reward_raw_penalty": "RewardPenalty/raw_penalty",
        }
        reward_keys = {
            "reward_total": "Reward/reward_total",
        }

        for info_key, tag in {**cost_keys, **penalty_keys, **reward_keys}.items():
            if info_key in infos:
                vals = infos[info_key]
                if isinstance(vals, np.ndarray):
                    writer.add_scalar(tag, float(np.mean(vals)), global_step)

        # --- 处理经验存储（统一批次遍历） ---
        for i in range(train_cfg.num_envs):
            done = dones[i] or truncateds[i]

            # 累积奖励和步数（为策略2兜底提供原始数据）
            episode_rewards[i] += rewards[i]
            episode_lengths[i] += 1

            # 提取真实的下一状态（防止自动 reset 破坏 MDP 连贯性）
            if done and isinstance(infos.get("final_observation"), (list, np.ndarray)):
                real_next_state = infos["final_observation"][i]
            else:
                real_next_state = next_states[i]

            # 存入 ReplayBuffer
            replay_buffer.push(states[i], actions[i], rewards[i], real_next_state, done)

            # --- 延迟重置：在两个策略解析完成后才能清零 ---
            if done:
                _reset_envs.add(i)

        # --- 策略 1：infos["final_info"]（标准 Gymnasium 向量环境结构）---
        final_infos = infos.get("final_info")
        if final_infos is not None and any(
            final_infos[i] is not None and final_infos[i].get("episode") is not None
            for i in range(train_cfg.num_envs)
        ):
            batch_rewards = []
            for i in range(train_cfg.num_envs):
                ep_info = final_infos[i]
                if ep_info is not None:
                    ep = ep_info.get("episode", {})
                    episode_reward = float(ep.get("r", 0.0))
                    episode_length = int(ep.get("l", 0))
                    if episode_reward != 0.0 or episode_length != 0:
                        batch_rewards.append(episode_reward)
                        _log_and_checkpoint(episode_reward, global_step)
            if len(batch_rewards) > 0:
                mean_reward = float(np.mean(batch_rewards))
                writer.add_scalar("Global/Mean_Reward", mean_reward, global_step)
                writer.flush()

        # --- 策略 2：备用（未开启 episode 记录时，或 final_info 未覆盖的部分 env）---
        else:
            batch_rewards = []
            for i in range(train_cfg.num_envs):
                done = dones[i] or truncateds[i]
                if done:
                    episode_reward = float(episode_rewards[i])
                    episode_length = int(episode_lengths[i])
                    if episode_reward != 0.0 or episode_length != 0:
                        batch_rewards.append(episode_reward)
                        _log_and_checkpoint(episode_reward, global_step)
            if len(batch_rewards) > 0:
                mean_reward = float(np.mean(batch_rewards))
                writer.add_scalar("Global/Mean_Reward", mean_reward, global_step)
                writer.flush()

        # --- 统一延迟重置（在读完 episode_rewards/lengths 后才能清零）---
        for i in _reset_envs:
            episode_rewards[i] = 0.0
            episode_lengths[i] = 0
        _reset_envs.clear()

        # --- 更新当前状态 ---
        states = next_states

        # --- 网络更新 ---
        if global_step >= train_cfg.start_steps and global_step % train_cfg.update_every == 0:
            for _ in range(train_cfg.update_times):
                actor_loss, c1_loss, c2_loss, actor_q_loss, bc_reg_loss, bc_mse_loss, bc_reg_active = agent.update(replay_buffer, train_cfg.batch_size)
                if actor_loss is not None:
                    writer.add_scalar("Loss/Actor", actor_loss, global_step)
                    writer.add_scalar("Loss/Actor_Total", actor_loss, global_step)
                    writer.add_scalar("Loss/Actor_Q", actor_q_loss, global_step)
                    writer.add_scalar("Loss/Actor_BC_Reg", bc_reg_loss, global_step)
                    writer.add_scalar("Loss/Actor_BC_MSE", bc_mse_loss, global_step)
                    writer.add_scalar("PolicyHealth/actor_update_flag", 1, global_step)
                else:
                    writer.add_scalar("PolicyHealth/actor_update_flag", 0, global_step)
                writer.add_scalar("PolicyHealth/bc_reg_active", bc_reg_active, global_step)
                writer.add_scalar("Loss/Critic_1", c1_loss, global_step)
                writer.add_scalar("Loss/Critic_2", c2_loss, global_step)
                writer.flush()

    # ============================================================================
    # 训练结束
    # ============================================================================
    elapsed = _time_module.time() - train_start_time
    elapsed_hms = str(datetime.timedelta(seconds=int(elapsed)))
    actual_steps = global_step + train_cfg.num_envs
    final_path = os.path.join(model_dir, f"final_step_{actual_steps}.pth")
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_1_state_dict': agent.critic_1.state_dict(),
        'critic_2_state_dict': agent.critic_2.state_dict(),
        'target_actor_state_dict': agent.target_actor.state_dict(),
        'target_critic_1_state_dict': agent.target_critic_1.state_dict(),
        'target_critic_2_state_dict': agent.target_critic_2.state_dict(),
        'global_step': actual_steps,
    }, final_path)
    print(f"[Checkpoint] 最终模型已保存至: {final_path}")
    best_str = f"{best_reward:.4f}@{best_step}" if best_step is not None else "None"
    print(f"[SaveSummary] type=final | run={run_name} | step={actual_steps} | best_so_far={best_str} | elapsed={elapsed_hms} | path={final_path}")

    writer.close()
    envs.close()

    # --- TrainSummary ---
    best_str = f"{best_reward:.4f}@{best_step}" if best_step is not None else "None"
    print("=" * 60)
    print("[TrainSummary]"
          f" run={run_name}"
          f" | steps={actual_steps}"
          f" | elapsed={elapsed_hms}"
          f" | device={train_cfg.device}"
          f" | num_envs={train_cfg.num_envs}"
          f" | best_ep_reward={best_str}"
          f" | final={final_path}")
    print("=" * 60)