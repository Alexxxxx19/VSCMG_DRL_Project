"""
VSCMG 姿态控制强化学习训练脚本
TD3 算法 — 多环境异步并行训练主循环 (v0.5.8)

参数集中化第二阶段：训练侧 + Agent 侧
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
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from envs.vscmg_env import VSCMGEnv
from agents.td3_agent import TD3, ReplayBuffer
from configs.train_config import TrainConfig, make_default_train_config
from configs.agent_config import AgentConfig, make_default_agent_config


def make_env():
    """
    顶层环境工厂函数（可序列化）
    用于 AsyncVectorEnv 实例化
    """
    return VSCMGEnv()


def parse_args():
    """
    解析命令行参数（CLI 优先级高于 train_config 默认值）

    所有这些参数在 train_config.py 中都有对应的默认值，
    此处 CLI 值会覆盖默认值。
    """
    parser = argparse.ArgumentParser(description="VSCMG TD3 并行训练脚本 v0.5.8")

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

    # --- 随机种子 ---
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（覆盖 train_config 默认值）；0 或负值表示不固定随机种子")

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
    if args.seed is not None:
        cfg.seed = args.seed
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
    print(f"    seed            = {train_cfg.seed}")
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
    print("=" * 60)


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
    # 设备同步到 agent_cfg
    agent_cfg.device = train_cfg.device

    # 设置全局随机种子（numpy + torch + random + cuda）
    set_global_seed(train_cfg.seed)

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
    )
    device_torch = torch.device(agent_cfg.device)
    print(f"[Tracer] TD3 神经网络已建立并载入 {'GPU' if device_torch.type == 'cuda' else 'CPU'}！")

    # ============================================================================
    # 第五步：打印配置快照
    # ============================================================================
    print_config_snapshot(train_cfg, agent_cfg, state_dim, action_dim)

    # ============================================================================
    # 第六步：TensorBoard 日志
    # ============================================================================
    ts = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_dir = f"{train_cfg.log_dir_base}/vscmg_parallel_{train_cfg.num_envs}envs_{ts}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=train_cfg.tb_flush_secs)
    print(f"\n[Storage] TensorBoard 日志正实时写入绝对路径: {os.path.abspath(writer.log_dir)}\n")

    # ============================================================================
    # 训练状态跟踪
    # ============================================================================
    best_reward = -1e6
    episode_rewards = np.zeros(train_cfg.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(train_cfg.num_envs, dtype=np.int32)
    first_episode_done = False  # 心跳静默开关

    def _log_and_checkpoint(ep_reward: float):
        """全局最佳模型保存（基于单 episode 得分触发，不写 TensorBoard）"""
        global best_reward, first_episode_done
        first_episode_done = True
        if ep_reward > best_reward:
            best_reward = ep_reward
            os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
            best_path = os.path.join(train_cfg.checkpoint_dir, "best_model_parallel.pth")
            agent.save_model(best_path)
            print(f"  >> [Checkpoint] New best reward: {best_reward:.4f} -> Model saved.")

    # ============================================================================
    # 训练主循环（全局步数驱动）
    # ============================================================================
    print("=" * 60)
    print("VSCMG TD3 异步并行训练已启动 (v0.5.8 参数集中化)")
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

        # --- 定期保存检查点（每 checkpoint_frequency 步） ---
        if global_step > 0 and global_step % train_cfg.checkpoint_frequency == 0:
            os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                train_cfg.checkpoint_dir,
                f"checkpoint_step_{global_step}.pth"
            )
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_1_state_dict': agent.critic_1.state_dict(),
                'critic_2_state_dict': agent.critic_2.state_dict(),
                'global_step': global_step,
            }, checkpoint_path)
            print(f"[Checkpoint] 模型已保存至: {checkpoint_path}")

        # --- 批量动作选择 ---
        if global_step < train_cfg.start_steps:
            # 纯随机探索阶段
            actions = envs.action_space.sample()
        else:
            # 批量策略推理
            states_tensor = torch.FloatTensor(states).to(device_torch)
            with torch.no_grad():
                actions = agent.actor(states_tensor).cpu().numpy()

            # 添加探索噪声
            noise = agent.sigma * np.random.randn(train_cfg.num_envs, agent.action_dim)
            actions = actions + noise
            actions = np.clip(actions, -agent.action_bound, agent.action_bound)

        # --- 异步并行环境交互 ---
        next_states, rewards, dones, truncateds, infos = envs.step(actions)

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
        if final_infos is not None:
            batch_rewards = []
            for i in range(train_cfg.num_envs):
                ep_info = final_infos[i]
                if ep_info is not None:
                    ep = ep_info.get("episode", {})
                    episode_reward = float(ep.get("r", 0.0))
                    episode_length = int(ep.get("l", 0))
                    if episode_reward != 0.0 or episode_length != 0:
                        batch_rewards.append(episode_reward)
                        _log_and_checkpoint(episode_reward)
            if len(batch_rewards) > 0:
                mean_reward = float(np.mean(batch_rewards))
                writer.add_scalar("Global/Mean_Reward", mean_reward, global_step)
                writer.flush()

        # --- 策略 2：备用（未开启 episode 记录时）---
        if final_infos is None or all(
            final_infos[i] is None or final_infos[i].get("episode") is None
            for i in range(train_cfg.num_envs)
        ):
            batch_rewards = []
            for i in range(train_cfg.num_envs):
                done = dones[i] or truncateds[i]
                if done:
                    episode_reward = float(episode_rewards[i])
                    episode_length = int(episode_lengths[i])
                    if episode_reward != 0.0 or episode_length != 0:
                        batch_rewards.append(episode_reward)
                        _log_and_checkpoint(episode_reward)
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
                actor_loss, c1_loss, c2_loss = agent.update(replay_buffer, train_cfg.batch_size)
                writer.add_scalar("Loss/Actor", actor_loss, global_step)
                writer.add_scalar("Loss/Critic_1", c1_loss, global_step)
                writer.add_scalar("Loss/Critic_2", c2_loss, global_step)
                writer.flush()

    # ============================================================================
    # 训练结束
    # ============================================================================
    writer.close()
    envs.close()
    print("=" * 60)
    print("训练完成 (v0.5.8 参数集中化版)")
    print(f"最佳奖励: {best_reward:.4f}")
    print("=" * 60)