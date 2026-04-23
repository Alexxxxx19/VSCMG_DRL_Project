"""
VSCMG 姿态控制强化学习训练脚本
TD3 算法 — ��环境异步并行训练主循环 (v0.5.12)

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

from envs.vscmg_env import VSCMGEnv, RewardConfig
from agents.td3_agent import TD3, ReplayBuffer
from configs.train_config import TrainConfig, make_default_train_config
from configs.agent_config import AgentConfig, make_default_agent_config
import json


def generate_run_name(train_cfg: TrainConfig, agent_cfg: AgentConfig, reward_cfg: RewardConfig) -> str:
    """
    生成唯一实验标识符

    包含：版本号 + 时间戳 + 并行数 + seed + reward 权重摘要
    """
    version = "v0.5.11"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    reward_summary = f"att{reward_cfg.w_att}_om{reward_cfg.w_omega}_wb{reward_cfg.w_wheel_bias}_ga{reward_cfg.w_gimbal_act}_wa{reward_cfg.w_wheel_act}"
    return f"{version}_{timestamp}_envs{train_cfg.num_envs}_seed{train_cfg.seed}_{reward_summary}"


def save_run_config(model_dir: str, run_name: str, train_cfg: TrainConfig,
                    agent_cfg: AgentConfig, reward_cfg: RewardConfig):
    """保存实验配置到 JSON"""
    config = {
        "run_name": run_name,
        "version": "v0.5.11",
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
        },
        "reward_config": {
            "w_att": reward_cfg.w_att,
            "w_omega": reward_cfg.w_omega,
            "w_wheel_bias": reward_cfg.w_wheel_bias,
            "w_gimbal_act": reward_cfg.w_gimbal_act,
            "w_wheel_act": reward_cfg.w_wheel_act,
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
    return VSCMGEnv()


def parse_args():
    """
    解析命令行参数（CLI 优先级高于 train_config 默认值）

    所有这些参数在 train_config.py 中都有对应的默认值，
    此处 CLI 值会覆盖默认值。
    """
    parser = argparse.ArgumentParser(description="VSCMG TD3 并行训练脚本 v0.5.12")

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
    # 第六步：生成 run_name 和模型目录
    # ============================================================================
    reward_cfg = RewardConfig()  # 读取当前默认 reward 配置
    run_name = generate_run_name(train_cfg, agent_cfg, reward_cfg)
    model_dir = os.path.join(train_cfg.checkpoint_dir, run_name)
    os.makedirs(model_dir, exist_ok=True)

    # 保存实验配置
    save_run_config(model_dir, run_name, train_cfg, agent_cfg, reward_cfg)

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
        global best_reward, best_step, first_episode_done
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
        _episode_summary(ep_reward, ep_step, saved, path)

    # ============================================================================
    # 训练主循环（全局步数驱动）
    # ============================================================================
    print("=" * 60)
    print("VSCMG TD3 异步并行训练已启动 (v0.5.12)")
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
                actor_loss, c1_loss, c2_loss = agent.update(replay_buffer, train_cfg.batch_size)
                writer.add_scalar("Loss/Actor", actor_loss, global_step)
                writer.add_scalar("Loss/Critic_1", c1_loss, global_step)
                writer.add_scalar("Loss/Critic_2", c2_loss, global_step)
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
    elapsed = _time_module.time() - train_start_time
    elapsed_hms = str(datetime.timedelta(seconds=int(elapsed)))
    actual_steps = global_step + train_cfg.num_envs
    final_path = os.path.join(model_dir, f"final_step_{actual_steps}.pth")
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_1_state_dict': agent.critic_1.state_dict(),
        'critic_2_state_dict': agent.critic_2.state_dict(),
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