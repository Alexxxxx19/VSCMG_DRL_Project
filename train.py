"""
VSCMG 姿态控制强化学习训练脚本
TD3 算法 — 多环境异步并行训练主循环 (v0.5.0)

[快速启动指南]
========================================

默认模式 (CPU单核):
    python train.py

调试模式 (CPU多核):
    python train.py --num_envs 4

极致压榨模式 (RTX 5070 + Ultra 7 专属):
    python train.py --num_envs 16 --batch_size 2048 --update_every 200 --device cuda

========================================
"""

import os
# 底层防御：强制单线程模式，切断线程风暴防止 Windows 多进程死锁
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from envs.vscmg_env import VSCMGEnv
from agents.td3_agent import TD3, ReplayBuffer


def make_env():
    """
    顶层环境工厂函数（可序列化）
    用于 AsyncVectorEnv 实例化
    """
    return VSCMGEnv()


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="VSCMG TD3 并行训练脚本 v0.5.0")
    parser.add_argument("--num_envs", type=int, default=1, help="并行环境数量")
    parser.add_argument("--device", type=str, default="cpu", help="计算设备 (cpu/cuda)")
    parser.add_argument("--max_steps", type=int, default=2_000_000, help="最大训练步数")
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--update_after", type=int, default=5000, help="前N步纯随机探索")
    parser.add_argument("--update_every", type=int, default=50, help="每N步更新一次网络")
    return parser.parse_args()


if __name__ == "__main__":
    # ============================================================================
    # 智能提示：检测基础配置并建议优化方案
    # ============================================================================
    # 解析参数
    args = parse_args()

    # 检测是否使用默认基础配置
    if args.num_envs == 1 and args.device == "cpu":
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
    # 环境与 Agent 实例化
    # ============================================================================
    # 创建向量环境（终极统一：无论单环境还是多环境，统一走向量接口）
    env_fns = [make_env for _ in range(args.num_envs)]
    if args.num_envs == 1:
        print("[Tracer] 准备唤醒 1 个同步物理引擎...")
        envs = SyncVectorEnv(env_fns)
        print("[Tracer] 同步引擎唤醒成功！")
    else:
        print(f"[Tracer] 准备唤醒 {args.num_envs} 个并行物理引擎 (可能需等待10-30秒，请耐心)...")
        envs = AsyncVectorEnv(env_fns)
        print("[Tracer] 并行引擎唤醒成功！")

    # 创建 ReplayBuffer
    replay_buffer = ReplayBuffer(capacity=100000)

    # 创建 TD3 Agent（注入设备参数）
    agent = TD3(
        state_dim=14,
        action_dim=8,
        hidden_dim=256,
        action_bound=1.0,
        sigma=0.1,
        actor_lr=3e-4,
        critic_lr=3e-4,
        tau=0.005,
        gamma=0.99,
        device=args.device,
        delay=2
    )
    print("[Tracer] TD3 神经网络已建立并载入 GPU！")

    # TensorBoard 日志（动态时间戳破解幽灵缓存）
    ts = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_dir = f"runs/vscmg_parallel_{args.num_envs}envs_{ts}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"\n[Storage] TensorBoard 日志正实时写入绝对路径: {os.path.abspath(writer.log_dir)}\n")

    # ============================================================================
    # [自证清白] 心跳前置探针：验证数据管道本身是否连通
    # ============================================================================
    writer.add_scalar("System/Ignition", 1, 0)
    writer.flush()
    print("[Probe] System/Ignition=1 已写入 TensorBoard，数据管道连通性已验证。\n")

    # ============================================================================
    # 训练状态跟踪
    # ============================================================================
    best_reward = -1e6
    episode_rewards = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)

    def _log_and_checkpoint(env_idx: int, ep_reward: float, ep_len: int, g_step: int,
                            mean_reward: float = None):
        """战报打印 + TensorBoard 写入（独立曲线 + 均值基准线）+ 最佳模型保存"""
        global best_reward
        print(f"\n[战报] 飞船 Env-{env_idx} 完赛！得分: {ep_reward:12.4f}, "
              f"坚持了 {ep_len:4d} 步 (当前全局总吞吐: {g_step} 步)\n")
        # 独立曲线：每个 Env 一条专属 Tag
        writer.add_scalar(f"Reward/Env_{env_idx}", ep_reward, g_step)
        writer.add_scalar("Environment/Steps_Survived", ep_len, g_step)
        # 均值基准线（仅当本批次存在多个完赛环境时写入）
        if mean_reward is not None:
            writer.add_scalar("Reward/Mean_Reward", mean_reward, g_step)
        writer.flush()
        if ep_reward > best_reward:
            best_reward = ep_reward
            os.makedirs("checkpoints", exist_ok=True)
            agent.save_model("checkpoints/best_model_parallel.pth")
            print(f"  >> New best reward: {best_reward:.4f} — checkpoint saved!")

    # ============================================================================
    # 训练主循环（全局步数驱动）
    # ============================================================================
    print("=" * 60)
    print("VSCMG TD3 异步并行训练已启动 (v0.5.0)")
    print("=" * 60)
    print(f"并行环境数: {args.num_envs}")
    print(f"计算设备: {args.device}")
    print(f"最大训练步数: {args.max_steps}")
    print("=" * 60)

    # 初始化环境
    states, infos = envs.reset()
    print("[Tracer] 环境初态重置完毕，正在冲击主循环！")

    # 全局步数驱动主循环
    _reset_envs: set = set()  # 延迟重置集合（完赛→先读后清）
    for global_step in range(0, args.max_steps, args.num_envs):
        # --- 心跳监测（每 1000 步输出一次进度） ---
        if global_step > 0 and global_step % 1000 == 0:
            print(f"[Heartbeat] 物理引擎正在全速运转，当前已吞吐 {global_step} 步数据...")

        # --- 定期保存检查点（每 10 万步） ---
        if global_step > 0 and global_step % 100000 == 0:
            os.makedirs("models", exist_ok=True)
            checkpoint_path = f"models/checkpoint_step_{global_step}.pth"
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_1_state_dict': agent.critic_1.state_dict(),
                'critic_2_state_dict': agent.critic_2.state_dict(),
                'global_step': global_step,
            }, checkpoint_path)
            print(f"[Checkpoint] 模型已保存至: {checkpoint_path}")

        # --- 批量动作选择 ---
        if global_step < args.update_after:
            # 纯随机探索阶段
            actions = envs.action_space.sample()
        else:
            # 批量策略推理
            states_tensor = torch.FloatTensor(states).to(args.device)
            with torch.no_grad():
                actions = agent.actor(states_tensor).cpu().numpy()

            # 添加探索噪声
            noise = agent.sigma * np.random.randn(args.num_envs, agent.action_dim)
            actions = actions + noise
            actions = np.clip(actions, -agent.action_bound, agent.action_bound)

        # --- 异步并行环境交互 ---
        next_states, rewards, dones, truncateds, infos = envs.step(actions)

        # --- 处理经验存储（统一批次遍历） ---
        for i in range(args.num_envs):
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
            #    （先记下来，解析完再统一清）
            if done:
                _reset_envs.add(i)

        # ============================================================================
        # [Bug Hunt] 计分板探针：只要有任意子环境完赛，立即打印 infos 全貌
        # ============================================================================
        any_done = bool(np.any(dones) or np.any(truncateds))
        if any_done:
            print(f"\n[探针] global_step={global_step} | 检测到完赛信号！")
            print(f"       dones     = {dones}")
            print(f"       truncateds= {truncateds}")
            print(f"       infos.keys() = {list(infos.keys())}")
            for _k, _v in infos.items():
                print(f"       infos['{_k}'] -> type={type(_v).__name__} | value={repr(_v)[:300]}")
            print()

        # --- 策略 1：infos["final_info"]（标准 Gymnasium 向量环境结构）---
        final_infos = infos.get("final_info")
        if final_infos is not None:
            batch_rewards = []  # 收集本批次所有完赛得分
            for i in range(args.num_envs):
                ep_info = final_infos[i]
                if ep_info is not None:
                    ep = ep_info.get("episode", {})
                    episode_reward = float(ep.get("r", 0.0))
                    episode_length = int(ep.get("l", 0))
                    if episode_reward != 0.0 or episode_length != 0:
                        batch_rewards.append(episode_reward)
                        _log_and_checkpoint(i, episode_reward, episode_length, global_step,
                                            mean_reward=None)  # 均值在批次结束后统一写
            # 批次完赛写入均值基准线
            if len(batch_rewards) > 0:
                mean_reward = float(np.mean(batch_rewards))
                writer.add_scalar("Reward/Mean_Reward", mean_reward, global_step)
                writer.flush()

        # --- 策略 2：infos["final_info"] 备用（未开启 episode 记录时）---
        #    当且仅当策略1完全失败时兜底：用本地累积计数器
        if final_infos is None or all(
            final_infos[i] is None or final_infos[i].get("episode") is None
            for i in range(args.num_envs)
        ):
            batch_rewards = []  # 收集本批次所有完赛得分
            for i in range(args.num_envs):
                done = dones[i] or truncateds[i]
                if done:
                    # 本地累积奖励（从 episode_rewards 数组取）
                    episode_reward = float(episode_rewards[i])
                    episode_length = int(episode_lengths[i])
                    if episode_reward != 0.0 or episode_length != 0:
                        batch_rewards.append(episode_reward)
                        _log_and_checkpoint(i, episode_reward, episode_length, global_step,
                                            mean_reward=None)
            # 批次完赛写入均值基准线
            if len(batch_rewards) > 0:
                mean_reward = float(np.mean(batch_rewards))
                writer.add_scalar("Reward/Mean_Reward", mean_reward, global_step)
                writer.flush()

        # --- 统一延迟重置（在读完 episode_rewards/lengths 后才能清零）---
        for i in _reset_envs:
            episode_rewards[i] = 0.0
            episode_lengths[i] = 0
        _reset_envs.clear()

        # --- 更新当前状态 ---
        states = next_states

        # --- 网络更新 + Loss 抽水机 ---
        if global_step >= args.update_after and global_step % args.update_every == 0:
            for _ in range(args.update_every):
                actor_loss, c1_loss, c2_loss = agent.update(replay_buffer, args.batch_size)
                # 强制写入 TensorBoard，防止缓存滞留
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
    print("训练完成 (v0.5.0 异步并行引擎)")
    print(f"最佳奖励: {best_reward:.4f}")
    print("=" * 60)