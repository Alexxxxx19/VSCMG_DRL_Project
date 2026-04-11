"""
VSCMG 姿态控制强化学习训练脚本
TD3 算法 — 工业级训练主循环
"""

import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from envs.vscmg_env import VSCMGEnv
from agents.td3_agent import TD3, ReplayBuffer

# ============================================================================
# 超参数配置
# ============================================================================
MAX_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 1000
BATCH_SIZE = 256
UPDATE_AFTER = 5000        # 前5000步纯随机探索，不更新网络
UPDATE_EVERY = 50         # 每50步更新一次网络
CAPACITY = 100000         # ReplayBuffer 容量

# ============================================================================
# 环境与算法实例化
# ============================================================================
env = VSCMGEnv()

replay_buffer = ReplayBuffer(capacity=CAPACITY)

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
    device="cpu",
    delay=2
)

# TensorBoard 日志
os.makedirs("runs/vscmg_experiment_1", exist_ok=True)
writer = SummaryWriter(log_dir="runs/vscmg_experiment_1")

# ============================================================================
# 训练主循环
# ============================================================================
best_reward = -1e6        # 初始基准，用于检查点保存
global_step = 0

print("=" * 60)
print("VSCMG TD3 Training Started")
print("=" * 60)

for episode in range(1, MAX_EPISODES + 1):
    state, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0

    for step in range(MAX_STEPS_PER_EPISODE):
        # --- 动作选择 ---
        if global_step < UPDATE_AFTER:
            # 纯随机探索阶段
            action = env.action_space.sample()
        else:
            action = agent.take_action(state)

        # --- 环境交互 ---
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # --- 存入 ReplayBuffer ---
        replay_buffer.push(state, action, reward, next_state, done)

        episode_reward += reward
        episode_length += 1
        global_step += 1
        state = next_state

        # --- 网络更新 ---
        if global_step >= UPDATE_AFTER and global_step % UPDATE_EVERY == 0:
            for _ in range(UPDATE_EVERY):
                agent.update(replay_buffer, BATCH_SIZE)

        if done:
            break

    # --- Episode 结束：日志输出与 TensorBoard 记录 ---
    print(f"Episode {episode:4d} | Reward: {episode_reward:12.4f} | Steps: {episode_length}")

    writer.add_scalar("Train/EpisodeReward", episode_reward, episode)
    writer.add_scalar("Train/EpisodeLength", episode_length, episode)

    # --- 模型检查点保存 ---
    if episode_reward > best_reward:
        best_reward = episode_reward
        os.makedirs("checkpoints", exist_ok=True)
        agent.save_model("checkpoints/best_model.pth")
        print(f"  >> New best reward: {best_reward:.4f} — checkpoint saved!")

writer.close()
print("=" * 60)
print("Training Completed.")
print("=" * 60)
