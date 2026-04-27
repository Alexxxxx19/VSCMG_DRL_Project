"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) 算法实现
基于 PyTorch 从零编写
"""

import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple


class ReplayBuffer:
    """
    经验回放池
    """

    def __init__(self, capacity: int):
        """
        初始化回放池

        Args:
            capacity: 最大容量
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        存入经验

        Args:
            state: 当前状态
            action: 执行动作
            reward: 获得奖励
            next_state: 下一状态
            done: 是否终止
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样批次经验

        Args:
            batch_size: 批次大小

        Returns:
            states, actions, rewards, next_states, dones (Tensor)
        """
        batch = random.sample(self.buffer, batch_size)
        # noinspection PyArgumentList
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class PolicyNet(nn.Module):
    """
    策略网络 (Actor)
    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, action_bound: float):
        """
        初始化策略网络

        Args:
            state_dim: 状态维度
            hidden_dim: 隐藏层维度
            action_dim: 动作维度
            action_bound: 动作边界
        """
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        """
        前向传播

        Args:
            x: 状态输入

        Returns:
            动作输出 (已缩放到 [-action_bound, action_bound])
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.action_bound


class QValueNet(nn.Module):
    """
    价值网络 (Critic)
    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        """
        初始化价值网络

        Args:
            state_dim: 状态维度
            hidden_dim: 隐藏层维度
            action_dim: 动作维度
        """
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        """
        前向传播

        Args:
            state: 状态输入
            action: 动作输入

        Returns:
            Q 值
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TD3:
    """
    TD3 算法主体
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        action_bound: float,
        sigma: float = 0.1,
        tau: float = 0.005,
        gamma: float = 0.99,
        critic_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        delay: int = 2,
        policy_noise: float = 0.5,
        noise_clip: float = 0.5,
        device: str = "cpu",
        actor_freeze_steps: int = 0
    ):
        """
        初始化 TD3 算法

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            action_bound: 动作边界
            sigma: 探索噪声标准差
            tau: 软更新系数
            gamma: 折扣因子
            critic_lr: Critic 学习率
            actor_lr: Actor 学习率
            delay: 策略更新延迟频率
            policy_noise: 目标策略平滑噪声标准差
            noise_clip: 目标策略平滑噪声截断范围
            device: 计算设备
            actor_freeze_steps: 前 N 次 update 只更新 critic，不更新 actor（0=关闭）
        """
        self.actor_freeze_steps = actor_freeze_steps
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.sigma = sigma
        self.tau = tau
        self.gamma = gamma
        self.delay = delay
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.device = device
        self.total_count = 0

        # 初始化 Actor 网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # 初始化双 Critic 网络
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())

        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

    def take_action(self, state: np.ndarray) -> np.ndarray:
        """
        生成动作（带探索噪声）

        Args:
            state: 当前状态

        Returns:
            动作 (已添加噪声并截断)
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).item() if self.action_dim == 1 else self.actor(state)[0].cpu().detach().numpy()

        # 添加探索噪声
        noise = self.sigma * np.random.randn(self.action_dim)
        action = action + noise

        # 截断到动作边界
        action = np.clip(action, -self.action_bound, self.action_bound)

        return action

    def soft_update(self, net, target_net):
        """
        软更新目标网络

        Args:
            net: 源网络
            target_net: 目标网络
        """
        for param_target, param in zip(list(target_net.parameters()), list(net.parameters())):
            param_target.data.copy_(self.tau * param.data + (1 - self.tau) * param_target.data)

    def update(self, replay_buffer: ReplayBuffer, batch_size: int):
        """
        更新网络

        Args:
            replay_buffer: 经验回放池
            batch_size: 批次大小

        Returns:
            tuple: (actor_loss, critic_loss_1, critic_loss_2)
        """
        self.total_count += 1

        # 采样批次经验
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 计算目标 Q 值（使用目标策略平滑）
        with torch.no_grad():
            # 目标 Actor 生成的下一动作
            next_actions = self.target_actor(next_states)

            # 目标策略平滑：添加截断噪声
            noise = torch.randn_like(next_actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = next_actions + noise
            next_actions = torch.clamp(next_actions, -self.action_bound, self.action_bound)

            # 双 Critic 取最小值
            target_q1 = self.target_critic_1(next_states, next_actions)
            target_q2 = self.target_critic_2(next_states, next_actions)
            next_q_values = torch.min(target_q1, target_q2)

            # TD 目标值（带终止状态掩码）
            q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        # 更新 Critic 1
        q_values_1 = self.critic_1(states, actions)
        critic_loss_1 = F.mse_loss(q_values_1, q_targets)
        self.critic_1_optimizer.zero_grad()
        critic_loss_1.backward()
        self.critic_1_optimizer.step()

        # 更新 Critic 2
        q_values_2 = self.critic_2(states, actions)
        critic_loss_2 = F.mse_loss(q_values_2, q_targets)
        self.critic_2_optimizer.zero_grad()
        critic_loss_2.backward()
        self.critic_2_optimizer.step()

        # 延迟更新 Actor 和目标网络（受 actor_freeze_steps 保护）
        actor_should_update = (
            self.total_count > self.actor_freeze_steps
            and self.total_count % self.delay == 0
        )

        if actor_should_update:
            # 更新 Actor（最大化 Q 值）
            actor_actions = self.actor(states)
            q_value = self.critic_1(states, actor_actions)
            actor_loss = -torch.mean(q_value)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            self.soft_update(self.actor, self.target_actor)
            self.soft_update(self.critic_1, self.target_critic_1)
            self.soft_update(self.critic_2, self.target_critic_2)

            return actor_loss.item(), critic_loss_1.item(), critic_loss_2.item()
        return None, critic_loss_1.item(), critic_loss_2.item()

    def save_model(self, path: str):
        """
        保存模型

        Args:
            path: 保存路径
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'target_critic_1': self.target_critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'target_critic_2': self.target_critic_2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
        }, path)
        print(f"模型已保存到 {path}")

    def load_model(self, path: str):
        """
        加载模型

        Args:
            path: 加载路径
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.target_critic_1.load_state_dict(checkpoint['target_critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.target_critic_2.load_state_dict(checkpoint['target_critic_2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer'])
        self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer'])
        print(f"模型已从 {path} 加载")
