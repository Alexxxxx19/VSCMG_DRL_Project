"""
VSCMG 航天器姿态控制强化学习环境
基于 Gymnasium API 实现
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.dynamics import SpacecraftDynamics
from geometry.pyramid_config import PyramidVSCMG


class VSCMGEnv(gym.Env):
    """
    VSCMG 航天器姿态控制环境

    状态空间 (14维):
    - MRPs 姿态 (3维)
    - 本体角速度 (3维)
    - 框架角 (4维)
    - 飞轮角动量 (4维)

    动作空间 (8维):
    - 框架角速度指令 (4维)
    - 飞轮角加速度指令 (4维)
    """

    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()

        # 时间步长
        self.dt = 0.01  # 秒

        # 实例化动力学底座
        self.dynamics = SpacecraftDynamics(j_sc=np.diag([100.0, 100.0, 100.0]))
        self.vscmg = PyramidVSCMG()

        # 动作空间: 8维连续 [-1, 1]
        # 前4维: 框架角速度指令, 后4维: 飞轮角加速度指令
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32
        )

        # 状态空间: 14维连续
        # MRPs(3) + omega(3) + delta(4) + h_w(4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
            dtype=np.float32
        )

        # 物理边界
        self.max_gimbal_rate = 1.0  # rad/s
        self.max_wheel_accel = 0.5  # Nms/s²
        self.max_omega = 20.0  # rad/s (终止条件)

        # 状态变量
        self.mrps = None  # MRPs 姿态 [3]
        self.omega = None  # 本体角速度 [3]
        self.delta = None  # 框架角 [4]
        self.h_w = None  # 飞轮角动量 [4]

    def reset(self, seed=None, options=None):
        """
        重置环境
        """
        super().reset(seed=seed)

        # 随机初始化 MRPs 姿态 [-0.1, 0.1]
        self.mrps = np.random.uniform(-0.1, 0.1, size=3)

        # 随机初始化本体角速度 [-0.01, 0.01]
        self.omega = np.random.uniform(-0.01, 0.01, size=3)

        # 框架角初始化为 0
        self.delta = np.zeros(4)

        # 飞轮角动量初始化为 0
        self.h_w = np.zeros(4)

        # 组装观测
        obs = self._get_obs().astype(np.float32)

        info = {}

        return obs, info

    def step(self, action):
        """
        环境步进
        """
        # 解析并反归一化动作
        gimbal_rate_cmd = action[:4] * self.max_gimbal_rate  # 框架角速度
        wheel_accel_cmd = action[4:] * self.max_wheel_accel  # 飞轮角加速度

        # 物理计算
        delta_vec = self.delta.reshape(4, 1)
        h_w_vec = self.h_w.reshape(4, 1)
        delta_dot_vec = gimbal_rate_cmd.reshape(4, 1)
        h_w_dot_vec = wheel_accel_cmd.reshape(4, 1)

        # 计算 VSCMG 输出力矩
        tau_vscmg = self.vscmg.calculate_output_torque(
            delta_vec, h_w_vec, delta_dot_vec, h_w_dot_vec
        )

        # 计算角加速度
        omega_vec = self.omega.reshape(3, 1)
        # 计算 VSCMG 总角动量: h_vscmg = A_s @ h_w
        a_s = self.vscmg.get_spin_matrix(delta_vec)
        h_vscmg_vec = a_s @ h_w_vec
        tau_ext = np.zeros((3, 1))  # 无外力矩

        omega_dot = self.dynamics.compute_angular_acceleration(
            omega_vec, h_vscmg_vec, tau_vscmg, tau_ext
        )

        # 数值积分（一阶欧拉法）
        # 更新角速度
        self.omega += omega_dot.flatten() * self.dt

        # 更新框架角
        self.delta += gimbal_rate_cmd * self.dt

        # 更新飞轮角动量
        self.h_w += wheel_accel_cmd * self.dt

        # 更新 MRPs 姿态
        # MRPs 运动学微分方程: σ̇ = 1/4 * [(1 - σᵀσ)I + 2[σ]× + 2σσᵀ] * ω
        sigma = self.mrps
        sigma_norm_sq = np.sum(sigma**2)

        # 构造 MRPs 运动学矩阵
        # G(sigma) = 1/4 * [(1 - σᵀσ)I + 2[σ]× + 2σσᵀ]
        I = np.eye(3)
        sigma_cross = np.array([
            [0, -sigma[2], sigma[1]],
            [sigma[2], 0, -sigma[0]],
            [-sigma[1], sigma[0], 0]
        ])
        sigma_outer = np.outer(sigma, sigma)

        G = 0.25 * ((1 - sigma_norm_sq) * I + 2 * sigma_cross + 2 * sigma_outer)

        # 计算 MRPs 导数
        sigma_dot = G @ self.omega

        # 更新 MRPs
        self.mrps += sigma_dot * self.dt

        # 影子集映射 (Shadow Set)
        # 如果 ||σ||² > 1, 映射到 σ = -σ / ||σ||²
        mrps_norm_sq = np.sum(self.mrps**2)
        if mrps_norm_sq > 1.0:
            self.mrps = -self.mrps / mrps_norm_sq

        # 计算奖励
        reward = -(np.sum(self.mrps**2) + 0.1 * np.sum(self.omega**2) + 0.01 * np.sum(action**2))

        # 终止条件
        terminated = np.linalg.norm(self.omega) > self.max_omega
        truncated = False

        # 组装观测
        obs = self._get_obs().astype(np.float32)

        info = {}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """
        组装观测向量
        """
        return np.concatenate([
            self.mrps,    # MRPs (3)
            self.omega,   # 角速度 (3)
            self.delta,   # 框架角 (4)
            self.h_w      # 飞轮角动量 (4)
        ])
