"""
VSCMG 航天器姿态控制强化学习环境
基于 Gymnasium API 实现
v1.0 前置：quaternion 内部积分 + MRP 误差观测 + 22 维观测接口
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.dynamics import SpacecraftDynamics
from geometry.pyramid_config import PyramidVSCMG


# =============================================================================
# 全项目统一姿态约定（scalar-first: [w, x, y, z]）
# =============================================================================

def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """
    归一化四元数（单位化）
    输入/输出：[w, x, y, z]（标量部在前）
    """
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    四元数乘法（Hamilton 积）
    q1 ⊗ q2
    输入/输出：[w, x, y, z]（scalar-first）
    """
    q1 = np.asarray(q1, dtype=np.float64).flatten()
    q2 = np.asarray(q2, dtype=np.float64).flatten()

    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # w
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # x
        w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,  # y
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,  # z
    ])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """
    四元数共轭 q* = [w, -x, -y, -z]
    输入/输出：[w, x, y, z]（scalar-first）
    """
    q = np.asarray(q, dtype=np.float64).flatten()
    return np.array([q[0], -q[1], -q[2], -q[3]])


def compute_orientation_error_quaternion(q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """
    计算误差四元数 q_err = q_target ⊗ q_current*

    返回 4 维四元数 [w, x, y, z]（scalar-first）
    物理意义：
    - q_err 标量部 w = cos(theta/2)
    - q_err 向量部 [x,y,z] = sin(theta/2) * axis

    注意：调用方负责做双覆盖规避（标量部 < 0 时整体取反）
    """
    q_current_star = quaternion_conjugate(q_current)
    q_err = quaternion_multiply(q_target, q_current_star)
    return quaternion_normalize(q_err)


def apply_scalar_sign_flip(q: np.ndarray) -> np.ndarray:
    """
    双覆盖规避（Double-Cover Avoidance）

    规则：若 q[0]（标量部）< 0，则整体取反
    这是全项目统一约定，用于确保误差四元数在短旋转方向上连续
    输入/��出：[w, x, y, z]（scalar-first）
    """
    q = np.asarray(q, dtype=np.float64).flatten()
    if q[0] < 0.0:
        return -q
    return q


def quaternion_to_mrp(q: np.ndarray) -> np.ndarray:
    """
    四元数 → MRP (Modified Rodrigues Parameters)

    输入：[w, x, y, z]（scalar-first）
    输出：sigma [3]（MRP 向量）

    若 ||q||² < 1，使用长形 MRP：sigma = q_v / (1 + q_w)
    若 ||q||² > 1，切换到 shadow set：sigma = -q_v / (1 - q_w)
    """
    q = np.asarray(q, dtype=np.float64).flatten()
    qw = q[0]  # 标量部
    qv = q[1:4]  # 向量部 [x, y, z]

    qv_norm_sq = np.dot(qv, qv)
    qw_sq = qw * qw

    if qv_norm_sq + qw_sq < 1e-12:
        return np.zeros(3)

    if qv_norm_sq <= qw_sq:
        denom = 1.0 + qw
        if denom < 1e-12:
            return np.zeros(3)
        sigma = qv / denom
    else:
        denom = 1.0 - qw
        if denom < 1e-12:
            return np.zeros(3)
        sigma = -qv / denom

    return sigma


def mrp_shadow(sigma: np.ndarray) -> np.ndarray:
    """
    MRP Shadow Set 映射
    当 ||sigma||² > 1 时，映射到 sigma = -sigma / ||sigma||²
    """
    sigma = np.asarray(sigma, dtype=np.float64).flatten()
    norm_sq = np.dot(sigma, sigma)
    if norm_sq > 1.0:
        return -sigma / norm_sq
    return sigma


def orientation_error_quaternion_to_sigma_err(q_err: np.ndarray) -> np.ndarray:
    """
    误差四元数 q_err → 误差 MRP sigma_err

    流程：
    1. 双覆盖规避（标量部 < 0 则取反）
    2. quaternion → MRP 转换
    3. shadow set 映射（若 ||sigma||² > 1）

    输入：4 维 q_err [w, x, y, z]（scalar-first）
    输出：3 维 sigma_err
    """
    q_err = apply_scalar_sign_flip(q_err)
    sigma_err = quaternion_to_mrp(q_err)
    sigma_err = mrp_shadow(sigma_err)
    return sigma_err


def quaternion_kinematics_dynamics(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    四元数运动学微分方程

    q̇ = 0.5 * Ω(ω) @ q

    输入：
        q：当前四元数 [w, x, y, z]（scalar-first）
        omega：本体角速度 [3]
    返回：
        q_dot：四元数导数 [w, x, y, z]
    """
    q = np.asarray(q, dtype=np.float64).flatten()
    omega = np.asarray(omega, dtype=np.float64).flatten()

    wx, wy, wz = omega[0], omega[1], omega[2]

    # Ω(ω) 矩阵（scalar-first 形式）
    omega_matrix = np.array([
        [ 0.0, -wx, -wy, -wz],
        [ wx,  0.0, -wz,  wy],
        [ wy,  wz,  0.0, -wx],
        [ wz, -wy,  wx,  0.0]
    ])

    q_dot = 0.5 * omega_matrix @ q
    return q_dot


# =============================================================================
# VSCMG 环境
# =============================================================================

class VSCMGEnv(gym.Env):
    """
    VSCMG 航天器姿态控制环境

    内部姿态表示：quaternion [w, x, y, z]（scalar-first）
    观测输出：MRP 误差 sigma_err [3] + 其他状态

    v1.0 前置要求：
    - 观测接口：22 维（sigma_err + omega_B + sin(delta) + cos(delta) + delta_dot + Omega_w_tilde）
    - 内部积分：quaternion
    - 飞轮状态：omega_w + I_w -> h_w（转速 + 转动惯量 -> 角动量）
    - reset：初始姿态误差 ≤ ±5°，初始角速度 = 0，飞轮 3000 rpm 偏置，外扰关闭
    - 动作缩放：框架指令 max 1 rad/s，飞轮指令 max 50 rad/s²
    """

    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()

        self.dt = 0.01

        self.dynamics = SpacecraftDynamics(j_sc=np.diag([100.0, 100.0, 100.0]))
        self.vscmg = PyramidVSCMG()

        # 动作空间: 8维连续 [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32
        )

        # v1.0 观测空间: 22维
        obs_bound = np.full(22, 1e30, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_bound,
            high=obs_bound,
            shape=(22,),
            dtype=np.float32
        )

        # v1.0 动作缩放
        self.max_gimbal_rate = 1.0   # rad/s  （框架指令）
        self.max_wheel_accel = 50.0  # rad/s² （飞轮指令）
        self.max_omega = 20.0        # rad/s

        self.max_episode_steps = 1000

        # --- 内部状态（quaternion 积分） ---
        self.q = None          # 当前姿态四元数 [w, x, y, z]（scalar-first）
        self.q_target = None   # 目标姿态四元数 [w, x, y, z]（固定为 [1, 0, 0, 0]）
        self.omega = None      # 本体角速度 [3]
        self.delta = None      # 框架角 [4]

        # --- 飞轮状态（omega_w + I_w -> h_w）---
        self.omega_w = None    # 飞轮角速度 [4]（rad/s）
        self.I_w = None        # 飞轮转动惯量 [4]（kg·m²）
        self.h_w = None        # 飞轮角动量 [4]（Nms）= I_w * omega_w

        # v1.0 标称转速：3000 rpm → rad/s
        self.omega_w_nominal = 3000.0 * (2.0 * np.pi / 60.0)  # ≈ 314.159 rad/s

        self.current_step = None

        # 缓存上一步框架角速度指令，用于观测中的 delta_dot
        self._delta_dot_cache = np.zeros(4)

    def reset(self, seed=None, options=None):
        """
        v1.0 前置 reset 条件：
        - 初始姿态误差：±5° 范围内随机
        - 初始本体角速度：强制为 0
        - 飞轮初始转速：锁定在 3000 rpm 标称偏置（直接设置 omega_w）
        - 外扰力矩：关闭（零向量）
        """
        super().reset(seed=seed)

        # 目标姿态固定为单位四元数 [w, x, y, z] = [1, 0, 0, 0]
        self.q_target = np.array([1.0, 0.0, 0.0, 0.0])

        # --- 姿态误差在 ±5° 范围内随机初始化 ---
        # 旋转误差角：随机方向，大小在 [0, 5°]
        angle_deg = self.np_random.uniform(0.0, 5.0)
        angle_rad = np.radians(angle_deg)
        axis = self.np_random.standard_normal(3)
        axis = axis / (np.linalg.norm(axis) + 1e-12)

        # 误差四元数 [w, x, y, z]（scalar-first）
        q_err = np.array([
            np.cos(angle_rad / 2.0),               # w
            axis[0] * np.sin(angle_rad / 2.0),     # x
            axis[1] * np.sin(angle_rad / 2.0),     # y
            axis[2] * np.sin(angle_rad / 2.0),     # z
        ])
        q_err = quaternion_normalize(q_err)

        # 当前姿态 = 误差四元数（因为目标 = 单位四元数）
        self.q = quaternion_normalize(q_err)

        # 初始本体角速度强制为 0
        self.omega = np.zeros(3)

        # 框架角初始化为 0
        self.delta = np.zeros(4)

        # --- 飞轮状态初始化 ---
        # 转动惯量显式保存
        self.I_w = np.full(4, 0.1)  # 每个飞轮 0.1 kg·m²

        # 角速度直接设置到 3000 rpm 标称偏置
        self.omega_w = np.full(4, self.omega_w_nominal)

        # 角动量由 I_w 和 omega_w 计算得到
        self.h_w = self.I_w * self.omega_w

        self.current_step = 0
        self._delta_dot_cache = np.zeros(4)

        obs = self._get_obs().astype(np.float32)
        info = {}
        return obs, info

    def step(self, action):
        """
        动作缩放（v1.0 前置）：
        - 前 4 维：框架指令 → max 1 rad/s
        - 后 4 维：飞轮指令 → max 50 rad/s²
        """
        gimbal_rate_cmd = action[:4] * self.max_gimbal_rate
        wheel_accel_cmd = action[4:] * self.max_wheel_accel

        # 缓存 delta_dot 用于观测
        self._delta_dot_cache = gimbal_rate_cmd.copy()

        delta_vec = self.delta.reshape(4, 1)
        h_w_vec = self.h_w.reshape(4, 1)
        delta_dot_vec = gimbal_rate_cmd.reshape(4, 1)
        h_w_dot_vec = wheel_accel_cmd.reshape(4, 1)

        tau_vscmg = self.vscmg.calculate_output_torque(
            delta_vec, h_w_vec, delta_dot_vec, h_w_dot_vec
        )

        omega_vec = self.omega.reshape(3, 1)
        a_s = self.vscmg.get_spin_matrix(delta_vec)
        h_vscmg_vec = a_s @ h_w_vec
        tau_ext = np.zeros((3, 1))

        omega_dot = self.dynamics.compute_angular_acceleration(
            omega_vec, h_vscmg_vec, tau_vscmg, tau_ext
        )

        # --- 一阶欧拉积分 ---
        # 1. 积分角速度
        self.omega += omega_dot.flatten() * self.dt

        # 2. 积分四元数（归一化保单位性）
        q_dot = quaternion_kinematics_dynamics(self.q, self.omega)
        self.q += q_dot * self.dt
        self.q = quaternion_normalize(self.q)

        # 3. 积分框架角
        self.delta += gimbal_rate_cmd * self.dt

        # 4. 积分飞轮角速度（由角加速度）
        self.omega_w += wheel_accel_cmd * self.dt

        # 5. 更新飞轮角动量（由 I_w 和 omega_w）
        self.h_w = self.I_w * self.omega_w

        # --- 计算奖励（保持 v0.5 旧版，不做最终 reward 设计）---
        q_err = compute_orientation_error_quaternion(self.q, self.q_target)
        sigma_err = orientation_error_quaternion_to_sigma_err(q_err)
        reward = -(
            np.sum(sigma_err ** 2)
            + 0.1 * np.sum(self.omega ** 2)
            + 0.01 * np.sum(action ** 2)
        )

        self.current_step += 1

        terminated = np.linalg.norm(self.omega) > self.max_omega
        truncated = self.current_step >= self.max_episode_steps

        obs = self._get_obs().astype(np.float32)
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """
        组装 22 维观测向量

        顺序：
        [0:3]   sigma_err     - 姿态误差 MRP [3]
        [3:6]   omega_B       - 本体角速度 [3]
        [6:10]  sin(delta)    - 框架角正弦 [4]
        [10:14] cos(delta)    - 框架角余弦 [4]
        [14:18] delta_dot     - 框架角速度（已缩放后的指令值）[4]
        [18:22] Omega_w_tilde - 飞轮转速相对 3000 rpm 标称偏置的归一化偏差 [4]

        计算方式：
        - Omega_w_tilde = (omega_w - omega_w_nominal) / omega_w_nominal
          （直接基于 omega_w，不是基于 h_w 反推）
        """
        # 误差四元数 + 双覆盖 + quaternion→MRP + shadow set
        q_err = compute_orientation_error_quaternion(self.q, self.q_target)
        sigma_err = orientation_error_quaternion_to_sigma_err(q_err)

        # omega_B
        omega_B = self.omega.copy()

        # sin(delta), cos(delta)
        sin_delta = np.sin(self.delta)
        cos_delta = np.cos(self.delta)

        # delta_dot（从 step 中缓存）
        delta_dot = self._delta_dot_cache.copy()

        # Omega_w_tilde：直接基于 omega_w 计算
        omega_w_tilde = (self.omega_w - self.omega_w_nominal) / self.omega_w_nominal

        return np.concatenate([
            sigma_err,    # 3
            omega_B,      # 3
            sin_delta,    # 4
            cos_delta,    # 4
            delta_dot,    # 4
            omega_w_tilde # 4
        ])