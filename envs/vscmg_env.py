"""
VSCMG 航天器姿态控制强化学习环境
基于 Gymnasium API 实现
v1.0 前置：quaternion 内部积分 + MRP 误差观测 + 22 维观测接口

v3.0 接口预留：
- 所有物理参数从 VSCMGEnvConfig 集中配置
- reset(options=...) 支持 episode 级随机化覆盖（options 不泄漏到后续 episode）
- dynamics 支持 episode 级 J 随机化（通过 update_inertia）
- 外扰/延迟接口已预留在 config 中（默认关闭）
"""

import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from envs.dynamics import SpacecraftDynamics
from geometry.pyramid_config import PyramidVSCMG
from configs.env_config import VSCMGEnvConfig, make_default_config


# =============================================================================
# 全项目统一姿态约定（scalar-first: [w, x, y, z]）
# =============================================================================

def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = np.asarray(q1, dtype=np.float64).flatten()
    q2 = np.asarray(q2, dtype=np.float64).flatten()
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
    ])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).flatten()
    return np.array([q[0], -q[1], -q[2], -q[3]])


def compute_orientation_error_quaternion(q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    q_current_star = quaternion_conjugate(q_current)
    q_err = quaternion_multiply(q_target, q_current_star)
    return quaternion_normalize(q_err)


def apply_scalar_sign_flip(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).flatten()
    if q[0] < 0.0:
        return -q
    return q


def quaternion_to_mrp(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).flatten()
    qw = q[0]
    qv = q[1:4]
    qv_norm_sq = np.dot(qv, qv)
    qw_sq = qw * qw
    if qv_norm_sq + qw_sq < 1e-12:
        return np.zeros(3)
    if qv_norm_sq <= qw_sq:
        denom = 1.0 + qw
        if denom < 1e-12:
            return np.zeros(3)
        return qv / denom
    else:
        denom = 1.0 - qw
        if denom < 1e-12:
            return np.zeros(3)
        return -qv / denom


def mrp_shadow(sigma: np.ndarray) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=np.float64).flatten()
    norm_sq = np.dot(sigma, sigma)
    if norm_sq > 1.0:
        return -sigma / norm_sq
    return sigma


def orientation_error_quaternion_to_sigma_err(q_err: np.ndarray) -> np.ndarray:
    q_err = apply_scalar_sign_flip(q_err)
    sigma_err = quaternion_to_mrp(q_err)
    return mrp_shadow(sigma_err)


def quaternion_kinematics_dynamics(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).flatten()
    omega = np.asarray(omega, dtype=np.float64).flatten()
    wx, wy, wz = omega[0], omega[1], omega[2]
    omega_matrix = np.array([
        [ 0.0, -wx, -wy, -wz],
        [ wx,  0.0, -wz,  wy],
        [ wy,  wz,  0.0, -wx],
        [ wz, -wy,  wx,  0.0]
    ])
    return 0.5 * omega_matrix @ q


# =============================================================================
# VSCMG 环境
# =============================================================================

class VSCMGEnv(gym.Env):
    """
    VSCMG 航天器姿态控制环境

    内部姿态表示：quaternion [w, x, y, z]（scalar-first）
    观测输出：22 维（sigma_err + omega_B + sin(delta) + cos(delta) + delta_dot + Omega_w_tilde）

    v1.0 前置要求：
    - 观测接口：22 维
    - 内部积分：quaternion
    - 飞轮状态：omega_w + I_w -> h_w
    - reset：初始姿态误差 ≤ ±5°，初始角速度 = 0，飞轮 3000 rpm 偏置，外扰关闭
    - 动作缩放：框架指令 max 1 rad/s，飞轮指令 max 50 rad/s²

    v3.0 接口预留：
    - 所有物理参数从 self.cfg（基线配置）读取
    - reset(options=...) 支持 episode 级随机化覆盖
    - self.episode_cfg 是每次 reset 从 self.cfg 克隆的 working copy，
      永不泄漏到后续 episode
    - dynamics 支持 episode 级 J 随机化（通过 update_inertia）
    - 外扰接口：episode_cfg.disturbance（默认关闭）
    - 延迟接口：episode_cfg.delay（默认关闭）
    """

    metadata = {'render_modes': []}

    def __init__(self, config: VSCMGEnvConfig = None):
        """
        初始化 VSCMG 环境

        Args:
            config: VSCMGEnvConfig 实例。
                    若为 None，使用 make_default_config()（所有随机化关闭 = v1.0 行为）。
                    传入后作为永久基线配置（self.cfg），不会被 reset 污染。
        """
        super().__init__()
        self.cfg = config if config is not None else make_default_config()

        # 动力学（使用 current_j_sc 初始化）
        self.dynamics = SpacecraftDynamics(j_sc=self.cfg.current_j_sc.copy())
        self.vscmg = PyramidVSCMG()

        # 动作空间（8维，固定不变）
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # 观测空间（22维，固定不变）
        obs_bound = np.full(22, 1e30, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_bound, high=obs_bound, shape=(22,), dtype=np.float32
        )

        # 内部状态（reset 时初始化）
        self.q = None           # 当前姿态四元数 [w, x, y, z]
        self.q_target = None    # 目标姿态四元数（固定 [1,0,0,0]）
        self.omega = None       # 本体角速度 [3]
        self.delta = None       # 框架角 [4]
        self.omega_w = None     # 飞轮角速度 [4]
        self.I_w = None         # 飞轮转动惯量 [4]
        self.h_w = None         # 飞轮角动量 [4]
        self.omega_w_nominal = None  # 飞轮标称转速（用于观测中 Omega_w_tilde）
        self.current_step = None

        # 每 episode 的 working config（reset 时构建）
        self.episode_cfg: Optional[VSCMGEnvConfig] = None

        # 缓存上一步框架角速度指令
        self._delta_dot_cache = np.zeros(4)

    # =======================================================================
    # 内部辅助方法
    # =======================================================================

    def _sync_dynamics(self):
        """
        把 episode_cfg.current_j_sc 同步到 self.dynamics（无条件执行）

        每次 reset 后调用，确保 dynamics 使用本 episode 实际的惯量矩阵。
        无论 J 是 randomization 采样还是 options 覆盖，都走此路径。
        """
        self.dynamics.update_inertia(self.episode_cfg.current_j_sc)

    def _build_episode_cfg(self, options):
        """
        从 self.cfg 克隆干净的 working copy，依次应用：
        1. randomization（基于 self.np_random）
        2. options 覆盖

        self.cfg 本身不被修改。
        """
        # 克隆：self.cfg 永远不被污染
        episode_cfg: VSCMGEnvConfig = copy.deepcopy(self.cfg)

        # randomization（使用 self.np_random；首次 reset 尚未 init 时跳过）
        if self.np_random is not None:
            episode_cfg.apply_randomization(self.np_random)

        # options 覆盖
        if options is not None:
            # 整体配置替换
            if "config" in options and options["config"] is not None:
                episode_cfg = copy.deepcopy(options["config"])

            # J_sc 覆盖 → 触发 dynamics 同步
            if "j_sc" in options and options["j_sc"] is not None:
                episode_cfg.current_j_sc = np.asarray(options["j_sc"], dtype=np.float64)

            # I_w 覆盖
            if "i_w" in options and options["i_w"] is not None:
                episode_cfg.current_i_w = np.asarray(options["i_w"], dtype=np.float64)

            # 飞轮偏置倍数覆盖
            if "omega_bias_factor" in options and options["omega_bias_factor"] is not None:
                factor = float(options["omega_bias_factor"])
                episode_cfg.current_omega_w_nominal = episode_cfg.nominal_omega_w[0] * factor
                episode_cfg.current_omega_w = np.full(4, episode_cfg.current_omega_w_nominal)

            # 初始姿态误差覆盖
            if "init_attitude_deg" in options and options["init_attitude_deg"] is not None:
                episode_cfg.current_init_attitude_deg = float(options["init_attitude_deg"])

            # 初始角速度覆盖
            if "init_omega" in options and options["init_omega"] is not None:
                episode_cfg.current_init_omega = np.asarray(options["init_omega"], dtype=np.float64)

            # 外扰开关覆盖
            if "disturbance_enabled" in options and options["disturbance_enabled"] is not None:
                episode_cfg.disturbance.enabled = bool(options["disturbance_enabled"])

            # 延迟开关覆盖
            if "delay_enabled" in options and options["delay_enabled"] is not None:
                episode_cfg.delay.enabled = bool(options["delay_enabled"])
            if "delay_tau" in options and options["delay_tau"] is not None:
                episode_cfg.delay.tau = float(options["delay_tau"])

        return episode_cfg

    # =======================================================================
    # Gymnasium 接口
    # =======================================================================

    def reset(self, seed=None, options=None):
        """
        重置环境，开始新 episode

        Args:
            seed: 随机种子
            options: 可选字典，支持以下 episode 级覆盖字段：
                - "config": VSCMGEnvConfig，替换整个基线配置
                - "j_sc": np.ndarray，覆盖 J_sc（触发 dynamics 同步）
                - "i_w": np.ndarray，覆盖飞轮转动惯量 [4]
                - "omega_bias_factor": float，飞轮偏置倍数（相对 3000 rpm）
                - "init_attitude_deg": float，初始姿态误差角（度）
                - "init_omega": np.ndarray，初始角速度 [3]
                - "disturbance_enabled": bool，开关外扰
                - "delay_enabled": bool，开关延迟
                - "delay_tau": float，延迟时间常数（秒）

        重要：options 仅作用于本 episode。下一个 episode 不传 options 时，
              回到 self.cfg 基线（v1.0 默认行为）。
        """
        super().reset(seed=seed)  # 初始化 self.np_random

        # 1. 构建本 episode working config（从基线克隆 → randomization → options 覆盖）
        self.episode_cfg = self._build_episode_cfg(options)

        # 2. dynamics J 同步（无条件）
        self._sync_dynamics()

        # --- 目标姿态（固定） ---
        self.q_target = np.array([1.0, 0.0, 0.0, 0.0])

        # --- 初始姿态误差 ---
        angle_deg = self.episode_cfg.current_init_attitude_deg
        angle_rad = np.radians(angle_deg)
        axis = self.np_random.standard_normal(3)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        q_err = np.array([
            np.cos(angle_rad / 2.0),
            axis[0] * np.sin(angle_rad / 2.0),
            axis[1] * np.sin(angle_rad / 2.0),
            axis[2] * np.sin(angle_rad / 2.0),
        ])
        self.q = quaternion_normalize(q_err)

        # --- 初始角速度 ---
        self.omega = self.episode_cfg.current_init_omega.copy()

        # --- 框架角初始值 ---
        self.delta = self.episode_cfg.current_init_gimbal.copy()

        # --- 飞轮状态 ---
        self.I_w = self.episode_cfg.current_i_w.copy()
        self.omega_w = self.episode_cfg.current_omega_w.copy()
        self.omega_w_nominal = self.episode_cfg.current_omega_w_nominal
        self.h_w = self.I_w * self.omega_w

        self.current_step = 0
        self._delta_dot_cache = np.zeros(4)

        obs = self._get_obs().astype(np.float32)
        return obs, {}

    def step(self, action):
        """
        执行一步仿真

        动作解析（8维，固定不变）：
        - 前 4 维：框架指令 → max_gimbal_rate rad/s
        - 后 4 维：飞轮指令 → max_wheel_accel rad/s²
        """
        # 动作缩放
        gimbal_rate_cmd = action[:4] * self.episode_cfg.max_gimbal_rate
        wheel_accel_cmd = action[4:] * self.episode_cfg.max_wheel_accel

        # 一阶滞后延迟（默认关闭）
        gimbal_rate_cmd = self.episode_cfg.apply_actuator_delay(
            gimbal_rate_cmd, self.episode_cfg.dt
        )
        wheel_accel_cmd = self.episode_cfg.apply_actuator_delay(
            wheel_accel_cmd, self.episode_cfg.dt
        )

        # 缓存 delta_dot
        self._delta_dot_cache = gimbal_rate_cmd.copy()

        # VSCMG 力矩
        delta_vec = self.delta.reshape(4, 1)
        h_w_vec = self.h_w.reshape(4, 1)
        delta_dot_vec = gimbal_rate_cmd.reshape(4, 1)
        h_w_dot_vec = wheel_accel_cmd.reshape(4, 1)

        tau_vscmg = self.vscmg.calculate_output_torque(
            delta_vec, h_w_vec, delta_dot_vec, h_w_dot_vec
        )

        # 外扰力矩（默认关闭）
        t = self.current_step * self.episode_cfg.dt
        tau_ext = self.episode_cfg.compute_disturbance_torque(t)

        # 动力学积分
        omega_vec = self.omega.reshape(3, 1)
        a_s = self.vscmg.get_spin_matrix(delta_vec)
        h_vscmg_vec = a_s @ h_w_vec

        omega_dot = self.dynamics.compute_angular_acceleration(
            omega_vec, h_vscmg_vec, tau_vscmg, tau_ext
        )

        # 一阶欧拉积分
        self.omega += omega_dot.flatten() * self.episode_cfg.dt

        q_dot = quaternion_kinematics_dynamics(self.q, self.omega)
        self.q += q_dot * self.episode_cfg.dt
        self.q = quaternion_normalize(self.q)

        self.delta += gimbal_rate_cmd * self.episode_cfg.dt
        self.omega_w += wheel_accel_cmd * self.episode_cfg.dt
        self.h_w = self.I_w * self.omega_w

        # 奖励（保持原样，不改 reward 结构）
        q_err = compute_orientation_error_quaternion(self.q, self.q_target)
        sigma_err = orientation_error_quaternion_to_sigma_err(q_err)
        reward = -(
            np.sum(sigma_err ** 2)
            + 0.1 * np.sum(self.omega ** 2)
            + 0.01 * np.sum(action ** 2)
        )

        self.current_step += 1

        terminated = np.linalg.norm(self.omega) > self.episode_cfg.max_omega
        truncated = self.current_step >= self.episode_cfg.max_episode_steps

        obs = self._get_obs().astype(np.float32)
        return obs, reward, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        """
        组装 22 维观测向量（v1.0 接口，固定不变）

        顺序：
        [0:3]   sigma_err     - 姿态误差 MRP [3]
        [3:6]   omega_B       - 本体角速度 [3]
        [6:10]  sin(delta)    - 框架角正弦 [4]
        [10:14] cos(delta)    - 框架角余弦 [4]
        [14:18] delta_dot     - 框架角速度 [4]
        [18:22] Omega_w_tilde - 飞轮转速相对标称偏置的归一化偏差 [4]
        """
        q_err = compute_orientation_error_quaternion(self.q, self.q_target)
        sigma_err = orientation_error_quaternion_to_sigma_err(q_err)
        omega_B = self.omega.copy()
        sin_delta = np.sin(self.delta)
        cos_delta = np.cos(self.delta)
        delta_dot = self._delta_dot_cache.copy()
        omega_w_tilde = (
            (self.omega_w - self.omega_w_nominal)
            / self.omega_w_nominal
        )
        return np.concatenate([
            sigma_err, omega_B, sin_delta, cos_delta, delta_dot, omega_w_tilde
        ])
