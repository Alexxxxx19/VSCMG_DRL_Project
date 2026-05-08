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
from dataclasses import dataclass
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
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * z2,
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
# v1.0 Reward 归一化参考尺度配置
# =============================================================================

# v1.0 姿态误差参考：5° 精确对应 MRP 范数
V1_ATTITUDE_REF_DEG: float = 5.0
V1_SIGMA_REF: float = float(np.tan(np.deg2rad(V1_ATTITUDE_REF_DEG) / 4.0))


@dataclass
class RewardNormalizationConfig:
    """
    Reward 归一化参考尺度配置

    设计原则：基于 v1.0 验收标准设定参考尺度。

    cost=1 的物理含义：
    - attitude_cost = 1  -> 姿态误差恰好等于 v1.0 初始上限 5°
    - omega_cost = 1     -> 角速度恰好等于 omega_ref（按各分量 RMS 意义）
    - wheel_bias_cost = 1 -> 飞轮归一化偏差平均等于 wheel_bias_ref
    - action_cost = 1    -> 动作平方和恰好等于 scale（满幅）
    """
    # 姿态误差参考：v1.0 初始误差上限 5° 对应的 MRP 范数
    # sigma_ref = tan(5°/4) ≈ 0.02182
    # 当 ||sigma_err|| = sigma_ref 时 attitude_cost = 1
    sigma_ref: float = V1_SIGMA_REF

    # 角速度参考：v1.0 诊断尺度
    # omega_ref = 0.1 rad/s，让 omega_cost 有诊断意义
    # 后续如 omega 项太强，通过 w_omega 调整
    omega_ref: float = 0.1

    # 飞轮偏置参考：v1.0 可接受偏置范围（±10%）
    # wheel_bias_cost = wheel_bias_sq / (wheel_bias_ref^2) / 4
    # 当每轮平均偏置 10% 时 cost = 1
    wheel_bias_ref: float = 0.10

    # 动作项：除以 4 抵消维度差异，scale=4 表示动作满幅(±1)时 cost=1
    gimbal_action_scale: float = 4.0
    wheel_action_scale: float = 4.0


# =============================================================================
# v1.0 Reward 权重配置
# =============================================================================

@dataclass
class RewardConfig:
    """
    v0.5.17 Stage A: 姿态单项训练

    仅保留姿态误差主项，关闭所有辅助 penalty：
    - w_omega = 0.00：角速度阻尼关闭（Stage A 专注姿态收敛）
    - w_gimbal_act = 0.00 / w_wheel_act = 0.00：动作正则关闭
    """
    w_att:        float = 1.00  # 姿态误差主项
    w_omega:      float = 0.00  # 角速度阻尼（Stage A 关闭）
    w_wheel_bias: float = 0.00  # 飞轮偏置（Stage A 关闭）
    w_gimbal_act: float = 0.00  # 框架动作正则（Stage A 关闭）
    w_wheel_act:  float = 0.00  # 飞轮动作正则（Stage A 关闭）
    reward_scale: float = 300.0  # reward 全局缩放因子
    w_att_progress: float = 0.00  # 姿态误差 progress 奖励（默认关闭，向后兼容）


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

    def __init__(self, config: VSCMGEnvConfig = None, reward_cfg: "RewardConfig" = None):
        """
        初始化 VSCMG 环境

        Args:
            config: VSCMGEnvConfig 实例。
                    若为 None，使用 make_default_config()（所有随机化关闭 = v1.0 行为）。
                    传入后作为永久基线配置（self.cfg），不会被 reset 污染。
            reward_cfg: RewardConfig 实例。若为 None，使用默认 RewardConfig()，
                        保持旧行为（w_att_progress=0.0）。
        """
        super().__init__()
        self.cfg = config if config is not None else make_default_config()

        # 动力学（使用 current_j_sc 初始化）
        self.dynamics = SpacecraftDynamics(j_sc=self.cfg.current_j_sc.copy())
        self.vscmg = PyramidVSCMG()

        # 动作空间（维度由 action_mode 决定）
        if self.cfg.action_mode == "gimbal_only":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

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

        # 缓存上一步 normalized attitude_cost（用于 progress reward）
        self._attitude_cost_prev = 0.0

        # ---- reward 归一化配置（v1.0 P1 方案）----
        self.reward_norm_cfg = RewardNormalizationConfig()
        self.reward_cfg = reward_cfg if reward_cfg is not None else RewardConfig()

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
                - "j_sc": np.ndarray，覆盖 J_sc（触发 dynamics 同��）
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

        # --- 目标姿态（固定）---
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

        # 初始化 progress reward 缓存：当前姿态误差的 normalized attitude_cost
        q_err_init = compute_orientation_error_quaternion(self.q, self.q_target)
        sigma_err_init = orientation_error_quaternion_to_sigma_err(q_err_init)
        sigma_err_sq_init = float(np.sum(sigma_err_init ** 2))
        self._attitude_cost_prev = sigma_err_sq_init / (self.reward_norm_cfg.sigma_ref ** 2)

        obs = self._get_obs().astype(np.float32)
        return obs, {}

    def step(self, action):
        """
        执行一步仿真

        动作解析：
        - full_8d（8维）: 前4维=框架指令 → max_gimbal_rate rad/s，后4维=飞轮指令 → max_wheel_accel rad/s²
        - gimbal_only（4维）: 仅前4维有效，飞轮指令强制为 0
        """
        # 动作维度安全检查
        expected_dim = 4 if self.cfg.action_mode == "gimbal_only" else 8
        if action.shape[0] != expected_dim:
            raise ValueError(
                f"action dim mismatch: got {action.shape[0]}, "
                f"expected {expected_dim} for action_mode='{self.cfg.action_mode}'"
            )
        # 动作解析
        gimbal_rate_cmd = action[:4] * self.episode_cfg.max_gimbal_rate
        if self.cfg.action_mode == "gimbal_only":
            wheel_accel_cmd = np.zeros(4, dtype=np.float64)
        else:
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

        # ---- v1.0 reward 计算（归一化版本）----
        q_err = compute_orientation_error_quaternion(self.q, self.q_target)
        sigma_err = orientation_error_quaternion_to_sigma_err(q_err)

        # 原始物理量（保留用于诊断）
        sigma_err_sq = np.sum(sigma_err ** 2)
        omega_sq = np.sum(self.omega ** 2)
        omega_w_tilde = (self.omega_w - self.omega_w_nominal) / self.omega_w_nominal
        wheel_bias_sq = np.sum(omega_w_tilde ** 2)

        # 动作项：网络输出 action [-1, 1]
        gimbal_action_sq = np.sum(action[:4] ** 2)
        wheel_action_sq = np.sum(action[4:] ** 2)
        action_sq = gimbal_action_sq + wheel_action_sq

        # ---- 归一化 cost ----
        # attitude_cost: sigma_err 范数平方 / sigma_ref^2
        # sigma_ref = tan(5°/4)，当 ||sigma_err|| = sigma_ref（即 5° 误差）时 cost = 1
        attitude_cost = sigma_err_sq / (self.reward_norm_cfg.sigma_ref ** 2)
        # omega_cost: omega 范数平方 / (omega_ref^2 * 3)
        omega_cost = omega_sq / (self.reward_norm_cfg.omega_ref ** 2) / 3.0
        # wheel_bias_cost: 飞轮归一化偏���相对于参考偏置的平方误差
        wheel_bias_cost = wheel_bias_sq / (self.reward_norm_cfg.wheel_bias_ref ** 2) / 4.0
        # gimbal_action_cost: 框架动作平方和相对于参考尺度的 cost
        gimbal_action_cost = gimbal_action_sq / self.reward_norm_cfg.gimbal_action_scale
        # wheel_action_cost: 飞轮动作平方和相对于参考尺度的 cost
        wheel_action_cost = wheel_action_sq / self.reward_norm_cfg.wheel_action_scale

        # 加权 penalty
        att_penalty = self.reward_cfg.w_att * attitude_cost
        omega_penalty = self.reward_cfg.w_omega * omega_cost
        wheel_bias_penalty = self.reward_cfg.w_wheel_bias * wheel_bias_cost
        gimbal_act_penalty = self.reward_cfg.w_gimbal_act * gimbal_action_cost
        wheel_act_penalty = self.reward_cfg.w_wheel_act * wheel_action_cost

        # ---- Stage A: reward scale 修复 ----
        raw_penalty = (
            att_penalty
            + omega_penalty
            + wheel_bias_penalty
            + gimbal_act_penalty
            + wheel_act_penalty
        )
        base_reward = -raw_penalty / self.reward_cfg.reward_scale

        # ---- attitude progress reward（normalized attitude_cost 差值，与 base 同尺度）----
        attitude_cost_prev = float(self._attitude_cost_prev)
        attitude_cost_now = float(attitude_cost)
        attitude_progress = attitude_cost_prev - attitude_cost_now
        attitude_progress_reward = (
            self.reward_cfg.w_att_progress
            * attitude_progress
            / self.reward_cfg.reward_scale
        )

        reward = base_reward + attitude_progress_reward

        # 更新 progress 缓存（供下一步使用）
        self._attitude_cost_prev = attitude_cost_now

        self.current_step += 1

        terminated = np.linalg.norm(self.omega) > self.episode_cfg.max_omega
        truncated = self.current_step >= self.episode_cfg.max_episode_steps

        obs = self._get_obs().astype(np.float32)
        info = {
            "reward_total": float(reward),
            # Stage A 新增诊断
            "reward_raw_penalty": float(raw_penalty),
            "reward_scale": float(self.reward_cfg.reward_scale),
            "reward_scaled_total": float(reward),
            # 原始物理量（保留诊断）
            "sigma_err_sq": float(sigma_err_sq),
            "omega_sq": float(omega_sq),
            "wheel_bias_sq": float(wheel_bias_sq),
            "action_sq": float(action_sq),
            "gimbal_action_sq": float(gimbal_action_sq),
            "wheel_action_sq": float(wheel_action_sq),
            # 归一化 cost（P1 方案）
            "reward_attitude_cost": float(attitude_cost),
            "reward_omega_cost": float(omega_cost),
            "reward_wheel_bias_cost": float(wheel_bias_cost),
            "reward_gimbal_action_cost": float(gimbal_action_cost),
            "reward_wheel_action_cost": float(wheel_action_cost),
            # 加权 penalty
            "reward_att_penalty": float(att_penalty),
            "reward_omega_penalty": float(omega_penalty),
            "reward_wheel_bias_penalty": float(wheel_bias_penalty),
            "reward_gimbal_act_penalty": float(gimbal_act_penalty),
            "reward_wheel_act_penalty": float(wheel_act_penalty),
            # attitude progress reward（P53-2 新增）
            "attitude_cost_prev": float(attitude_cost_prev),
            "attitude_cost_now": float(attitude_cost_now),
            "attitude_progress": float(attitude_progress),
            "attitude_progress_reward": float(attitude_progress_reward),
            "w_att_progress": float(self.reward_cfg.w_att_progress),
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """
        组装 22 维观测向量��v1.0 接口，固定不变）

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
