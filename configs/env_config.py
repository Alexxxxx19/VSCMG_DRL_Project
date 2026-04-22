"""
VSCMG 环境物理参数配置入口
===========================

nominal_*   ：标称值（设计基准值）
current_*   ：当前 episode 实际值（reset 时由 nominal + randomization 采样得到）
randomization：随机化配置（随机化幅度、分布、是否启用）

所有物理参数统一在此定义，env 初始化时从配置读取。
以后修改 J / I_w / 外扰 / 延迟 / 初始值等，无需进入 env 文件深处。
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ============================================================================
# 分布定义
# ============================================================================

@dataclass
class UniformRange:
    """闭区间均匀采样 [low, high]"""
    low:  float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return rng.uniform(self.low, self.high)

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


@dataclass
class GaussianRange:
    """以 nominal 为均值、sigma 为标准差的高斯采样（截断到 [low, high]）"""
    nominal:  float
    sigma:    float
    low:      float = -np.inf
    high:     float = np.inf

    def sample(self, rng: np.random.Generator) -> float:
        value = rng.normal(self.nominal, self.sigma)
        return float(np.clip(value, self.low, self.high))


@dataclass
class DisabledRange:
    """禁用随机化——始终返回标称值"""
    nominal: float

    def sample(self, rng: np.random.Generator) -> float:
        return self.nominal


# ============================================================================
# 外扰配置
# ============================================================================

@dataclass
class DisturbanceConfig:
    """外扰力矩配置"""
    enabled:         bool   = False
    # 外扰类型: "constant", "sinusoidal", "white_noise"
    disturbance_type: str   = "constant"
    # 幅值上限（Nm），类型不同解释不同：
    #   constant   : 恒定向量模长（每次 episode 随机方向）
    #   sinusoidal : 正弦幅度（频率固定）
    #   white_noise: 噪声标准差
    magnitude:      float  = 0.0
    # sinusoidal 专用：频率（Hz）
    frequency_hz:   float  = 0.1
    # 当前 episode 实际外扰力矩向量（reset 时采样，step 中累加）
    _current_torque: np.ndarray = field(default_factory=lambda: np.zeros(3), repr=False)


# ============================================================================
# 延迟（伺服一阶滞后）配置
# ============================================================================

@dataclass
class DelayConfig:
    """控制延迟 / 伺服一阶滞后配置"""
    enabled:      bool   = False
    # 一阶滞后时间常数（秒）
    # u_delayed(t) = alpha * u_delayed(t-dt) + (1-alpha) * u_current(t)
    # 其中 alpha = exp(-dt / tau)
    tau:     float = 0.0    # 滞后时间常数（秒）
    # 延迟步数（整数，dt 的整数倍近似；0 = 纯一阶滞后无整数延迟）
    delay_steps: int   = 0


# ============================================================================
# 随机化配置（reset 时采样）
# ============================================================================

@dataclass
class RandomizationConfig:
    """Episode 级随机化配置（reset 时应用）"""
    # 航天器转动惯量 J_sc 随机化
    # 每个对角元独立采样，形状 (3,)
    j_sc_enabled:  bool            = False
    j_sc_range:    GaussianRange   = field(default_factory=lambda: GaussianRange(nominal=100.0, sigma=10.0, low=50.0, high=200.0))

    # 飞轮转动惯量 I_w 随机化
    i_w_enabled:   bool            = False
    i_w_range:     GaussianRange   = field(default_factory=lambda: GaussianRange(nominal=0.1, sigma=0.01, low=0.05, high=0.2))

    # 飞轮标称偏置随机化（3000 rpm ± 百分比）
    omega_bias_enabled:     bool          = False
    omega_bias_range:       UniformRange  = field(default_factory=lambda: UniformRange(low=0.95, high=1.05))

    # 初始姿态误差角（度）随机化范围（默认 [0, 5°]）
    init_attitude_enabled:  bool          = False    # False = 保持当前随机范围 [0, 5°]
    init_attitude_range:   UniformRange  = field(default_factory=lambda: UniformRange(low=0.0, high=5.0))

    # 初始角速度范围（rad/s）随机化
    init_omega_enabled:     bool          = False
    init_omega_range:       UniformRange  = field(default_factory=lambda: UniformRange(low=0.0, high=0.0))

    # 框架角初始值随机化（rad）
    init_gimbal_enabled:    bool          = False
    init_gimbal_range:      UniformRange  = field(default_factory=lambda: UniformRange(low=-0.1, high=0.1))


# ============================================================================
# VSCMG 环境主配置
# ============================================================================

@dataclass
class VSCMGEnvConfig:
    """
    VSCMG 环境物理参数集中配置

    用法：
        from configs.env_config import VSCMGEnvConfig, make_default_config
        cfg = make_default_config()   # v1.0 默认行为（无随机化）
        cfg = make_v3_robust_config()  # v3.0 域随机化配置
    """

    # --- 标称值（物理基准） ---
    nominal_j_sc:            np.ndarray = field(
        default_factory=lambda: np.diag([100.0, 100.0, 100.0])
    )
    nominal_i_w:            np.ndarray = field(
        default_factory=lambda: np.full(4, 0.1)
    )
    nominal_omega_w:        np.ndarray = field(
        default_factory=lambda: np.full(4, 3000.0 * (2.0 * np.pi / 60.0))
    )
    dt:                     float  = 0.01        # 积分步长（秒）
    max_gimbal_rate:        float  = 1.0         # rad/s
    max_wheel_accel:        float  = 50.0        # rad/s²
    max_omega:              float  = 20.0        # rad/s（截断阈值）
    max_episode_steps:      int    = 1000

    # --- 随机化配置 ---
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)

    # --- 外扰配置 ---
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)

    # --- 延迟配置 ---
    delay: DelayConfig = field(default_factory=DelayConfig)

    # --- 当前 episode 实际值（reset 后由 nominal + randomization 采样填充）---
    current_j_sc:     np.ndarray = field(default_factory=lambda: np.diag([100.0, 100.0, 100.0]))
    current_i_w:      np.ndarray = field(default_factory=lambda: np.full(4, 0.1))
    current_omega_w:  np.ndarray = field(default_factory=lambda: np.full(4, 3000.0 * (2.0 * np.pi / 60.0)))
    current_omega_w_nominal: float = field(
        default_factory=lambda: 3000.0 * (2.0 * np.pi / 60.0)
    )

    # 初始姿态误差（度）
    current_init_attitude_deg: float = 0.0

    # 初始角速度（rad/s）
    current_init_omega: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # 框架角初始值（rad）
    current_init_gimbal: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # 上一时刻动作（用于一阶滞后延迟和动作平滑）
    _prev_action: np.ndarray = field(default_factory=lambda: np.zeros(8), repr=False)

    # --- 内部方法 ---

    def apply_randomization(self, rng: np.random.Generator) -> None:
        """
        根据 randomization 配置，采样并填充所有 current_* 值。
        在 env.reset() 时调用。
        """
        # J_sc（每个 episode 独立采样）
        j_diag = np.array([
            self.randomization.j_sc_range.sample(rng) if self.randomization.j_sc_enabled
            else self.nominal_j_sc[i, i]
            for i in range(3)
        ])
        self.current_j_sc = np.diag(j_diag)

        # I_w
        i_w_vals = np.array([
            self.randomization.i_w_range.sample(rng) if self.randomization.i_w_enabled
            else self.nominal_i_w[i]
            for i in range(4)
        ])
        self.current_i_w = i_w_vals

        # 飞轮偏置
        bias_factor = (
            self.randomization.omega_bias_range.sample(rng)
            if self.randomization.omega_bias_enabled
            else 1.0
        )
        self.current_omega_w_nominal = self.nominal_omega_w[0] * bias_factor
        self.current_omega_w = np.full(4, self.current_omega_w_nominal)

        # 初始姿态误差
        self.current_init_attitude_deg = (
            self.randomization.init_attitude_range.sample(rng)
            if self.randomization.init_attitude_enabled
            else self.randomization.init_attitude_range.sample(rng)  # 始终用 [0,5°]
        )

        # 初始角速度
        self.current_init_omega = np.array([
            self.randomization.init_omega_range.sample(rng) if self.randomization.init_omega_enabled
            else 0.0
            for _ in range(3)
        ])

        # 框架角初始值
        self.current_init_gimbal = np.array([
            self.randomization.init_gimbal_range.sample(rng) if self.randomization.init_gimbal_enabled
            else 0.0
            for _ in range(4)
        ])

        # 动作缓存归零
        self._prev_action = np.zeros(8)

    def compute_disturbance_torque(self, t: float) -> np.ndarray:
        """
        根据 disturbance 配置，在时刻 t 计算外扰力矩向量 [3]

        默认返回零向量（无外扰）。
        """
        if not self.disturbance.enabled:
            return np.zeros(3)

        d = self.disturbance
        mag = d.magnitude

        if d.disturbance_type == "constant":
            return d._current_torque  # reset 时已采样方向和幅值
        elif d.disturbance_type == "sinusoidal":
            return mag * np.array([
                np.sin(2 * np.pi * d.frequency_hz * t),
                np.cos(2 * np.pi * d.frequency_hz * t),
                0.0
            ])
        elif d.disturbance_type == "white_noise":
            rng = np.random.default_rng()
            return rng.normal(0.0, mag, size=3)
        else:
            return np.zeros(3)

    def apply_actuator_delay(self, action: np.ndarray, dt: float) -> np.ndarray:
        """
        应用控制延迟/伺服一阶滞后，返回处理后的动作

        默认（disabled）返回原始动作（identity）。
        """
        if not self.delay.enabled:
            return action.copy()

        prev = self._prev_action
        alpha = np.exp(-dt / max(self.delay.tau, 1e-12))
        delayed = alpha * prev + (1.0 - alpha) * action
        self._prev_action = delayed.copy()
        return delayed


# ============================================================================
# 预设配置工厂
# ============================================================================

def make_default_config() -> VSCMGEnvConfig:
    """
    v1.0 默认配置：所有随机化关闭，所有外扰关闭，所有延迟关闭，
    与 v1.0 验收基线完全一致。
    """
    return VSCMGEnvConfig(
        nominal_j_sc=np.diag([100.0, 100.0, 100.0]),
        nominal_i_w=np.full(4, 0.1),
        nominal_omega_w=np.full(4, 3000.0 * (2.0 * np.pi / 60.0)),
        dt=0.01,
        max_gimbal_rate=1.0,
        max_wheel_accel=50.0,
        max_omega=20.0,
        max_episode_steps=1000,
        randomization=RandomizationConfig(
            j_sc_enabled=False,
            i_w_enabled=False,
            omega_bias_enabled=False,
            init_attitude_enabled=False,     # False = 使用默认 [0, 5°]
            init_omega_enabled=False,
            init_gimbal_enabled=False,
        ),
        disturbance=DisturbanceConfig(enabled=False),
        delay=DelayConfig(enabled=False),
    )


def make_v3_robust_config() -> VSCMGEnvConfig:
    """
    v3.0 域随机化配置：
    - J_sc ±20% 随机化
    - I_w ±10% 随机化
    - 飞轮偏置 ±5% 随机化
    - 初始角速度 ±0.1 rad/s 随机化
    - 初始姿态误差 [0, 10°] 扩大范围
    - 外扰开启（白噪声，0.01 Nm）
    """
    return VSCMGEnvConfig(
        nominal_j_sc=np.diag([100.0, 100.0, 100.0]),
        nominal_i_w=np.full(4, 0.1),
        nominal_omega_w=np.full(4, 3000.0 * (2.0 * np.pi / 60.0)),
        dt=0.01,
        max_gimbal_rate=1.0,
        max_wheel_accel=50.0,
        max_omega=20.0,
        max_episode_steps=1000,
        randomization=RandomizationConfig(
            j_sc_enabled=True,
            j_sc_range=GaussianRange(nominal=100.0, sigma=20.0, low=50.0, high=200.0),
            i_w_enabled=True,
            i_w_range=GaussianRange(nominal=0.1, sigma=0.01, low=0.05, high=0.2),
            omega_bias_enabled=True,
            omega_bias_range=UniformRange(low=0.95, high=1.05),
            init_attitude_enabled=True,
            init_attitude_range=UniformRange(low=0.0, high=10.0),
            init_omega_enabled=True,
            init_omega_range=UniformRange(low=-0.1, high=0.1),
            init_gimbal_enabled=False,
        ),
        disturbance=DisturbanceConfig(
            enabled=True,
            disturbance_type="white_noise",
            magnitude=0.01,
        ),
        delay=DelayConfig(enabled=False),
    )
