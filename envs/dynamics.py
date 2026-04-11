"""
航天器刚体动力学引擎
基于牛顿-欧拉方程的航天器姿态动力学求解器
"""

import numpy as np


class SpacecraftDynamics:
    """
    航天器刚体动力学求解器

    职责：
    - 计算航天器角加速度
    - 处理陀螺效应和VSCMG耦合力矩
    - 求解牛顿-欧拉方程
    """

    def __init__(self, j_sc: np.ndarray):
        """
        初始化航天器动力学

        Args:
            j_sc: 航天器本体转动惯量矩阵 [3 x 3]，单位：kg·m²
                  必须是对称正定矩阵
        """
        self.J_sc = j_sc.astype(np.float64)

        # 预计算惯量矩阵的逆，避免重复求逆
        self.J_sc_inv = np.linalg.inv(self.J_sc)

    def compute_angular_acceleration(
        self,
        omega: np.ndarray,
        h_vscmg: np.ndarray,
        tau_vscmg: np.ndarray,
        tau_external: np.ndarray = None
    ) -> np.ndarray:
        """
        计算航天器角加速度（牛顿-欧拉方程）

        动力学方程：
        J · ω̇ + ω × (J · ω + h_vscmg) = τ_external - τ_vscmg

        整理得：
        ω̇ = J⁻¹ · [τ_external - τ_vscmg - ω × (J · ω) - ω × h_vscmg]

        其中各项物理意义：
        - ω̇：本体角加速度 [rad/s²]
        - J：航天器转动惯量矩阵 [kg·m²]
        - ω：本体角速度 [rad/s]
        - h_vscmg：VSCMG阵列总角动量 [Nms]
        - τ_vscmg：VSCMG输出力矩（动量变化率）[Nm]
        - τ_external：外部干扰力矩 [Nm]
        - ω × (J · ω)：航天器自身陀螺力矩
        - ω × h_vscmg：VSCMG牵连耦合力矩

        Args:
            omega: 本体角速度向量 [3 x 1]，单位：rad/s
            h_vscmg: VSCMG阵列总角动量 [3 x 1]，单位：Nms
            tau_vscmg: VSCMG输出力矩 [3 x 1]，单位：Nm
            tau_external: 外部干扰力矩 [3 x 1]，单位：Nm，默认为零向量

        Returns:
            角加速度向量 [3 x 1]，单位：rad/s²
        """
        # 确保输入为列向量
        omega = omega.reshape(3, 1)
        h_vscmg = h_vscmg.reshape(3, 1)
        tau_vscmg = tau_vscmg.reshape(3, 1)

        # 外部力矩默认为零
        if tau_external is None:
            tau_external = np.zeros((3, 1))
        else:
            tau_external = tau_external.reshape(3, 1)

        # 1. 计算航天器自身陀螺力矩：τ_gyro_sc = ω × (J · ω)
        j_omega = self.J_sc @ omega  # J · ω
        tau_gyro_sc = np.cross(omega.flatten(), j_omega.flatten()).reshape(3, 1)

        # 2. 计算VSCMG牵连耦合力矩：τ_coup = ω × h_vscmg
        tau_coupling = np.cross(omega.flatten(), h_vscmg.flatten()).reshape(3, 1)

        # 3. 牛顿-欧拉方程：ω̇ = J⁻¹ · [τ_ext - τ_vscmg - τ_gyro_sc - τ_coup]
        total_torque = tau_external - tau_vscmg - tau_gyro_sc - tau_coupling
        omega_dot = self.J_sc_inv @ total_torque

        return omega_dot



# ==================== 测试模块 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("航天器动力学求解器测试")
    print("=" * 60)

    # 创建动力学实例（假设航天器参数）
    test_j_sc = np.diag([100.0, 120.0, 80.0])  # 对角惯量矩阵 [kg·m²]
    dynamics = SpacecraftDynamics(test_j_sc)

    print(f"\n航天器转动惯量矩阵 J [kg·m²]:")
    print(test_j_sc)
    print(f"\n惯量矩阵逆 J⁻¹ [(kg·m²)⁻¹]:")
    print(dynamics.J_sc_inv)

    # 测试工况1：静态
    print("\n" + "-" * 60)
    print("测试工况1：静态（角速度为0）")
    print("-" * 60)

    test_omega = np.zeros((3, 1))
    test_h_vscmg = np.array([[5.0], [0.0], [0.0]])  # X方向5Nms角动量
    test_tau_vscmg = np.zeros((3, 1))
    test_tau_ext = np.array([[0.1], [0.0], [0.0]])   # X方向0.1Nm外力矩

    print(f"角速度 ω = {test_omega.flatten()} rad/s")
    print(f"VSCMG角动量 h = {test_h_vscmg.flatten()} Nms")
    print(f"VSCMG输出力矩 τ_vscmg = {test_tau_vscmg.flatten()} Nm")
    print(f"外力矩 τ_ext = {test_tau_ext.flatten()} Nm")

    test_omega_dot = dynamics.compute_angular_acceleration(test_omega, test_h_vscmg, test_tau_vscmg, test_tau_ext)
    print(f"\n角加速度 ω̇ = {test_omega_dot.flatten()} rad/s²")
    print(f"\n角加速度 ω̇ = {omega_dot.flatten()} rad/s²")

    # 测试工况2：旋转状态
    print("\n" + "-" * 60)
    print("测试工况2：旋转状态（Y轴0.1 rad/s）")
    print("-" * 60)

    test_omega = np.array([[0.0], [0.1], [0.0]])  # 绕Y轴旋转
    test_h_vscmg = np.array([[0.0], [0.0], [0.0]])
    test_tau_vscmg = np.array([[-0.05], [0.0], [0.0]])  # 反向控制力矩
    test_tau_ext = np.zeros((3, 1))

    print(f"角速度 ω = {test_omega.flatten()} rad/s")
    print(f"VSCMG角动量 h = {test_h_vscmg.flatten()} Nms")
    print(f"VSCMG输出力矩 τ_vscmg = {test_tau_vscmg.flatten()} Nm")
    print(f"外力矩 τ_ext = {test_tau_ext.flatten()} Nm")

    test_omega_dot = dynamics.compute_angular_acceleration(test_omega, test_h_vscmg, test_tau_vscmg, test_tau_ext)
    print(f"\n角加速度 ω̇ = {test_omega_dot.flatten()} rad/s²")

    # 分项显示
    test_j_omega = test_j_sc @ test_omega
    test_tau_gyro = np.cross(test_omega.flatten(), test_j_omega.flatten()).reshape(3, 1)
    print(f"\n分项验证：")
    print(f"  航天器陀螺力矩 ω×(J·ω) = {test_tau_gyro.flatten()} Nm")
    print(f"  总力矩 = {(test_tau_ext - test_tau_vscmg - test_tau_gyro).flatten()} Nm")

    print("\n" + "=" * 60)
    print("测试完成！请人工核对角加速度计算是否符合预期。")
    print("=" * 60)
