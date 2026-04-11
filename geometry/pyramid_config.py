"""
金字塔构型VSCMG阵列实现
标准4-VSCMG金字塔构型，侧倾角β=54.73°
"""

import numpy as np
from geometry.base_config import BaseVSCMGArray


class PyramidVSCMG(BaseVSCMGArray):
    """
    金字塔构型VSCMG阵列

    几何配置：
    - 4个VSCMG单元
    - 侧倾角β = 54.73°（约等于arcsin(√(2/3))，实现3轴等转动惯量）
    - 框架轴分布：绕X、Y轴交替排列

    坐标系定义：
    - Z轴：金字塔中心轴（垂直于底面）
    - X、Y轴：底面坐标轴
    """

    def __init__(self, beta_deg: float = 54.73):
        """
        初始化金字塔构型VSCMG阵列

        Args:
            beta_deg: 侧倾角（度），默认54.73°
        """
        super().__init__(num_vscmg=4)

        # 转换为弧度
        self.beta = np.radians(beta_deg)

        # 预计算三角函数值以优化性能
        self.sin_beta = np.sin(self.beta)
        self.cos_beta = np.cos(self.beta)

    def get_spin_matrix(self, delta: np.ndarray) -> np.ndarray:
        """
        获取自旋轴矩阵（齿轮轴方向矩阵）

        对于4-VSCMG金字塔构型，自旋轴矩阵为：
        A_s = [
            [-cβ·sδ₁,  -cδ₂,   cβ·sδ₃,   cδ₄    ]
            [ cδ₁,    -cβ·sδ₂,  -cδ₃,    cβ·sδ₄ ]
            [ sβ·sδ₁,  sβ·sδ₂,  sβ·sδ₃,  sβ·sδ₄ ]
        ]

        其中：cβ = cos(β), sβ = sin(β), cδ = cos(δ), sδ = sin(δ)

        Args:
            delta: 框架角向量 [4 x 1]，顺序为[VSCMG1, VSCMG2, VSCMG3, VSCMG4]

        Returns:
            自旋轴矩阵 [3 x 4]
        """
        d1, d2, d3, d4 = delta.flatten()

        # 预计算各框架角的三角函数
        s1, c1 = np.sin(d1), np.cos(d1)
        s2, c2 = np.sin(d2), np.cos(d2)
        s3, c3 = np.sin(d3), np.cos(d3)
        s4, c4 = np.sin(d4), np.cos(d4)

        # 构建自旋轴矩阵 [3 x 4]
        a_s = np.array([
            [-self.cos_beta * s1,  -c2,              self.cos_beta * s3,   c4             ],
            [ c1,                 -self.cos_beta * s2, -c3,                self.cos_beta * s4],
            [ self.sin_beta * s1,  self.sin_beta * s2,  self.sin_beta * s3,  self.sin_beta * s4]
        ])

        return a_s

    def get_transverse_matrix(self, delta: np.ndarray) -> np.ndarray:
        """
        获取横向轴矩阵（雅可比矩阵）

        对于4-VSCMG金字塔构型，横向轴矩阵为：
        A_t = [
            [-cβ·cδ₁,  sδ₂,   cβ·cδ₃,  -sδ₄    ]
            [-sδ₁,    -cβ·cδ₂,  sδ₃,    cβ·cδ₄ ]
            [ sβ·cδ₁,  sβ·cδ₂,  sβ·cδ₃,  sβ·cδ₄ ]
        ]

        物理意义：描述框架角速度如何转化为角动量变化

        Args:
            delta: 框架角向量 [4 x 1]

        Returns:
            横向轴矩阵 [3 x 4]
        """
        d1, d2, d3, d4 = delta.flatten()

        # 预计算各框架角的三角函数
        s1, c1 = np.sin(d1), np.cos(d1)
        s2, c2 = np.sin(d2), np.cos(d2)
        s3, c3 = np.sin(d3), np.cos(d3)
        s4, c4 = np.sin(d4), np.cos(d4)

        # 构建横向轴矩阵 [3 x 4]
        a_t = np.array([
            [-self.cos_beta * c1,  s2,               self.cos_beta * c3,  -s4              ],
            [-s1,                 -self.cos_beta * c2,  s3,                self.cos_beta * c4],
            [ self.sin_beta * c1,  self.sin_beta * c2,  self.sin_beta * c3,  self.sin_beta * c4]
        ])

        return a_t


# ==================== 测试模块 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("金字塔构型VSCMG测试")
    print("=" * 60)

    # 创建金字塔构型实例
    pyramid = PyramidVSCMG(beta_deg=54.73)

    # 测试用例：框架角全为0度
    test_delta_zero = np.zeros((4, 1))  # [0, 0, 0, 0]弧度
    test_delta_zero_deg = np.array([0, 0, 0, 0]) * 180 / np.pi

    print(f"\n测试条件：框架角 δ = {test_delta_zero_deg}°")
    print(f"侧倾角 β = 54.73° (sin(beta) = {pyramid.sin_beta:.6f}, cos(beta) = {pyramid.cos_beta:.6f})")

    # 获取矩阵
    test_a_s_zero = pyramid.get_spin_matrix(test_delta_zero)
    test_a_t_zero = pyramid.get_transverse_matrix(test_delta_zero)

    print("\n" + "-" * 60)
    print("自旋轴矩阵 A_s (3x4):")
    print("-" * 60)
    print(test_a_s_zero)

    print("\n" + "-" * 60)
    print("横向轴矩阵 A_t (3x4):")
    print("-" * 60)
    print(test_a_t_zero)

    # 测试力矩计算
    print("\n" + "-" * 60)
    print("力矩计算测试")
    print("-" * 60)

    # 假设飞轮参数
    test_h_w = np.array([[1.0], [1.0], [1.0], [1.0]])      # 飞轮角动量 [Nms]
    test_delta_dot = np.array([[0.1], [0.1], [0.1], [0.1]])  # 框架角速度 [rad/s]
    test_h_w_dot = np.array([[0.5], [0.5], [0.5], [0.5]])    # 飞轮角加速度 [Nms/s²]

    print(f"Flywheel angular momentum h_w = {test_h_w.flatten()} Nms")
    print(f"Gimbal rate delta_dot = {test_delta_dot.flatten()} rad/s")
    print(f"Flywheel acceleration h_w_dot = {test_h_w_dot.flatten()} Nms/s^2")

    # 计算输出力矩
    test_tau_out = pyramid.calculate_output_torque(test_delta_zero, test_h_w, test_delta_dot, test_h_w_dot)

    print(f"\nTotal torque tau = {test_tau_out.flatten()} Nm")

    # 分项验证
    print("\nVerification:")
    test_tau_gyro = test_a_t_zero @ (test_h_w * test_delta_dot)
    test_tau_spin = test_a_s_zero @ test_h_w_dot
    print(f"  Gyro torque = {test_tau_gyro.flatten()} Nm")
    print(f"  Spin torque = {test_tau_spin.flatten()} Nm")
    print(f"  Total = {(test_tau_gyro + test_tau_spin).flatten()} Nm")

    print("\n" + "=" * 60)
    print("Test completed! Please verify the matrix values.")
    print("=" * 60)
