"""
VSCMG阵列底层物理抽象基类
定义了VSCMG系统的通用接口和核心物理计算逻辑
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseVSCMGArray(ABC):
    """
    VSCMG阵列抽象基类

    职责：
    - 定义VSCMG阵列的通用接口
    - 实现系统总输出力矩计算
    - 声明必须由子类实现的抽象方法
    """

    def __init__(self, num_vscmg: int):
        """
        初始化VSCMG阵列

        Args:
            num_vscmg: VSCMG的数量
        """
        self.num_vscmg = num_vscmg

    @abstractmethod
    def get_spin_matrix(self, delta: np.ndarray) -> np.ndarray:
        """
        获取自旋轴矩阵（齿轮轴方向矩阵）

        物理意义：描述每个VSCMG自旋轴在本体坐标系中的方向
        维度：3 x N（N为VSCMG数量）

        Args:
            delta: 框架角向量 [N x 1]

        Returns:
            自旋轴矩阵 [3 x N]
        """
        pass

    @abstractmethod
    def get_transverse_matrix(self, delta: np.ndarray) -> np.ndarray:
        """
        获取横向轴矩阵（雅可比矩阵）

        物理意义：描述框架角变化对角动量的传递关系
        维度：3 x N

        Args:
            delta: 框架角向量 [N x 1]

        Returns:
            横向轴矩阵 [3 x N]
        """
        pass

    def calculate_output_torque(
        self,
        delta: np.ndarray,
        h_w: np.ndarray,
        delta_dot: np.ndarray,
        h_w_dot: np.ndarray
    ) -> np.ndarray:
        """
        计算VSCMG阵列的总输出力矩

        物理公式：
        τ_total = A_t(δ) · (h_w ⊙ δ̇) + A_s(δ) · ḣ_w

        其中：
        - A_t(δ)：横向轴矩阵（雅可比矩阵）
        - h_w：飞轮角动量向量 [N x 1]
        - δ̇：框架角速度向量 [N x 1]
        - ⊙：逐元素乘积（哈达玛积）
        - A_s(δ)：自旋轴矩阵
        - ḣ_w：飞轮角动量变化率（角加速度乘转动惯量）[N x 1]

        Args:
            delta: 框架角向量 [N x 1]
            h_w: 飞轮角动量向量 [N x 1]
            delta_dot: 框架角速度向量 [N x 1]
            h_w_dot: 飞轮角动量变化率向量 [N x 1]

        Returns:
            总输出力矩向量 [3 x 1]
        """
        # 获取当前构型的矩阵
        a_s = self.get_spin_matrix(delta)      # 自旋轴矩阵 [3 x N]
        a_t = self.get_transverse_matrix(delta)  # 横向轴矩阵 [3 x N]

        # 计算框架轴产生的力矩（陀螺效应部分）
        # τ_gyro = A_t · (h_w ⊙ δ̇)
        h_w_times_delta_dot = h_w * delta_dot  # 逐元素乘积 [N x 1]
        torque_gyro = a_t @ h_w_times_delta_dot  # [3 x N] @ [N x 1] = [3 x 1]

        # 计算自旋轴产生的力矩（控制力矩部分）
        # τ_spin = A_s · ḣ_w
        torque_spin = a_s @ h_w_dot  # [3 x N] @ [N x 1] = [3 x 1]

        # 总输出力矩
        total_torque = torque_gyro + torque_spin

        return total_torque
