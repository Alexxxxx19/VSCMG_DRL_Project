"""
VSCMG 训练调度参数集中配置
===========================

用途：所有训练调度相关的超参数统一在此定义。
以后修改训练步数 / batch_size / 并行数 / 保存频率等，只需改这里。

优先级规则：
    train.py CLI 参数 > 本文件默认值
    即：CLI 存在时覆盖这里的默认值。

用法：
    from configs.train_config import TrainConfig, make_default_train_config
    cfg = make_default_train_config()
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    """
    训练调度参数配置

    包含：并行数、步数、batch、更新频率、ReplayBuffer、日志/检查点路径等。
    """

    # --- 并行与设备 ---
    num_envs:           int    = 1          # 并行环境数量（1=sync, >1=async）
    device:             str    = "cpu"      # 计算设备 (cpu/cuda)

    # --- 训练步数与采样 ---
    total_steps:        int    = 2_000_000  # 最大训练步数
    start_steps:        int    = 5_000      # 前N步纯随机探索（无需策略）
    update_every:       int    = 50         # 每N步触发一次网络更新
    update_times:       int    = 50         # 每次触发更新 N 轮（与原始 train.py 行为一致）

    # --- Batch 与 Replay ---
    batch_size:         int    = 256        # 每次 update 的 batch 大小
    replay_capacity:    int    = 100_000    # ReplayBuffer 容量
    replay_prefill_path: Optional[str] = None  # npz 文件路径，训练开始前预填充 replay buffer

    # --- 评估与保存 ---
    eval_frequency:     int    = 100_000    # 每N步评估一次（预留，当前未使用）
    checkpoint_frequency: int  = 100_000    # 每N步保存一次 checkpoint
    checkpoint_dir:     str    = "models"   # checkpoint 保存目录

    # --- TensorBoard 日志 ---
    log_dir_base:       str    = "runs"     # TensorBoard 日志根目录
    tb_flush_secs:      int    = 30         # TensorBoard writer flush 间隔（秒）

    # --- 随机种子 ---
    seed:               int    = 42         # 全局随机种子（固定种子便于复现，CLI 可覆盖）


def make_default_train_config() -> TrainConfig:
    """
    v1.0 默认训练配置（面向第一阶段正式训练的保守起点）

    默认行为：
        - num_envs=1（CLI 可覆盖，如 --num_envs 16）
        - device=cpu（CLI 可覆盖，如 --device cuda）
        - 200万步训练，batch_size=256，update_every=50，update_times=50
        - ReplayBuffer=100000，每10万步保存 checkpoint
        - seed=42（固定种子便于复现，CLI 可覆盖，如 --seed 123）

    注意：
        - update_times=50 是为了保持原始 train.py 的训练强度不变
          （每 50 步触发 50 轮更新，等于每步平均 1 次更新）
        - eval_frequency 当前未使用，预留字段
        - 固定种子是为了第一阶段 reward / 配置对比时更易复现；
          如需不同随机性，通过 CLI 设置 --seed 0 或其他值

    CLI 覆盖示例：
        python train.py --num_envs 16 --device cuda --batch_size 2048
        python train.py --seed 123  # 切换为其他固定种子
    """
    return TrainConfig(
        num_envs=1,
        device="cpu",
        total_steps=2_000_000,
        start_steps=5_000,
        update_every=50,
        update_times=50,
        batch_size=256,
        replay_capacity=100_000,
        eval_frequency=100_000,
        checkpoint_frequency=100_000,
        checkpoint_dir="models",
        log_dir_base="runs",
        tb_flush_secs=30,
        seed=42,  # v1.0 第一阶段固定种子便于复现
    )