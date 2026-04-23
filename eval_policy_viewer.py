"""
通用策略控制效果评估工具
========================

功能：
- 加载任意训练好的模型文件
- 运行单回合评估，记录详细物理量
- 生成可视化图表和CSV数据

用法：
    python eval_policy_viewer.py --model models/<run_name>/best_episode_reward.pth
    python eval_policy_viewer.py --model models/<run_name>/final_step_100000.pth --seed 123 --max_steps 1500

说明：
    --model 必填，需指向 models/<run_name>/ 下的具体 .pth 文件
"""

import argparse
import os
import csv
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from envs.vscmg_env import VSCMGEnv
from agents.td3_agent import TD3
from configs.agent_config import make_default_agent_config


def setup_chinese_font():
    """配置 matplotlib 中文字体（Windows 优先，失败则回退到英文）"""
    try:
        # Windows 常见中文字体
        system_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
        plt.rcParams['font.sans-serif'] = system_fonts + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except:
        return False


def load_actor_from_checkpoint(model_path: str, device: str = "cpu"):
    """
    从 checkpoint 中加载 actor 网络（兼容多种保存格式）

    支持的格式：
    1. agent.save_model() 保存的完整格式：键为 'actor'
    2. 训练脚本保存的 checkpoint 格式：键为 'actor_state_dict'
    """
    checkpoint = torch.load(model_path, map_location=device)

    # 创建临时 TD3 实例（只用于提取 actor 结构）
    cfg = make_default_agent_config()
    cfg.device = device
    agent = TD3(
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        hidden_dim=cfg.hidden_dim,
        action_bound=cfg.action_bound,
        sigma=cfg.sigma,
        tau=cfg.tau,
        gamma=cfg.gamma,
        critic_lr=cfg.critic_lr,
        actor_lr=cfg.actor_lr,
        delay=cfg.policy_delay,
        policy_noise=cfg.policy_noise,
        noise_clip=cfg.noise_clip,
        device=device,
    )

    # 兼容两种键名格式
    if 'actor' in checkpoint:
        agent.actor.load_state_dict(checkpoint['actor'])
    elif 'actor_state_dict' in checkpoint:
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    else:
        raise ValueError(f"无法识别的 checkpoint 格式，可用键: {list(checkpoint.keys())}")

    agent.actor.eval()
    return agent.actor


def quaternion_to_mrp_array(q: np.ndarray) -> np.ndarray:
    """四元数转 MRP（复用环境逻辑）"""
    from envs.vscmg_env import (
        quaternion_conjugate, quaternion_multiply,
        quaternion_normalize, quaternion_to_mrp, mrp_shadow,
        apply_scalar_sign_flip
    )
    q_target = np.array([1.0, 0.0, 0.0, 0.0])
    q_err = quaternion_multiply(q_target, quaternion_conjugate(q))
    q_err = quaternion_normalize(q_err)
    sigma_err = quaternion_to_mrp(apply_scalar_sign_flip(q_err))
    return mrp_shadow(sigma_err)


def quaternion_to_euler_deg(q: np.ndarray) -> np.ndarray:
    """
    四元数转 Euler 角（ZYX/yaw-pitch-roll 约定，单位：度）

    约定：
    - 绕 Z 轴旋转 yaw (ψ)
    - 绕 Y 轴旋转 pitch (θ)
    - 绕 X 轴旋转 roll (φ)
    - 旋转顺序：ZYX（航空常用约定）

    输入：四元数 [w, x, y, z] (scalar-first)
    输出：[roll, pitch, yaw] (deg)
    """
    q = q / np.linalg.norm(q)  # 归一化
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Roll (φ)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (θ)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # 90度时奇异点
    else:
        pitch = np.arcsin(sinp)

    # Yaw (ψ)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # 转换为度
    return np.array([roll, pitch, yaw]) * 180.0 / np.pi


def rad_to_deg(rad_value):
    """弧度转度"""
    return rad_value * 180.0 / np.pi


def run_episode_with_logging(env, actor, device, seed=42, max_steps=1000):
    """跑一个完整 episode，记录每一步的状态（内部单位+显示层单位）"""
    obs, _ = env.reset(seed=seed)

    # 记录容器（内部单位：rad, MRP）
    history = {
        'step': [],
        # 内部单位（MRP / rad）
        'sigma_err_x': [], 'sigma_err_y': [], 'sigma_err_z': [], 'sigma_err_norm': [],
        'omega_x': [], 'omega_y': [], 'omega_z': [], 'omega_norm': [],
        'delta_1': [], 'delta_2': [], 'delta_3': [], 'delta_4': [],
        'delta_dot_1': [], 'delta_dot_2': [], 'delta_dot_3': [], 'delta_dot_4': [],
        'omega_w_1': [], 'omega_w_2': [], 'omega_w_3': [], 'omega_w_4': [],
        'omega_w_tilde_1': [], 'omega_w_tilde_2': [], 'omega_w_tilde_3': [], 'omega_w_tilde_4': [],
        'wheel_bias_sq': [],
        # 动作原始数据
        'action_1': [], 'action_2': [], 'action_3': [], 'action_4': [],
        'action_5': [], 'action_6': [], 'action_7': [], 'action_8': [],
        'action_abs_mean': [], 'action_sat_rate': [],
        # Reward breakdown（包含新增的拆项指标）
        'reward_total': [], 'sigma_err_sq': [], 'omega_sq': [], 'action_sq': [],
        'wheel_bias_sq_info': [],
        'gimbal_action_sq': [], 'wheel_action_sq': [],  # 新增：拆项指标
        # 显示层单位（deg / deg/s）
        'roll_deg': [], 'pitch_deg': [], 'yaw_deg': [],
        'omega_x_deg_s': [], 'omega_y_deg_s': [], 'omega_z_deg_s': [], 'omega_norm_deg_s': [],
        'delta_1_deg': [], 'delta_2_deg': [], 'delta_3_deg': [], 'delta_4_deg': [],
        'delta_dot_1_deg_s': [], 'delta_dot_2_deg_s': [], 'delta_dot_3_deg_s': [], 'delta_dot_4_deg_s': [],
    }

    states_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

    for step in range(max_steps):
        # Actor 推理（无噪声）
        with torch.no_grad():
            actions = actor(states_tensor).cpu().numpy()[0]

        # 执行
        obs, reward, terminated, truncated, info = env.step(actions)

        # 提取物理量（内部单位）
        q = env.q  # 当前四元数 [w,x,y,z]
        omega = env.omega.copy()
        omega_w = env.omega_w.copy()
        omega_w_nominal = env.omega_w_nominal
        delta = env.delta.copy()
        delta_dot = env._delta_dot_cache.copy()

        # 计算 MRP 误差（内部单位）
        sigma_err = quaternion_to_mrp_array(q)

        # 计算 Euler 角（显示层：deg）
        euler_deg = quaternion_to_euler_deg(q)

        # 计算 Omega_w_tilde 和 wheel_bias_sq
        omega_w_tilde = (omega_w - omega_w_nominal) / omega_w_nominal
        wheel_bias_sq = float(np.sum(omega_w_tilde ** 2))

        # 计算动作统计
        action_abs_mean = float(np.mean(np.abs(actions)))
        action_sat_rate = float(np.mean(np.abs(actions) >= 0.95))

        # 记录内部单位数据
        history['step'].append(step)
        history['sigma_err_x'].append(sigma_err[0])
        history['sigma_err_y'].append(sigma_err[1])
        history['sigma_err_z'].append(sigma_err[2])
        history['sigma_err_norm'].append(np.linalg.norm(sigma_err))
        history['omega_x'].append(omega[0])
        history['omega_y'].append(omega[1])
        history['omega_z'].append(omega[2])
        history['omega_norm'].append(np.linalg.norm(omega))
        for i in range(4):
            history[f'delta_{i+1}'].append(delta[i])
            history[f'delta_dot_{i+1}'].append(delta_dot[i])
            history[f'omega_w_{i+1}'].append(omega_w[i])
            history[f'omega_w_tilde_{i+1}'].append(omega_w_tilde[i])
        history['wheel_bias_sq'].append(wheel_bias_sq)
        for i in range(8):
            history[f'action_{i+1}'].append(actions[i])
        history['action_abs_mean'].append(action_abs_mean)
        history['action_sat_rate'].append(action_sat_rate)
        history['reward_total'].append(reward)

        # 从 info 提取 reward breakdown（包含新增的拆项指标）
        if isinstance(info, dict):
            for key in ['sigma_err_sq', 'omega_sq', 'action_sq']:
                history[key].append(float(info.get(key, 0.0)))
            history['wheel_bias_sq_info'].append(float(info.get('wheel_bias_sq', 0.0)))
            # 新增：拆项动作指标
            history['gimbal_action_sq'].append(float(info.get('gimbal_action_sq', 0.0)))
            history['wheel_action_sq'].append(float(info.get('wheel_action_sq', 0.0)))
        else:
            for key in ['sigma_err_sq', 'omega_sq', 'action_sq']:
                history[key].append(0.0)
            history['wheel_bias_sq_info'].append(0.0)
            history['gimbal_action_sq'].append(0.0)
            history['wheel_action_sq'].append(0.0)

        # 记录显示层数据（deg / deg/s）
        history['roll_deg'].append(euler_deg[0])
        history['pitch_deg'].append(euler_deg[1])
        history['yaw_deg'].append(euler_deg[2])
        history['omega_x_deg_s'].append(rad_to_deg(omega[0]))
        history['omega_y_deg_s'].append(rad_to_deg(omega[1]))
        history['omega_z_deg_s'].append(rad_to_deg(omega[2]))
        history['omega_norm_deg_s'].append(rad_to_deg(np.linalg.norm(omega)))
        for i in range(4):
            history[f'delta_{i+1}_deg'].append(rad_to_deg(delta[i]))
            history[f'delta_dot_{i+1}_deg_s'].append(rad_to_deg(delta_dot[i]))

        if terminated or truncated:
            break

        states_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

    return history, terminated, truncated


def plot_spacecraft_metrics(history, output_path, use_chinese, model_name=None):
    """生成航天器状态图表（Euler角 + 角速度 + reward）"""
    steps = history['step']
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    if model_name:
        fig.suptitle(f"Eval Model: {model_name}", fontsize=10, fontweight='bold')

    # 1. Euler 角（deg）
    axes[0].plot(steps, history['roll_deg'], label='roll', alpha=0.7)
    axes[0].plot(steps, history['pitch_deg'], label='pitch', alpha=0.7)
    axes[0].plot(steps, history['yaw_deg'], label='yaw', alpha=0.7)
    title = '1. Attitude (Euler Angles, ZYX)' if not use_chinese else '1. 姿态 (Euler角, ZYX约定)'
    axes[0].set_title(title)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Angle (deg)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 角速度（deg/s）
    axes[1].plot(steps, history['omega_x_deg_s'], label='ωx', alpha=0.7)
    axes[1].plot(steps, history['omega_y_deg_s'], label='ωy', alpha=0.7)
    axes[1].plot(steps, history['omega_z_deg_s'], label='ωz', alpha=0.7)
    axes[1].plot(steps, history['omega_norm_deg_s'], 'k--', label='||ω||', linewidth=1.5)
    title = '2. Angular Velocity' if not use_chinese else '2. 角速度'
    axes[1].set_title(title)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('deg/s')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Reward 和 breakdown
    ax3_twin = axes[2].twinx()
    axes[2].plot(steps, history['reward_total'], 'b-', label='reward_total', alpha=0.7)
    ax3_twin.plot(steps, history['sigma_err_sq'], 'r--', label='sigma_err_sq', alpha=0.5)
    ax3_twin.plot(steps, history['omega_sq'], 'g--', label='omega_sq', alpha=0.5)
    title = '3. Reward & Breakdown' if not use_chinese else '3. 奖励与分解项'
    axes[2].set_title(title)
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Total Reward', color='b')
    ax3_twin.set_ylabel('Squared Terms', color='r')
    axes[2].legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  >> 航天器图表已保存到: {output_path}")


def plot_actuator_metrics(history, output_path, use_chinese, model_name=None):
    """生成执行器状态图表（框架 + 飞轮 + 健康度）"""
    steps = history['step']
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    if model_name:
        fig.suptitle(f"Eval Model: {model_name}", fontsize=10, fontweight='bold')

    # 1. 框架角（deg）
    for i in range(4):
        axes[0].plot(steps, history[f'delta_{i+1}_deg'], label=f'delta_{i+1}', alpha=0.7)
    title = '1. Gimbal Angles' if not use_chinese else '1. 框架角'
    axes[0].set_title(title)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Angle (deg)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 框架角速度（deg/s）
    for i in range(4):
        axes[1].plot(steps, history[f'delta_dot_{i+1}_deg_s'], label=f'delta_dot_{i+1}', alpha=0.7)
    title = '2. Gimbal Rates' if not use_chinese else '2. 框架角速度'
    axes[1].set_title(title)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Rate (deg/s)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. 飞轮转速与偏置
    ax3_twin = axes[2].twinx()
    for i in range(4):
        axes[2].plot(steps, history[f'omega_w_{i+1}'], label=f'omega_w_{i+1}', alpha=0.7)
    ax3_twin.plot(steps, history['wheel_bias_sq'], 'k--', label='||Omega_tilde||^2', linewidth=1.5)
    title = '3. Wheel Speeds & Bias' if not use_chinese else '3. 飞轮转速与偏置'
    axes[2].set_title(title)
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('omega_w (rad/s)')
    axes[2].legend(loc='upper left')
    ax3_twin.set_ylabel('||Omega_tilde||^2')
    ax3_twin.legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    # 4. 执行器健康度（action相关指标 + 新增拆项指标）
    axes[3].plot(steps, history['action_sq'], label='action_sq', alpha=0.7)
    axes[3].plot(steps, history['gimbal_action_sq'], '--', label='gimbal_action_sq', alpha=0.7)
    axes[3].plot(steps, history['wheel_action_sq'], '--', label='wheel_action_sq', alpha=0.7)
    axes[3].plot(steps, history['wheel_bias_sq'], label='wheel_bias_sq', alpha=0.5)
    axes[3].plot(steps, history['wheel_bias_sq_info'], ':', label='wheel_bias_sq_info', alpha=0.4)
    ax4_twin = axes[3].twinx()
    ax4_twin.plot(steps, history['action_abs_mean'], 'g-', label='action_abs_mean', alpha=0.5)
    ax4_twin.plot(steps, history['action_sat_rate'], 'm-', label='action_sat_rate', alpha=0.5)
    title = '4. Actuator Health Scores' if not use_chinese else '4. 执行器健康度'
    axes[3].set_title(title)
    axes[3].set_xlabel('Step')
    axes[3].set_ylabel('Squared Terms')
    ax4_twin.set_ylabel('Action Stats')
    axes[3].legend(loc='upper left', fontsize='small')
    ax4_twin.legend(loc='upper right', fontsize='small')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  >> 执行器图表已保存到: {output_path}")


def save_csv(history, output_path):
    """保存评估数据到 CSV（包含内部单位+显示层单位）"""
    fieldnames = [
        'step',
        # 内部单位（MRP / rad）
        'sigma_err_x', 'sigma_err_y', 'sigma_err_z', 'sigma_err_norm',
        'omega_x', 'omega_y', 'omega_z', 'omega_norm',
        'delta_1', 'delta_2', 'delta_3', 'delta_4',
        'delta_dot_1', 'delta_dot_2', 'delta_dot_3', 'delta_dot_4',
        'omega_w_1', 'omega_w_2', 'omega_w_3', 'omega_w_4',
        'omega_w_tilde_1', 'omega_w_tilde_2', 'omega_w_tilde_3', 'omega_w_tilde_4',
        'wheel_bias_sq',
        # 动作原始数据
        'action_1', 'action_2', 'action_3', 'action_4',
        'action_5', 'action_6', 'action_7', 'action_8',
        'action_abs_mean', 'action_sat_rate',
        # Reward breakdown
        'reward_total', 'sigma_err_sq', 'omega_sq', 'action_sq',
        'wheel_bias_sq_info',
        'gimbal_action_sq', 'wheel_action_sq',  # 新增：拆项指标
        # 显示层单位（deg / deg/s）
        'roll_deg', 'pitch_deg', 'yaw_deg',
        'omega_x_deg_s', 'omega_y_deg_s', 'omega_z_deg_s', 'omega_norm_deg_s',
        'delta_1_deg', 'delta_2_deg', 'delta_3_deg', 'delta_4_deg',
        'delta_dot_1_deg_s', 'delta_dot_2_deg_s', 'delta_dot_3_deg_s', 'delta_dot_4_deg_s',
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        n_steps = len(history['step'])
        for i in range(n_steps):
            row = {key: history[key][i] for key in fieldnames}
            writer.writerow(row)

    print(f"  >> CSV数据已保存到: {output_path}")


def save_summary(history, terminated, truncated, model_path, seed, max_steps,
                 png_paths, csv_path, output_path):
    """保存评估结果摘要到文本文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("VSCMG 策略控制效果评估摘要\n")
        f.write("=" * 70 + "\n\n")

        f.write("【配置信息】\n")
        f.write(f"  模型路径:     {model_path}\n")
        f.write(f"  随机种子:     {seed}\n")
        f.write(f"  最大步数:     {max_steps}\n\n")

        f.write("【Episode 结果】\n")
        f.write(f"  实际长度:     {len(history['step'])} 步\n")
        f.write(f"  终止状态:     terminated={terminated}, truncated={truncated}\n\n")

        f.write("【最终状态】\n")
        f.write(f"  ||sigma_err||:       {history['sigma_err_norm'][-1]:.6f}\n")
        f.write(f"  ||omega||:           {history['omega_norm'][-1]:.6f} rad/s\n")
        f.write(f"  ||Omega_tilde||^2:   {history['wheel_bias_sq'][-1]:.6f}\n\n")

        f.write("【动作统计】\n")
        f.write(f"  平均 |action|:       {np.mean(history['action_abs_mean']):.6f}\n")
        f.write(f"  平均饱和率:          {np.mean(history['action_sat_rate']):.2%}\n")
        f.write(f"  平均 gimbal_action_sq: {np.mean(history['gimbal_action_sq']):.6f}\n")
        f.write(f"  平均 wheel_action_sq: {np.mean(history['wheel_action_sq']):.6f}\n\n")

        f.write("【输出文件】\n")
        for png_path in png_paths:
            f.write(f"  {os.path.basename(png_path)}\n")
        f.write(f"  {os.path.basename(csv_path)}\n")
        f.write(f"  {os.path.basename(output_path)} (本文件)\n\n")

        f.write("=" * 70 + "\n")
        f.write("说明：\n")
        f.write("- CSV 文件包含逐步记录的表格数据，每行对应一个时间步\n")
        f.write("- 可用 Excel / pandas / MATLAB 打开进行后续分析\n")
        f.write("=" * 70 + "\n")

    print(f"  >> 摘要已保存到: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="VSCMG 策略控制效果评估工具")
    parser.add_argument("--model", type=str, required=True,
                        help="模型文件路径（必填）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认 42）")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="最大评估步数（默认 1000）")
    parser.add_argument("--device", type=str, default="cpu",
                        help="计算设备 cpu/cuda（默认 cpu）")
    parser.add_argument("--output_dir", type=str, default="eval_outputs",
                        help="输出根目录（默认 eval_outputs）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 配置中文字体
    use_chinese = setup_chinese_font()

    print("=" * 70)
    print("VSCMG 策略控制效果评估工具")
    print("=" * 70)

    # 加载模型
    print(f"\n[1/5] 加载模型: {args.model}")
    actor = load_actor_from_checkpoint(args.model, args.device)

    # 创建环境
    print("[2/5] 创建环境...")
    env = VSCMGEnv()

    # 跑 episode
    print(f"[3/5] 运行评估 (seed={args.seed}, max_steps={args.max_steps})...")
    history, terminated, truncated = run_episode_with_logging(
        env, actor, args.device, seed=args.seed, max_steps=args.max_steps
    )

    # 创建独立输出文件夹（模型名_seed_时间戳）
    model_name = os.path.basename(args.model).replace('.pth', '')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{model_name}_seed{args.seed}_{timestamp}"
    output_folder = os.path.join(args.output_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # 生成文件路径
    spacecraft_png = os.path.join(output_folder, "spacecraft_plots.png")
    actuator_png = os.path.join(output_folder, "actuator_plots.png")
    csv_path = os.path.join(output_folder, "rollout.csv")
    summary_path = os.path.join(output_folder, "summary.txt")

    # 保存结果
    print(f"[4/5] 保存结果到文件夹: {output_folder}")
    model_basename = os.path.basename(args.model)
    plot_spacecraft_metrics(history, spacecraft_png, use_chinese, model_name=model_basename)
    plot_actuator_metrics(history, actuator_png, use_chinese, model_name=model_basename)
    save_csv(history, csv_path)
    save_summary(history, terminated, truncated, args.model, args.seed, args.max_steps,
                 [spacecraft_png, actuator_png], csv_path, summary_path)

    # 控制台汇总
    print("\n" + "=" * 70)
    print("评估结果汇总")
    print("=" * 70)
    print(f"模型路径:          {args.model}")
    print(f"Episode 长度:      {len(history['step'])} 步")
    print(f"终止状态:          terminated={terminated}, truncated={truncated}")
    print(f"最终 ||sigma_err||: {history['sigma_err_norm'][-1]:.6f}")
    print(f"最终 ||omega||:     {history['omega_norm'][-1]:.6f} rad/s")
    print(f"最终 ||Omega_tilde||^2: {history['wheel_bias_sq'][-1]:.6f}")
    print(f"平均 |action|:      {np.mean(history['action_abs_mean']):.6f}")
    print(f"平均饱和率:        {np.mean(history['action_sat_rate']):.2%}")
    print(f"平均 gimbal_action_sq: {np.mean(history['gimbal_action_sq']):.6f}")
    print(f"平均 wheel_action_sq: {np.mean(history['wheel_action_sq']):.6f}")
    print(f"\n输出文件夹: {output_folder}")
    print(f"  - spacecraft_plots.png (航天器状态)")
    print(f"  - actuator_plots.png (执行器状态)")
    print(f"  - rollout.csv (逐步数据)")
    print(f"  - summary.txt (评估摘要)")
    print("=" * 70)