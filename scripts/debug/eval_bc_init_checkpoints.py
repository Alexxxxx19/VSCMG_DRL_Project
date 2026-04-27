"""
BC-init TD3 Checkpoint 完整评估脚本（v2: 修正版）
==================================================

关键修正：使用 env.reset(options={"init_attitude_deg": angle_deg})
与 BC 训练数据生成方式和旧 BC 评估方式保持一致。

双 horizon 输出：
- max_steps=320：对齐训练 episode horizon
- max_steps=1000：对齐旧 BC rollout 评估
"""

import numpy as np
import torch
from envs.vscmg_env import VSCMGEnv
from configs.env_config import make_default_config
from agents.td3_agent import TD3

# =============================================================================
# 配置
# =============================================================================
MODEL_DIR = (
    "models/v0.5.17-dirty_20260427_004215_envs16_seed42_gr0.5_g0.997"
    "_gimbal_only_att1.0_om0.05_wb0.0_ga0.05_wa0.05"
)
BC_PATH = "_tmp_bc_gimbal_only_actor.pth"

CHECKPOINTS = [
    ("BC_origin",       BC_PATH),
    ("checkpoint_10k", f"{MODEL_DIR}/checkpoint_step_10000.pth"),
    ("checkpoint_20k", f"{MODEL_DIR}/checkpoint_step_20000.pth"),
    ("checkpoint_30k", f"{MODEL_DIR}/checkpoint_step_30000.pth"),
    ("checkpoint_40k", f"{MODEL_DIR}/checkpoint_step_40000.pth"),
    ("final_step_50k", f"{MODEL_DIR}/final_step_50000.pth"),
    ("best_ep_reward", f"{MODEL_DIR}/best_episode_reward.pth"),
]

TEST_ANGLES_DEG = [5, 10, 20, 30, 45]
EPISODES_PER_ANGLE = 30
HORIZONS = [320, 1000]

CONVERGE_THRESHOLDS = {
    5:  1.0,
    10: 2.0,
    20: 3.0,
    30: 4.5,
    45: 6.75,
}


# =============================================================================
# 辅助函数
# =============================================================================
def sigma_to_theta_deg(sigma):
    """MRP sigma 向量 -> 等效旋转角 (度)"""
    return float(np.degrees(4.0 * np.arctan(np.linalg.norm(sigma))))


def make_agent(state_dim, action_dim, device="cpu"):
    return TD3(
        state_dim=state_dim, action_dim=action_dim, hidden_dim=256,
        action_bound=1.0, sigma=0.0, tau=0.005, gamma=0.997,
        critic_lr=3e-4, actor_lr=1e-4, delay=2,
        policy_noise=0.0, noise_clip=0.0, device=device,
    )


def load_actor_sd(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    if "actor" in ckpt:
        return ckpt["actor"]
    elif "actor_state_dict" in ckpt:
        return ckpt["actor_state_dict"]
    elif "actor_target" in ckpt:
        return ckpt["actor_target"]
    else:
        raise ValueError(f"Unknown keys: {list(ckpt.keys())}")


def load_actor_agent(path, state_dim, action_dim, device="cpu"):
    agent = make_agent(state_dim, action_dim, device)
    sd = load_actor_sd(path, device)
    agent.actor.load_state_dict(sd, strict=True)
    agent.target_actor.load_state_dict(sd, strict=True)
    agent.actor.eval()
    return agent


# =============================================================================
# 单 episode rollout
# =============================================================================
def rollout(env, actor_net, angle_deg, seed, max_steps, device="cpu"):
    """
    使用 env.reset(options={"init_attitude_deg": ...}) 重置环境。
    与 BC 训练数据生成方式完全一致。
    """
    obs, _ = env.reset(seed=seed, options={"init_attitude_deg": angle_deg})
    init_theta_deg = sigma_to_theta_deg(obs[0:3])

    actions_list = []
    cum_reward = 0.0
    steps = 0
    terminated = truncated = False

    while not (terminated or truncated) and steps < max_steps:
        s_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action = actor_net(s_tensor).cpu().numpy().squeeze(0)
        action = np.clip(action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward
        actions_list.append(action)
        steps += 1

    final_theta_deg = sigma_to_theta_deg(obs[0:3])
    actions_arr = np.array(actions_list) if actions_list else np.zeros((1, 4))
    omega_final = float(np.linalg.norm(obs[3:6]))

    return {
        "init_theta_deg": init_theta_deg,
        "final_theta_deg": final_theta_deg,
        "theta_improvement_deg": init_theta_deg - final_theta_deg,
        "cumulative_reward": cum_reward,
        "action_abs_mean": float(np.mean(np.abs(actions_arr))),
        "action_sat_rate": float(np.mean(np.abs(actions_arr) >= 0.95)),
        "omega_final": omega_final,
        "steps": steps,
    }


# =============================================================================
# 主评估
# =============================================================================
def main():
    device = "cpu"

    env_cfg = make_default_config()
    env_cfg.action_mode = "gimbal_only"
    env_cfg.max_gimbal_rate = 0.5

    dummy_env = VSCMGEnv(config=env_cfg)
    state_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.shape[0]
    del dummy_env
    print(f"state_dim={state_dim}, action_dim={action_dim}")

    # 加载所有模型
    print("Loading actors...")
    actors = {}
    for name, path in CHECKPOINTS:
        actors[name] = load_actor_agent(path, state_dim, action_dim, device)
        print(f"  {name}: OK")

    # 存储所有结果
    all_results = {}

    for max_steps in HORIZONS:
        print(f"\n{'=' * 120}")
        print(f"=== max_steps = {max_steps} ===")
        print(f"{'=' * 120}")

        results = {}

        for angle_deg in TEST_ANGLES_DEG:
            print(f"\n--- angle={angle_deg}° ---")

            # zero_action baseline
            env_zero = VSCMGEnv(config=env_cfg)
            zero_improvements = []
            for ep in range(EPISODES_PER_ANGLE):
                seed = 1000 + angle_deg * 1000 + ep
                obs, _ = env_zero.reset(seed=seed, options={"init_attitude_deg": angle_deg})
                init_theta = sigma_to_theta_deg(obs[0:3])
                s = obs.copy()
                terminated = truncated = False
                steps = 0
                while not (terminated or truncated) and steps < max_steps:
                    s, _, terminated, truncated, _ = env_zero.step(np.zeros(4))
                    steps += 1
                final_theta = sigma_to_theta_deg(s[0:3])
                zero_improvements.append(init_theta - final_theta)
            zero_improve_mean = float(np.mean(zero_improvements))
            print(f"  [ZeroAction] theta_improve_mean={zero_improve_mean:.4f}°")
            env_zero.close()

            # 各模型 rollout
            for ckpt_name, agent in actors.items():
                ep_data = {
                    "final_theta_deg": [], "theta_improvement_deg": [],
                    "cumulative_reward": [], "action_abs_mean": [],
                    "action_sat_rate": [], "omega_final": [],
                }

                for ep in range(EPISODES_PER_ANGLE):
                    seed = 1000 + angle_deg * 1000 + ep
                    env = VSCMGEnv(config=env_cfg)
                    r = rollout(env, agent.actor, angle_deg, seed, max_steps, device)
                    env.close()
                    ep_data["final_theta_deg"].append(r["final_theta_deg"])
                    ep_data["theta_improvement_deg"].append(r["theta_improvement_deg"])
                    ep_data["cumulative_reward"].append(r["cumulative_reward"])
                    ep_data["action_abs_mean"].append(r["action_abs_mean"])
                    ep_data["action_sat_rate"].append(r["action_sat_rate"])
                    ep_data["omega_final"].append(r["omega_final"])

                ft = np.array(ep_data["final_theta_deg"])
                ti = np.array(ep_data["theta_improvement_deg"])
                cr = np.array(ep_data["cumulative_reward"])
                aa = np.array(ep_data["action_abs_mean"])
                sr = np.array(ep_data["action_sat_rate"])
                om = np.array(ep_data["omega_final"])

                threshold = CONVERGE_THRESHOLDS[angle_deg]
                converged = bool(np.mean(ft) < threshold)
                beats_zero = bool(np.mean(ti) > zero_improve_mean)

                key = (ckpt_name, angle_deg, max_steps)
                all_results[key] = {
                    "final_theta_mean": float(np.mean(ft)),
                    "final_theta_std": float(np.std(ft)),
                    "theta_improvement_mean": float(np.mean(ti)),
                    "theta_improvement_std": float(np.std(ti)),
                    "cumulative_reward_mean": float(np.mean(cr)),
                    "cumulative_reward_std": float(np.std(cr)),
                    "action_abs_mean": float(np.mean(aa)),
                    "action_sat_rate": float(np.mean(sr)),
                    "omega_final_mean": float(np.mean(om)),
                    "beats_zero_action": beats_zero,
                    "converges": converged,
                    "converge_threshold": threshold,
                    "zero_improve_mean": zero_improve_mean,
                }

                print(f"  [{ckpt_name}] "
                      f"fin_theta={np.mean(ft):.3f}±{np.std(ft):.3f}° "
                      f"improve={np.mean(ti):.3f}° "
                      f"reward={np.mean(cr):.1f} "
                      f"act_abs={np.mean(aa):.4f} "
                      f"sat={np.mean(sr):.4f} "
                      f"omega={np.mean(om):.4f} "
                      f"beats0={beats_zero} conv={converged}")

    # =========================================================================
    # 汇总表格
    # =========================================================================
    for max_steps in HORIZONS:
        print(f"\n{'=' * 140}")
        print(f"=== 汇总表 max_steps={max_steps} ===")
        print(f"{'=' * 140}")

        header = (
            f"{'Model':<20} {'Angle':>5} "
            f"{'fin_theta':>10} {'±':>6} "
            f"{'improve':>9} {'cum_rew':>10} "
            f"{'act_abs':>8} {'sat_rate':>9} {'omega':>8} "
            f"{'beats0':>6} {'conv':>5}"
        )

        for angle_deg in TEST_ANGLES_DEG:
            print(f"\n--- angle={angle_deg}° (threshold={CONVERGE_THRESHOLDS[angle_deg]}°) ---")
            print(header)
            print("-" * 140)
            for ckpt_name, _ in CHECKPOINTS:
                r = all_results[(ckpt_name, angle_deg, max_steps)]
                print(
                    f"{ckpt_name:<20} {angle_deg:>5} "
                    f"{r['final_theta_mean']:>10.3f} {r['final_theta_std']:>6.3f} "
                    f"{r['theta_improvement_mean']:>9.3f} {r['cumulative_reward_mean']:>10.1f} "
                    f"{r['action_abs_mean']:>8.4f} {r['action_sat_rate']:>9.4f} {r['omega_final_mean']:>8.4f} "
                    f"{'Y' if r['beats_zero_action'] else 'N':>6} "
                    f"{'Y' if r['converges'] else 'N':>5}"
                )

    # =========================================================================
    # 关键结论
    # =========================================================================
    print("\n" + "=" * 140)
    print("=== 关键结论 ===")
    print("=" * 140)

    for max_steps in HORIZONS:
        print(f"\n--- max_steps = {max_steps} ---")

        bc_final = {deg: all_results[("BC_origin", deg, max_steps)]["final_theta_mean"]
                    for deg in TEST_ANGLES_DEG}

        # Q1
        bc_is_best = True
        for ckpt_name, _ in CHECKPOINTS[1:]:
            for deg in TEST_ANGLES_DEG:
                if all_results[(ckpt_name, deg, max_steps)]["final_theta_mean"] < bc_final[deg] - 0.3:
                    bc_is_best = False
        print(f"  Q1: BC 是否仍然最好? -> {'是' if bc_is_best else '否（存在更优 checkpoint）'}")

        # Q2
        r10k = all_results[("checkpoint_10k", 20, max_steps)]
        rBC = all_results[("BC_origin", 20, max_steps)]
        degraded = (r10k["action_abs_mean"] > rBC["action_abs_mean"] * 2
                    or r10k["final_theta_mean"] > rBC["final_theta_mean"] + 1.0)
        print(f"  Q2: checkpoint_10k 是否已退化? -> {'是' if degraded else '否'}")
        print(f"      BC: fin_theta={rBC['final_theta_mean']:.3f}°, act_abs={rBC['action_abs_mean']:.4f}")
        print(f"      10k: fin_theta={r10k['final_theta_mean']:.3f}°, act_abs={r10k['action_abs_mean']:.4f}")

        # Q3
        best_r = all_results[("best_ep_reward", 20, max_steps)]
        final_r = all_results[("final_step_50k", 20, max_steps)]
        best_better = best_r["final_theta_mean"] < final_r["final_theta_mean"] - 0.3
        print(f"  Q3: best_ep_reward 是否比 final 好? -> "
              f"{'是' if best_better else '否'} "
              f"(best={best_r['final_theta_mean']:.3f}°, final={final_r['final_theta_mean']:.3f}°)")

        # Q4
        all_degraded = all(
            all_results[(ckpt_name, 20, max_steps)]["action_abs_mean"] > 0.15
            for ckpt_name, _ in CHECKPOINTS[1:]
        )
        print(f"  Q4: 所有 TD3 checkpoint 是否走向大动作? -> {'是' if all_degraded else '否'}")

        # Q5
        beats_bc_list = []
        for ckpt_name, _ in CHECKPOINTS[1:]:
            for deg in TEST_ANGLES_DEG:
                if all_results[(ckpt_name, deg, max_steps)]["final_theta_mean"] < bc_final[deg] - 0.5:
                    beats_bc_list.append(f"{ckpt_name}@{deg}°")
        print(f"  Q5: 是否存在超越 BC 的 checkpoint? -> "
              f"{'否' if not beats_bc_list else '是: ' + ', '.join(beats_bc_list)}")

        # Q6
        print(f"\n  --- action_abs_mean / sat_rate 汇总 (20°, max_steps={max_steps}) ---")
        for ckpt_name, _ in CHECKPOINTS:
            r = all_results[(ckpt_name, 20, max_steps)]
            print(f"    {ckpt_name:<20} act_abs={r['action_abs_mean']:.4f}  sat_rate={r['action_sat_rate']:.4f}")

    print("\n" + "=" * 140)
    print("评估完成")
    print("=" * 140)


if __name__ == "__main__":
    main()