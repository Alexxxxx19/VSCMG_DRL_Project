"""
BC-init 评估一致性审计脚本
==========================

三项审计：
A. Oracle baseline 在 reset_fixed_angle_random_axis 下的表现
B. 对比旧评估方式（env.reset + options）vs 新评估方式（手动修改 env 状态）
C. BC_origin fixed_axis vs random_axis 对比
"""

import numpy as np
import torch
from envs.vscmg_env import (
    VSCMGEnv, quaternion_normalize,
    quaternion_to_mrp,
    compute_orientation_error_quaternion,
    orientation_error_quaternion_to_sigma_err,
)
from configs.env_config import make_default_config
from agents.td3_agent import TD3
from geometry.pyramid_config import PyramidVSCMG

# =============================================================================
# 配置
# =============================================================================
TEST_ANGLES_DEG = [5, 10, 20, 30, 45]
EPISODES_PER_ANGLE = 30
VSCMG = PyramidVSCMG()

# Oracle config
SIGN_P = -1; SIGN_D = +1; KP = 10; KD = 0.5; LAMBDA = 1e-4

BC_PATH = "_tmp_bc_gimbal_only_actor.pth"
MODEL_DIR = (
    "models/v0.5.17-dirty_20260427_004215_envs16_seed42_gr0.5_g0.997"
    "_gimbal_only_att1.0_om0.05_wb0.0_ga0.05_wa0.05"
)


# =============================================================================
# 辅助函数
# =============================================================================
def sigma_to_theta_deg(sigma):
    return float(np.degrees(4.0 * np.arctan(np.linalg.norm(sigma))))


def q_to_theta_deg(q):
    s = quaternion_to_mrp(q)
    return sigma_to_theta_deg(s)


def reset_fixed_angle_random_axis(env, angle_deg, seed):
    """新评估方式：reset 后手动修改内部状态 + 随机轴"""
    obs, _ = env.reset(seed=seed)
    angle_rad = np.radians(angle_deg)
    rng = np.random.default_rng(seed + 9999)
    axis = rng.standard_normal(3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    q_err = np.array([
        np.cos(angle_rad / 2.0),
        axis[0] * np.sin(angle_rad / 2.0),
        axis[1] * np.sin(angle_rad / 2.0),
        axis[2] * np.sin(angle_rad / 2.0),
    ])
    env.q = quaternion_normalize(q_err)
    env.omega = np.zeros(3)
    env.delta = np.zeros(4)
    env.omega_w = env.omega_w_nominal * np.ones(4)
    env._delta_dot_cache = np.zeros(4)
    obs = env._get_obs().astype(np.float32)
    assert obs.shape == (22,)
    theta_check = sigma_to_theta_deg(obs[0:3])
    if abs(theta_check - angle_deg) > 0.2:
        raise RuntimeError(f"theta_check={theta_check:.3f}° != target={angle_deg}°")
    return obs


def reset_via_env_options(env, angle_deg, seed):
    """旧评估方式：使用 env.reset(options={"init_attitude_deg": ...})"""
    obs, _ = env.reset(seed=seed, options={"init_attitude_deg": angle_deg})
    return obs


def reset_fixed_axis(env, angle_deg, seed):
    """固定轴 [0,0,1] 版本"""
    obs, _ = env.reset(seed=seed)
    angle_rad = np.radians(angle_deg)
    axis = np.array([0.0, 0.0, 1.0])
    q_err = np.array([
        np.cos(angle_rad / 2.0),
        axis[0] * np.sin(angle_rad / 2.0),
        axis[1] * np.sin(angle_rad / 2.0),
        axis[2] * np.sin(angle_rad / 2.0),
    ])
    env.q = quaternion_normalize(q_err)
    env.omega = np.zeros(3)
    env.delta = np.zeros(4)
    env.omega_w = env.omega_w_nominal * np.ones(4)
    env._delta_dot_cache = np.zeros(4)
    obs = env._get_obs().astype(np.float32)
    return obs


def make_env():
    cfg = make_default_config()
    cfg.action_mode = "gimbal_only"
    cfg.max_gimbal_rate = 0.5
    return VSCMGEnv(config=cfg)


def make_bc_actor():
    agent = TD3(
        state_dim=22, action_dim=4, hidden_dim=256,
        action_bound=1.0, sigma=0.0, tau=0.005, gamma=0.997,
        critic_lr=3e-4, actor_lr=1e-4, delay=2,
        policy_noise=0.0, noise_clip=0.0, device="cpu",
    )
    ckpt = torch.load(BC_PATH, map_location="cpu")
    sd = ckpt["actor"] if "actor" in ckpt else ckpt["actor_state_dict"]
    agent.actor.load_state_dict(sd, strict=True)
    agent.actor.eval()
    return agent.actor


def oracle_action(env):
    q_err = compute_orientation_error_quaternion(env.q, env.q_target)
    sigma_err = orientation_error_quaternion_to_sigma_err(q_err)
    omega = env.omega.copy()
    tau_ref = SIGN_P * KP * sigma_err + SIGN_D * KD * omega
    delta = env.delta.copy()
    h_w = env.h_w.copy()
    A_t = VSCMG.get_transverse_matrix(delta)
    h_w_safe = np.where(np.abs(h_w) < 1e-10, np.where(h_w >= 0, 1e-10, -1e-10), h_w)
    M = A_t * h_w_safe[np.newaxis, :]
    MMT = M @ M.T + LAMBDA * np.eye(3)
    try:
        inv_MMT = np.linalg.inv(MMT)
    except np.linalg.LinAlgError:
        return np.zeros(4)
    delta_dot_cmd = M.T @ inv_MMT @ tau_ref
    return np.clip(delta_dot_cmd / 0.5, -1.0, 1.0)


def rollout_with_reset_fn(env, action_fn, angle_deg, seed, reset_fn, max_steps):
    obs = reset_fn(env, angle_deg, seed)
    init_theta = sigma_to_theta_deg(obs[0:3])
    actions_list = []
    cum_reward = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated) and steps < max_steps:
        action = action_fn(env, obs)
        action = np.clip(action, -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward
        actions_list.append(action)
        steps += 1
    final_theta = sigma_to_theta_deg(obs[0:3])
    actions_arr = np.array(actions_list) if actions_list else np.zeros((1, 4))
    return {
        "init_theta": init_theta,
        "final_theta": final_theta,
        "improvement": init_theta - final_theta,
        "cum_reward": cum_reward,
        "action_abs_mean": float(np.mean(np.abs(actions_arr))),
        "action_sat_rate": float(np.mean(np.abs(actions_arr) >= 0.95)),
        "omega_final": float(np.linalg.norm(obs[3:6])),
        "steps": steps,
    }


def oracle_action_wrapper(env, obs):
    return oracle_action(env)


def make_bc_action_wrapper(bc_actor):
    def fn(env, obs):
        with torch.no_grad():
            a = bc_actor(torch.FloatTensor(obs).unsqueeze(0)).cpu().numpy().squeeze(0)
        return a
    return fn


def zero_action_wrapper(env, obs):
    return np.zeros(4)


# =============================================================================
# 主逻辑
# =============================================================================
def main():
    bc_actor = make_bc_actor()
    bc_action_fn = make_bc_action_wrapper(bc_actor)

    # =========================================================================
    # A. Oracle 在新评估方式 (reset_fixed_angle_random_axis) 下的表现
    # =========================================================================
    print("=" * 100)
    print("A. Oracle baseline — reset_fixed_angle_random_axis, max_steps=320")
    print("=" * 100)
    print(f"{'Angle':>6} {'fin_theta':>10} {'±':>6} {'improve':>9} "
          f"{'cum_rew':>10} {'act_abs':>8} {'sat_rate':>9} {'omega':>8} {'steps':>6}")
    print("-" * 100)

    for angle_deg in TEST_ANGLES_DEG:
        results = []
        for ep in range(EPISODES_PER_ANGLE):
            seed = 1000 + angle_deg * 1000 + ep
            env = make_env()
            r = rollout_with_reset_fn(env, oracle_action_wrapper, angle_deg, seed,
                                      reset_fixed_angle_random_axis, max_steps=320)
            results.append(r)
            env.close()

        ft = np.array([r["final_theta"] for r in results])
        ti = np.array([r["improvement"] for r in results])
        cr = np.array([r["cum_reward"] for r in results])
        aa = np.array([r["action_abs_mean"] for r in results])
        sr = np.array([r["action_sat_rate"] for r in results])
        om = np.array([r["omega_final"] for r in results])
        st = np.array([r["steps"] for r in results])

        print(f"{angle_deg:>6} {np.mean(ft):>10.3f} {np.std(ft):>6.3f} {np.mean(ti):>9.3f} "
              f"{np.mean(cr):>10.1f} {np.mean(aa):>8.4f} {np.mean(sr):>9.4f} "
              f"{np.mean(om):>8.4f} {np.mean(st):>6.0f}")

    # A2: Oracle max_steps=1000 (与旧评估一致)
    print("\n" + "=" * 100)
    print("A2. Oracle baseline — reset_fixed_angle_random_axis, max_steps=1000")
    print("=" * 100)
    print(f"{'Angle':>6} {'fin_theta':>10} {'±':>6} {'improve':>9} "
          f"{'cum_rew':>10} {'act_abs':>8} {'sat_rate':>9} {'omega':>8} {'steps':>6}")
    print("-" * 100)

    for angle_deg in TEST_ANGLES_DEG:
        results = []
        for ep in range(EPISODES_PER_ANGLE):
            seed = 1000 + angle_deg * 1000 + ep
            env = make_env()
            r = rollout_with_reset_fn(env, oracle_action_wrapper, angle_deg, seed,
                                      reset_fixed_angle_random_axis, max_steps=1000)
            results.append(r)
            env.close()

        ft = np.array([r["final_theta"] for r in results])
        ti = np.array([r["improvement"] for r in results])
        cr = np.array([r["cum_reward"] for r in results])
        aa = np.array([r["action_abs_mean"] for r in results])
        sr = np.array([r["action_sat_rate"] for r in results])
        om = np.array([r["omega_final"] for r in results])
        st = np.array([r["steps"] for r in results])

        print(f"{angle_deg:>6} {np.mean(ft):>10.3f} {np.std(ft):>6.3f} {np.mean(ti):>9.3f} "
              f"{np.mean(cr):>10.1f} {np.mean(aa):>8.4f} {np.mean(sr):>9.4f} "
              f"{np.mean(om):>8.4f} {np.mean(st):>6.0f}")

    # =========================================================================
    # B. 对比旧评估方式 vs 新评估方式 (BC actor, max_steps=1000)
    # =========================================================================
    print("\n" + "=" * 100)
    print("B. BC actor — 旧方式 (env.reset options) vs 新方式 (手动修改状态)")
    print("   max_steps=1000, 随机轴")
    print("=" * 100)
    print(f"{'Angle':>6} {'Method':>25} {'fin_theta':>10} {'±':>6} {'improve':>9} "
          f"{'act_abs':>8} {'sat_rate':>9} {'steps':>6}")
    print("-" * 100)

    for angle_deg in TEST_ANGLES_DEG:
        for method_name, reset_fn in [
            ("旧:env.reset+options", reset_via_env_options),
            ("新:手动修改+随机轴", reset_fixed_angle_random_axis),
        ]:
            results = []
            for ep in range(EPISODES_PER_ANGLE):
                seed = 1000 + angle_deg * 1000 + ep
                env = make_env()
                r = rollout_with_reset_fn(env, bc_action_fn, angle_deg, seed,
                                          reset_fn, max_steps=1000)
                results.append(r)
                env.close()

            ft = np.array([r["final_theta"] for r in results])
            ti = np.array([r["improvement"] for r in results])
            aa = np.array([r["action_abs_mean"] for r in results])
            sr = np.array([r["action_sat_rate"] for r in results])
            st = np.array([r["steps"] for r in results])

            print(f"{angle_deg:>6} {method_name:>25} {np.mean(ft):>10.3f} {np.std(ft):>6.3f} "
                  f"{np.mean(ti):>9.3f} {np.mean(aa):>8.4f} {np.mean(sr):>9.4f} {np.mean(st):>6.0f}")

    # =========================================================================
    # C. BC_origin: fixed_axis vs random_axis (max_steps=1000)
    # =========================================================================
    print("\n" + "=" * 100)
    print("C. BC_origin — fixed_axis vs random_axis, max_steps=1000")
    print("=" * 100)
    print(f"{'Angle':>6} {'Axis':>12} {'fin_theta':>10} {'±':>6} {'improve':>9} "
          f"{'act_abs':>8} {'sat_rate':>9} {'omega':>8}")
    print("-" * 100)

    for angle_deg in [10, 20, 30, 45]:
        for axis_name, reset_fn in [
            ("fixed", reset_fixed_axis),
            ("random", reset_fixed_angle_random_axis),
        ]:
            results = []
            for ep in range(EPISODES_PER_ANGLE):
                seed = 1000 + angle_deg * 1000 + ep
                env = make_env()
                r = rollout_with_reset_fn(env, bc_action_fn, angle_deg, seed,
                                          reset_fn, max_steps=1000)
                results.append(r)
                env.close()

            ft = np.array([r["final_theta"] for r in results])
            ti = np.array([r["improvement"] for r in results])
            aa = np.array([r["action_abs_mean"] for r in results])
            sr = np.array([r["action_sat_rate"] for r in results])
            om = np.array([r["omega_final"] for r in results])

            print(f"{angle_deg:>6} {axis_name:>12} {np.mean(ft):>10.3f} {np.std(ft):>6.3f} "
                  f"{np.mean(ti):>9.3f} {np.mean(aa):>8.4f} {np.mean(sr):>9.4f} {np.mean(om):>8.4f}")

    print("\n" + "=" * 100)
    print("审计完成")
    print("=" * 100)


if __name__ == "__main__":
    main()