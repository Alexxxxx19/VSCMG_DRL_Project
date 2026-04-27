"""
BC-init TD3 checkpoint 评估 - 预备验证
验证项：
1. reset_fixed_angle 返回 22 维 obs，sigma_err 范数对应指定角度
2. 三种 checkpoint strict=True 加载到真实 TD3 actor
3. 加载后 actor 输出维度正确，action 规模可对比
"""

import numpy as np
import torch
from envs.vscmg_env import VSCMGEnv, quaternion_normalize
from configs.env_config import make_default_config
from agents.td3_agent import TD3


def reset_fixed_angle(env: VSCMGEnv, angle_deg: float, seed: int):
    """
    将环境重置到固定初始角度，返回 env._get_obs() 构造的 22 维 obs。
    固定旋转轴 [0,0,1]，omega=0，delta=0，omega_w=nominal。
    """
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


def main():
    device = "cpu"

    print("=" * 70)
    print("验证 1: reset_fixed_angle -> 22 维 obs, theta_equiv 正确")
    print("=" * 70)

    env_cfg = make_default_config()
    env_cfg.action_mode = "gimbal_only"
    env_cfg.max_gimbal_rate = 0.5
    env = VSCMGEnv(config=env_cfg)

    print(f"env.observation_space.shape = {env.observation_space.shape}")
    assert env.observation_space.shape == (22,)

    for angle_deg in [5, 10, 20, 30, 45]:
        obs = reset_fixed_angle(env, angle_deg, seed=1000)
        sigma_norm = float(np.linalg.norm(obs[0:3]))
        theta_equiv_deg = float(np.degrees(4.0 * np.arctan(sigma_norm)))
        print(f"  angle={angle_deg:>2}° -> obs.shape={obs.shape}, "
              f"||sigma_err||={sigma_norm:.6f}, "
              f"theta_equiv={theta_equiv_deg:.3f}°")

    env.close()

    # =========================================================================
    print("\n" + "=" * 70)
    print("验证 2: BC actor strict=True 加载到真实 TD3 actor")
    print("=" * 70)

    state_dim, action_dim = 22, 4
    bc_path = "_tmp_bc_gimbal_only_actor.pth"
    agent = make_agent(state_dim, action_dim, device)
    sd = load_actor_sd(bc_path, device)
    print(f"  checkpoint keys: {list(sd.keys())[:6]}")
    print(f"  fc3.weight.shape = {sd['fc3.weight'].shape}")

    agent.actor.load_state_dict(sd, strict=True)
    agent.target_actor.load_state_dict(sd, strict=True)
    print("  [PASS] BC actor loaded to agent.actor and agent.target_actor")

    torch.manual_seed(42)
    x = torch.randn(8, state_dim)
    with torch.no_grad():
        out = agent.actor(x)
    print(f"  actor output shape = {out.shape}")
    assert out.shape == (8, action_dim)

    env2 = VSCMGEnv(config=env_cfg)
    obs2 = reset_fixed_angle(env2, 20.0, seed=5000)
    obs2_t = torch.FloatTensor(obs2).unsqueeze(0).to(device)
    with torch.no_grad():
        a2 = agent.actor(obs2_t)
    print(f"  20° obs -> action = {a2.squeeze().tolist()}")
    print(f"  |action| = {float(torch.norm(a2)):.4f}")
    env2.close()

    # =========================================================================
    print("\n" + "=" * 70)
    print("验证 3: checkpoint_10k / final_step_50k strict=True 加载")
    print("=" * 70)

    MODEL_DIR = (
        "models/v0.5.17-dirty_20260427_004215_envs16_seed42_gr0.5_g0.997"
        "_gimbal_only_att1.0_om0.05_wb0.0_ga0.05_wa0.05"
    )

    for name, path_suffix in [
        ("checkpoint_10k", "checkpoint_step_10000.pth"),
        ("final_step_50k", "final_step_50000.pth"),
    ]:
        path = f"{MODEL_DIR}/{path_suffix}"
        sd = load_actor_sd(path, device)
        print(f"  [{name}] fc3.weight.shape = {sd['fc3.weight'].shape}")

        ag = make_agent(state_dim, action_dim, device)
        ag.actor.load_state_dict(sd, strict=True)
        ag.target_actor.load_state_dict(sd, strict=True)
        print(f"  [PASS] {name} loaded strict=True")

        env3 = VSCMGEnv(config=env_cfg)
        obs3 = reset_fixed_angle(env3, 20.0, seed=5000)
        obs3_t = torch.FloatTensor(obs3).unsqueeze(0).to(device)
        with torch.no_grad():
            a3 = ag.actor(obs3_t)
        print(f"  [{name}] 20° obs -> action = {a3.squeeze().tolist()}")
        print(f"  [{name}] |action| = {float(torch.norm(a3)):.4f}")
        env3.close()

    # =========================================================================
    print("\n" + "=" * 70)
    print("验证 4: 三种模型 20° 单步 action_norm 对比 (10 episodes)")
    print("=" * 70)

    agents_dict = {}
    for label, path in [
        ("BC_origin", bc_path),
        ("checkpoint_10k", f"{MODEL_DIR}/checkpoint_step_10000.pth"),
        ("final_step_50k", f"{MODEL_DIR}/final_step_50000.pth"),
    ]:
        ag = make_agent(state_dim, action_dim, device)
        sd = load_actor_sd(path, device)
        ag.actor.load_state_dict(sd, strict=True)
        ag.actor.eval()
        agents_dict[label] = ag

    eval_env = VSCMGEnv(config=env_cfg)
    for label, ag in agents_dict.items():
        norms = []
        for ep in range(10):
            obs = reset_fixed_angle(eval_env, 20.0, seed=1000 + ep)
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                a_out = ag.actor(obs_t)
            norms.append(float(torch.norm(a_out)))
        print(f"  [{label}] action_norm: mean={np.mean(norms):.4f}, "
              f"std={np.std(norms):.4f}, min={np.min(norms):.4f}, max={np.max(norms):.4f}")

    eval_env.close()

    print("\n" + "=" * 70)
    print("预备验证完成")
    print("=" * 70)


if __name__ == "__main__":
    main()