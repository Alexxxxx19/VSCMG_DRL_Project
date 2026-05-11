"""
P57-1e Senior-Inspired Reward Rollout Sanity Check (正式接入版)
================================================================
验证 progress-only senior_inspired reward_mode 在 env 中正式生效后的表现。

用法:
    python scripts/diagnostics/check_senior_inspired_reward_rollout.py

输出:
    eval_senior_inspired_rollout_<timestamp>/
        summary.csv   — 汇总表

PASS 标准:
    1. 四个角度 BC cumulative_reward > zero_action
    2. 四个角度 BC final_theta_deg < zero_action
    3. unsafe_count = 0
    4. abs(cumulative_reward) <= 1000
    5. default reward 未改变
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
import csv
from math import atan, degrees, sqrt
from datetime import datetime

from configs.env_config import make_default_config
from envs.vscmg_env import VSCMGEnv, RewardConfig
from agents.td3_agent import PolicyNet


# ── 配置 ──────────────────────────────────────────────
BC_CKPT = "_tmp_bc_gimbal_only_actor.pth"
ANGLES = [10, 20, 30, 45]
CONTROLLERS = ["zero", "bc"]
MAX_STEPS = 1000
SEED = 42


# ── 辅助函数 ──────────────────────────────────────────

def theta_deg(obs):
    s = obs[:3]
    return degrees(4 * atan(sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2)))


def make_env_senior(angle_deg):
    """创建 gimbal_only + senior_inspired reward 环境."""
    cfg = make_default_config()
    cfg.action_mode = "gimbal_only"
    cfg.max_gimbal_rate = 0.5
    rcfg = RewardConfig()
    rcfg.reward_mode = "senior_inspired"
    env = VSCMGEnv(config=cfg, reward_cfg=rcfg)
    obs, _ = env.reset(seed=SEED, options={"init_attitude_deg": float(angle_deg)})
    return env, obs


def make_env_default(angle_deg):
    """创建 gimbal_only + default reward 环境（用于确认 default 未变）."""
    cfg = make_default_config()
    cfg.action_mode = "gimbal_only"
    cfg.max_gimbal_rate = 0.5
    rcfg = RewardConfig()
    env = VSCMGEnv(config=cfg, reward_cfg=rcfg)
    obs, _ = env.reset(seed=SEED, options={"init_attitude_deg": float(angle_deg)})
    return env, obs


def load_bc_actor(ckpt_path):
    """加载 BC actor."""
    ck = torch.load(ckpt_path, map_location="cpu")
    if "actor" in ck:
        actor_sd = ck["actor"]
    elif "actor_state_dict" in ck:
        actor_sd = ck["actor_state_dict"]
    else:
        raise KeyError(
            f"Checkpoint {ckpt_path} has no 'actor' or 'actor_state_dict' key. "
            f"Available keys: {list(ck.keys())}"
        )
    actor = PolicyNet(22, 256, 4, 1.0)
    actor.load_state_dict(actor_sd, strict=True)
    actor.eval()
    return actor


def rollout(env, obs, actor, controller, max_steps):
    """Deterministic rollout."""
    records = []
    cr = 0.0

    for step in range(max_steps):
        if controller == "bc":
            st = torch.FloatTensor(obs).unsqueeze(0)
            ac = actor(st).squeeze(0).detach().numpy()
        else:
            ac = np.zeros(4, dtype=np.float64)

        obs_next, r, term, trunc, info = env.step(ac)
        cr += r

        records.append({
            "reward": r,
            "cumulative_reward": cr,
            "theta_deg": theta_deg(obs_next),
            "omega_norm": float(np.linalg.norm(obs_next[3:6])),
            "action_abs_mean": float(np.abs(ac).mean()),
            "action_abs_max": float(np.abs(ac).max()),
            "senior_r1": info.get("senior_r1", 0.0),
            "senior_r2": info.get("senior_r2", 0.0),
            "senior_r3": info.get("senior_r3", 0.0),
            "senior_progress": info.get("senior_progress", 0.0),
            "senior_progress_reward": info.get("senior_progress_reward", 0.0),
            "senior_control_sq": info.get("senior_control_sq", 0.0),
            "senior_near_goal": info.get("senior_near_goal", 0.0),
            "senior_unsafe": info.get("senior_unsafe", 0.0),
            "done": bool(term or trunc),
        })

        obs = obs_next
        if term or trunc:
            break

    return records


# ── 主流程 ────────────────────────────────────────────

def main():
    ts = datetime.now().strftime("%m%d_%H%M")
    outdir = f"eval_senior_inspired_rollout_{ts}"
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}/")

    actor = load_bc_actor(BC_CKPT)
    print(f"BC actor loaded from {BC_CKPT}")

    # 1. 验证 default reward 未改变（10° 一个 rollout 即可）
    print("\n=== CHECKING DEFAULT REWARD (sanity) ===")
    env_def, obs_def = make_env_default(10)
    rec_def = rollout(env_def, obs_def, actor, "zero", MAX_STEPS)
    env_def.close()
    def_cr = rec_def[-1]["cumulative_reward"]
    print(f"  default reward (10 deg zero): {def_cr:.4f}")
    # P53-2 已知 baseline: 10° zero ~ -40 左右（无 progress）
    # 带 progress 的 default: 10° zero 应该约 -40
    # 如果明显不同说明 default 被意外改动了
    if abs(def_cr) > 500:
        print(f"  WARNING: default reward seems abnormal (abs > 500). Default may be corrupted.")
    else:
        print(f"  default reward looks reasonable.")

    # 2. Senior-inspired rollout
    print("\n=== ROLLOUT (senior_inspired reward_mode) ===")
    summary_rows = []

    for angle in ANGLES:
        for ctrl in CONTROLLERS:
            env, obs = make_env_senior(angle)
            records = rollout(env, obs, actor, ctrl, MAX_STEPS)
            env.close()

            thetas = [r["theta_deg"] for r in records]
            omegas = [r["omega_norm"] for r in records]
            ameans = [r["action_abs_mean"] for r in records]
            sat_rate = sum(1 for r in records if r["action_abs_max"] >= 0.95) / len(records)

            sr1_sum = sum(r["senior_r1"] for r in records)
            sr2_sum = sum(r["senior_r2"] for r in records)
            sr3_sum = sum(r["senior_r3"] for r in records)
            sp_sum = sum(r["senior_progress_reward"] for r in records)
            ctrl_sq_sum = sum(r["senior_control_sq"] for r in records)
            near_goal_count = sum(1 for r in records if r["senior_near_goal"] > 0.5)
            unsafe_count = sum(1 for r in records if r["senior_unsafe"] > 0.5)

            env2, obs0 = make_env_senior(angle)
            t0 = theta_deg(obs0)
            env2.close()

            row = {
                "init_attitude_deg": angle,
                "controller": ctrl,
                "cumulative_reward": records[-1]["cumulative_reward"],
                "theta0_deg": t0,
                "final_theta_deg": thetas[-1],
                "min_theta_deg": min(thetas),
                "omega_max": max(omegas),
                "action_abs_mean": float(np.mean(ameans)),
                "sat_rate": sat_rate,
                "senior_r1_sum": sr1_sum,
                "senior_progress_reward_sum": sp_sum,
                "senior_r2_sum": sr2_sum,
                "senior_r3_sum": sr3_sum,
                "senior_control_sq_sum": ctrl_sq_sum,
                "near_goal_count": near_goal_count,
                "unsafe_count": unsafe_count,
                "nsteps": len(records),
            }
            summary_rows.append(row)

            print(
                f"  {ctrl:>4} angle={angle:2d}  "
                f"cr={row['cumulative_reward']:10.2f}  "
                f"final_theta={thetas[-1]:7.2f}  "
                f"min_theta={min(thetas):7.2f}  "
                f"omega_max={max(omegas):.4f}  "
                f"sr1={sr1_sum:10.2f}  sp={sp_sum:10.2f}  "
                f"near_goal={near_goal_count}  unsafe={unsafe_count}"
            )

    # 3. 写 CSV
    csv_path = os.path.join(outdir, "summary.csv")
    fields = [
        "init_attitude_deg", "controller", "cumulative_reward",
        "theta0_deg", "final_theta_deg", "min_theta_deg",
        "omega_max", "action_abs_mean", "sat_rate",
        "senior_r1_sum", "senior_progress_reward_sum",
        "senior_r2_sum", "senior_r3_sum", "senior_control_sq_sum",
        "near_goal_count", "unsafe_count", "nsteps",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in summary_rows:
            w.writerow([r[k] for k in fields])

    # 4. 打印汇总表
    print()
    print("=" * 100)
    print("SENIOR-INSPIRED REWARD ROLLOUT SUMMARY")
    print("=" * 100)
    print(
        f"{'angle':>5} {'ctrl':>4} {'cr':>10} {'theta0':>8} {'final':>8} {'min':>8} "
        f"{'om_max':>8} {'sr1':>10} {'sp':>10} {'r3':>8} {'ng':>4} {'un':>3}"
    )
    print("-" * 100)
    for r in summary_rows:
        print(
            f"{r['init_attitude_deg']:5d} {r['controller']:>4} "
            f"{r['cumulative_reward']:10.2f} {r['theta0_deg']:8.2f} "
            f"{r['final_theta_deg']:8.2f} {r['min_theta_deg']:8.2f} "
            f"{r['omega_max']:8.4f} "
            f"{r['senior_r1_sum']:10.2f} {r['senior_progress_reward_sum']:10.2f} "
            f"{r['senior_r3_sum']:8.2f} "
            f"{r['near_goal_count']:4d} {r['unsafe_count']:3d}"
        )

    # 5. Sanity 判断
    print()
    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    all_pass = True
    for angle in ANGLES:
        bc = next(r for r in summary_rows if r["init_attitude_deg"] == angle and r["controller"] == "bc")
        zero = next(r for r in summary_rows if r["init_attitude_deg"] == angle and r["controller"] == "zero")

        cr_ok = bc["cumulative_reward"] > zero["cumulative_reward"]
        ft_ok = bc["final_theta_deg"] < zero["final_theta_deg"]
        unsafe_ok = bc["unsafe_count"] == 0 and zero["unsafe_count"] == 0
        scale_ok = all(abs(r["cumulative_reward"]) <= 1000 for r in [bc, zero])
        default_ok = abs(def_cr) <= 500  # rough check

        status = "PASS" if (cr_ok and ft_ok and unsafe_ok and scale_ok and default_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(
            f"  angle={angle:2d}: cr_bc>zero={'Y' if cr_ok else 'N'}  "
            f"ft_bc<zero={'Y' if ft_ok else 'N'}  "
            f"unsafe={'Y' if unsafe_ok else 'N'}  "
            f"scale={'Y' if scale_ok else 'N'}  "
            f"default={'Y' if default_ok else 'N'}  "
            f"->{status}"
        )

    print()
    if all_pass:
        print("ALL SANITY CHECKS PASSED -> OK to proceed to P57-2 short 5k A/B training.")
    else:
        print("SOME CHECKS FAILED -> DO NOT train. Investigate first.")

    print(f"\nSummary CSV: {csv_path}")


if __name__ == "__main__":
    main()
