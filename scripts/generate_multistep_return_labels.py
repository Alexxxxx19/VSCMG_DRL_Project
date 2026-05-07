"""
Multistep Return Label Generator
=================================

从 candidate transition dataset 中抽取同状态多动作 group，
对每个 candidate 的 next_obs 恢复环境状态后接 BC actor rollout，
计算多步真实回报标签，用于 critic ranking 校正诊断。

用法示例（dry-run）：
    python scripts/generate_multistep_return_labels.py --dry_run

用法示例（固定参数）：
    python scripts/generate_multistep_return_labels.py \
        --dataset_path _tmp_near_bc_candidate_transition_dataset.npz \
        --bc_actor_path _tmp_bc_gimbal_only_actor.pth \
        --checkpoint_path "models/.../final_step_2000.pth" \
        --output _tmp_multistep_return_labels.npz

安全机制：
    - 默认不写文件（需要 --output 且非 --dry_run）
    - 不覆盖已存在文件（需要 --force）
    - 保护关键文件和目录不被覆盖
"""

import os
import sys
import io
import argparse

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch

from envs.vscmg_env import VSCMGEnv
from configs.env_config import VSCMGEnvConfig
from agents.td3_agent import PolicyNet, QValueNet

# =============================================================================
# Constants
# =============================================================================
ALL_CANDIDATE_SOURCES = [
    "bc_actor", "zero_action", "p37_final_actor", "final_x2_clipped",
    "bc_x2_clipped", "bc_plus_noise_small", "bc_minus_noise_small",
]
PROTECTED_FILES = [
    "_tmp_near_bc_candidate_transition_dataset.npz",
    "_tmp_bc_gimbal_only_actor.pth",
]
PROTECTED_DIR_PREFIXES = ["models", "runs"]
OMEGA_W_NOMINAL = 3000.0 * (2.0 * np.pi / 60.0)

DEFAULT_CHECKPOINT_DIR = (
    "v0.5.18-debug-residual-safe-rl-roadmap-3-gf50e1f5-dirty_20260429_202142"
    "_envs1_seed42_gr0.5_g0.997_gimbal_only_att1.0_om0.0_wb0.0_ga0.0_wa0.0"
)


# =============================================================================
# State restoration (verified in P49-1b-pre2)
# =============================================================================
def mrp_to_quat(sigma):
    s = np.asarray(sigma, dtype=np.float64).flatten()
    p = np.dot(s, s)
    d = 1.0 + p
    q = np.array([(1.0 - p) / d, 2.0 * s[0] / d, 2.0 * s[1] / d, 2.0 * s[2] / d])
    return q / np.linalg.norm(q)


def restore_env_from_obs(env, ob):
    ob = np.asarray(ob, dtype=np.float64)
    env.q = mrp_to_quat(-ob[0:3])
    env.q /= np.linalg.norm(env.q)
    env.q_target = np.array([1.0, 0.0, 0.0, 0.0])
    env.omega = ob[3:6].copy()
    env.delta = np.arctan2(ob[6:10], ob[10:14])
    env._delta_dot_cache = ob[14:18].copy()
    env.omega_w_nominal = OMEGA_W_NOMINAL
    env.omega_w = (1.0 + ob[18:22]) * OMEGA_W_NOMINAL
    env.I_w = np.full(4, 0.1)
    env.h_w = env.I_w * env.omega_w
    env.current_step = 0


def theta_deg(ob):
    sig = ob[0:3].astype(np.float64)
    return np.degrees(4.0 * np.arctan(np.linalg.norm(sig)))


def omega_norm(ob):
    return np.linalg.norm(ob[3:6].astype(np.float64))


# =============================================================================
# Model loading
# =============================================================================
def load_bc_actor(path, device="cpu"):
    bc_actor = PolicyNet(state_dim=22, hidden_dim=256, action_dim=4, action_bound=1.0)
    ckpt = torch.load(path, map_location=device)
    if "actor_state_dict" in ckpt:
        bc_actor.load_state_dict(ckpt["actor_state_dict"])
    elif "actor" in ckpt:
        bc_actor.load_state_dict(ckpt["actor"])
    else:
        raise ValueError("Cannot find actor weights in: {}".format(path))
    bc_actor.eval()
    return bc_actor


def load_critics(path, device="cpu"):
    if not os.path.exists(path):
        return None, None
    ckpt = torch.load(path, map_location=device)
    if "critic_1_state_dict" not in ckpt:
        return None, None
    q1 = QValueNet(22, 256, 4)
    q2 = QValueNet(22, 256, 4)
    q1.load_state_dict(ckpt["critic_1_state_dict"])
    q2.load_state_dict(ckpt["critic_2_state_dict"])
    q1.eval()
    q2.eval()
    return q1, q2


# =============================================================================
# Group construction
# =============================================================================
def build_groups(d):
    ep = d["episode_id"]
    bs = d["base_step_id"]
    iad = d["init_attitude_deg"]
    asrc = d["action_source"]
    N = len(ep)

    gkey = np.array(
        [(int(ep[i]), int(bs[i]), float(iad[i])) for i in range(N)],
        dtype=[("ep", "i8"), ("bs", "i8"), ("iad", "f8")],
    )
    ug, ginv = np.unique(gkey, return_inverse=True)

    all_src_set = set(ALL_CANDIDATE_SOURCES)
    gm = {}
    for g in range(len(ug)):
        rows = np.where(ginv == g)[0]
        if len(rows) != len(ALL_CANDIDATE_SOURCES):
            continue
        ss = {}
        for r in rows:
            ss[str(asrc[r])] = int(r)
        if set(ss.keys()) == all_src_set:
            gm[g] = ss
    return gm, ug


# =============================================================================
# Exclude seen groups from previous label file
# =============================================================================
def load_exclude_group_keys(path):
    if path is None:
        return set()
    if not os.path.exists(path):
        raise FileNotFoundError("exclude_labels_path not found: {}".format(path))
    data = np.load(path, allow_pickle=False)
    required = ["group_episode_id", "group_base_step_id", "init_attitude_deg"]
    for k in required:
        if k not in data.files:
            raise ValueError("exclude labels missing field: {}".format(k))
    keys = set()
    for i in range(len(data["group_episode_id"])):
        keys.add((
            int(data["group_episode_id"][i]),
            int(data["group_base_step_id"][i]),
            float(data["init_attitude_deg"][i]),
        ))
    return keys


# =============================================================================
# Output safety (Windows + absolute path compatible)
# =============================================================================
def is_protected(path):
    norm_abs = os.path.abspath(os.path.normpath(path))
    basename = os.path.basename(norm_abs)

    if basename in PROTECTED_FILES:
        return True

    project_abs = os.path.abspath(_PROJECT_ROOT)
    try:
        rel = os.path.relpath(norm_abs, project_abs)
    except ValueError:
        rel = norm_abs

    # Normalize to forward slashes for consistent prefix matching
    rel_posix = os.path.normpath(rel).replace("\\", "/")

    for prefix in PROTECTED_DIR_PREFIXES:
        if rel_posix == prefix or rel_posix.startswith(prefix + "/"):
            return True

    return False


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Multistep Return Label Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset_path", type=str,
        default="_tmp_near_bc_candidate_transition_dataset.npz",
        help="Candidate transition dataset",
    )
    parser.add_argument(
        "--bc_actor_path", type=str,
        default="_tmp_bc_gimbal_only_actor.pth",
        help="BC actor checkpoint path",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="P37 checkpoint path for Q-value computation (default: auto-detect)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output npz path (if not provided, only print summary)",
    )
    parser.add_argument(
        "--angles", type=str, default="5,10,20,30",
        help="Comma-separated initial attitude angles in degrees",
    )
    parser.add_argument(
        "--groups_per_angle", type=int, default=5,
        help="Number of groups to sample per angle",
    )
    parser.add_argument(
        "--horizon", type=int, default=80,
        help="Max continuation rollout steps",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.997,
        help="Discount factor",
    )
    parser.add_argument(
        "--max_gimbal_rate", type=float, default=0.5,
        help="Max gimbal rate for continuation env",
    )
    parser.add_argument(
        "--seed", type=int, default=20260506,
        help="Random seed for group sampling",
    )
    parser.add_argument(
        "--action_sources", type=str,
        default="bc_actor,zero_action,p37_final_actor,final_x2_clipped",
        help="Comma-separated action sources to evaluate",
    )
    parser.add_argument(
        "--exclude_labels_path", type=str, default=None,
        help=(
            "Optional npz with previously generated labels; groups matching "
            "(group_episode_id, group_base_step_id, init_attitude_deg) will be "
            "excluded from sampling."
        ),
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only print plan, do not execute rollouts or write files",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Allow overwriting existing output file",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()

    angles = [float(x.strip()) for x in args.angles.split(",") if x.strip()]
    action_sources = [x.strip() for x in args.action_sources.split(",") if x.strip()]

    print("=" * 60)
    print("Multistep Return Label Generator")
    print("=" * 60)
    print("  dataset_path:    {}".format(args.dataset_path))
    print("  bc_actor_path:   {}".format(args.bc_actor_path))
    print("  checkpoint_path: {}".format(args.checkpoint_path or "auto-detect"))
    print("  output:          {}".format(args.output or "(none, print only)"))
    print("  angles:          {}".format(angles))
    print("  groups_per_angle: {}".format(args.groups_per_angle))
    print("  horizon:         {}".format(args.horizon))
    print("  gamma:           {}".format(args.gamma))
    print("  max_gimbal_rate: {}".format(args.max_gimbal_rate))
    print("  seed:            {}".format(args.seed))
    print("  action_sources:  {}".format(action_sources))
    print("  exclude_labels:  {}".format(
        args.exclude_labels_path or "(none)"))
    print("  dry_run:         {}".format(args.dry_run))
    print("  force:           {}".format(args.force))

    for s in action_sources:
        if s not in ALL_CANDIDATE_SOURCES:
            print("\nERROR: unknown action_source: {}".format(s))
            print("  valid: {}".format(ALL_CANDIDATE_SOURCES))
            sys.exit(1)

    # Output safety
    if args.output is not None:
        if is_protected(args.output):
            print("\nERROR: output path is protected: {}".format(args.output))
            sys.exit(1)
        if os.path.exists(args.output) and not args.force:
            print("\nERROR: output already exists: {}".format(args.output))
            print("  Use --force to overwrite.")
            sys.exit(1)

    # Load dataset
    print("\n[Load] Dataset...")
    d = np.load(args.dataset_path, allow_pickle=False)
    obs_all = d["obs"].astype(np.float64)
    nxt_all = d["next_obs"].astype(np.float64)
    act_all = d["actions"].astype(np.float64)
    rew_all = d["rewards"].astype(np.float64)
    done_all = d["dones"]
    asrc_all = d["action_source"]
    N = len(obs_all)
    print("[Load] {} transitions".format(N))

    # Build groups
    print("[Groups] Building...")
    gm, ug = build_groups(d)
    print("[Groups] {} complete groups (7 sources each)".format(len(gm)))

    # Load exclude set
    exclude_keys = load_exclude_group_keys(args.exclude_labels_path)
    if args.exclude_labels_path is not None:
        print("\n[Exclude] exclude_labels_path: {}".format(
            args.exclude_labels_path))
        print("[Exclude] exclude group keys: {}".format(len(exclude_keys)))

    # Sample groups
    rng = np.random.default_rng(args.seed)
    sampled = []
    for ang in angles:
        ag = []
        for g in gm:
            key = (int(ug[g]["ep"]), int(ug[g]["bs"]), float(ug[g]["iad"]))
            if key in exclude_keys:
                continue
            if abs(float(ug[g]["iad"]) - ang) < 0.1:
                ag.append(g)
        if not ag:
            print("  WARNING: no groups for angle={}".format(ang))
            continue
        take = min(args.groups_per_angle, len(ag))
        chosen = rng.choice(ag, size=take, replace=False)
        sampled.extend(chosen.tolist())

    n_sampled = len(sampled)
    n_branches = n_sampled * len(action_sources)
    print("[Sample] {} groups, {} branches".format(n_sampled, n_branches))

    if args.dry_run:
        complete_key_set = set(
            (int(ug[g]["ep"]), int(ug[g]["bs"]), float(ug[g]["iad"]))
            for g in gm
        )
        excluded_in_pool = len(complete_key_set & exclude_keys)
        remaining_groups = len(gm) - excluded_in_pool
        per_angle = {}
        for g in sampled:
            a = float(ug[g]["iad"])
            per_angle[a] = per_angle.get(a, 0) + 1
        print("\n[DRY RUN] Plan:")
        print("  complete_groups:  {}".format(len(gm)))
        print("  exclude_groups:   {}".format(len(exclude_keys)))
        print("  excluded_in_pool: {}".format(excluded_in_pool))
        print("  remaining_groups: {}".format(remaining_groups))
        print("  sampled_groups:   {}".format(n_sampled))
        print("  branches:         {}".format(n_branches))
        print("  action_sources:   {}".format(action_sources))
        print("  horizon:          {}".format(args.horizon))
        print("  gamma:            {}".format(args.gamma))
        print("  max_gimbal_rate:  {}".format(args.max_gimbal_rate))
        for a in sorted(per_angle):
            print("  sampled_angle_{}: {}".format(a, per_angle[a]))
        print("  No rollouts executed. No files written.")
        sys.exit(0)

    # Load models
    print("\n[Load] BC actor...")
    bc_actor = load_bc_actor(args.bc_actor_path)
    print("[Load] BC actor loaded.")

    ckpt_path = args.checkpoint_path
    if ckpt_path is None:
        ckpt_path = os.path.join("models", DEFAULT_CHECKPOINT_DIR, "final_step_2000.pth")

    print("[Load] Critics from: {}".format(ckpt_path))
    q1, q2 = load_critics(ckpt_path)
    if q1 is not None:
        print("[Load] Critics loaded.")
    else:
        print("[Load] Critics not available. q_current will be NaN.")

    # Compute q_current
    sampled_rows = []
    for g in sampled:
        for s in action_sources:
            sampled_rows.append(gm[g][s])

    q_vals = np.full(len(sampled_rows), np.nan, dtype=np.float64)
    if q1 is not None:
        with torch.no_grad():
            ot = torch.FloatTensor(obs_all[sampled_rows])
            at = torch.FloatTensor(act_all[sampled_rows])
            q1v = q1(ot, at).squeeze(1).numpy()
            q2v = q2(ot, at).squeeze(1).numpy()
            q_vals = np.minimum(q1v, q2v).astype(np.float64)
        print("[Q] Computed for {} pairs".format(len(sampled_rows)))

    # Env config
    env_cfg = VSCMGEnvConfig(action_mode="gimbal_only")
    env_cfg.max_gimbal_rate = args.max_gimbal_rate

    # Rollout
    print("\n[Rollout] Starting {} branches...".format(n_branches))
    all_data = {
        "group_episode_id": [],
        "group_base_step_id": [],
        "init_attitude_deg": [],
        "action_source": [],
        "row_index": [],
        "q_current": [],
        "candidate_reward": [],
        "discounted_return": [],
        "undiscounted_return": [],
        "final_theta_deg": [],
        "final_omega_norm": [],
        "rollout_steps": [],
        "done": [],
    }

    bi = 0
    for g_idx, g in enumerate(sampled):
        for s in action_sources:
            row = gm[g][s]
            q_cur = float(q_vals[bi])
            cand_rew = float(rew_all[row])
            cand_done = bool(done_all[row])
            nxt_ob = nxt_all[row]

            if cand_done:
                all_data["group_episode_id"].append(int(ug[g]["ep"]))
                all_data["group_base_step_id"].append(int(ug[g]["bs"]))
                all_data["init_attitude_deg"].append(float(ug[g]["iad"]))
                all_data["action_source"].append(s)
                all_data["row_index"].append(row)
                all_data["q_current"].append(q_cur)
                all_data["candidate_reward"].append(cand_rew)
                all_data["discounted_return"].append(cand_rew)
                all_data["undiscounted_return"].append(cand_rew)
                all_data["final_theta_deg"].append(theta_deg(nxt_ob))
                all_data["final_omega_norm"].append(omega_norm(nxt_ob))
                all_data["rollout_steps"].append(0)
                all_data["done"].append(True)
                bi += 1
                continue

            env = VSCMGEnv(config=env_cfg)
            env.reset(seed=args.seed, options={"init_attitude_deg": 5.0})
            restore_env_from_obs(env, nxt_ob)

            disc_ret = cand_rew
            undisc_ret = cand_rew
            cont_steps = 0
            cur_gamma = args.gamma

            for _ in range(args.horizon):
                with torch.no_grad():
                    ob_t = torch.FloatTensor(
                        env._get_obs().astype(np.float32)
                    ).unsqueeze(0)
                    a_bc = bc_actor(ob_t).cpu().numpy().squeeze(0)
                    a_bc = np.clip(a_bc, -1.0, 1.0)
                _, r, term, trunc, _ = env.step(a_bc)
                disc_ret += cur_gamma * r
                undisc_ret += r
                cur_gamma *= args.gamma
                cont_steps += 1
                if term or trunc:
                    break

            final_ob = env._get_obs().astype(np.float64)
            all_data["group_episode_id"].append(int(ug[g]["ep"]))
            all_data["group_base_step_id"].append(int(ug[g]["bs"]))
            all_data["init_attitude_deg"].append(float(ug[g]["iad"]))
            all_data["action_source"].append(s)
            all_data["row_index"].append(row)
            all_data["q_current"].append(q_cur)
            all_data["candidate_reward"].append(cand_rew)
            all_data["discounted_return"].append(float(disc_ret))
            all_data["undiscounted_return"].append(float(undisc_ret))
            all_data["final_theta_deg"].append(theta_deg(final_ob))
            all_data["final_omega_norm"].append(omega_norm(final_ob))
            all_data["rollout_steps"].append(cont_steps)
            all_data["done"].append(False)
            env.close()
            bi += 1

        if (g_idx + 1) % 5 == 0:
            print("  {}/{} groups done".format(g_idx + 1, n_sampled))

    print("[Rollout] Complete.")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    dr = np.array(all_data["discounted_return"])
    for s in action_sources:
        mask = np.array([x == s for x in all_data["action_source"]])
        if mask.sum() == 0:
            continue
        print("  {:<28s} n={:3d} disc_ret={:+.6f}".format(
            s, mask.sum(), dr[mask].mean()))

    # Save
    if args.output is not None:
        save_data = {}
        for k, v in all_data.items():
            if k == "action_source":
                save_data[k] = np.array(v, dtype="<U32")
            elif k in ("row_index", "rollout_steps", "group_episode_id",
                        "group_base_step_id"):
                save_data[k] = np.array(v, dtype=np.int32)
            elif k == "done":
                save_data[k] = np.array(v, dtype=np.bool_)
            else:
                save_data[k] = np.array(v, dtype=np.float64)
        save_data["horizon"] = np.int32(args.horizon)
        save_data["gamma"] = np.float64(args.gamma)
        save_data["max_gimbal_rate"] = np.float64(args.max_gimbal_rate)

        np.savez_compressed(args.output, **save_data)
        print("\n[Save] Written to: {}".format(args.output))
    else:
        print("\n[Info] No --output provided. Nothing saved.")

    print("[DONE]")


if __name__ == "__main__":
    main()
