"""
Near-BC Candidate Transition Dataset Generator
===============================================

用于收集 branching one-step candidate transition 数据：
在同一个 base obs 上，用 deepcopy(env) 分叉执行多种 candidate action，
生成用于 critic ranking 校正的 transition dataset。

每个 base step：
  1. 在当前 base obs 上构造 7 种 candidate actions
  2. 对每个 candidate action，用 deepcopy(base_env) 分叉执行一步
  3. 记录所有 candidate 的 (obs, action, reward, next_obs, done, action_source)
  4. base env 再用 bc_actor 前进一步，进入下一个 base obs

candidate actions（7 种）：
  1. bc_actor:         BC actor(obs)
  2. zero_action:      zeros(4)
  3. p37_final_actor:  P37 final actor(obs)
  4. final_x2_clipped: clip(2.0 * p37_final_actor(obs), -1, 1)
  5. bc_x2_clipped:    clip(2.0 * bc_actor(obs), -1, 1)
  6. bc_plus_noise:    clip(bc_actor(obs) + noise, -1, 1)
  7. bc_minus_noise:   clip(bc_actor(obs) - noise, -1, 1)

用法示例（dry-run）：
    python scripts/generate_bc_candidate_transition_dataset.py --dry_run

用法示例（固定角度列表模式）：
    python scripts/generate_bc_candidate_transition_dataset.py \
        --bc_actor_path _tmp_bc_gimbal_only_actor.pth \
        --final_actor_path "models/.../final_step_2000.pth" \
        --angle_list_deg "5,10,20,30" \
        --episodes_per_angle 5 \
        --max_steps 320 \
        --seed 42 \
        --noise_std 0.05 \
        --output _tmp_near_bc_candidate_transition_dataset.npz

本脚本默认只支持 gimbal_only（4D）action 模式。
"""

import os
import sys
import io
import copy

# 确保 stdout 使用 utf-8（Windows GBK 终端兼容）
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 确保项目根目录在 sys.path 中
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import numpy as np
import torch

from envs.vscmg_env import VSCMGEnv
from configs.env_config import VSCMGEnvConfig, make_default_config
from agents.td3_agent import TD3


# =============================================================================
# 配置
# =============================================================================
DEFAULT_OUTPUT = "_tmp_near_bc_candidate_transition_dataset.npz"
PROTECTED_FILES = ["_tmp_oracle_bc_dataset.npz"]

# Candidate action sources（按优先级排序）
CANDIDATE_SOURCES = [
    "bc_actor",
    "zero_action",
    "p37_final_actor",
    "final_x2_clipped",
    "bc_x2_clipped",
    "bc_plus_noise_small",
    "bc_minus_noise_small",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Near-BC Candidate Transition Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
默认不覆盖已存在的 output 文件，需要 --force 显式确认。
默认不覆盖 _tmp_oracle_bc_dataset.npz。
默认 action_mode=gimbal_only（4D），其他模式会报错退出。
        """
    )
    parser.add_argument(
        "--bc_actor_path", type=str, default="_tmp_bc_gimbal_only_actor.pth",
        help="BC actor checkpoint 路径（默认：_tmp_bc_gimbal_only_actor.pth）"
    )
    parser.add_argument(
        "--final_actor_path", type=str, default=None,
        help="P37 final actor checkpoint 路径（必填）"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"输出 npz 路径（默认：{DEFAULT_OUTPUT}）"
    )
    parser.add_argument(
        "--angle_list_deg", type=str, default="5,10,20,30",
        help="逗号分隔的固定初始姿态角列表，默认为 5,10,20,30"
    )
    parser.add_argument(
        "--episodes_per_angle", type=int, default=5,
        help="每个固定角度 rollout 的 episode 数量（默认：5）"
    )
    parser.add_argument(
        "--max_steps", type=int, default=320,
        help="单 episode 最大步数（默认：320）"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="初始随机种子（默认：42）"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="计算设备（默认：cpu）"
    )
    parser.add_argument(
        "--noise_std", type=float, default=0.05,
        help="BC action 小扰动噪声标准差（默认：0.05）"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="只打印 rollout 预览，不保存 npz"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="允许覆盖已存在的 output 文件"
    )
    parser.add_argument(
        "--action_mode", type=str, default="gimbal_only",
        choices=["gimbal_only"],
        help="动作模式（默认：gimbal_only，当前只支持 gimbal_only）"
    )
    return parser.parse_args()


def parse_angle_list(angle_list_str):
    """解析逗号分隔的角度字符串，返回 float 列表"""
    parts = angle_list_str.split(",")
    angles = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        val = float(p)
        if val <= 0 or val > 180:
            raise ValueError(f"Invalid angle {val}. Each angle must be in (0, 180].")
        angles.append(val)
    if not angles:
        raise ValueError("angle_list_deg is empty after parsing.")
    return angles


def load_actor(actor_path: str, expected_action_dim: int, device: str, label: str):
    """从 checkpoint 加载 actor"""
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"Actor checkpoint not found for {label}: {actor_path}")

    checkpoint = torch.load(actor_path, map_location=device)

    if "actor" in checkpoint:
        actor_sd = checkpoint["actor"]
    elif "actor_state_dict" in checkpoint:
        actor_sd = checkpoint["actor_state_dict"]
    elif "actor_target" in checkpoint:
        actor_sd = checkpoint["actor_target"]
    else:
        raise ValueError(
            f"Unknown checkpoint keys: {list(checkpoint.keys())}. "
            f"Need 'actor', 'actor_state_dict', or 'actor_target'."
        )

    if "fc3.weight" in actor_sd:
        actual_dim = actor_sd["fc3.weight"].shape[0]
        if actual_dim != expected_action_dim:
            raise RuntimeError(
                f"Action dim mismatch for {label}: actor output dim = {actual_dim}, "
                f"expected {expected_action_dim} for gimbal_only mode."
            )

    agent = TD3(
        state_dim=22,
        action_dim=expected_action_dim,
        hidden_dim=256,
        action_bound=1.0,
        sigma=0.0,
        tau=0.005,
        gamma=0.997,
        critic_lr=3e-4,
        actor_lr=1e-4,
        delay=2,
        policy_noise=0.0,
        noise_clip=0.0,
        device=device,
    )
    agent.actor.load_state_dict(actor_sd, strict=True)
    agent.actor.eval()
    return agent.actor


def get_candidate_actions(bc_actor, final_actor, obs, rng, noise_std):
    """为给定 obs 生成 7 种 candidate action"""
    with torch.no_grad():
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(next(bc_actor.parameters()).device)

        a_bc = bc_actor(obs_t).cpu().numpy().squeeze(0)
        a_final = final_actor(obs_t).cpu().numpy().squeeze(0)

    noise = rng.normal(0, noise_std, size=4).astype(np.float64)

    return {
        "bc_actor": np.clip(a_bc, -1.0, 1.0),
        "zero_action": np.zeros(4, dtype=np.float64),
        "p37_final_actor": np.clip(a_final, -1.0, 1.0),
        "final_x2_clipped": np.clip(2.0 * a_final, -1.0, 1.0),
        "bc_x2_clipped": np.clip(2.0 * a_bc, -1.0, 1.0),
        "bc_plus_noise_small": np.clip(a_bc + noise, -1.0, 1.0),
        "bc_minus_noise_small": np.clip(a_bc - noise, -1.0, 1.0),
    }


def collect_candidate_transitions(
    bc_actor, final_actor, env_cfg,
    angle_list, episodes_per_angle, max_steps, seed_start, noise_std, device
):
    """执行 branching rollout，收集 candidate transition 数据。

    每个 base step：
    1. 在当前 base obs 上构造 7 种 candidate actions
    2. 对每个 candidate action，用 deepcopy(base_env) 分叉执行一步
    3. 记录所有 candidate 的 transition
    4. base env 再用 bc_actor 前进一步，进入下一个 base obs
    """
    obs_list = []
    action_list = []
    reward_list = []
    next_obs_list = []
    done_list = []
    episode_id_list = []
    base_step_id_list = []
    branch_id_list = []
    init_attitude_deg_list = []
    action_source_list = []

    # 可选的 info 字段
    optional_fields = [
        "sigma_err_sq", "omega_sq", "gimbal_action_sq",
        "wheel_action_sq", "wheel_bias_sq", "reward_total"
    ]
    optional_data = {k: [] for k in optional_fields}

    rng = np.random.default_rng(seed_start)

    # action_source 到 id 的映射
    source_to_id = {s: i for i, s in enumerate(CANDIDATE_SOURCES)}

    episode_counter = 0
    total_base_steps = 0

    for angle in angle_list:
        for ep in range(episodes_per_angle):
            seed = seed_start + episode_counter
            episode_counter += 1

            base_env = VSCMGEnv(config=env_cfg)
            obs, _ = base_env.reset(seed=seed, options={"init_attitude_deg": float(angle)})

            base_step = 0
            while base_step < max_steps:
                # 1. 在当前 base obs 上构造 candidate actions
                candidate_actions = get_candidate_actions(
                    bc_actor, final_actor, obs, rng, noise_std
                )

                # 2. 对每个 candidate action 用 deepcopy 分叉执行一步
                for branch_idx, source in enumerate(CANDIDATE_SOURCES):
                    action = candidate_actions[source]

                    branch_env = copy.deepcopy(base_env)
                    try:
                        next_obs, reward, terminated, truncated, info = branch_env.step(action)
                    finally:
                        branch_env.close()

                    done = bool(terminated or truncated)

                    obs_list.append(obs.copy())
                    action_list.append(action.copy())
                    reward_list.append(float(reward))
                    next_obs_list.append(next_obs.copy())
                    done_list.append(done)
                    episode_id_list.append(episode_counter - 1)
                    base_step_id_list.append(base_step)
                    branch_id_list.append(branch_idx)
                    init_attitude_deg_list.append(float(angle))
                    action_source_list.append(source)

                    # 可选 info 字段
                    if isinstance(info, dict):
                        for key in optional_fields:
                            if key in info:
                                optional_data[key].append(float(info[key]))
                            else:
                                optional_data[key].append(0.0)
                    else:
                        for key in optional_fields:
                            optional_data[key].append(0.0)

                # 3. base env 再用 bc_actor 前进一步，进入下一 base obs
                a_bc = candidate_actions["bc_actor"]
                obs, reward_base, terminated, truncated, info_base = base_env.step(a_bc)
                done_base = bool(terminated or truncated)

                total_base_steps += 1
                base_step += 1

                if done_base:
                    break

            base_env.close()

    # 构建返回字典
    result = {
        "obs": np.array(obs_list, dtype=np.float32),
        "actions": np.array(action_list, dtype=np.float32),
        "rewards": np.array(reward_list, dtype=np.float32),
        "next_obs": np.array(next_obs_list, dtype=np.float32),
        "dones": np.array(done_list, dtype=np.bool_),
        "episode_id": np.array(episode_id_list, dtype=np.int32),
        "base_step_id": np.array(base_step_id_list, dtype=np.int32),
        "branch_id": np.array(branch_id_list, dtype=np.int32),
        "init_attitude_deg": np.array(init_attitude_deg_list, dtype=np.float32),
        # action_source 用 unicode string，allow_pickle=False 可读
        "action_source": np.array(action_source_list, dtype="<U32"),
        "action_source_id": np.array(
            [source_to_id[s] for s in action_source_list], dtype=np.int32
        ),
        **{k: np.array(v, dtype=np.float32) for k, v in optional_data.items()},
    }

    return result, total_base_steps


def audit_dataset(data, total_base_steps):
    """打印数据集统计信息"""
    print("\n" + "=" * 70)
    print("Candidate Dataset Audit Summary")
    print("=" * 70)

    print(f"\n  Basic shapes:")
    print(f"    obs shape:          {data['obs'].shape}")
    print(f"    actions shape:     {data['actions'].shape}")
    print(f"    rewards shape:     {data['rewards'].shape}")
    print(f"    next_obs shape:    {data['next_obs'].shape}")
    print(f"    dones shape:       {data['dones'].shape}")
    print(f"    episode_id shape:  {data['episode_id'].shape}")
    print(f"    base_step_id:      {data['base_step_id'].shape}")
    print(f"    branch_id shape:   {data['branch_id'].shape}")
    print(f"    init_attitude_deg: {data['init_attitude_deg'].shape}")

    # action_source dtype
    print(f"\n  action_source dtype: {data['action_source'].dtype}")
    print(f"    action_source_id dtype: {data['action_source_id'].dtype}")
    print(f"    unique sources: {sorted(set(data['action_source']))}")
    print(f"    total_base_steps: {total_base_steps}")

    # per-source stats
    print(f"\n  Per-source statistics:")
    print("  " + "-" * 68)

    for source in CANDIDATE_SOURCES:
        mask = data['action_source'] == source
        if not np.any(mask):
            continue

        cnt = int(np.sum(mask))
        r = data['rewards'][mask]
        a = data['actions'][mask]
        sig = data.get('sigma_err_sq', np.zeros(0))[mask]
        omg = data.get('omega_sq', np.zeros(0))[mask]

        print(
            f"  {source:28s}  n={cnt:4d}"
            f"  reward={float(r.mean()):+.6f}±{float(r.std()):.6f}"
            f"  |a|_mean={float(np.mean(np.abs(a))):.6f}"
            f"  |a|_max={float(np.max(np.abs(a))):.6f}"
            f"  sat={float(np.mean(np.abs(a) > 0.95)):.6f}"
        )

    # init_attitude_deg 分布
    iad = data.get("init_attitude_deg")
    if iad is not None:
        unique_angles = np.unique(iad[~np.isnan(iad)])
        if len(unique_angles) > 0:
            print(f"\n  init_attitude_deg unique values: {sorted(unique_angles.tolist())}")
            for angle in sorted(unique_angles):
                count = int(np.sum(iad == angle))
                print(f"    angle={float(angle):.1f} deg: {count} transitions")

    print("=" * 70)


def main():
    args = parse_args()

    if args.final_actor_path is None:
        print("ERROR: --final_actor_path is required")
        print("Run with: python scripts/generate_bc_candidate_transition_dataset.py --help")
        sys.exit(1)

    action_mode = args.action_mode
    if action_mode != "gimbal_only":
        print(f"ERROR: only gimbal_only mode is supported. Got: {action_mode}")
        sys.exit(1)

    if os.path.exists(args.output):
        protected = any(args.output.endswith(p) for p in PROTECTED_FILES)
        if protected:
            print(f"ERROR: output file '{args.output}' is protected and will not be overwritten.")
            sys.exit(1)
        if not args.force:
            print(f"ERROR: output file '{args.output}' already exists.")
            print("       Use --force to overwrite, or specify a different --output path.")
            sys.exit(1)

    env_cfg = VSCMGEnvConfig(action_mode="gimbal_only")
    env = VSCMGEnv(config=env_cfg)
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    env.close()

    if action_dim != 4:
        print(f"ERROR: gimbal_only mode expects action_dim=4, got {action_dim}")
        sys.exit(1)
    if state_dim != 22:
        print(f"ERROR: expected state_dim=22, got {state_dim}")
        sys.exit(1)

    angle_list = parse_angle_list(args.angle_list_deg)
    total_planned_episodes = len(angle_list) * args.episodes_per_angle

    print(f"[Config] bc_actor_path      = {args.bc_actor_path}")
    print(f"[Config] final_actor_path   = {args.final_actor_path}")
    print(f"[Config] output             = {args.output}")
    print(f"[Config] angle_list_deg     = {angle_list}")
    print(f"[Config] episodes_per_angle= {args.episodes_per_angle}")
    print(f"[Config] total_episodes    = {total_planned_episodes}")
    print(f"[Config] max_steps         = {args.max_steps}")
    print(f"[Config] seed_start        = {args.seed}")
    print(f"[Config] device            = {args.device}")
    print(f"[Config] noise_std         = {args.noise_std}")
    print(f"[Config] dry_run           = {args.dry_run}")
    print(f"[Config] action_mode        = {action_mode}")
    print(f"[Config] action_dim        = {action_dim}")
    print(f"[Config] candidate_sources = {CANDIDATE_SOURCES}")

    print(f"\n[Load] Loading BC actor...")
    bc_actor = load_actor(args.bc_actor_path, action_dim, args.device, "BC actor")
    print("[Load] BC actor loaded successfully.")

    print(f"\n[Load] Loading P37 final actor...")
    final_actor = load_actor(args.final_actor_path, action_dim, args.device, "P37 final actor")
    print("[Load] P37 final actor loaded successfully.")

    print(f"\n[Rollout] Collecting branching candidate transitions...")
    data, total_base_steps = collect_candidate_transitions(
        bc_actor, final_actor, env_cfg,
        angle_list, args.episodes_per_angle, args.max_steps,
        args.seed, args.noise_std, args.device
    )
    total_transitions = len(data["obs"])
    print(f"[Rollout] Collected {total_transitions} transitions over {total_base_steps} base steps.")

    audit_dataset(data, total_base_steps)

    if args.dry_run:
        print("\n[DRY RUN] No file written. Re-run without --dry_run to save npz.")
        sys.exit(0)

    print(f"\n[Save] Writing to {args.output}...")
    np.savez_compressed(args.output, **data)
    print(f"[Save] Saved successfully.")

    print(f"\n[Verify] Re-loading {args.output} for integrity check...")
    verify = np.load(args.output, allow_pickle=False)
    print(f"[Verify] keys: {sorted(verify.keys())}")
    print(f"[Verify] obs shape: {verify['obs'].shape}")
    print(f"[Verify] actions shape: {verify['actions'].shape}")
    print(f"[Verify] rewards shape: {verify['rewards'].shape}")
    print(f"[Verify] next_obs shape: {verify['next_obs'].shape}")
    print(f"[Verify] dones shape: {verify['dones'].shape}")
    print(f"[Verify] action_source dtype: {verify['action_source'].dtype}")
    print(f"[Verify] action_source_id dtype: {verify['action_source_id'].dtype}")
    print(f"[Verify] allow_pickle=False read: SUCCESS")
    verify.close()

    print(f"\n[DONE] Dataset saved to: {args.output}")


if __name__ == "__main__":
    main()
