"""
BC Transition Dataset Generator
================================

用于收集 BC actor 产生的完整 transition 数据（obs/action/reward/next_obs/done），
作为 replay prefill / critic warmup 的数据源。

用法示例（dry-run）：
    python scripts/generate_bc_transition_dataset.py --actor_path _tmp_bc_gimbal_only_actor.pth --dry_run

用法示例（固定角度列表模式，推荐用于 replay prefill）：
    python scripts/generate_bc_transition_dataset.py \
        --actor_path _tmp_bc_gimbal_only_actor.pth \
        --angle_list_deg "5,10,20,30" \
        --episodes_per_angle 5 \
        --max_steps 320 \
        --seed 42 \
        --output _bc_transition_dataset_for_prefill.npz

用法示例（默认 reset 分布模式）：
    python scripts/generate_bc_transition_dataset.py \
        --actor_path _tmp_bc_gimbal_only_actor.pth \
        --episodes 5 \
        --max_steps 320 \
        --seed 42 \
        --output _bc_transition_dataset_for_prefill.npz

本脚本默认只支持 gimbal_only（4D）action 模式。
"""

import os
import sys
import io

# 确保 stdout 使用 utf-8（Windows GBK 终端兼容）
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 确保项目根目录在 sys.path 中（支持从 scripts/ 子目录运行）
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
DEFAULT_OUTPUT = "_tmp_bc_transition_dataset_for_prefill.npz"
PROTECTED_FILES = ["_tmp_oracle_bc_dataset.npz"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="BC Transition Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
默认不覆盖已存在的 output 文件，需要 --force 显式确认。
默认不覆盖 _tmp_oracle_bc_dataset.npz。
默认 action_mode=gimbal_only（4D），其他模式会报错退出。
        """
    )
    parser.add_argument(
        "--actor_path", type=str, default=None,
        help="BC actor checkpoint 路径（必填，除非使用 --help）"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"输出 npz 路径（默认：{DEFAULT_OUTPUT}）"
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="rollout episode 数量（默认：1，dry-run 用）。仅在未指定 --angle_list_deg 时生效。"
    )
    parser.add_argument(
        "--angle_list_deg", type=str, default=None,
        help="逗号分隔的固定初始姿态角列表，例��� 5,10,20,30。"
             "指定后对每个角度运行 episodes_per_angle 个 episode。"
             "不指定则使用环境默认 reset 分布（0~5 度随机）。"
    )
    parser.add_argument(
        "--episodes_per_angle", type=int, default=1,
        help="每个固定角度 rollout 的 episode 数量（默认：1，仅在 --angle_list_deg 指定时生效）"
    )
    parser.add_argument(
        "--max_steps", type=int, default=200,
        help="单 episode 最大步数（默认：200）"
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
    if angle_list_str is None:
        return None
    parts = angle_list_str.split(",")
    angles = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        val = float(p)
        if val <= 0 or val > 180:
            raise ValueError(
                f"Invalid angle {val} in angle_list_deg. "
                f"Each angle must be in (0, 180]."
            )
        angles.append(val)
    if not angles:
        raise ValueError("angle_list_deg is empty after parsing.")
    return angles


def load_bc_actor(actor_path: str, expected_action_dim: int, device: str):
    """从 checkpoint 加载 BC actor"""
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"Actor checkpoint not found: {actor_path}")

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
                f"Action dim mismatch: actor output dim = {actual_dim}, "
                f"expected {expected_action_dim} for gimbal_only mode. "
                f"Use a compatible gimbal_only checkpoint."
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
    agent.target_actor.load_state_dict(actor_sd, strict=True)
    agent.actor.eval()

    return agent.actor


def collect_transitions(actor, env_cfg, episodes, max_steps, seed_start, device,
                        angle_list=None, episodes_per_angle=1):
    """执行 rollout，收集完整 transition 数据。

    两种模式：
    - angle_list=None：使用环境默认 reset 分布（0~5°），运行 episodes 个 episode。
    - angle_list 非 None：对每个角度运行 episodes_per_angle 个 episode，
      reset 时使用 options={"init_attitude_deg": angle}。
    """
    obs_list = []
    action_list = []
    reward_list = []
    next_obs_list = []
    done_list = []
    episode_id_list = []
    step_id_list = []
    init_attitude_deg_list = []

    # 可选的 reward breakdown 字段（从 info 中尝试提取）
    optional_fields = [
        "sigma_err_sq", "omega_sq", "gimbal_action_sq",
        "wheel_action_sq", "wheel_bias_sq", "reward_total"
    ]
    optional_data = {k: [] for k in optional_fields}

    # 构建 episode 计划：(seed, angle_or_nan)
    episode_plan = []
    if angle_list is not None:
        episode_counter = 0
        for angle in angle_list:
            for _ in range(episodes_per_angle):
                episode_plan.append((seed_start + episode_counter, angle))
                episode_counter += 1
    else:
        for ep in range(episodes):
            episode_plan.append((seed_start + ep, float("nan")))

    for ep, (seed, init_angle) in enumerate(episode_plan):
        env = VSCMGEnv(config=env_cfg)

        # 根据是否有固定角度决定 reset 方式
        if not np.isnan(init_angle):
            obs, _ = env.reset(seed=seed, options={"init_attitude_deg": init_angle})
        else:
            obs, _ = env.reset(seed=seed)

        for step in range(max_steps):
            with torch.no_grad():
                s_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = actor(s_tensor).cpu().numpy().squeeze(0)
            action = np.clip(action, -1.0, 1.0)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            obs_list.append(obs.copy())
            action_list.append(action.copy())
            reward_list.append(float(reward))
            next_obs_list.append(next_obs.copy())
            done_list.append(done)
            step_id_list.append(step)
            episode_id_list.append(ep)
            init_attitude_deg_list.append(init_angle)

            # 可选字段
            if isinstance(info, dict):
                for key in optional_fields:
                    if key in info:
                        optional_data[key].append(float(info[key]))
                    else:
                        optional_data[key].append(0.0)
            else:
                for key in optional_fields:
                    optional_data[key].append(0.0)

            obs = next_obs

            if done:
                break

        env.close()

    return {
        "obs": np.array(obs_list, dtype=np.float32),
        "actions": np.array(action_list, dtype=np.float32),
        "rewards": np.array(reward_list, dtype=np.float32),
        "next_obs": np.array(next_obs_list, dtype=np.float32),
        "dones": np.array(done_list, dtype=np.bool_),
        "episode_id": np.array(episode_id_list, dtype=np.int32),
        "step_id": np.array(step_id_list, dtype=np.int32),
        "init_attitude_deg": np.array(init_attitude_deg_list, dtype=np.float32),
        **{k: np.array(v, dtype=np.float32) for k, v in optional_data.items()},
    }


def audit_dataset(data):
    """打印数据集统计信息"""
    print("\n" + "=" * 70)
    print("Dataset Audit Summary")
    print("=" * 70)

    print(f"  obs shape:         {data['obs'].shape}")
    print(f"  actions shape:    {data['actions'].shape}")
    print(f"  rewards shape:     {data['rewards'].shape}")
    print(f"  next_obs shape:   {data['next_obs'].shape}")
    print(f"  dones shape:       {data['dones'].shape}")
    print(f"  episode_id shape: {data['episode_id'].shape}")
    print(f"  step_id shape:    {data['step_id'].shape}")

    r = data["rewards"]
    print(f"\n  reward stats:")
    print(f"    min:    {float(r.min()):.4f}")
    print(f"    max:    {float(r.max()):.4f}")
    print(f"    mean:   {float(r.mean()):.4f}")
    print(f"    std:    {float(r.std()):.4f}")

    a = data["actions"]
    print(f"\n  action stats:")
    print(f"    action_abs_mean:        {float(np.mean(np.abs(a))):.6f}")
    print(f"    action_abs_max:         {float(np.max(np.abs(a))):.6f}")
    print(f"    action_sat_rate>0.95:   {float(np.mean(np.abs(a) > 0.95)):.6f}")

    d = data["dones"]
    print(f"\n  done stats:")
    print(f"    done_true_ratio:        {float(np.mean(d.astype(bool))):.4f}")

    n_obs = data["obs"]
    print(f"\n  state stats:")
    print(f"    obs_dim:                 {n_obs.shape[1]}")
    print(f"    obs_min:                 {float(n_obs.min()):.4f}")
    print(f"    obs_max:                 {float(n_obs.max()):.4f}")
    print(f"    obs_mean:                {float(n_obs.mean()):.4f}")

    print(f"\n  optional fields present:")
    for key in ["sigma_err_sq", "omega_sq", "gimbal_action_sq", "wheel_action_sq",
                "wheel_bias_sq", "reward_total"]:
        arr = data.get(key)
        if arr is not None and len(arr) > 0:
            print(f"    {key}: mean={float(arr.mean()):.4f}, std={float(arr.std()):.4f}")
        else:
            print(f"    {key}: (not available)")

    # init_attitude_deg 分布
    iad = data.get("init_attitude_deg")
    if iad is not None:
        unique_angles = np.unique(iad[~np.isnan(iad)])
        if len(unique_angles) > 0:
            print(f"\n  init_attitude_deg unique values: {unique_angles.tolist()}")
            for angle in unique_angles:
                count = int(np.sum(iad == angle))
                print(f"    angle={float(angle):.1f} deg: {count} transitions")
        else:
            print(f"\n  init_attitude_deg: all NaN (using default reset distribution)")

    print("=" * 70)


def main():
    args = parse_args()

    if args.actor_path is None:
        print("ERROR: --actor_path is required (unless using --help)")
        print("Run with: python scripts/generate_bc_transition_dataset.py --help")
        sys.exit(1)

    action_mode = args.action_mode
    if action_mode != "gimbal_only":
        print(f"ERROR: only gimbal_only mode is supported. Got: {action_mode}")
        sys.exit(1)

    if os.path.exists(args.output):
        protected = any(args.output.endswith(p) for p in PROTECTED_FILES)
        if protected:
            print(f"ERROR: output file '{args.output}' is protected and will not be overwritten.")
            print("       Use --actor_path to specify a different output, or manually remove the file.")
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

    print(f"[Config] actor_path  = {args.actor_path}")
    print(f"[Config] output      = {args.output}")
    print(f"[Config] max_steps   = {args.max_steps}")
    print(f"[Config] seed_start  = {args.seed}")
    print(f"[Config] device      = {args.device}")
    print(f"[Config] dry_run     = {args.dry_run}")
    print(f"[Config] action_mode = {action_mode}")
    print(f"[Config] action_dim  = {action_dim} (gimbal_only)")

    # 解析角度列表
    angle_list = parse_angle_list(args.angle_list_deg)

    if angle_list is not None:
        total_episodes = len(angle_list) * args.episodes_per_angle
        print(f"[Config] angle_list_deg     = {angle_list}")
        print(f"[Config] episodes_per_angle = {args.episodes_per_angle}")
        print(f"[Config] total_planned_episodes = {total_episodes}")
    else:
        total_episodes = args.episodes
        print(f"[Config] episodes = {args.episodes} (default reset distribution)")

    print("\n[Load] Loading BC actor...")
    actor = load_bc_actor(args.actor_path, action_dim, args.device)
    print("[Load] BC actor loaded successfully.")

    print(f"\n[Rollout] Collecting transitions ({total_episodes} episodes, max_steps={args.max_steps})...")
    data = collect_transitions(
        actor, env_cfg, args.episodes, args.max_steps, args.seed, args.device,
        angle_list=angle_list,
        episodes_per_angle=args.episodes_per_angle,
    )
    total_transitions = len(data["obs"])
    print(f"[Rollout] Collected {total_transitions} transitions.")

    audit_dataset(data)

    if args.dry_run:
        print("\n[DRY RUN] No file written. Re-run without --dry_run to save npz.")
        sys.exit(0)

    print(f"\n[Save] Writing to {args.output}...")
    np.savez_compressed(args.output, **data)

    print(f"[Verify] Re-loading {args.output} for integrity check...")
    verify = np.load(args.output, allow_pickle=False)
    print(f"[Verify] keys: {list(verify.keys())}")
    print(f"[Verify] obs shape: {verify['obs'].shape}, actions shape: {verify['actions'].shape}")
    print(f"[Verify] rewards shape: {verify['rewards'].shape}, next_obs shape: {verify['next_obs'].shape}")
    print(f"[Verify] dones shape: {verify['dones'].shape}, episode_id shape: {verify['episode_id'].shape}")
    verify.close()

    print(f"\n[DONE] Dataset saved to: {args.output}")


if __name__ == "__main__":
    main()