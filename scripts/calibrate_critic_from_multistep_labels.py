"""
Offline Critic Calibration from Multistep Return Labels
=========================================================

加载 P37 checkpoint 的 critic_1 / critic_2，
用多步真实回报标签做离线校准，目标：
  1. Q(bc_actor) > Q(bad_action) + margin
  2. Q(bc_actor) 轻量回归 discounted_return_bc

用法示例（dry-run）：
    python scripts/calibrate_critic_from_multistep_labels.py --dry_run

用法示例（实际校准）：
    python scripts/calibrate_critic_from_multistep_labels.py \
        --labels_path _tmp_multistep_return_labels_p50_small.npz \
        --candidate_dataset_path _tmp_near_bc_candidate_transition_dataset.npz \
        --checkpoint_path "models/.../final_step_2000.pth" \
        --output _tmp_calibrated_critic.pth

安全机制：
    - 默认不训练（需要 --output 且非 --dry_run）
    - 不覆盖已存在文件（需要 --force）
    - 保护原始 checkpoint 和关键数据集不被覆盖
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
import torch.nn.functional as F

from agents.td3_agent import QValueNet

# =============================================================================
# Constants
# =============================================================================
PROTECTED_FILES = [
    "_tmp_near_bc_candidate_transition_dataset.npz",
    "_tmp_multistep_return_labels_p50_small.npz",
    "_tmp_bc_gimbal_only_actor.pth",
]
PROTECTED_DIR_PREFIXES = ["models", "runs"]
EXPECTED_STATE_DIM = 22
EXPECTED_ACTION_DIM = 4

DEFAULT_CHECKPOINT_DIR = (
    "v0.5.18-debug-residual-safe-rl-roadmap-3-gf50e1f5-dirty_20260429_202142"
    "_envs1_seed42_gr0.5_g0.997_gimbal_only_att1.0_om0.0_wb0.0_ga0.0_wa0.0"
)


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

    rel_posix = os.path.normpath(rel).replace("\\", "/")

    for prefix in PROTECTED_DIR_PREFIXES:
        if rel_posix == prefix or rel_posix.startswith(prefix + "/"):
            return True

    return False


# =============================================================================
# Data loading and validation
# =============================================================================
def load_data(labels_path, candidate_path):
    labels = np.load(labels_path, allow_pickle=False)
    candidate = np.load(candidate_path, allow_pickle=False)

    required_label_fields = [
        "row_index", "action_source", "group_episode_id",
        "group_base_step_id", "init_attitude_deg",
        "discounted_return", "q_current",
    ]
    for f in required_label_fields:
        if f not in labels:
            raise ValueError("Missing field in labels: {}".format(f))

    required_candidate_fields = ["obs", "actions", "action_source"]
    for f in required_candidate_fields:
        if f not in candidate:
            raise ValueError("Missing field in candidate dataset: {}".format(f))

    n_labels = len(labels["row_index"])
    n_cand = len(candidate["obs"])

    max_row = int(labels["row_index"].max())
    if max_row >= n_cand:
        raise ValueError("row_index {} >= candidate size {}".format(max_row, n_cand))

    for i in range(n_labels):
        ri = int(labels["row_index"][i])
        lbl_src = str(labels["action_source"][i])
        cand_src = str(candidate["action_source"][ri])
        if lbl_src != cand_src:
            raise ValueError(
                "Row {}: label source '{}' != candidate source '{}'".format(
                    ri, lbl_src, cand_src))

    obs_shape = candidate["obs"].shape
    act_shape = candidate["actions"].shape
    if obs_shape[1] != EXPECTED_STATE_DIM:
        raise ValueError("Expected state_dim={}, got {}".format(
            EXPECTED_STATE_DIM, obs_shape[1]))
    if act_shape[1] != EXPECTED_ACTION_DIM:
        raise ValueError("Expected action_dim={}, got {}".format(
            EXPECTED_ACTION_DIM, act_shape[1]))

    if not np.all(np.isfinite(labels["discounted_return"])):
        raise ValueError("discounted_return contains non-finite values")
    if not np.all(np.isfinite(labels["q_current"])):
        raise ValueError("q_current contains non-finite values")

    return labels, candidate, n_labels, n_cand


# =============================================================================
# Group construction
# =============================================================================
def build_groups_from_labels(labels, bad_sources, bc_source):
    n = len(labels["row_index"])
    gkey = np.array(
        [(int(labels["group_episode_id"][i]),
          int(labels["group_base_step_id"][i]),
          float(labels["init_attitude_deg"][i]))
         for i in range(n)],
        dtype=[("ep", "i8"), ("bs", "i8"), ("iad", "f8")],
    )
    ug, ginv = np.unique(gkey, return_inverse=True)

    groups = []
    skipped = 0
    for g in range(len(ug)):
        rows = np.where(ginv == g)[0]
        src_map = {}
        for r in rows:
            src_map[str(labels["action_source"][r])] = r

        if bc_source not in src_map:
            skipped += 1
            continue

        missing = [s for s in bad_sources if s not in src_map]
        if missing:
            skipped += 1
            continue

        groups.append({
            "bc": src_map[bc_source],
            "bad": {s: src_map[s] for s in bad_sources},
        })

    return groups, skipped


# =============================================================================
# Critic loading
# =============================================================================
def load_critics(checkpoint_path, device="cpu"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))

    ckpt = torch.load(checkpoint_path, map_location=device)
    required = ["critic_1_state_dict", "critic_2_state_dict"]
    for k in required:
        if k not in ckpt:
            raise ValueError("Missing key in checkpoint: {}".format(k))

    c1 = QValueNet(EXPECTED_STATE_DIM, 256, EXPECTED_ACTION_DIM).to(device)
    c2 = QValueNet(EXPECTED_STATE_DIM, 256, EXPECTED_ACTION_DIM).to(device)
    c1.load_state_dict(ckpt["critic_1_state_dict"])
    c2.load_state_dict(ckpt["critic_2_state_dict"])
    c1.eval()
    c2.eval()

    tc1 = QValueNet(EXPECTED_STATE_DIM, 256, EXPECTED_ACTION_DIM).to(device)
    tc2 = QValueNet(EXPECTED_STATE_DIM, 256, EXPECTED_ACTION_DIM).to(device)
    if "target_critic_1_state_dict" in ckpt:
        tc1.load_state_dict(ckpt["target_critic_1_state_dict"])
    else:
        tc1.load_state_dict(ckpt["critic_1_state_dict"])
    if "target_critic_2_state_dict" in ckpt:
        tc2.load_state_dict(ckpt["target_critic_2_state_dict"])
    else:
        tc2.load_state_dict(ckpt["critic_2_state_dict"])
    tc1.eval()
    tc2.eval()

    return c1, c2, tc1, tc2, ckpt


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_ranking(c1, c2, groups, labels, candidate, device="cpu", bc_source="bc_actor"):
    c1.eval(); c2.eval()

    bc_q1_list = []
    bc_q2_list = []
    bad_q1 = {s: [] for s in groups[0]["bad"].keys()}
    bad_q2 = {s: [] for s in groups[0]["bad"].keys()}

    with torch.no_grad():
        for g in groups:
            bc_row = g["bc"]
            obs_bc = torch.FloatTensor(
                candidate["obs"][labels["row_index"][bc_row]]
            ).unsqueeze(0).to(device)
            act_bc = torch.FloatTensor(
                candidate["actions"][labels["row_index"][bc_row]]
            ).unsqueeze(0).to(device)
            q1_bc = float(c1(obs_bc, act_bc).cpu().item())
            q2_bc = float(c2(obs_bc, act_bc).cpu().item())
            bc_q1_list.append(q1_bc)
            bc_q2_list.append(q2_bc)

            for s, row in g["bad"].items():
                obs_b = torch.FloatTensor(
                    candidate["obs"][labels["row_index"][row]]
                ).unsqueeze(0).to(device)
                act_b = torch.FloatTensor(
                    candidate["actions"][labels["row_index"][row]]
                ).unsqueeze(0).to(device)
                q1_b = float(c1(obs_b, act_b).cpu().item())
                q2_b = float(c2(obs_b, act_b).cpu().item())
                bad_q1[s].append(q1_b)
                bad_q2[s].append(q2_b)

    bc_q1 = np.array(bc_q1_list)
    bc_q2 = np.array(bc_q2_list)

    results = {}
    for s in bad_q1:
        bq1 = np.array(bad_q1[s])
        bq2 = np.array(bad_q2[s])
        q1_pos = (bc_q1 > bq1).mean()
        q2_pos = (bc_q2 > bq2).mean()
        q_min_pos = (np.minimum(bc_q1, bc_q2) > np.minimum(bq1, bq2)).mean()
        results[s] = {
            "q1_pos_rate": q1_pos,
            "q2_pos_rate": q2_pos,
            "q_min_pos_rate": q_min_pos,
        }

    top_counts = {s: 0 for s in list(bad_q1.keys()) + [bc_source]}
    for gi in range(len(groups)):
        scores = {bc_source: min(bc_q1[gi], bc_q2[gi])}
        for s in bad_q1:
            scores[s] = min(bad_q1[s][gi], bad_q2[s][gi])
        winner = max(scores, key=scores.get)
        top_counts[winner] += 1

    return results, top_counts


# =============================================================================
# Training
# =============================================================================
def train_critics(c1, c2, tc1, tc2, groups, labels, candidate,
                  steps, lr, margin, rank_weight, reg_weight,
                  batch_groups, seed, device="cpu"):
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(list(c1.parameters()) + list(c2.parameters()), lr=lr)

    n_groups = len(groups)
    print("\n[Train] Starting {} steps, {} groups, batch_groups={}".format(
        steps, n_groups, batch_groups))

    for step in range(steps):
        c1.train(); c2.train()

        if n_groups <= batch_groups:
            batch_idx = list(range(n_groups))
        else:
            batch_idx = rng.choice(n_groups, size=batch_groups, replace=False).tolist()

        rank_losses = []
        reg_losses = []

        for gi in batch_idx:
            g = groups[gi]
            bc_row = g["bc"]
            obs_bc = torch.FloatTensor(
                candidate["obs"][labels["row_index"][bc_row]]
            ).unsqueeze(0).to(device)
            act_bc = torch.FloatTensor(
                candidate["actions"][labels["row_index"][bc_row]]
            ).unsqueeze(0).to(device)
            ret_bc = float(labels["discounted_return"][bc_row])
            ret_bc_t = torch.FloatTensor([ret_bc]).to(device)

            q1_bc = c1(obs_bc, act_bc)
            q2_bc = c2(obs_bc, act_bc)

            reg_losses.append(F.mse_loss(q1_bc.view(-1), ret_bc_t.view(-1)))
            reg_losses.append(F.mse_loss(q2_bc.view(-1), ret_bc_t.view(-1)))

            for s, row in g["bad"].items():
                obs_b = torch.FloatTensor(
                    candidate["obs"][labels["row_index"][row]]
                ).unsqueeze(0).to(device)
                act_b = torch.FloatTensor(
                    candidate["actions"][labels["row_index"][row]]
                ).unsqueeze(0).to(device)

                q1_b = c1(obs_b, act_b)
                q2_b = c2(obs_b, act_b)

                rank_losses.append(torch.clamp(margin - (q1_bc - q1_b), min=0.0).mean())
                rank_losses.append(torch.clamp(margin - (q2_bc - q2_b), min=0.0).mean())

        rank_loss = torch.stack(rank_losses).mean() if rank_losses else torch.tensor(0.0, device=device)
        reg_loss = torch.stack(reg_losses).mean() if reg_losses else torch.tensor(0.0, device=device)
        loss = rank_weight * rank_loss + reg_weight * reg_loss

        if not torch.isfinite(loss):
            raise RuntimeError("Loss became non-finite at step {}".format(step + 1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Soft update targets
        tau = 0.005
        with torch.no_grad():
            for param, target_param in zip(c1.parameters(), tc1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(c2.parameters(), tc2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        if (step + 1) % 100 == 0 or step == 0:
            print("  step {:4d}/{}  loss={:.6f}  rank={:.6f}  reg={:.6f}".format(
                step + 1, steps, loss.item(), rank_loss.item(), reg_loss.item()))

    print("[Train] Complete.")
    return c1, c2, tc1, tc2


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline Critic Calibration from Multistep Return Labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--labels_path", type=str,
        default="_tmp_multistep_return_labels_p50_small.npz",
        help="Multistep return labels npz",
    )
    parser.add_argument(
        "--candidate_dataset_path", type=str,
        default="_tmp_near_bc_candidate_transition_dataset.npz",
        help="Candidate transition dataset npz",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="P37 checkpoint path (default: auto-detect)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output calibrated critic pth (required for training)",
    )
    parser.add_argument(
        "--steps", type=int, default=500,
        help="Number of calibration steps",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Critic learning rate",
    )
    parser.add_argument(
        "--margin", type=float, default=0.01,
        help="Rank margin",
    )
    parser.add_argument(
        "--rank_weight", type=float, default=1.0,
        help="Weight for rank loss",
    )
    parser.add_argument(
        "--reg_weight", type=float, default=0.1,
        help="Weight for regression loss",
    )
    parser.add_argument(
        "--batch_groups", type=int, default=8,
        help="Number of groups per batch",
    )
    parser.add_argument(
        "--seed", type=int, default=20260506,
        help="Random seed",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Device",
    )
    parser.add_argument(
        "--bc_source", type=str, default="bc_actor",
        help="Source name for the good action",
    )
    parser.add_argument(
        "--bad_sources", type=str,
        default="zero_action,p37_final_actor,final_x2_clipped",
        help="Comma-separated bad action sources",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only load and evaluate, no training",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Allow overwriting existing output",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    bad_sources = [s.strip() for s in args.bad_sources.split(",") if s.strip()]

    print("=" * 60)
    print("Offline Critic Calibration")
    print("=" * 60)
    print("  labels_path:    {}".format(args.labels_path))
    print("  candidate_path: {}".format(args.candidate_dataset_path))
    print("  checkpoint:     {}".format(args.checkpoint_path or "auto-detect"))
    print("  output:         {}".format(args.output or "(none)"))
    print("  steps:          {}".format(args.steps))
    print("  lr:             {}".format(args.lr))
    print("  margin:         {}".format(args.margin))
    print("  rank_weight:   {}".format(args.rank_weight))
    print("  reg_weight:    {}".format(args.reg_weight))
    print("  batch_groups:  {}".format(args.batch_groups))
    print("  seed:           {}".format(args.seed))
    print("  device:         {}".format(args.device))
    print("  bc_source:      {}".format(args.bc_source))
    print("  bad_sources:    {}".format(bad_sources))
    print("  dry_run:        {}".format(args.dry_run))

    # Output safety
    if args.output is not None:
        if is_protected(args.output):
            print("\nERROR: output path is protected: {}".format(args.output))
            sys.exit(1)
        if os.path.exists(args.output) and not args.force:
            print("\nERROR: output already exists: {}".format(args.output))
            sys.exit(1)

    if not args.dry_run and args.output is None:
        print("\nERROR: --output is required for training. Use --dry_run to preview.")
        sys.exit(1)

    # Load data
    print("\n[Load] Labels...")
    labels, candidate, n_labels, n_cand = load_data(
        args.labels_path, args.candidate_dataset_path)
    print("[Load] {} labels, {} candidate transitions".format(n_labels, n_cand))

    # Build groups
    print("[Groups] Building...")
    groups, skipped = build_groups_from_labels(labels, bad_sources, args.bc_source)
    print("[Groups] {} valid groups, {} skipped".format(len(groups), skipped))

    if len(groups) == 0:
        print("ERROR: No valid groups found.")
        sys.exit(1)

    # Load critics
    ckpt_path = args.checkpoint_path
    if ckpt_path is None:
        ckpt_path = os.path.join("models", DEFAULT_CHECKPOINT_DIR, "final_step_2000.pth")

    print("[Load] Critics from: {}".format(ckpt_path))
    c1, c2, tc1, tc2, ckpt = load_critics(ckpt_path, device=args.device)
    print("[Load] Critics loaded.")

    # Initial evaluation
    print("\n[Eval] Initial Q ranking (before calibration)...")
    rank_results, top_counts = evaluate_ranking(
        c1, c2, groups, labels, candidate,
        device=args.device, bc_source=args.bc_source)

    for s in bad_sources:
        r = rank_results[s]
        print("  {}: Q1_pos={:.1f}%  Q2_pos={:.1f}%  Qmin_pos={:.1f}%".format(
            s, r["q1_pos_rate"] * 100, r["q2_pos_rate"] * 100,
            r["q_min_pos_rate"] * 100))

    print("  top_by_Q:")
    for s, cnt in sorted(top_counts.items(), key=lambda x: -x[1]):
        print("    {}: {}/{} ({:.0f}%)".format(
            s, cnt, len(groups), cnt / len(groups) * 100))

    if args.dry_run:
        print("\n[DRY RUN] Plan:")
        print("  groups:        {}".format(len(groups)))
        print("  steps:         {}".format(args.steps))
        print("  lr:            {}".format(args.lr))
        print("  margin:         {}".format(args.margin))
        print("  rank_weight:   {}".format(args.rank_weight))
        print("  reg_weight:    {}".format(args.reg_weight))
        print("  batch_groups:   {}".format(args.batch_groups))
        print("  DRY RUN: no training, no optimizer step, no checkpoint written")
        sys.exit(0)

    # Training
    c1, c2, tc1, tc2 = train_critics(
        c1, c2, tc1, tc2, groups, labels, candidate,
        steps=args.steps, lr=args.lr, margin=args.margin,
        rank_weight=args.rank_weight, reg_weight=args.reg_weight,
        batch_groups=args.batch_groups, seed=args.seed, device=args.device,
    )

    # Final evaluation
    print("\n[Eval] Final Q ranking (after calibration)...")
    rank_results, top_counts = evaluate_ranking(
        c1, c2, groups, labels, candidate,
        device=args.device, bc_source=args.bc_source)

    for s in bad_sources:
        r = rank_results[s]
        print("  {}: Q1_pos={:.1f}%  Q2_pos={:.1f}%  Qmin_pos={:.1f}%".format(
            s, r["q1_pos_rate"] * 100, r["q2_pos_rate"] * 100,
            r["q_min_pos_rate"] * 100))

    print("  top_by_Q:")
    for s, cnt in sorted(top_counts.items(), key=lambda x: -x[1]):
        print("    {}: {}/{} ({:.0f}%)".format(
            s, cnt, len(groups), cnt / len(groups) * 100))

    # Save
    save_dict = {
        "critic_1_state_dict": c1.state_dict(),
        "critic_2_state_dict": c2.state_dict(),
        "target_critic_1_state_dict": tc1.state_dict(),
        "target_critic_2_state_dict": tc2.state_dict(),
        "source_checkpoint_path": ckpt_path,
        "labels_path": args.labels_path,
        "candidate_dataset_path": args.candidate_dataset_path,
        "steps": args.steps,
        "lr": args.lr,
        "margin": args.margin,
        "rank_weight": args.rank_weight,
        "reg_weight": args.reg_weight,
        "batch_groups": args.batch_groups,
        "seed": args.seed,
        "bc_source": args.bc_source,
        "bad_sources": bad_sources,
        "n_groups": len(groups),
        "metadata": "Offline critic calibration from multistep return labels",
    }

    torch.save(save_dict, args.output)
    print("\n[Save] Written to: {}".format(args.output))
    print("[DONE]")


if __name__ == "__main__":
    main()
