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
def build_groups_from_labels(labels, bad_sources, bc_source, ranking_mode="bc_vs_bad"):
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

        if ranking_mode == "bc_vs_bad":
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
                "rows": src_map,
            })
        else:
            if len(src_map) < 2:
                skipped += 1
                continue
            entry = {
                "rows": src_map,
                "bad": {s: src_map[s] for s in bad_sources if s in src_map},
            }
            if bc_source in src_map:
                entry["bc"] = src_map[bc_source]
            groups.append(entry)

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
def evaluate_ranking(c1, c2, groups, labels, candidate, device="cpu",
                     bc_source="bc_actor", return_margin_eps=1e-8):
    c1.eval(); c2.eval()

    n_groups = len(groups)
    row_index = labels["row_index"]
    disc_ret = labels["discounted_return"]

    all_sources = set()
    all_bad_sources = set()
    for g in groups:
        all_sources.update(g["rows"].keys())
        if "bad" in g:
            all_bad_sources.update(g["bad"].keys())

    group_qmaps = []
    group_retmaps = []

    with torch.no_grad():
        for g in groups:
            qmap = {}
            retmap = {}
            for src, lbl_row in g["rows"].items():
                obs_t = torch.FloatTensor(
                    candidate["obs"][int(row_index[lbl_row])]
                ).unsqueeze(0).to(device)
                act_t = torch.FloatTensor(
                    candidate["actions"][int(row_index[lbl_row])]
                ).unsqueeze(0).to(device)
                q1 = float(c1(obs_t, act_t).cpu().item())
                q2 = float(c2(obs_t, act_t).cpu().item())
                qmap[src] = {"q1": q1, "q2": q2, "qmin": min(q1, q2)}
                retmap[src] = float(disc_ret[lbl_row])
            group_qmaps.append(qmap)
            group_retmaps.append(retmap)

    # --- Old bc-vs-bad results (per-group) ---
    results = {}
    for s in all_bad_sources:
        q1_pos_list = []
        q2_pos_list = []
        qmin_pos_list = []
        for gi in range(n_groups):
            qmap = group_qmaps[gi]
            if bc_source in qmap and s in qmap:
                q1_pos_list.append(float(qmap[bc_source]["q1"] > qmap[s]["q1"]))
                q2_pos_list.append(float(qmap[bc_source]["q2"] > qmap[s]["q2"]))
                qmin_pos_list.append(float(qmap[bc_source]["qmin"] > qmap[s]["qmin"]))
        if q1_pos_list:
            results[s] = {
                "q1_pos_rate": np.mean(q1_pos_list),
                "q2_pos_rate": np.mean(q2_pos_list),
                "q_min_pos_rate": np.mean(qmin_pos_list),
                "n_compared": len(q1_pos_list),
            }
        else:
            results[s] = {
                "q1_pos_rate": 0.0,
                "q2_pos_rate": 0.0,
                "q_min_pos_rate": 0.0,
                "n_compared": 0,
            }

    # --- Top-by-Q ---
    top_counts = {s: 0 for s in all_sources}
    for gi in range(n_groups):
        qmap = group_qmaps[gi]
        scores = {src: qmap[src]["qmin"] for src in qmap}
        winner = max(scores, key=scores.get)
        top_counts[winner] += 1

    # --- Return-based metrics ---
    ret_top_counts = {s: 0 for s in all_sources}
    top_alignment = 0
    pairwise_agree = 0
    pairwise_total = 0

    for gi in range(n_groups):
        retmap = group_retmaps[gi]
        qmap = group_qmaps[gi]
        srcs = list(retmap.keys())

        ret_winner = max(retmap, key=retmap.get)
        ret_top_counts[ret_winner] += 1

        q_scores = {src: qmap[src]["qmin"] for src in qmap}
        q_winner = max(q_scores, key=q_scores.get)
        if q_winner == ret_winner:
            top_alignment += 1

        for i_idx in range(len(srcs)):
            for j_idx in range(i_idx + 1, len(srcs)):
                si = srcs[i_idx]
                sj = srcs[j_idx]
                ri = retmap[si]
                rj = retmap[sj]
                if abs(ri - rj) < return_margin_eps:
                    continue
                q_i = qmap[si]["qmin"]
                q_j = qmap[sj]["qmin"]
                pairwise_total += 1
                if (ri > rj) == (q_i > q_j):
                    pairwise_agree += 1

    if pairwise_total > 0:
        pairwise_rate = pairwise_agree / pairwise_total * 100
    else:
        pairwise_rate = 0.0

    return_metrics = {
        "top_by_return": ret_top_counts,
        "top_by_Q": top_counts,
        "top_alignment": (top_alignment, n_groups, top_alignment / n_groups * 100),
        "pairwise_agreement": (pairwise_agree, pairwise_total, pairwise_rate),
    }

    return results, top_counts, return_metrics


# =============================================================================
# Training
# =============================================================================
def train_critics(c1, c2, tc1, tc2, groups, labels, candidate,
                  steps, lr, margin, rank_weight, reg_weight,
                  batch_groups, seed, device="cpu",
                  ranking_mode="bc_vs_bad",
                  return_margin_eps=1e-8,
                  zero_rank_weight=2.0,
                  regression_mode="bc",
                  bc_source="bc_actor"):
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(list(c1.parameters()) + list(c2.parameters()), lr=lr)

    n_groups = len(groups)
    row_index = labels["row_index"]
    disc_ret = labels["discounted_return"]

    print("\n[Train] Starting {} steps, {} groups, batch_groups={}".format(
        steps, n_groups, batch_groups))
    print("        ranking_mode={}  regression_mode={}".format(
        ranking_mode, regression_mode))

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
            rows_map = g["rows"]

            if ranking_mode == "bc_vs_bad":
                if "bc" not in g:
                    continue

                bc_lbl = g["bc"]
                obs_bc = torch.FloatTensor(
                    candidate["obs"][int(row_index[bc_lbl])]
                ).unsqueeze(0).to(device)
                act_bc = torch.FloatTensor(
                    candidate["actions"][int(row_index[bc_lbl])]
                ).unsqueeze(0).to(device)
                ret_bc = float(disc_ret[bc_lbl])
                ret_bc_t = torch.FloatTensor([ret_bc]).to(device)

                q1_bc = c1(obs_bc, act_bc)
                q2_bc = c2(obs_bc, act_bc)

                reg_losses.append(F.mse_loss(q1_bc.view(-1), ret_bc_t.view(-1)))
                reg_losses.append(F.mse_loss(q2_bc.view(-1), ret_bc_t.view(-1)))

                for s, bad_lbl in g["bad"].items():
                    obs_b = torch.FloatTensor(
                        candidate["obs"][int(row_index[bad_lbl])]
                    ).unsqueeze(0).to(device)
                    act_b = torch.FloatTensor(
                        candidate["actions"][int(row_index[bad_lbl])]
                    ).unsqueeze(0).to(device)
                    q1_b = c1(obs_b, act_b)
                    q2_b = c2(obs_b, act_b)
                    rank_losses.append(
                        torch.clamp(margin - (q1_bc - q1_b), min=0.0).mean())
                    rank_losses.append(
                        torch.clamp(margin - (q2_bc - q2_b), min=0.0).mean())

            else:
                srcs = list(rows_map.keys())

                retmap = {}
                q1_map = {}
                q2_map = {}
                for src in srcs:
                    lbl_row = rows_map[src]
                    obs_t = torch.FloatTensor(
                        candidate["obs"][int(row_index[lbl_row])]
                    ).unsqueeze(0).to(device)
                    act_t = torch.FloatTensor(
                        candidate["actions"][int(row_index[lbl_row])]
                    ).unsqueeze(0).to(device)
                    retmap[src] = float(disc_ret[lbl_row])
                    q1_map[src] = c1(obs_t, act_t)
                    q2_map[src] = c2(obs_t, act_t)

                for i_idx in range(len(srcs)):
                    for j_idx in range(i_idx + 1, len(srcs)):
                        si = srcs[i_idx]
                        sj = srcs[j_idx]
                        ri = retmap[si]
                        rj = retmap[sj]

                        if abs(ri - rj) < return_margin_eps:
                            continue

                        if ri > rj:
                            q1_w, q2_w = q1_map[si], q2_map[si]
                            q1_l, q2_l = q1_map[sj], q2_map[sj]
                            loser_src = sj
                        else:
                            q1_w, q2_w = q1_map[sj], q2_map[sj]
                            q1_l, q2_l = q1_map[si], q2_map[si]
                            loser_src = si

                        pair_loss_q1 = torch.clamp(
                            margin - (q1_w - q1_l), min=0.0).mean()
                        pair_loss_q2 = torch.clamp(
                            margin - (q2_w - q2_l), min=0.0).mean()

                        w = zero_rank_weight if loser_src == "zero_action" else 1.0
                        rank_losses.append(w * pair_loss_q1)
                        rank_losses.append(w * pair_loss_q2)

                if regression_mode == "top":
                    top_src = max(retmap, key=retmap.get)
                    q1_top = q1_map[top_src]
                    q2_top = q2_map[top_src]
                    ret_top_t = torch.FloatTensor([retmap[top_src]]).to(device)
                    reg_losses.append(F.mse_loss(q1_top.view(-1), ret_top_t.view(-1)))
                    reg_losses.append(F.mse_loss(q2_top.view(-1), ret_top_t.view(-1)))

                elif regression_mode == "bc":
                    if "bc" in g:
                        bc_lbl = g["bc"]
                        obs_bc = torch.FloatTensor(
                            candidate["obs"][int(row_index[bc_lbl])]
                        ).unsqueeze(0).to(device)
                        act_bc = torch.FloatTensor(
                            candidate["actions"][int(row_index[bc_lbl])]
                        ).unsqueeze(0).to(device)
                        ret_bc_t = torch.FloatTensor([float(disc_ret[bc_lbl])]).to(device)
                        q1_bc = c1(obs_bc, act_bc)
                        q2_bc = c2(obs_bc, act_bc)
                        reg_losses.append(F.mse_loss(q1_bc.view(-1), ret_bc_t.view(-1)))
                        reg_losses.append(F.mse_loss(q2_bc.view(-1), ret_bc_t.view(-1)))

                elif regression_mode == "all":
                    for src in srcs:
                        lbl_row = rows_map[src]
                        obs_r = torch.FloatTensor(
                            candidate["obs"][int(row_index[lbl_row])]
                        ).unsqueeze(0).to(device)
                        act_r = torch.FloatTensor(
                            candidate["actions"][int(row_index[lbl_row])]
                        ).unsqueeze(0).to(device)
                        ret_r_t = torch.FloatTensor([retmap[src]]).to(device)
                        q1_r = c1(obs_r, act_r)
                        q2_r = c2(obs_r, act_r)
                        reg_losses.append(F.mse_loss(q1_r.view(-1), ret_r_t.view(-1)))
                        reg_losses.append(F.mse_loss(q2_r.view(-1), ret_r_t.view(-1)))

        rank_loss = torch.stack(rank_losses).mean() if rank_losses else torch.tensor(0.0, device=device)
        reg_loss = torch.stack(reg_losses).mean() if reg_losses else torch.tensor(0.0, device=device)
        loss = rank_weight * rank_loss + reg_weight * reg_loss

        if not torch.isfinite(loss):
            raise RuntimeError("Loss became non-finite at step {}".format(step + 1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tau = 0.005
        with torch.no_grad():
            for p, tp in zip(c1.parameters(), tc1.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
            for p, tp in zip(c2.parameters(), tc2.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

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
    parser.add_argument(
        "--ranking_mode", type=str, default="bc_vs_bad",
        choices=["bc_vs_bad", "return_pairwise"],
        help=(
            "Ranking loss mode. bc_vs_bad keeps the old behavior. "
            "return_pairwise ranks actions by discounted_return within each group."
        ),
    )
    parser.add_argument(
        "--return_margin_eps", type=float, default=1e-8,
        help="Minimum return difference required to create a pairwise ranking constraint.",
    )
    parser.add_argument(
        "--zero_rank_weight", type=float, default=2.0,
        help=(
            "Extra multiplier for ranking pairs where zero_action is the loser. "
            "Used only in return_pairwise mode."
        ),
    )
    parser.add_argument(
        "--regression_mode", type=str, default="bc",
        choices=["none", "bc", "top", "all"],
        help=(
            "Which actions receive light return regression. "
            "For return_pairwise, top is recommended. For bc_vs_bad, bc is the default."
        ),
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
    print("  ranking_mode:   {}".format(args.ranking_mode))
    print("  return_margin_eps: {}".format(args.return_margin_eps))
    print("  zero_rank_weight: {}".format(args.zero_rank_weight))
    print("  regression_mode: {}".format(args.regression_mode))
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
    groups, skipped = build_groups_from_labels(
        labels, bad_sources, args.bc_source, ranking_mode=args.ranking_mode
    )
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
    rank_results, top_counts, init_rm = evaluate_ranking(
        c1, c2, groups, labels, candidate,
        device=args.device, bc_source=args.bc_source,
        return_margin_eps=args.return_margin_eps)

    for s in bad_sources:
        if s in rank_results:
            r = rank_results[s]
            print("  {}: Q1_pos={:.1f}%  Q2_pos={:.1f}%  Qmin_pos={:.1f}%".format(
                s, r["q1_pos_rate"] * 100, r["q2_pos_rate"] * 100,
                r["q_min_pos_rate"] * 100))

    print("  top_by_Q:")
    for s, cnt in sorted(top_counts.items(), key=lambda x: -x[1]):
        print("    {}: {}/{} ({:.0f}%)".format(
            s, cnt, len(groups), cnt / len(groups) * 100))

    print("  top_by_return:")
    for s, cnt in sorted(init_rm["top_by_return"].items(), key=lambda x: -x[1]):
        print("    {}: {}/{} ({:.0f}%)".format(
            s, cnt, len(groups), cnt / len(groups) * 100))
    align_n, align_tot, align_pct = init_rm["top_alignment"]
    print("  top_alignment_with_return: {}/{} ({:.1f}%)".format(
        align_n, align_tot, align_pct))
    pa_n, pa_tot, pa_pct = init_rm["pairwise_agreement"]
    print("  pairwise_return_agreement: {}/{} ({:.1f}%)".format(
        pa_n, pa_tot, pa_pct))

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
        ranking_mode=args.ranking_mode,
        return_margin_eps=args.return_margin_eps,
        zero_rank_weight=args.zero_rank_weight,
        regression_mode=args.regression_mode,
        bc_source=args.bc_source,
    )

    # Final evaluation
    print("\n[Eval] Final Q ranking (after calibration)...")
    rank_results, top_counts, final_rm = evaluate_ranking(
        c1, c2, groups, labels, candidate,
        device=args.device, bc_source=args.bc_source,
        return_margin_eps=args.return_margin_eps)

    for s in bad_sources:
        if s in rank_results:
            r = rank_results[s]
            print("  {}: Q1_pos={:.1f}%  Q2_pos={:.1f}%  Qmin_pos={:.1f}%".format(
                s, r["q1_pos_rate"] * 100, r["q2_pos_rate"] * 100,
                r["q_min_pos_rate"] * 100))

    print("  top_by_Q:")
    for s, cnt in sorted(top_counts.items(), key=lambda x: -x[1]):
        print("    {}: {}/{} ({:.0f}%)".format(
            s, cnt, len(groups), cnt / len(groups) * 100))

    print("  top_by_return:")
    for s, cnt in sorted(final_rm["top_by_return"].items(), key=lambda x: -x[1]):
        print("    {}: {}/{} ({:.0f}%)".format(
            s, cnt, len(groups), cnt / len(groups) * 100))
    align_n, align_tot, align_pct = final_rm["top_alignment"]
    print("  top_alignment_with_return: {}/{} ({:.1f}%)".format(
        align_n, align_tot, align_pct))
    pa_n, pa_tot, pa_pct = final_rm["pairwise_agreement"]
    print("  pairwise_return_agreement: {}/{} ({:.1f}%)".format(
        pa_n, pa_tot, pa_pct))

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
        "ranking_mode": args.ranking_mode,
        "return_margin_eps": args.return_margin_eps,
        "zero_rank_weight": args.zero_rank_weight,
        "regression_mode": args.regression_mode,
        "n_groups": len(groups),
        "metadata": "Offline critic calibration from multistep return labels",
    }

    torch.save(save_dict, args.output)
    print("\n[Save] Written to: {}".format(args.output))
    print("[DONE]")


if __name__ == "__main__":
    main()
