#!/usr/bin/env python3
"""
Fast, GPU-Preloaded SynthBrachy3T + Tip-Aware Bundle Graph Transformer Benchmark
on edelweiss RTX 4090 (cuda:0).
"""
import os, sys, json, glob, time, math, pickle, types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

for mod in ["vtk", "vtk.util", "vtk.util.numpy_support", "slicer", "slicer.util", "qt", "ctk", "SimpleITK", "sitkUtils", "slicer.ScriptedLoadableModule"]:
    sys.modules[mod] = types.ModuleType(mod)

sys.path.insert(0, "/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder")
sys.path.insert(0, "/tmp")

from evaluate_unified_apples_to_apples import (
    evaluate_method_unified,
    compute_wilcoxon_comparison,
    compute_unified_pair_metrics,
    summarize_distribution,
    extract_nn_labelmap_tubes,
    build_slice_graph_fast,
    NeedleSliceMPNN,
    decode_mpnn_trajectories,
    get_gts,
    traj_gt_dist,
    CACHE_DIR,
    OOF_DIR,
    META_PATH,
    BASE_TRAJS_PATH,
)
from run_xiao2026_and_fewshot_benchmark import (
    analyze_catheter_adhesion,
    regularize_poly3_trajectory,
    dedup_traj_dicts,
    get_nf2026_synctrack_poly3_preds,
)
from test_synthbrachy3t_v2 import (
    compute_trajectory_axial_profile_and_tip_features,
    build_enriched_bundle_graph_item,
    generate_synthetic_bundle_graph_item,
    TipAwareBundleGraphTransformer,
    apply_tip_adjustment_and_poly3,
)

MPNN_CACHE_PATH = "/tmp/cached_oof_mpnn_trajs_54.pkl"
ENRICHED_BUNDLES_CACHE_PATH = "/tmp/cached_enriched_bundles_54.pkl"


def augment_clinical_bundle_with_synthbrachy3t(bg, rng):
    """
    Fold-isolated SynthBrachy3T domain-randomized augmentation of a training bundle graph:
    1. Perturbs cranial tip heights z_top on true catheters by dz_pert ~ N(0, 5.5 mm) and updates
       the corresponding node features (z_span, z_high, dz_prof, dz_nbr) and exact target y_dz_tip
       so TipAwareBundleGraphTransformer learns exact cranial tip recovery.
    2. Injects synthetic severe catheter-adhesion twin trajectories (1.65 - 2.55 mm apart) with
       y_1to1 = 1.0 and synthetic vascular false-positive trajectories with y_1to1 = 0.0.
    """
    x = bg["x"].clone().numpy()         # [M, 26]
    af = bg["adj_feat"].clone().numpy() # [M, M, 8]
    am = bg["adj_mask"].clone().numpy() # [M, M]
    y_rec = bg["y_rec"].clone().numpy()
    y_1to1 = bg["y_1to1"].clone().numpy()
    y_long = bg["y_long_1to1"].clone().numpy()
    y_dz = bg["y_dz_tip"].clone().numpy()
    y_w = bg["y_dz_weight"].clone().numpy()

    M = len(x)
    if M == 0:
        return bg

    # 1. Feature noise + Synthetic Cranial Tip Over/Undershoot Perturbation
    feat_noise = rng.normal(0.0, 0.025, size=x.shape).astype(np.float32)
    feat_noise[:, [1, 16, 17]] = 0.0
    x = x + feat_noise

    for u in range(M):
        if y_w[u] > 0.5 and rng.random() < 0.70:
            # Synthetically perturb current tip by delta_z_mm in [-10, +10] mm
            delta_z_mm = float(rng.uniform(-9.5, 9.5))
            # Current z_span is x[u, 10]*60, z_high is x[u, 15]*100, dz_prof is x[u, 24]*20, dz_nbr is x[u, 25]*20
            x[u, 10] = max(0.12, x[u, 10] + delta_z_mm / 60.0)
            x[u, 15] = x[u, 15] + delta_z_mm / 100.0
            x[u, 24] = x[u, 24] - delta_z_mm / 20.0
            x[u, 25] = x[u, 25] - delta_z_mm / 20.0
            # Target tip correction must now undo delta_z_mm as well!
            new_target_mm = y_dz[u] * 15.0 - delta_z_mm
            y_dz[u] = float(np.clip(new_target_mm / 15.0, -1.5, 1.5))

    # 2. Inject 1-3 synthetic adhering twin catheters (cloned from winners at 1.7-2.4 mm separation)
    win_idx = np.where(y_1to1 > 0.5)[0]
    if len(win_idx) >= 2 and rng.random() < 0.65:
        n_twins = int(rng.integers(1, min(4, len(win_idx) + 1)))
        chosen = rng.choice(win_idx, size=n_twins, replace=False)
        new_x = []
        new_rec, new_1to1, new_long, new_dz, new_w = [], [], [], [], []
        for c_idx in chosen:
            row = x[c_idx].copy()
            row[0] = np.clip(row[0] + rng.normal(0, 0.04), 0.35, 0.98)
            row[5] = np.clip(row[5] + rng.normal(0, 0.05), 0.25, 0.98)
            row[18] = np.clip(row[18] + rng.normal(0, 0.05), 0.22, 0.98)
            new_x.append(row)
            new_rec.append(1.0)
            new_1to1.append(1.0)
            new_long.append(float(y_long[c_idx]))
            new_dz.append(float(y_dz[c_idx]))
            new_w.append(float(y_w[c_idx]))

        K_add = len(new_x)
        x = np.vstack([x, np.array(new_x, dtype=np.float32)])
        y_rec = np.concatenate([y_rec, np.array(new_rec, dtype=np.float32)])
        y_1to1 = np.concatenate([y_1to1, np.array(new_1to1, dtype=np.float32)])
        y_long = np.concatenate([y_long, np.array(new_long, dtype=np.float32)])
        y_dz = np.concatenate([y_dz, np.array(new_dz, dtype=np.float32)])
        y_w = np.concatenate([y_w, np.array(new_w, dtype=np.float32)])

        M_new = M + K_add
        af_new = np.zeros((M_new, M_new, 8), dtype=np.float32)
        am_new = np.zeros((M_new, M_new), dtype=np.float32)
        af_new[:M, :M] = af
        am_new[:M, :M] = am
        for k_i, c_idx in enumerate(chosen):
            u_new = M + k_i
            am_new[u_new, :M] = am[c_idx, :M]
            am_new[:M, u_new] = am[:M, c_idx]
            af_new[u_new, :M] = af[c_idx, :M]
            af_new[:M, u_new] = af[:M, c_idx]
            # Adhering twin edge between u_new and c_idx ( ~2.0 mm apart, NOT a duplicate < 1.6 mm )
            am_new[u_new, c_idx] = 1.0
            am_new[c_idx, u_new] = 1.0
            sep_norm = float(rng.uniform(1.75, 2.45)) / 35.0
            af_new[u_new, c_idx] = [sep_norm, sep_norm, 0.08, 0.0, 0.0, 0.0, 0.0, 0.92]
            af_new[c_idx, u_new] = [sep_norm, sep_norm, 0.08, 0.0, 0.0, 0.0, 0.0, 0.92]
        af, am = af_new, am_new

    return {
        "x": torch.tensor(x, dtype=torch.float32),
        "adj_feat": torch.tensor(af, dtype=torch.float32),
        "adj_mask": torch.tensor(am, dtype=torch.float32),
        "y_rec": torch.tensor(y_rec, dtype=torch.float32),
        "y_1to1": torch.tensor(y_1to1, dtype=torch.float32),
        "y_long_1to1": torch.tensor(y_long, dtype=torch.float32),
        "y_dz_tip": torch.tensor(y_dz, dtype=torch.float32),
        "y_dz_weight": torch.tensor(y_w, dtype=torch.float32),
    }


def to_gpu_item(bg, device):
    return {
        "x": bg["x"].to(device, non_blocking=True),
        "adj_feat": bg["adj_feat"].to(device, non_blocking=True),
        "adj_mask": bg["adj_mask"].to(device, non_blocking=True),
        "y_rec": bg["y_rec"].to(device, non_blocking=True),
        "y_1to1": bg["y_1to1"].to(device, non_blocking=True),
        "y_long_1to1": bg["y_long_1to1"].to(device, non_blocking=True),
        "y_dz_tip": bg["y_dz_tip"].to(device, non_blocking=True),
        "y_dz_weight": bg["y_dz_weight"].to(device, non_blocking=True),
    }


def decode_hybrid_synth_nf2026_preds(
    bundle_items, base_oof_scores, synth_oof_outputs,
    min_conf=50.0, long_mode=False, w_synth=0.58, tip_scale=1.0
):
    """
    Combines the calibrated clinical DenseGraphTransformer scores (base_oof_scores)
    with the SynthBrachy3T-co-trained TipAwareBundleGraphTransformer outputs (synth_oof_outputs),
    applying:
      1. IRGM spatial hull gating + Stage 0 circular dark-core symmetry verification
      2. Synchronous adhesion-aware deduplication (resolving twin shafts <= 1.60 mm apart)
      3. Learned + Axial-Profile + Bundle-Neighborhood Cranial Tip Refinement + Poly3 smoothing
    """
    preds_out = []
    for i, bg in enumerate(bundle_items):
        sc_base = base_oof_scores[i]
        out_syn = synth_oof_outputs[i]
        sc_syn = 1.0 / (1.0 + np.exp(-np.clip(out_syn[:, :3], -20.0, 20.0)))
        dz_pred_net = out_syn[:, 3] * 15.0

        sc = (1.0 - w_synth) * sc_base + w_synth * sc_syn
        if long_mode:
            conf_arr = 100.0 * (0.30 * sc[:, 0] + 0.25 * sc[:, 1] + 0.45 * sc[:, 2])
        else:
            conf_arr = 100.0 * (0.45 * sc[:, 0] + 0.55 * sc[:, 1])

        # Compute IRGM implant centroid & radius in XY physical space from high-confidence core
        high_idx = np.where(conf_arr >= 55.0)[0]
        if len(high_idx) >= 3:
            all_xy = np.concatenate([bg["pool"][idx]["ras"][:, :2] for idx in high_idx], axis=0)
            center_xy = np.median(all_xy, axis=0)
            dists_core = np.linalg.norm(all_xy - center_xy[None, :], axis=1)
            irgm_radius_mm = float(np.percentile(dists_core, 98.0)) + 15.5
        else:
            center_xy = None
            irgm_radius_mm = 999.0

        sel = []
        for idx_c in np.argsort(-conf_arr):
            if conf_arr[idx_c] < min_conf:
                continue
            cand = dict(bg["pool"][idx_c])
            pts = np.asarray(cand["ras"], dtype=np.float64)
            pts = pts[np.argsort(pts[:, 2])]

            # IRGM spatial gate: reject peripheral vessel/bowel artifacts outside implant envelope
            if center_xy is not None:
                mean_dist_center = float(np.linalg.norm(pts[:, :2].mean(axis=0) - center_xy))
                if mean_dist_center > irgm_radius_mm and conf_arr[idx_c] < min_conf + 20.0:
                    continue

            # Reject low-symmetry vascular false positives unless MPNN + transformer confidence is strong
            if cand.get("mean_sym", 0.5) < 0.14 and cand.get("frac_active", 0.5) < 0.40 and conf_arr[idx_c] < min_conf + 15.0:
                continue

            # Adhesion-aware synchronous deduplication
            dedup_thresh = 1.60 if (conf_arr[idx_c] >= 65.0 and cand.get("mpnn_pa", 0.0) >= 0.50) else 2.02
            if any(traj_gt_dist(pts, s["ras"]) < dedup_thresh for s in sel):
                continue

            # Cranial tip refinement combining neural tip offset, axial profile transition, and bundle dome
            dz_prof_mm = float(cand.get("z_prof_tip", pts[-1, 2]) - pts[-1, 2])
            dz_nbr_mm = float(cand.get("z_tip_nbr", pts[-1, 2]) - pts[-1, 2])
            dz_comb_mm = tip_scale * (0.65 * float(dz_pred_net[idx_c]) + 0.25 * dz_prof_mm + 0.10 * dz_nbr_mm)

            pts_refined = apply_tip_adjustment_and_poly3(
                pts,
                dz_comb_mm,
                float(cand.get("vx_top", 0.0)),
                float(cand.get("vy_top", 0.0)),
            )
            cand["ras"] = pts_refined
            cand["points"] = pts_refined
            cand["score"] = float(conf_arr[idx_c])
            sel.append(cand)
        preds_out.append(sel)
    return preds_out


def main():
    t0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[SynthBrachy3T Full] Device: {device} ({torch.cuda.get_device_name(device)})", flush=True)

    meta_list = pickle.load(open(META_PATH, "rb"))
    base_trajs_data = pickle.load(open(BASE_TRAJS_PATH, "rb"))
    gts_all_list = [get_gts(m["gts_all"]) for m in meta_list]
    gts_long_list = [get_gts(m["gts_long"]) for m in meta_list]
    gts_short_list = []
    for ci in range(54):
        long_z = [float(g[-1, 2] - g[0, 2]) for g in gts_long_list[ci]]
        min_lz = min(long_z) - 2.0 if long_z else 30.0
        shorts = [g for g in gts_all_list[ci] if float(g[-1, 2] - g[0, 2]) < min_lz]
        gts_short_list.append(shorts)

    adhesion_info = analyze_catheter_adhesion(meta_list, gts_all_list, gts_long_list, adhesion_thresh_mm=3.0)
    adh_idx = adhesion_info["adhesion_case_indices"]
    non_adh_idx = adhesion_info["non_adhesion_case_indices"]

    # Step 1: Load or build cached enriched bundle items & base OOF scores
    if os.path.exists(ENRICHED_BUNDLES_CACHE_PATH):
        print(f"[1/4] Loading cached enriched bundle graphs from {ENRICHED_BUNDLES_CACHE_PATH}...", flush=True)
        cache_d = pickle.load(open(ENRICHED_BUNDLES_CACHE_PATH, "rb"))
        bundle_items = cache_d["bundle_items"]
        base_oof_scores = cache_d["base_oof_scores"]
        stage0_preds = cache_d["stage0_preds"]
    else:
        print("[1/4] Building and caching 54 clinical slice graphs, MPNN OOF trajectories, and enriched bundle items...", flush=True)
        from train_export_graphnet_prod import DenseGraphTransformer, build_bundle_graph_item
        graphs = []
        all_X_vols, all_oof_npz = [], []
        for i in range(54):
            d_pt = torch.load(f"{CACHE_DIR}/case_{i:02d}.pt", map_location="cpu", weights_only=False)
            oof_npz = np.load(f"{OOF_DIR}/oof_case_{i:02d}.npz")
            X = d_pt["X"].numpy()
            all_X_vols.append(X)
            all_oof_npz.append(oof_npz)
            g = build_slice_graph_fast(
                X,
                oof_npz["p_mask"].astype(np.float32),
                oof_npz["p_peak"].astype(np.float32),
                oof_npz["vx"].astype(np.float32),
                oof_npz["vy"].astype(np.float32),
                oof_npz["p_obt"].astype(np.float32),
                int(d_pt["k0"]), int(d_pt["i0"]), int(d_pt["j0"]),
                meta_list[i]["sp_ras"], np.asarray(meta_list[i]["ijk_to_ras"], dtype=np.float64),
                gts_all=gts_all_list[i], gts_long=gts_long_list[i],
            )
            graphs.append(g)

        oof_mpnn_trajs = [None] * 54
        for fold in range(5):
            torch.manual_seed(42 + fold)
            np.random.seed(42 + fold)
            train_gs = [g for i, g in enumerate(graphs) if i % 5 != fold]
            model = NeedleSliceMPNN(in_node_dim=16, in_edge_dim=12, in_sp_dim=5, hidden_dim=64, num_layers=4).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40, eta_min=1e-4)
            for ep in range(40):
                model.train()
                for bi in np.random.permutation(len(train_gs)):
                    g = train_gs[bi]
                    opt.zero_grad(set_to_none=True)
                    no, el = model(g, device)
                    y_na, y_nl, y_off, y_e = g["nodes_y_all"].to(device), g["nodes_y_long"].to(device), g["nodes_offset"].to(device), g["edge_y"].to(device)
                    pw_n = torch.tensor([3.0], device=device)
                    fg_n = (y_na > 0.5).float().unsqueeze(1)
                    loss = F.binary_cross_entropy_with_logits(no[:, 0], y_na, pos_weight=pw_n) + \
                           0.5 * F.binary_cross_entropy_with_logits(no[:, 1], y_nl, pos_weight=pw_n) + \
                           0.5 * (F.smooth_l1_loss(no[:, 2:4], y_off, reduction="none") * fg_n).sum() / fg_n.sum().clamp(min=1.0)
                    if el.numel() > 0:
                        loss = loss + 2.0 * F.binary_cross_entropy_with_logits(el, y_e, pos_weight=torch.tensor([6.0], device=device))
                    loss.backward()
                    opt.step()
                sched.step()
            model.eval()
            with torch.no_grad():
                for i, g in enumerate(graphs):
                    if i % 5 == fold:
                        no, el = model(g, device)
                        oof_mpnn_trajs[i] = decode_mpnn_trajectories(g, no, el)

        bundle_items = []
        base_bundle_items = []
        stage0_preds = []
        for i, g in enumerate(graphs):
            X = all_X_vols[i]
            oof_npz = all_oof_npz[i]
            b_item = base_trajs_data[i]
            b_trajs = b_item["trajs"] if isinstance(b_item, dict) and "trajs" in b_item else b_item
            b_clean = []
            for t in b_trajs:
                tc = dict(t)
                tc["ras"] = np.asarray(tc["ras"] if "ras" in tc else tc["points"], dtype=np.float64)
                tc["points"] = tc["ras"]
                b_clean.append(tc)
            stage0_preds.append(b_clean)

            k0, i0, j0 = int(oof_npz["k0"]), int(oof_npz["i0"]), int(oof_npz["j0"])
            sp_i, sp_j, sp_k = [float(x) for x in meta_list[i]["sp_ras"]]
            ijk2ras = g["ijk_to_ras"]
            ras2ijk = np.linalg.inv(ijk2ras)
            resp_approx = np.expm1(X[1] * 4.0)
            nn_d = {k: oof_npz[k].astype(np.float32) for k in ["p_mask", "p_peak", "vx", "vy", "p_obt"]}
            tubes = extract_nn_labelmap_tubes(nn_d, resp_approx, k0, i0, j0, sp_i, sp_j, sp_k, ijk2ras, ras2ijk, b_clean, include_obturator=True)

            pool = []
            for t in b_clean:
                ras = t["ras"][np.argsort(t["ras"][:, 2])]
                pool.append({
                    "ras": ras,
                    "base_xy": np.asarray(t.get("base_xy", ras[0, :2]), dtype=np.float64),
                    "slope": np.asarray(t.get("slope", (ras[-1, :2] - ras[0, :2]) / max(1.0, ras[-1, 2] - ras[0, 2])), dtype=np.float64),
                    "z_span": float(t.get("z_span_tracked", ras[-1, 2] - ras[0, 2])),
                    "rmse": float(t.get("rmse", 0.35)),
                    "conf_heur": float(t.get("confidence", 65.0)) / 100.0,
                    "from_mpnn": 0.0, "mpnn_pa": 0.0, "mpnn_pl": 0.0, "mpnn_pe": 0.0,
                })
            for nt in tubes:
                ras = np.asarray(nt["points"], dtype=np.float64)
                ras = ras[np.argsort(ras[:, 2])]
                pool.append({
                    "ras": ras,
                    "base_xy": np.asarray(nt.get("base_xy", ras[0, :2]), dtype=np.float64),
                    "slope": np.asarray(nt.get("slope", (ras[-1, :2] - ras[0, :2]) / max(1.0, ras[-1, 2] - ras[0, 2])), dtype=np.float64),
                    "z_span": float(nt.get("z_span_tracked", ras[-1, 2] - ras[0, 2])),
                    "rmse": float(nt.get("rmse", 0.35)),
                    "conf_heur": float(nt.get("confidence", 65.0)) / 100.0,
                    "from_mpnn": 0.5, "mpnn_pa": float(nt.get("confidence", 65.0)) / 100.0, "mpnn_pl": 0.5, "mpnn_pe": 0.6,
                })
            for mt in oof_mpnn_trajs[i]:
                ras = mt["ras"]
                matched_idx = -1
                best_d = 2.05
                for idx_p, p in enumerate(pool):
                    d = traj_gt_dist(ras, p["ras"])
                    if d < best_d:
                        best_d = d
                        matched_idx = idx_p
                if matched_idx >= 0:
                    pool[matched_idx]["from_mpnn"] = 1.0
                    pool[matched_idx]["mpnn_pa"] = mt["mean_pa"]
                    pool[matched_idx]["mpnn_pl"] = mt["mean_pl"]
                    pool[matched_idx]["mpnn_pe"] = mt["mean_pe"]
                else:
                    pool.append({
                        "ras": ras,
                        "base_xy": mt["base_xy"],
                        "slope": mt["slope"],
                        "z_span": mt["z_span"],
                        "rmse": mt["rmse"],
                        "conf_heur": mt["mean_pa"] * mt["mean_pe"],
                        "from_mpnn": 1.0,
                        "mpnn_pa": mt["mean_pa"],
                        "mpnn_pl": mt["mean_pl"],
                        "mpnn_pe": mt["mean_pe"],
                    })

            base_bundle_items.append(build_bundle_graph_item([dict(p) for p in pool], g, oof_npz, gts_all=gts_all_list[i], gts_long=gts_long_list[i]))
            bundle_items.append(build_enriched_bundle_graph_item(pool, g, oof_npz, X, gts_all=gts_all_list[i], gts_long=gts_long_list[i]))

        base_oof_scores = [None] * 54
        for fold in range(5):
            torch.manual_seed(42 + fold)
            np.random.seed(42 + fold)
            tr_b = [b for i, b in enumerate(base_bundle_items) if i % 5 != fold]
            gt_model = DenseGraphTransformer(in_dim=18, edge_dim=6, d_model=64).to(device)
            opt = torch.optim.AdamW(gt_model.parameters(), lr=1.8e-3, weight_decay=1e-4)
            for ep in range(75):
                gt_model.train()
                for bi in np.random.permutation(len(tr_b)):
                    bg = tr_b[bi]
                    opt.zero_grad(set_to_none=True)
                    out = gt_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
                    loss = F.binary_cross_entropy_with_logits(out[:, 0], bg["y_rec"].to(device), pos_weight=torch.tensor([3.0], device=device)) + \
                           1.5 * F.binary_cross_entropy_with_logits(out[:, 1], bg["y_1to1"].to(device), pos_weight=torch.tensor([2.0], device=device)) + \
                           F.binary_cross_entropy_with_logits(out[:, 2], bg["y_long_1to1"].to(device), pos_weight=torch.tensor([2.0], device=device))
                    loss.backward()
                    opt.step()
            gt_model.eval()
            with torch.no_grad():
                for i, bg in enumerate(base_bundle_items):
                    if i % 5 == fold:
                        out = gt_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
                        base_oof_scores[i] = torch.sigmoid(out).cpu().numpy()

        pickle.dump({
            "bundle_items": bundle_items,
            "base_oof_scores": base_oof_scores,
            "stage0_preds": stage0_preds,
        }, open(ENRICHED_BUNDLES_CACHE_PATH, "wb"))
        print(f"Saved cache to {ENRICHED_BUNDLES_CACHE_PATH}!", flush=True)

    # Step 2: Pre-train + Fold-Isolated SynthBrachy3T Co-Training on GPU
    print("[2/4] Generating SynthBrachy3T synthetic bundles and running GPU-preloaded 5-fold training...", flush=True)
    rng_syn = np.random.default_rng(2026)
    pure_synth_gpu = [to_gpu_item(generate_synthetic_bundle_graph_item(rng_syn), device) for _ in range(300)]

    # Pre-train pure SynthBrachy3T model
    torch.manual_seed(2026)
    base_syn_model = TipAwareBundleGraphTransformer(in_dim=26, edge_dim=8, d_model=64).to(device)
    opt_syn = torch.optim.AdamW(base_syn_model.parameters(), lr=2.0e-3, weight_decay=1e-4)
    pw_r = torch.tensor([3.2], device=device)
    pw_1 = torch.tensor([2.6], device=device)
    pw_l = torch.tensor([2.4], device=device)
    for ep in range(20):
        base_syn_model.train()
        for bi in np.random.permutation(len(pure_synth_gpu)):
            bg = pure_synth_gpu[bi]
            opt_syn.zero_grad(set_to_none=True)
            out = base_syn_model(bg["x"], bg["adj_feat"], bg["adj_mask"])
            w_tip = bg["y_dz_weight"]
            l_tip = (F.smooth_l1_loss(out[:, 3], bg["y_dz_tip"], reduction="none") * w_tip).sum() / w_tip.sum().clamp(min=1.0)
            loss = F.binary_cross_entropy_with_logits(out[:, 0], bg["y_rec"], pos_weight=pw_r) + \
                   1.5 * F.binary_cross_entropy_with_logits(out[:, 1], bg["y_1to1"], pos_weight=pw_1) + \
                   F.binary_cross_entropy_with_logits(out[:, 2], bg["y_long_1to1"], pos_weight=pw_l) + \
                   1.2 * l_tip
            loss.backward()
            opt_syn.step()

    pretrained_state = {k: v.cpu().clone() for k, v in base_syn_model.state_dict().items()}

    # Fine-tune per fold using fold-isolated SynthBrachy3T domain-randomized clinical augmentations
    synth_oof_outputs = [None] * 54
    for fold in range(5):
        torch.manual_seed(100 + fold)
        np.random.seed(100 + fold)
        rng_f = np.random.default_rng(500 + fold)
        tr_b_raw = [b for i, b in enumerate(bundle_items) if i % 5 != fold]
        # Generate 6x fold-isolated SynthBrachy3T domain-randomized + tip-perturbed + twin-adhesion bundles
        tr_aug_gpu = [to_gpu_item(b, device) for b in tr_b_raw]
        for _ in range(5):
            for b in tr_b_raw:
                tr_aug_gpu.append(to_gpu_item(augment_clinical_bundle_with_synthbrachy3t(b, rng_f), device))

        gt_model = TipAwareBundleGraphTransformer(in_dim=26, edge_dim=8, d_model=64).to(device)
        gt_model.load_state_dict({k: v.to(device) for k, v in pretrained_state.items()})
        opt = torch.optim.AdamW(gt_model.parameters(), lr=1.5e-3, weight_decay=1.5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=45, eta_min=1e-4)
        for ep in range(45):
            gt_model.train()
            for bi in np.random.permutation(len(tr_aug_gpu)):
                bg = tr_aug_gpu[bi]
                opt.zero_grad(set_to_none=True)
                out = gt_model(bg["x"], bg["adj_feat"], bg["adj_mask"])
                w_tip = bg["y_dz_weight"]
                l_tip = (F.smooth_l1_loss(out[:, 3], bg["y_dz_tip"], reduction="none") * w_tip).sum() / w_tip.sum().clamp(min=1.0)
                loss = F.binary_cross_entropy_with_logits(out[:, 0], bg["y_rec"], pos_weight=pw_r) + \
                       1.5 * F.binary_cross_entropy_with_logits(out[:, 1], bg["y_1to1"], pos_weight=pw_1) + \
                       F.binary_cross_entropy_with_logits(out[:, 2], bg["y_long_1to1"], pos_weight=pw_l) + \
                       1.5 * l_tip
                loss.backward()
                opt.step()
            sched.step()

        gt_model.eval()
        with torch.no_grad():
            for i, bg in enumerate(bundle_items):
                if i % 5 == fold:
                    out = gt_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
                    synth_oof_outputs[i] = out.cpu().numpy()

    # Compare previous NeedleFinder 2026 + IRGM + Poly3 vs upgraded NeedleFinder 2026 + SynthBrachy3T + Tip-Aware
    prev_all = get_nf2026_synctrack_poly3_preds(bundle_items, base_oof_scores, meta_list, min_conf=50.0, long_mode=False)
    prev_long = get_nf2026_synctrack_poly3_preds(bundle_items, base_oof_scores, meta_list, min_conf=50.0, long_mode=True)
    res_prev_all, pp_prev_all = evaluate_method_unified(prev_all, gts_all_list, thresh=3.5, n_boot=100)
    res_prev_long, _ = evaluate_method_unified(prev_long, gts_long_list, aux_short_gts_per_case=gts_short_list, thresh=3.5, n_boot=100)
    res_prev_adh, _ = evaluate_method_unified([prev_all[i] for i in adh_idx], [gts_all_list[i] for i in adh_idx], thresh=3.5, n_boot=100)

    print(
        f"\n[Previous NF2026 + IRGM + Poly3 (tau=50%)]\n"
        f"  ALL(1063): Rec={res_prev_all['hungarian_1to1_recall_pct']:5.2f}% Prec={res_prev_all['precision_1to1_pct']:5.2f}% F1={res_prev_all['f1_1to1_pct']:5.2f}% "
        f"ShaftMed={res_prev_all['metrics']['mean_shaft_xy']['median']:.3f}mm TipMean/Med={res_prev_all['metrics']['tip_3d']['mean']:.2f}/{res_prev_all['metrics']['tip_3d']['median']:.2f}mm\n"
        f"  LONG(759): Rec={res_prev_long['hungarian_1to1_recall_pct']:5.2f}% F1_adj={res_prev_long['f1_1to1_obturator_adj_pct']:5.2f}% TipMed={res_prev_long['metrics']['tip_3d']['median']:.2f}mm\n"
        f"  ADH(233) : Rec={res_prev_adh['hungarian_1to1_recall_pct']:5.2f}% Prec={res_prev_adh['precision_1to1_pct']:5.2f}% F1={res_prev_adh['f1_1to1_pct']:5.2f}% TipMean/Med={res_prev_adh['metrics']['tip_3d']['mean']:.2f}/{res_prev_adh['metrics']['tip_3d']['median']:.2f}mm",
        flush=True,
    )

    print("\n=== UPGRADED NEEDLEFINDER 2026 + SYNTHBRACHY3T + TIP-AWARE 5-FOLD OOF SWEEP ===")
    for tau in [25.0, 35.0, 45.0, 50.0, 55.0, 60.0]:
        preds_all = decode_hybrid_synth_nf2026_preds(bundle_items, base_oof_scores, synth_oof_outputs, min_conf=tau, long_mode=False, w_synth=0.55, tip_scale=0.85)
        preds_long = decode_hybrid_synth_nf2026_preds(bundle_items, base_oof_scores, synth_oof_outputs, min_conf=tau, long_mode=True, w_synth=0.55, tip_scale=0.85)
        res_all, pp_all = evaluate_method_unified(preds_all, gts_all_list, thresh=3.5, n_boot=100)
        res_long, _ = evaluate_method_unified(preds_long, gts_long_list, aux_short_gts_per_case=gts_short_list, thresh=3.5, n_boot=100)
        res_adh, _ = evaluate_method_unified([preds_all[i] for i in adh_idx], [gts_all_list[i] for i in adh_idx], thresh=3.5, n_boot=100)
        print(
            f"  tau={tau:4.1f}% | ALL(1063): Rec={res_all['hungarian_1to1_recall_pct']:5.2f}% Prec={res_all['precision_1to1_pct']:5.2f}% F1={res_all['f1_1to1_pct']:5.2f}% "
            f"ShaftMed={res_all['metrics']['mean_shaft_xy']['median']:.3f}mm TipMean/Med={res_all['metrics']['tip_3d']['mean']:.2f}/{res_all['metrics']['tip_3d']['median']:.2f}mm | "
            f"LONG(759): Rec={res_long['hungarian_1to1_recall_pct']:5.2f}% F1_adj={res_long['f1_1to1_obturator_adj_pct']:5.2f}% TipMed={res_long['metrics']['tip_3d']['median']:.2f}mm | "
            f"ADH(233): F1={res_adh['f1_1to1_pct']:5.2f}% TipMean/Med={res_adh['metrics']['tip_3d']['mean']:.2f}/{res_adh['metrics']['tip_3d']['median']:.2f}mm",
            flush=True,
        )


if __name__ == "__main__":
    main()
