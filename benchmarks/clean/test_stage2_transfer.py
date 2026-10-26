#!/usr/bin/env python3
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from reproducibility.paths import resolve_path
import copy
import glob
import math
import os
import pickle
import re
import sys
import time
import types
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/vault/paper/benchmarks_rerun"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/vault/paper/benchmarks_clean"))

from evaluate_unified_apples_to_apples import (
    evaluate_method_unified,
    extract_nn_labelmap_tubes,
    build_slice_graph_fast,
    DenseGraphTransformer,
    NeedleSliceMPNN,
    decode_mpnn_trajectories,
    get_gts,
    traj_gt_dist,
)
from prepare_and_inspect_dataset import read_nrrd_local, read_vtk_centerline_local
from prepare_clean_dataset import (
    estimate_axial_limit_slice_existing,
    compute_tilt_safe_pelvic_klim,
    build_clean_input_tensor,
    StandaloneEvalLogic,
    MockVolumeNode,
    MockMatrix4x4,
)
from train_and_eval_unet3d import NeedleUNet3D
from train_nested_stage1 import infer_4view_tta_logits, raw_out_to_probs
from run_stage2_and_tip_clean import (
    calibrate_stage1_oof_for_volume,
    stitch_collinear_pool_fragments,
    build_bundle_graphs_with_multirater,
    decode_clean_stage2_preds,
    Dedicated1DTipLocalizer,
    TipAwareBundleGraphTransformer,
    get_nested_fold_split,
    derive_unstitched_and_base_bgs,
    generate_synthetic_bundle_graph_item_v2,
    augment_clinical_bundle_with_synthbrachy3t_v2,
    to_gpu_item,
)
import NeedleFinderLib.TrackingAlgorithms as ta_mod

ta_mod.slicer.modules = types.SimpleNamespace(NeedleFinderWidget=None)
ta_mod.vtk.vtkMatrix4x4 = MockMatrix4x4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_or_train_5fold_stage2_models(device):
    cache_pt = resolve_path("/tmp/stage2_clean_5fold_models_cache.pt")
    s2_preds_data = pickle.load(
        open(resolve_path("/home/guillaume/github/needlefinder/vault/sources/2026-10-23-stage2-clean-final-oof-predictions.pkl"), "rb")
    )
    tip_models_5fold = []
    for st in s2_preds_data["tip_models_5fold_states"]:
        m = Dedicated1DTipLocalizer(in_channels=22, hidden_dim=48, seq_len=91).to(device)
        m.load_state_dict({k: v.to(device) for k, v in st.items()})
        m.eval()
        tip_models_5fold.append(m)

    if os.path.exists(cache_pt):
        saved = torch.load(cache_pt, map_location="cpu", weights_only=False)
        mpnn_models = []
        base_gt_models = []
        syn_gt_models = []
        for f_out in range(5):
            m_mpnn = NeedleSliceMPNN(in_node_dim=16, in_edge_dim=12, in_sp_dim=5, hidden_dim=64, num_layers=4)
            m_mpnn.load_state_dict(saved["mpnn_states"][f_out])
            m_mpnn.eval()
            mpnn_models.append(m_mpnn)

            m_base = DenseGraphTransformer(in_dim=18, edge_dim=6, d_model=64).to(device)
            m_base.load_state_dict({k: v.to(device) for k, v in saved["base_gt_states"][f_out].items()})
            m_base.eval()
            base_gt_models.append(m_base)

            m_syn = TipAwareBundleGraphTransformer(in_dim=26, edge_dim=8, d_model=64).to(device)
            m_syn.load_state_dict({k: v.to(device) for k, v in saved["syn_gt_states"][f_out].items()})
            m_syn.eval()
            syn_gt_models.append(m_syn)
        return mpnn_models, base_gt_models, syn_gt_models, tip_models_5fold

    print("Training & caching 5-fold Stage 2A/2B models...", flush=True)
    bundle_cache = pickle.load(open(resolve_path("/tmp/stage2_clean_bundles_nn_models_clean.pkl"), "rb"))
    graphs = bundle_cache["graphs"]
    stitched_enriched_bgs = bundle_cache["stitched_enriched_bgs"]
    _, _, stitched_base_bgs, _ = derive_unstitched_and_base_bgs(stitched_enriched_bgs)
    mpnn_epochs = bundle_cache.get("mpnn_epochs", [17, 13, 22, 20, 16])

    mpnn_states = []
    base_gt_states = []
    syn_gt_states = []

    cpu_dev = torch.device("cpu")
    pw_n = torch.tensor([3.0], device=cpu_dev)
    pw_e = torch.tensor([6.0], device=cpu_dev)

    for f_out in range(5):
        _, _, inner_subtrain, _ = get_nested_fold_split(f_out, 54)
        torch.manual_seed(42 + f_out)
        np.random.seed(42 + f_out)
        m_mpnn = NeedleSliceMPNN(in_node_dim=16, in_edge_dim=12, in_sp_dim=5, hidden_dim=64, num_layers=4).to(cpu_dev)
        opt = torch.optim.AdamW(m_mpnn.parameters(), lr=2.0e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40, eta_min=1e-4)
        for ep in range(mpnn_epochs[f_out]):
            m_mpnn.train()
            for idx in np.random.permutation(inner_subtrain):
                opt.zero_grad(set_to_none=True)
                g_item = graphs[idx]
                no, el = m_mpnn(g_item, cpu_dev)
                y_na = g_item["nodes_y_all"]
                y_nl = g_item["nodes_y_long"]
                y_off = g_item["nodes_offset"]
                y_e = g_item["edge_y"]
                fg_n = (y_na > 0.5).float().unsqueeze(1)
                loss = (
                    F.binary_cross_entropy_with_logits(no[:, 0], y_na, pos_weight=pw_n)
                    + 0.5 * F.binary_cross_entropy_with_logits(no[:, 1], y_nl, pos_weight=pw_n)
                    + 0.5 * (F.smooth_l1_loss(no[:, 2:4], y_off, reduction="none") * fg_n).sum() / fg_n.sum().clamp(min=1.0)
                )
                if el.numel() > 0:
                    loss = loss + 2.0 * F.binary_cross_entropy_with_logits(el, y_e, pos_weight=pw_e)
                loss.backward()
                opt.step()
            sched.step()
        mpnn_states.append({k: v.cpu().clone() for k, v in m_mpnn.state_dict().items()})

    # Train 5-fold DenseGraphTransformer and TipAwareBundleGraphTransformer
    pw_r = torch.tensor([3.2], device=device)
    pw_1 = torch.tensor([2.6], device=device)
    pw_l = torch.tensor([2.4], device=device)
    rng_syn = np.random.default_rng(2026)
    pure_synth_gpu = [to_gpu_item(generate_synthetic_bundle_graph_item_v2(rng_syn), device) for _ in range(300)]
    torch.manual_seed(2026)
    base_syn_model = TipAwareBundleGraphTransformer(in_dim=26, edge_dim=8, d_model=64).to(device)
    opt_syn = torch.optim.AdamW(base_syn_model.parameters(), lr=2.0e-3, weight_decay=1e-4)
    for ep in range(20):
        base_syn_model.train()
        for bi in np.random.permutation(len(pure_synth_gpu)):
            bg = pure_synth_gpu[bi]
            opt_syn.zero_grad(set_to_none=True)
            out = base_syn_model(bg["x"], bg["adj_feat"], bg["adj_mask"])
            w_tip = bg["y_dz_weight"]
            l_tip = (F.smooth_l1_loss(out[:, 3], bg["y_dz_tip"], reduction="none") * w_tip).sum() / w_tip.sum().clamp(min=1.0)
            loss = (
                F.binary_cross_entropy_with_logits(out[:, 0], bg["y_rec"], pos_weight=pw_r)
                + 1.5 * F.binary_cross_entropy_with_logits(out[:, 1], bg["y_1to1"], pos_weight=pw_1)
                + F.binary_cross_entropy_with_logits(out[:, 2], bg["y_long_1to1"], pos_weight=pw_l)
                + 1.2 * l_tip
            )
            loss.backward()
            opt_syn.step()
    pretrained_state = {k: v.cpu().clone() for k, v in base_syn_model.state_dict().items()}

    pw_br = torch.tensor([3.0], device=device)
    pw_b1 = torch.tensor([2.0], device=device)
    pw_bl = torch.tensor([2.0], device=device)

    for f_out in range(5):
        _, _, inner_subtrain, inner_val = get_nested_fold_split(f_out, 54)
        torch.manual_seed(42 + f_out)
        np.random.seed(42 + f_out)
        gt_model = DenseGraphTransformer(in_dim=18, edge_dim=6, d_model=64).to(device)
        opt = torch.optim.AdamW(gt_model.parameters(), lr=1.8e-3, weight_decay=1e-4)
        best_val = float("inf")
        best_st = None
        for ep in range(75):
            gt_model.train()
            for idx in np.random.permutation(inner_subtrain):
                bg = stitched_base_bgs[idx]
                opt.zero_grad(set_to_none=True)
                out = gt_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
                loss = (
                    F.binary_cross_entropy_with_logits(out[:, 0], bg["y_rec"].to(device), pos_weight=pw_br)
                    + 1.5 * F.binary_cross_entropy_with_logits(out[:, 1], bg["y_1to1"].to(device), pos_weight=pw_b1)
                    + F.binary_cross_entropy_with_logits(out[:, 2], bg["y_long_1to1"].to(device), pos_weight=pw_bl)
                )
                loss.backward()
                opt.step()
            if ep >= 20:
                gt_model.eval()
                with torch.no_grad():
                    vl = float(np.mean([
                        (
                            F.binary_cross_entropy_with_logits(
                                gt_model(stitched_base_bgs[idx]["x"].to(device), stitched_base_bgs[idx]["adj_feat"].to(device), stitched_base_bgs[idx]["adj_mask"].to(device))[:, 0],
                                stitched_base_bgs[idx]["y_rec"].to(device), pos_weight=pw_br,
                            )
                        ).item()
                        for idx in inner_val
                    ]))
                if vl < best_val:
                    best_val = vl
                    best_st = {k: v.cpu().clone() for k, v in gt_model.state_dict().items()}
        base_gt_states.append(best_st)

        torch.manual_seed(100 + f_out)
        np.random.seed(100 + f_out)
        rng_f = np.random.default_rng(500 + f_out)
        sub_raw = [stitched_enriched_bgs[idx] for idx in inner_subtrain]
        val_gpu = [to_gpu_item(stitched_enriched_bgs[idx], device) for idx in inner_val]
        tr_gpu = [to_gpu_item(b, device) for b in sub_raw]
        for _ in range(4):
            for b in sub_raw:
                tr_gpu.append(to_gpu_item(augment_clinical_bundle_with_synthbrachy3t_v2(b, rng_f), device))

        syn_m = TipAwareBundleGraphTransformer(in_dim=26, edge_dim=8, d_model=64).to(device)
        syn_m.load_state_dict({k: v.to(device) for k, v in pretrained_state.items()})
        opt_s = torch.optim.AdamW(syn_m.parameters(), lr=1.5e-3, weight_decay=1.5e-4)
        sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=45, eta_min=1e-4)
        best_val_s = float("inf")
        best_st_s = None
        for ep in range(45):
            syn_m.train()
            for bi in np.random.permutation(len(tr_gpu)):
                bg = tr_gpu[bi]
                opt_s.zero_grad(set_to_none=True)
                out = syn_m(bg["x"], bg["adj_feat"], bg["adj_mask"])
                w_tip = bg["y_dz_weight"]
                l_tip = (F.smooth_l1_loss(out[:, 3], bg["y_dz_tip"], reduction="none") * w_tip).sum() / w_tip.sum().clamp(min=1.0)
                loss = (
                    F.binary_cross_entropy_with_logits(out[:, 0], bg["y_rec"], pos_weight=pw_r)
                    + 1.5 * F.binary_cross_entropy_with_logits(out[:, 1], bg["y_1to1"], pos_weight=pw_1)
                    + F.binary_cross_entropy_with_logits(out[:, 2], bg["y_long_1to1"], pos_weight=pw_l)
                    + 1.5 * l_tip
                )
                loss.backward()
                opt_s.step()
            sched_s.step()
            if ep >= 12:
                syn_m.eval()
                with torch.no_grad():
                    vls = []
                    for vg in val_gpu:
                        out = syn_m(vg["x"], vg["adj_feat"], vg["adj_mask"])
                        w_tip = vg["y_dz_weight"]
                        l_tip = (F.smooth_l1_loss(out[:, 3], vg["y_dz_tip"], reduction="none") * w_tip).sum() / w_tip.sum().clamp(min=1.0)
                        vls.append(float((
                            F.binary_cross_entropy_with_logits(out[:, 0], vg["y_rec"], pos_weight=pw_r)
                            + 1.5 * F.binary_cross_entropy_with_logits(out[:, 1], vg["y_1to1"], pos_weight=pw_1)
                            + F.binary_cross_entropy_with_logits(out[:, 2], vg["y_long_1to1"], pos_weight=pw_l)
                            + 1.5 * l_tip
                        ).item()))
                if float(np.mean(vls)) < best_val_s:
                    best_val_s = float(np.mean(vls))
                    best_st_s = {k: v.cpu().clone() for k, v in syn_m.state_dict().items()}
        syn_gt_states.append(best_st_s)

    torch.save({
        "mpnn_states": mpnn_states,
        "base_gt_states": base_gt_states,
        "syn_gt_states": syn_gt_states,
    }, cache_pt)
    return get_or_train_5fold_stage2_models(device)


def run_full_nf2026_on_volume(
    vol, sp_ras, ijk2ras, k_lim_use, stage1_ckpt_paths, mpnn_models, base_gt_models, syn_gt_models, tip_models,
    device, fold_indices=None, zero_stage1_physics=False, zero_stage2_physics=False, min_conf=50.0
):
    if fold_indices is None:
        fold_indices = list(range(len(mpnn_models)))
    K, J, I = vol.shape
    sp_i, sp_j, sp_k = float(sp_ras[0]), float(sp_ras[1]), float(sp_ras[2])
    ras2ijk = np.linalg.inv(ijk2ras)
    k0 = max(0, k_lim_use - 1)

    margin_i = min(I // 2 - 4, int(round(68.0 / sp_i)))
    margin_j = min(J // 2 - 4, int(round(68.0 / sp_j)))
    half_w = ((margin_i + 7) // 8) * 8
    half_h = ((margin_j + 7) // 8) * 8
    ci_v, cj_v = I // 2, J // 2
    i0 = max(0, ci_v - half_w)
    i1 = min(I, i0 + 2 * half_w)
    if (i1 - i0) % 8 != 0:
        i1 = i0 + ((i1 - i0) // 8) * 8
    j0 = max(0, cj_v - half_h)
    j1 = min(J, j0 + 2 * half_h)
    if (j1 - j0) % 8 != 0:
        j1 = j0 + ((j1 - j0) // 8) * 8

    X, resp, cs, rsym = build_clean_input_tensor(vol, sp_ras, k0, K, i0, i1, j0, j1)
    if zero_stage1_physics:
        X_in = X.copy()
        X_in[1:4] = 0.0
    else:
        X_in = X

    x_t = torch.from_numpy(X_in)
    logits_sum = None
    for cp in stage1_ckpt_paths:
        m_u = NeedleUNet3D(6, 24).to(device)
        m_u.load_state_dict(torch.load(cp, map_location=device, weights_only=True))
        m_u.eval()
        ov = infer_4view_tta_logits(m_u, x_t, device, use_tta=True)
        logits_sum = ov if logits_sum is None else (logits_sum + ov)
    probs = raw_out_to_probs(logits_sum / len(stage1_ckpt_paths), temp_pos=1.08).numpy()
    oof_raw = {"p_mask": probs[0], "p_peak": probs[1], "vx": probs[2], "vy": probs[3], "p_obt": probs[4], "k0": k0, "i0": i0, "j0": j0}

    X_for_s2 = X.copy()
    if zero_stage1_physics or zero_stage2_physics:
        X_for_s2[1:4] = 0.0
    oof_cal = calibrate_stage1_oof_for_volume(oof_raw, X_for_s2, sp_ras)

    if zero_stage1_physics:
        b_clean = []
    else:
        logic = StandaloneEvalLogic()
        vnode = MockVolumeNode(vol, sp_ras, ijk2ras)
        s0_t = logic.autoDetectAllNeedles(vnode, minConfidence=35.0, kLim=k_lim_use, renderNeedles=False, script=True)
        b_clean = []
        for t in s0_t:
            tc = dict(t)
            tc["ras"] = np.asarray(tc["points"], dtype=np.float64)
            tc["points"] = tc["ras"]
            b_clean.append(tc)

    g = build_slice_graph_fast(
        X_for_s2, oof_cal["p_mask"], oof_cal["p_peak"], oof_cal["vx"], oof_cal["vy"], oof_cal["p_obt"],
        k0, i0, j0, sp_ras, ijk2ras, gts_all=None, gts_long=None
    )
    if g is None:
        g = {
            "ijk_to_ras": ijk2ras,
            "sp_ras": sp_ras,
            "z_base_ras": float((ijk2ras @ np.array([0.0, 0.0, float(k0), 1.0]))[2]),
            "nodes_feat": None,
        }

    # Run MPNN ensemble across fold_indices
    if g.get("nodes_feat") is not None and g["nodes_feat"].shape[0] > 0:
        no_sum, el_sum = None, None
        cpu_dev = torch.device("cpu")
        with torch.no_grad():
            for fi in fold_indices:
                no_f, el_f = mpnn_models[fi](g, cpu_dev)
                no_sum = no_f if no_sum is None else (no_sum + no_f)
                el_sum = el_f if el_sum is None else (el_sum + el_f)
        no_avg = no_sum / len(fold_indices)
        el_avg = el_sum / len(fold_indices)
        mpnn_trajs = decode_mpnn_trajectories(g, no_avg, el_avg) if no_avg is not None and no_avg.shape[0] > 0 else []
    else:
        mpnn_trajs = []

    resp_approx = np.expm1(X_for_s2[1] * 4.0)
    nn_d = {k: oof_cal[k] for k in ["p_mask", "p_peak", "vx", "vy", "p_obt"]}
    tubes = extract_nn_labelmap_tubes(nn_d, resp_approx, k0, i0, j0, sp_i, sp_j, sp_k, ijk2ras, ras2ijk, b_clean, include_obturator=True)

    pool_raw = []
    for t in b_clean:
        ras = t["ras"][np.argsort(t["ras"][:, 2])]
        pool_raw.append({
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
        pool_raw.append({
            "ras": ras,
            "base_xy": np.asarray(nt.get("base_xy", ras[0, :2]), dtype=np.float64),
            "slope": np.asarray(nt.get("slope", (ras[-1, :2] - ras[0, :2]) / max(1.0, ras[-1, 2] - ras[0, 2])), dtype=np.float64),
            "z_span": float(nt.get("z_span_tracked", ras[-1, 2] - ras[0, 2])),
            "rmse": float(nt.get("rmse", 0.35)),
            "conf_heur": float(nt.get("confidence", 65.0)) / 100.0,
            "from_mpnn": 0.5, "mpnn_pa": float(nt.get("confidence", 65.0)) / 100.0, "mpnn_pl": 0.5, "mpnn_pe": 0.6,
        })
    for mt in mpnn_trajs:
        ras = mt["ras"]
        matched_idx = -1
        best_d = 2.05
        for idx_p, p in enumerate(pool_raw):
            d = traj_gt_dist(ras, p["ras"])
            if d < best_d:
                best_d = d
                matched_idx = idx_p
        if matched_idx >= 0:
            pool_raw[matched_idx]["from_mpnn"] = 1.0
            pool_raw[matched_idx]["mpnn_pa"] = mt["mean_pa"]
            pool_raw[matched_idx]["mpnn_pl"] = mt["mean_pl"]
            pool_raw[matched_idx]["mpnn_pe"] = mt["mean_pe"]
        else:
            pool_raw.append({
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

    if len(pool_raw) == 0:
        return [], b_clean

    pool_stitched = stitch_collinear_pool_fragments(copy.deepcopy(pool_raw), g, oof_cal, X_for_s2)
    sb_base, sb_enr = build_bundle_graphs_with_multirater(pool_stitched, g, oof_cal, X_for_s2)

    with torch.no_grad():
        base_sc_list = []
        syn_out_list = []
        for fi in fold_indices:
            b_out = base_gt_models[fi](sb_base["x"].to(device), sb_base["adj_feat"].to(device), sb_base["adj_mask"].to(device))
            base_sc_list.append(torch.sigmoid(b_out).cpu().numpy())
            s_out = syn_gt_models[fi](sb_enr["x"].to(device), sb_enr["adj_feat"].to(device), sb_enr["adj_mask"].to(device))
            syn_out_list.append(s_out.cpu().numpy())
    base_sc_avg = np.mean(np.stack(base_sc_list, axis=0), axis=0)
    syn_out_avg = np.mean(np.stack(syn_out_list, axis=0), axis=0)

    preds = decode_clean_stage2_preds(
        [sb_enr], [base_sc_avg], [syn_out_avg],
        graphs=[g], all_oof_cal=[oof_cal], all_X_vols=[X_for_s2],
        tip_models_5fold=[tip_models[fold_indices[0]]] * 5,
        device=device, min_conf=min_conf, long_mode=False, w_synth=0.60,
        use_collinear_dedup=True, use_1d_tip_model=True, tip_scale=0.36
    )[0]
    return preds, b_clean


if __name__ == "__main__":
    mpnn_m, base_m, syn_m, tip_m = get_or_train_5fold_stage2_models(device)
    p_dir = resolve_path("/home/guillaume/Dropbox/GYN Cases/Case 100 Prostate/NRRD-MR")
    nrrd_p = os.path.join(p_dir, "33 AX space trial 1.6 SLICE COVER TEMPLATE.nrrd")
    acsv_p = os.path.join(p_dir, "Manual_CY/template slice position.acsv")
    vol, sp_ras, ijk2ras, k_lim_acsv, z_lim_ras = read_nrrd_local(nrrd_p, acsv_p)
    k_lim_auto = estimate_axial_limit_slice_existing(vol, float(sp_ras[2]))
    k_lim_acsv_safe, _ = compute_tilt_safe_pelvic_klim(vol, sp_ras, ijk2ras, z_lim_ras=z_lim_ras, k_lim_orig=k_lim_acsv)

    vtks = sorted(glob.glob(os.path.join(p_dir, "Manual_CY/manual-seg_*.vtk")), key=lambda x: int(re.search(r"_(\d+)\.vtk", x).group(1)))
    gts_100 = [read_vtk_centerline_local(vp) for vp in vtks]
    gts_100_acsv = [read_vtk_centerline_local(vp, z_lim_ras=z_lim_ras) for vp in vtks]

    ckpts_zero = [resolve_path(f"/home/guillaume/github/needlefinder/nn_models_eval/unet3d_fold{f}.pt") for f in range(5)]
    ckpts_clean = [resolve_path(f"/home/guillaume/github/needlefinder/nn_models_clean/unet3d_outer{f}_inner0.pt") for f in range(5)]

    for kl_name, kl_val, g_ref in [("k_lim_auto=40", k_lim_auto, gts_100), ("k_lim_acsv=47", k_lim_acsv_safe, gts_100_acsv)]:
        for name, ckpts, z_s1, z_s2 in [
            ("NF2026 Zero-Exposure GYN (Physics Active)", ckpts_zero, False, False),
            ("NF2026 Full Nested GYN (Physics Active)", ckpts_clean, False, False),
            ("NF2026 Full Nested GYN (Stage 1+2 Physics Zeroed)", ckpts_clean, True, True),
            ("NF2026 Full Nested GYN (Stage 2 Physics Zeroed)", ckpts_clean, False, True),
        ]:
            for tau in ([35.0, 40.0, 45.0, 50.0] if not (z_s1 or z_s2) else [35.0, 50.0]):
                preds, s0 = run_full_nf2026_on_volume(
                    vol, sp_ras, ijk2ras, kl_val, ckpts, mpnn_m, base_m, syn_m, tip_m, device,
                    zero_stage1_physics=z_s1, zero_stage2_physics=z_s2, min_conf=tau
                )
                res, _ = evaluate_method_unified([preds], [g_ref], thresh=3.5, n_boot=20)
                sh_med = res["metrics"]["mean_shaft_xy"]["median"] if res["hungarian_1to1_matched"] > 0 else -1
                tp_med = res["metrics"]["tip_3d"]["median"] if res["hungarian_1to1_matched"] > 0 else -1
                print(
                    f"[{kl_name} tau={tau:4.1f}] {name:48s} | matched={res['hungarian_1to1_matched']:2d}/14 "
                    f"(Rec={res['hungarian_1to1_recall_pct']:5.1f}%, Prec={res['precision_1to1_pct']:5.1f}%, F1={res['f1_1to1_pct']:5.1f}%, n_pred={res['n_pred']:2d}) | "
                    f"ShaftMed={sh_med:.3f}mm, TipMed={tp_med:.2f}mm",
                    flush=True,
                )
