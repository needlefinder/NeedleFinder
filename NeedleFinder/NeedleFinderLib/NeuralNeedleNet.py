"""
3D Multi-Head Neural Network (NeedleUNet3D) Inference & Stage 2 Labelmap-to-Needle Reconstruction
for 3D Slicer NeedleFinder.

Stage 1: Runs the trained 3D Residual U-Net (`needle_unet3d.onnx`) via ONNX Runtime (or PyTorch)
         to predict:
         - P_mask: 3D Needle Binary/Soft Labelmap Probability [0, 1]
         - P_peak: Sharp Sub-Voxel Centerline Gaussian Peak Map (sigma = 0.70 mm) [0, 1]
         - V_x, V_y: 3D Physical Tangent Vector Field (dx/dz, dy/dz)
         - P_obt: Short Obturator Needle Probability Map [0, 1]
Stage 2: Constructs 3D needle trajectories from the predicted labelmap & tangent field,
         suppressing anatomical false positives (vessels, bowel gas, perineal creases)
         and recovering both missed long needles and short obturator needles.
"""

import math
import os
import time
import numpy as np
from scipy.ndimage import maximum_filter

LAST_STAGE2_TIMINGS = {"s2a_mpnn_sec": 0.0, "s2b_bgt_poly3_sec": 0.0}

# Reason-coded decisions of the last Stage 2 run (AUD-520). Every candidate track receives a typed
# reason, so the physicist can see why a track was kept or removed and restore a rejected one.
REASON_KEPT = "kept"
REASON_BELOW_THRESHOLD = "below_threshold"
REASON_OUTSIDE_IMPLANT = "outside_implant_region"
REASON_LOW_SYMMETRY = "low_dark_core_symmetry"
REASON_DUPLICATE = "duplicate_of_kept_track"
REASON_COLLINEAR_FRAGMENT = "collinear_fragment"
REASON_COLLINEAR_MERGED = "collinear_fragment_merged"
REASON_NO_GRAPH_MODEL = "kept_without_graph_model"
REASON_LABELS = {
    REASON_KEPT: "kept",
    REASON_BELOW_THRESHOLD: "confidence below threshold",
    REASON_OUTSIDE_IMPLANT: "outside the implant region",
    REASON_LOW_SYMMETRY: "low dark-core symmetry",
    REASON_DUPLICATE: "duplicate of a kept track",
    REASON_COLLINEAR_FRAGMENT: "fragment of a kept track",
    REASON_COLLINEAR_MERGED: "merged into a kept track",
    REASON_NO_GRAPH_MODEL: "kept (graph model unavailable)",
}
LAST_STAGE2_DECISIONS = {"n_candidates": 0, "kept": [], "rejected": []}


def _reset_stage2_decisions(n_candidates=0):
    LAST_STAGE2_DECISIONS["n_candidates"] = int(n_candidates)
    LAST_STAGE2_DECISIONS["kept"] = []
    LAST_STAGE2_DECISIONS["rejected"] = []


def _record_stage2_decision(traj, reason, confidence, rank):
    """Attach the typed decision to the trajectory dict and log it for the UI."""
    traj["decision"] = reason
    traj["decision_label"] = REASON_LABELS.get(reason, reason)
    traj["confidence_rank"] = int(rank)
    entry = {"traj": traj, "reason": reason, "label": REASON_LABELS.get(reason, reason),
             "confidence": float(confidence), "rank": int(rank),
             "length_mm": float(traj.get("length_mm", 0.0))}
    (LAST_STAGE2_DECISIONS["kept"] if reason in (REASON_KEPT, REASON_NO_GRAPH_MODEL) else LAST_STAGE2_DECISIONS["rejected"]).append(entry)

_ORT_SESSION = None
_ORT_MODEL_PATH = None


def get_onnx_model_path():
    lib_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(lib_dir, "needle_unet3d.onnx")


def _get_ort_session():
    global _ORT_SESSION, _ORT_MODEL_PATH
    model_path = get_onnx_model_path()
    if not os.path.exists(model_path):
        return None
    if _ORT_SESSION is not None and _ORT_MODEL_PATH == model_path:
        return _ORT_SESSION
    try:
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = max(1, min(16, (os.cpu_count() or 8)))
        avail = set(ort.get_available_providers()) if hasattr(ort, "get_available_providers") else set()
        providers = [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider") if p in avail]
        providers.append("CPUExecutionProvider")
        _ORT_SESSION = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
        _ORT_MODEL_PATH = model_path
        return _ORT_SESSION
    except Exception:
        return None


def _upsample_2x_xy(arr, target_h, target_w):
    """Fast bilinear 2x upsampling in H, W for 3D array [K, H_s, W_s]."""
    K_s, H_s, W_s = arr.shape
    out_h = np.empty((K_s, H_s * 2, W_s), dtype=np.float32)
    out_h[:, 0::2, :] = arr
    out_h[:, 1:-1:2, :] = 0.5 * (arr[:, :-1, :] + arr[:, 1:, :])
    out_h[:, -1, :] = arr[:, -1, :]

    out_hw = np.empty((K_s, H_s * 2, W_s * 2), dtype=np.float32)
    out_hw[:, :, 0::2] = out_h
    out_hw[:, :, 1:-1:2] = 0.5 * (out_h[:, :, :-1] + out_h[:, :, 1:])
    out_hw[:, :, -1] = out_h[:, :, -1]

    res = np.zeros((K_s, target_h, target_w), dtype=np.float32)
    ch = min(target_h, out_hw.shape[1])
    cw = min(target_w, out_hw.shape[2])
    res[:, :ch, :cw] = out_hw[:, :ch, :cw]
    return res


def _upsample_2x_k(arr, target_k):
    """Fast linear 2x upsampling along K for 3D array [K_s, H, W]."""
    K_s, H_s, W_s = arr.shape
    out_k = np.empty((K_s * 2, H_s, W_s), dtype=np.float32)
    out_k[0::2] = arr
    out_k[1:-1:2] = 0.5 * (arr[:-1] + arr[1:])
    out_k[-1] = arr[-1]
    if out_k.shape[0] >= target_k:
        return out_k[:target_k]
    res = np.empty((target_k, H_s, W_s), dtype=np.float32)
    res[:out_k.shape[0]] = out_k
    res[out_k.shape[0]:] = out_k[-1:]
    return res


def run_nn_3d_segmentation(vol, sp_i, sp_j, sp_k, i0, i1, j0, j1, k0, K, resp_sub, cs_sub, rsym_sub):
    """
    Run Stage 1 3D Multi-Head U-Net inference on the pelvic ROI.
    Uses 2x in-plane striding when ROI > 180 px (e.g. 0.5 mm 640x416 scans) and
    2x Z-striding when K_sub > 68 and sp_k < 2.2 mm so ONNX inference completes
    in ~1.5-2.5s while covering the full pelvic FOV.
    """
    sess = _get_ort_session()
    if sess is None:
        return None

    H_in, W_in = j1 - j0, i1 - i0
    sub_roi_full = vol[k0:K, j0:j1, i0:i1]
    K_in = sub_roi_full.shape[0]

    stride_xy = 2 if (H_in > 180 or W_in > 180) else 1
    stride_k = 2 if (K_in > 68 and sp_k < 2.2) else 1

    if stride_xy == 2 or stride_k == 2:
        sub_roi = sub_roi_full[::stride_k, ::stride_xy, ::stride_xy]
        r_s = resp_sub[::stride_k]
        c_s = cs_sub[::stride_k]
        if stride_xy == 2 and H_in >= 2 and W_in >= 2:
            H_ev, W_ev = (H_in // 2) * 2, (W_in // 2) * 2
            r_ev = np.maximum(
                np.maximum(r_s[:, 0:H_ev:2, 0:W_ev:2], r_s[:, 1:H_ev:2, 0:W_ev:2]),
                np.maximum(r_s[:, 0:H_ev:2, 1:W_ev:2], r_s[:, 1:H_ev:2, 1:W_ev:2]),
            )
            c_ev = np.maximum(
                np.maximum(c_s[:, 0:H_ev:2, 0:W_ev:2], c_s[:, 1:H_ev:2, 0:W_ev:2]),
                np.maximum(c_s[:, 0:H_ev:2, 1:W_ev:2], c_s[:, 1:H_ev:2, 1:W_ev:2]),
            )
            resp_c = r_s[:, ::2, ::2].copy()
            cs_c = c_s[:, ::2, ::2].copy()
            resp_c[:, :r_ev.shape[1], :r_ev.shape[2]] = r_ev
            cs_c[:, :c_ev.shape[1], :c_ev.shape[2]] = c_ev
        else:
            resp_c = r_s[:, ::stride_xy, ::stride_xy]
            cs_c = c_s[:, ::stride_xy, ::stride_xy]
        rsym_c = rsym_sub[::stride_k, ::stride_xy, ::stride_xy]
        sp_i_eff, sp_j_eff, sp_k_eff = sp_i * stride_xy, sp_j * stride_xy, sp_k * stride_k
    else:
        sub_roi = sub_roi_full
        resp_c = resp_sub
        cs_c = cs_sub
        rsym_c = rsym_sub
        sp_i_eff, sp_j_eff, sp_k_eff = sp_i, sp_j, sp_k

    K_sub, H_full, W_full = sub_roi.shape
    # Crop to central 112 x 128 pelvic window if strided FOV is still very wide
    max_h, max_w = 112, 128
    if H_full > max_h or W_full > max_w:
        hc0 = max(0, (H_full - max_h) // 2)
        hc1 = min(H_full, hc0 + max_h)
        wc0 = max(0, (W_full - max_w) // 2)
        wc1 = min(W_full, wc0 + max_w)
        sub_roi = sub_roi[:, hc0:hc1, wc0:wc1]
        resp_c = resp_c[:, hc0:hc1, wc0:wc1]
        cs_c = cs_c[:, hc0:hc1, wc0:wc1]
        rsym_c = rsym_c[:, hc0:hc1, wc0:wc1]
    else:
        hc0, hc1 = 0, H_full
        wc0, wc1 = 0, W_full

    _, H, W = sub_roi.shape
    non_zero = sub_roi[sub_roi > 0]
    p10 = float(np.percentile(non_zero, 10)) if len(non_zero) > 100 else 0.0
    p95 = float(np.percentile(non_zero, 95)) if len(non_zero) > 100 else 100.0

    ch0 = np.clip((sub_roi - p10) / max(1.0, p95 - p10), -0.5, 2.5).astype(np.float32)

    pos_r = resp_c[resp_c > 0]
    r_scale = float(np.percentile(pos_r, 99)) if len(pos_r) > 20 else 50.0
    ch1 = np.clip(np.log1p(np.maximum(0.0, resp_c)) / max(1.0, math.log1p(r_scale)), 0.0, 1.5).astype(np.float32)

    pos_c = cs_c[cs_c > 0]
    c_scale = float(np.percentile(pos_c, 99)) if len(pos_c) > 20 else 25.0
    ch2 = np.clip(cs_c / max(1.0, c_scale), 0.0, 1.5).astype(np.float32)

    ch3 = np.clip((rsym_c - sub_roi) / (rsym_c + 15.0), -0.5, 1.0).astype(np.float32)

    z_top = min(1.0, (K_sub * sp_k_eff) / 120.0)
    ks_norm = np.linspace(0.0, z_top, K_sub, dtype=np.float32)[:, None, None]
    ch4 = np.broadcast_to(ks_norm, (K_sub, H, W)).copy()

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r_mm = np.hypot((xx - W * 0.5) * sp_i_eff, (yy - H * 0.5) * sp_j_eff) / 50.0
    ch5 = np.broadcast_to(r_mm[None, :, :], (K_sub, H, W)).copy()

    X = np.stack([ch0, ch1, ch2, ch3, ch4, ch5], axis=0)

    def _infer_unet_tensor(X_in, K_target):
        pad_k = (2 - (K_target % 2)) % 2
        pad_h = (8 - (H % 8)) % 8
        pad_w = (8 - (W % 8)) % 8
        if pad_k > 0 or pad_h > 0 or pad_w > 0:
            X_p = np.pad(X_in, ((0, 0), (0, pad_k), (0, pad_h), (0, pad_w)), mode="edge")
        else:
            X_p = X_in
        K_pad = X_p.shape[1]
        if K_pad <= 72:
            out_raw = sess.run(None, {"input": X_p[None, ...]})[0][0]
        else:
            out_raw = np.zeros((5, K_pad, X_p.shape[2], X_p.shape[3]), dtype=np.float32)
            w_acc = np.zeros((1, K_pad, 1, 1), dtype=np.float32)
            chunk = 64
            stride = 52
            for k_s in range(0, K_pad, stride):
                k_e = min(K_pad, k_s + chunk)
                if k_e - k_s < 8:
                    k_s = max(0, k_e - chunk)
                sub_x = X_p[:, k_s:k_e]
                sub_o = sess.run(None, {"input": sub_x[None, ...]})[0][0]
                out_raw[:, k_s:k_e] += sub_o
                w_acc[:, k_s:k_e] += 1.0
                if k_e == K_pad:
                    break
            out_raw /= np.maximum(1.0, w_acc)
        return out_raw[:, :K_target, :H, :W]

    out = _infer_unet_tensor(X, K_sub)
    p_mask_w = 1.0 / (1.0 + np.exp(-np.clip(out[0], -20.0, 20.0)))

    # Zero-shot thick-slice canonical Z-upsampling guard: only triggers on out-of-distribution
    # thick-slice scans (sp_k_eff > 2.2 mm with weak native U-Net response p99.9 < 0.35).
    if sp_k_eff > 2.2 and float(np.percentile(p_mask_w, 99.9)) < 0.35:
        K_up = K_sub * 2
        sp_k_up = sp_k_eff * 0.5
        ch0_up = _upsample_2x_k(ch0, K_up)
        ch1_up = _upsample_2x_k(ch1, K_up)
        ch2_up = _upsample_2x_k(ch2, K_up)
        ch3_up = _upsample_2x_k(ch3, K_up)
        z_top_up = min(1.0, (K_up * sp_k_up) / 120.0)
        ks_norm_up = np.linspace(0.0, z_top_up, K_up, dtype=np.float32)[:, None, None]
        ch4_up = np.broadcast_to(ks_norm_up, (K_up, H, W)).copy()
        ch5_up = np.broadcast_to(r_mm[None, :, :], (K_up, H, W)).copy()
        X_up = np.stack([ch0_up, ch1_up, ch2_up, ch3_up, ch4_up, ch5_up], axis=0)
        out_up_full = _infer_unet_tensor(X_up, K_up)
        out_up = 0.5 * (out_up_full[:, 0::2, :, :] + out_up_full[:, 1::2, :, :])
        p_mask_up = 1.0 / (1.0 + np.exp(-np.clip(out_up[0], -20.0, 20.0)))
        use_up = p_mask_up > p_mask_w
        out = np.where(use_up[None, ...], out_up, out)
        p_mask_w = np.maximum(p_mask_w, p_mask_up)

    p_peak_w = 1.0 / (1.0 + np.exp(-np.clip(out[1], -20.0, 20.0)))
    vx_w = np.tanh(out[2]) * 0.75
    vy_w = np.tanh(out[3]) * 0.75
    p_obt_w = 1.0 / (1.0 + np.exp(-np.clip(out[4], -20.0, 20.0)))

    def _restore_full(arr_w):
        arr_u = arr_w.astype(np.float32)
        if stride_k == 2:
            arr_u = _upsample_2x_k(arr_u, K_in)
        if stride_xy == 2:
            arr_u = _upsample_2x_xy(arr_u, (hc1 - hc0) * 2, (wc1 - wc0) * 2)
        if H_full > max_h or W_full > max_w or arr_u.shape[1] != H_in or arr_u.shape[2] != W_in:
            res = np.zeros((K_in, H_in, W_in), dtype=np.float32)
            j_s = hc0 * stride_xy
            i_s = wc0 * stride_xy
            ch = min(H_in - j_s, arr_u.shape[1])
            cw = min(W_in - i_s, arr_u.shape[2])
            res[:, j_s:j_s + ch, i_s:i_s + cw] = arr_u[:, :ch, :cw]
            return res
        return arr_u

    return {
        "p_mask": _restore_full(p_mask_w),
        "p_peak": _restore_full(p_peak_w),
        "vx": _restore_full(vx_w),
        "vy": _restore_full(vy_w),
        "p_obt": _restore_full(p_obt_w),
    }


def create_or_update_slicer_labelmap(volumeNode, nn_preds, k0, i0, i1, j0, j1, K, J, I, threshold=0.40):
    """
    Create or update a 3D Slicer LabelMapVolumeNode ('Needle-NN-LabelMap') in the MRML scene
    from the Stage 1 3D U-Net probability map so the user can visualize the binary labelmap.
    """
    try:
        import slicer
        import vtk.util.numpy_support as vtk_np
        if not hasattr(slicer, "mrmlScene") or slicer.mrmlScene is None:
            return None

        label_name = "Needle-NN-LabelMap"
        labelNode = None
        col = slicer.mrmlScene.GetNodesByName(label_name)
        if col and col.GetNumberOfItems() > 0:
            labelNode = col.GetItemAsObject(0)
        if labelNode is None:
            labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", label_name)
            labelNode.CreateDefaultDisplayNodes()

        import vtk
        mat = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(mat)
        labelNode.SetIJKToRASMatrix(mat)
        labelNode.SetSpacing(volumeNode.GetSpacing())
        labelNode.SetOrigin(volumeNode.GetOrigin())

        mask_full = np.zeros((K, J, I), dtype=np.uint8)
        p_comb = np.maximum(nn_preds["p_mask"], nn_preds["p_obt"] * 0.90)
        K_sub = p_comb.shape[0]
        mask_full[k0:k0 + K_sub, j0:j1, i0:i1] = (p_comb >= threshold).astype(np.uint8)

        if hasattr(slicer.util, "updateVolumeFromArray"):
            slicer.util.updateVolumeFromArray(labelNode, mask_full)
        return labelNode
    except Exception:
        return None


def extract_nn_labelmap_tubes(
    nn_preds, resp_roi, k0, i0, j0, sp_i, sp_j, sp_k,
    ijk_to_ras, ras_to_ijk, existing_trajs,
    include_obturator=True,
):
    """
    Stage 2 Direct Labelmap + Tangent-Field Tube Reconstructor:
    Traces 3D trajectories along residual ridges of the 3D U-Net labelmap (p_mask * p_peak)
    guided by the predicted 3D tangent vector field (vx, vy).
    Recovers both missed long perineal needles and central short obturator needles.
    """
    p_mask = nn_preds["p_mask"]
    p_peak = nn_preds["p_peak"]
    vx_map = nn_preds["vx"]
    vy_map = nn_preds["vy"]
    p_obt = nn_preds["p_obt"]

    K_sub, H, W = p_mask.shape
    rot_mat = ijk_to_ras[:3, :3]
    trans_vec = ijk_to_ras[:3, 3]
    z_base_ras = float((rot_mat @ np.array([i0 + W * 0.5, j0 + H * 0.5, float(k0 + 1)]) + trans_vec)[2])

    if len(existing_trajs) >= 2:
        ex_pts = []
        ex_slopes = []
        for tr in existing_trajs:
            pr = tr["ras"] if "ras" in tr else tr["points"]
            ex_pts.append(pr[0, :2])
            ex_pts.append(pr[-1, :2])
            dz_e = float(pr[-1, 2] - pr[0, 2])
            if abs(dz_e) > 2.0:
                ex_slopes.append((pr[-1, :2] - pr[0, :2]) / dz_e)
        bundle_center = np.median(np.array(ex_pts), axis=0)
        med_slope = np.median(np.array(ex_slopes), axis=0) if ex_slopes else np.zeros(2)
    else:
        bundle_center = (rot_mat @ np.array([i0 + W * 0.5, j0 + H * 0.5, float(k0 + 1)]) + trans_vec)[:2]
        med_slope = np.zeros(2)

    work_peak = (p_peak * (0.45 + 0.55 * p_mask)).copy()
    if include_obturator:
        work_peak = np.maximum(work_peak, p_peak * p_obt * 1.15)

    # Mask out existing trajectories (1.85 mm radius cylinder)
    ri_m = max(1, int(round(1.85 / sp_i)))
    rj_m = max(1, int(round(1.85 / sp_j)))
    yy_m, xx_m = np.mgrid[-rj_m:rj_m + 1, -ri_m:ri_m + 1]
    disk_m = ((xx_m * sp_i) ** 2 + (yy_m * sp_j) ** 2) <= (1.85 ** 2)

    for tr in existing_trajs:
        pts_r = tr["points"] if "points" in tr else tr["ras"]
        pts_h = np.column_stack([pts_r, np.ones(len(pts_r))])
        pts_ijk = (ras_to_ijk @ pts_h.T).T[:, :3]
        order = np.argsort(pts_ijk[:, 2])
        pts_ijk = pts_ijk[order]
        k_lo = max(0, int(math.floor(pts_ijk[0, 2])) - k0)
        k_hi = min(K_sub - 1, int(math.ceil(pts_ijk[-1, 2])) - k0)
        for kr in range(k_lo, k_hi + 1):
            ic = int(round(float(np.interp(kr + k0, pts_ijk[:, 2], pts_ijk[:, 0])) - i0))
            jc = int(round(float(np.interp(kr + k0, pts_ijk[:, 2], pts_ijk[:, 1])) - j0))
            if 0 <= ic < W and 0 <= jc < H:
                j0_p, j1_p = max(0, jc - rj_m), min(H, jc + rj_m + 1)
                i0_p, i1_p = max(0, ic - ri_m), min(W, ic + ri_m + 1)
                d_sub = disk_m[(j0_p - (jc - rj_m)):(j1_p - (jc - rj_m)), (i0_p - (ic - ri_m)):(i1_p - (ic - ri_m))]
                work_peak[kr, j0_p:j1_p, i0_p:i1_p][d_sub] = 0.0

    nms_px = max(1, int(round(1.35 / min(sp_i, sp_j))))
    cand_fg = (work_peak >= 0.42) & ((p_mask >= 0.45) | (p_obt >= 0.45))
    fg_k, fg_j, fg_i = np.where(cand_fg)
    if len(fg_k) == 0:
        return []
    j_b0 = max(0, int(np.min(fg_j)) - nms_px - 1)
    j_b1 = min(H, int(np.max(fg_j)) + nms_px + 2)
    i_b0 = max(0, int(np.min(fg_i)) - nms_px - 1)
    i_b1 = min(W, int(np.max(fg_i)) + nms_px + 2)
    wp_sub = work_peak[:, j_b0:j_b1, i_b0:i_b1]
    lm_sub = wp_sub == maximum_filter(wp_sub, size=(1, 2 * nms_px + 1, 2 * nms_px + 1))
    seed_sub = lm_sub & cand_fg[:, j_b0:j_b1, i_b0:i_b1]
    sk, sj_rel, si_rel = np.where(seed_sub)
    if len(sk) == 0:
        return []
    sj = sj_rel + j_b0
    si = si_rel + i_b0

    scores = work_peak[sk, sj, si]
    order_seeds = np.argsort(-scores)
    visited = np.zeros_like(work_peak, dtype=bool)
    search_r = max(2, int(round(2.4 / min(sp_i, sp_j))))

    new_trajs = []
    min_slices = 4 if sp_k >= 3.5 else (5 if sp_k >= 2.2 else 7)

    for idx_s in order_seeds[:350]:
        ks0, js0, is0 = int(sk[idx_s]), int(sj[idx_s]), int(si[idx_s])
        if visited[ks0, js0, is0]:
            continue

        pts = {ks0: (float(is0), float(js0), float(work_peak[ks0, js0, is0]))}

        for direction in (1, -1):
            curr_k = ks0
            curr_i, curr_j = float(is0), float(js0)
            missed = 0
            while True:
                next_k = curr_k + direction
                if not (0 <= next_k < K_sub):
                    break
                ci_int = int(np.clip(round(curr_i), 0, W - 1))
                cj_int = int(np.clip(round(curr_j), 0, H - 1))
                vx_loc = float(vx_map[curr_k, cj_int, ci_int])
                vy_loc = float(vy_map[curr_k, cj_int, ci_int])
                pt_curr_ras = rot_mat @ np.array([curr_i + i0, curr_j + j0, float(curr_k + k0)]) + trans_vec
                dz_step = float((rot_mat @ np.array([curr_i + i0, curr_j + j0, float(next_k + k0)]) + trans_vec)[2] - pt_curr_ras[2])
                target_ras = pt_curr_ras + np.array([vx_loc * dz_step, vy_loc * dz_step, dz_step])
                target_ijk = ras_to_ijk @ np.append(target_ras, 1.0)
                pred_i = float(target_ijk[0] - i0)
                pred_j = float(target_ijk[1] - j0)

                pi_int, pj_int = int(round(pred_i)), int(round(pred_j))
                if not (1 <= pi_int < W - 1 and 1 <= pj_int < H - 1):
                    break

                j_lo, j_hi = max(0, pj_int - search_r), min(H, pj_int + search_r + 1)
                i_lo, i_hi = max(0, pi_int - search_r), min(W, pi_int + search_r + 1)
                patch = work_peak[next_k, j_lo:j_hi, i_lo:i_hi]
                yy_p, xx_p = np.mgrid[j_lo:j_hi, i_lo:i_hi]
                dist_mm = np.hypot((xx_p - pred_i) * sp_i, (yy_p - pred_j) * sp_j)
                weighted = patch * np.exp(-0.5 * (dist_mm / 1.65) ** 2) * (dist_mm <= 2.45)
                if weighted.size == 0 or float(np.max(weighted)) < 0.24:
                    missed += 1
                    if missed > 2:
                        break
                    curr_k = next_k
                    curr_i, curr_j = pred_i, pred_j
                    continue

                lj, li = np.unravel_index(int(np.argmax(weighted)), weighted.shape)
                best_j, best_i = j_lo + lj, i_lo + li
                raw_p = float(work_peak[next_k, best_j, best_i])
                if raw_p < 0.25 or (float(p_mask[next_k, best_j, best_i]) < 0.28 and float(p_obt[next_k, best_j, best_i]) < 0.30):
                    missed += 1
                    if missed > 2:
                        break
                    curr_k = next_k
                    curr_i, curr_j = pred_i, pred_j
                    continue

                if 1 <= best_j < H - 1 and 1 <= best_i < W - 1:
                    sub3 = np.maximum(0.0, work_peak[next_k, best_j - 1:best_j + 2, best_i - 1:best_i + 2])
                    wsum = float(np.sum(sub3))
                    if wsum > 1e-4:
                        ym, xm = np.mgrid[-1:2, -1:2]
                        best_i_f = float(best_i) + float(np.sum(xm * sub3)) / wsum
                        best_j_f = float(best_j) + float(np.sum(ym * sub3)) / wsum
                    else:
                        best_i_f, best_j_f = float(best_i), float(best_j)
                else:
                    best_i_f, best_j_f = float(best_i), float(best_j)

                pts[next_k] = (best_i_f, best_j_f, raw_p)
                curr_k = next_k
                curr_i, curr_j = best_i_f, best_j_f
                missed = 0

        if len(pts) < min_slices:
            continue

        ks = sorted(pts.keys())
        k_min_rel = min(ks)
        if not include_obturator and k_min_rel > int(round(32.0 / sp_k)):
            continue

        arr_pts = np.array([[pts[k][0] + i0, pts[k][1] + j0, k + k0, pts[k][2]] for k in ks], dtype=np.float64)
        ras_pts = (rot_mat @ arr_pts[:, :3].T).T + trans_vec
        order_z = np.argsort(ras_pts[:, 2])
        ras_pts = ras_pts[order_z]
        ks = [ks[i] for i in order_z]

        z_vals = ras_pts[:, 2]
        z_span = float(z_vals[-1] - z_vals[0])
        min_span = 11.5 if include_obturator else (12.0 if K_sub <= 28 else 16.0)
        if z_span < min_span:
            continue

        deg = 2 if len(z_vals) >= 8 else 1
        px = np.polyfit(z_vals, ras_pts[:, 0], deg)
        py = np.polyfit(z_vals, ras_pts[:, 1], deg)
        fit_x = np.polyval(px, z_vals)
        fit_y = np.polyval(py, z_vals)
        rmse = float(np.percentile(np.hypot(ras_pts[:, 0] - fit_x, ras_pts[:, 1] - fit_y), 85))
        if rmse > 1.35:
            continue

        dz = float(z_vals[-1] - z_vals[0])
        sx = float(fit_x[-1] - fit_x[0]) / dz
        sy = float(fit_y[-1] - fit_y[0]) / dz
        if math.hypot(sx, sy) > 0.42:
            continue
        if float(np.hypot(sx - med_slope[0], sy - med_slope[1])) > 0.26:
            continue
        mid_xy = np.array([float(np.mean(fit_x)), float(np.mean(fit_y))])
        if float(np.hypot(mid_xy[0] - bundle_center[0], mid_xy[1] - bundle_center[1])) > 38.0:
            continue

        mean_pm = float(np.mean([p_mask[k, int(round(pts[k][1])), int(round(pts[k][0]))] for k in ks]))
        mean_po = float(np.mean([p_obt[k, int(round(pts[k][1])), int(round(pts[k][0]))] for k in ks]))
        mean_r = float(np.mean([resp_roi[k, int(round(pts[k][1])), int(round(pts[k][0]))] for k in ks]))
        if mean_pm < 0.55 and mean_po < 0.55:
            continue
        if mean_r <= 2.5:
            continue

        if not include_obturator and z_vals[0] > z_base_ras + 0.5 * sp_k and abs(z_vals[0] - z_base_ras) <= 22.0:
            bx = float(fit_x[0] + sx * (z_base_ras - z_vals[0]))
            by = float(fit_y[0] + sy * (z_base_ras - z_vals[0]))
            smooth_ras = np.vstack([[bx, by, z_base_ras], np.column_stack([fit_x, fit_y, z_vals])])
        else:
            smooth_ras = np.column_stack([fit_x, fit_y, z_vals])

        # Check 3D overlap against existing_trajs and new_trajs to prevent duplicate instances
        is_dup = False
        for prev in list(existing_trajs) + new_trajs:
            pr = prev["ras"] if "ras" in prev else prev["points"]
            z0 = max(float(smooth_ras[0, 2]), float(pr[0, 2]))
            z1 = min(float(smooth_ras[-1, 2]), float(pr[-1, 2]))
            if z1 >= z0 + 2.0:
                zs = np.linspace(z0, z1, 10)
                dx = np.interp(zs, smooth_ras[:, 2], smooth_ras[:, 0]) - np.interp(zs, pr[:, 2], pr[:, 0])
                dy = np.interp(zs, smooth_ras[:, 2], smooth_ras[:, 1]) - np.interp(zs, pr[:, 2], pr[:, 1])
                if float(np.mean(np.hypot(dx, dy))) < 2.65:
                    is_dup = True
                    break
        if is_dup:
            continue

        for k_v in range(max(0, min(ks) - 1), min(K_sub, max(ks) + 2)):
            z_kv = float((rot_mat @ np.array([i0, j0, float(k_v + k0)]) + trans_vec)[2])
            x_kv = float(np.polyval(px, z_kv))
            y_kv = float(np.polyval(py, z_kv))
            ijk_kv = ras_to_ijk @ np.array([x_kv, y_kv, z_kv, 1.0])
            iv, jv = int(round(float(ijk_kv[0] - i0))), int(round(float(ijk_kv[1] - j0)))
            if 0 <= iv < W and 0 <= jv < H:
                j0_p, j1_p = max(0, jv - rj_m), min(H, jv + rj_m + 1)
                i0_p, i1_p = max(0, iv - ri_m), min(W, iv + ri_m + 1)
                d_sub = disk_m[(j0_p - (jv - rj_m)):(j1_p - (jv - rj_m)), (i0_p - (iv - ri_m)):(i1_p - (iv - ri_m))]
                visited[k_v, j0_p:j1_p, i0_p:i1_p][d_sub] = True
                work_peak[k_v, j0_p:j1_p, i0_p:i1_p][d_sub] = 0.0

        new_trajs.append({
            "ras": smooth_ras,
            "points": smooth_ras,
            "base_xy": np.array([float(smooth_ras[0, 0]), float(smooth_ras[0, 1])]),
            "slope": np.array([sx, sy]),
            "z_span_tracked": dz,
            "rmse": rmse,
            "confidence": float(np.clip(max(mean_pm, mean_po) * 100.0, 50.0, 99.9)),
            "length_mm": float(smooth_ras[-1, 2] - smooth_ras[0, 2]),
            "num_points": len(ks),
            "pass": 4,
        })

    return new_trajs


_GRAPHNET_WEIGHTS = None
_ORT_GT_SESSION = None


def _get_graphnet_weights():
    global _GRAPHNET_WEIGHTS
    if _GRAPHNET_WEIGHTS is not None:
        return _GRAPHNET_WEIGHTS
    lib_dir = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(lib_dir, "needle_graphnet.npz")
    if not os.path.exists(npz_path):
        return None
    try:
        data = np.load(npz_path)
        _GRAPHNET_WEIGHTS = {k: data[k].astype(np.float32) for k in data.files}
        return _GRAPHNET_WEIGHTS
    except Exception:
        return None


def _get_gt_ort_session():
    global _ORT_GT_SESSION
    if _ORT_GT_SESSION is not None:
        return _ORT_GT_SESSION
    lib_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(lib_dir, "needle_graph_transformer.onnx")
    if not os.path.exists(onnx_path):
        return None
    try:
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        _ORT_GT_SESSION = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
        return _ORT_GT_SESSION
    except Exception:
        return None


def _silu_np(x):
    return x / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _ln_np(x, w, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def _lin_np(x, w, b):
    return x @ w.T + b


def _scatter_sum_np(src, idx, dim_size):
    out = np.zeros((dim_size, src.shape[1]), dtype=np.float32)
    np.add.at(out, idx, src)
    return out


def _scatter_softmax_np(scores, idx, dim_size):
    max_per = np.full((dim_size, 1), -1e9, dtype=np.float32)
    np.maximum.at(max_per, idx, scores)
    exp_s = np.exp(np.clip(scores - max_per[idx], -20.0, 20.0))
    denom = _scatter_sum_np(exp_s, idx, dim_size)
    return exp_s / np.maximum(1e-6, denom[idx])


def _forward_mpnn_numpy(nodes_feat, edge_feat, src, dst, sp_src, sp_dst, sp_feat, W):
    N = nodes_feat.shape[0]
    E = edge_feat.shape[0]

    h = _silu_np(_ln_np(_lin_np(nodes_feat, W["mpnn.node_enc.0.weight"], W["mpnn.node_enc.0.bias"]), W["mpnn.node_enc.1.weight"], W["mpnn.node_enc.1.bias"]))
    h = _silu_np(_ln_np(_lin_np(h, W["mpnn.node_enc.3.weight"], W["mpnn.node_enc.3.bias"]), W["mpnn.node_enc.4.weight"], W["mpnn.node_enc.4.bias"]))

    if E > 0:
        e = _silu_np(_ln_np(_lin_np(edge_feat, W["mpnn.edge_enc.0.weight"], W["mpnn.edge_enc.0.bias"]), W["mpnn.edge_enc.1.weight"], W["mpnn.edge_enc.1.bias"]))
        e = _silu_np(_ln_np(_lin_np(e, W["mpnn.edge_enc.3.weight"], W["mpnn.edge_enc.3.bias"]), W["mpnn.edge_enc.4.weight"], W["mpnn.edge_enc.4.bias"]))
    else:
        e = np.zeros((0, 64), dtype=np.float32)

    if len(sp_src) > 0:
        sp_e = _silu_np(_ln_np(_lin_np(sp_feat, W["mpnn.sp_enc.0.weight"], W["mpnn.sp_enc.0.bias"]), W["mpnn.sp_enc.1.weight"], W["mpnn.sp_enc.1.bias"]))
    else:
        sp_e = np.zeros((0, 32), dtype=np.float32)

    for l_idx in range(4):
        pfx = f"mpnn.layers.{l_idx}"
        if E > 0:
            in_cat = np.concatenate([h[src], e], axis=-1)
            w_in = _scatter_softmax_np(_lin_np(in_cat, W[f"{pfx}.in_attn.weight"], W[f"{pfx}.in_attn.bias"]), dst, N)
            m_in_v = _silu_np(_ln_np(_lin_np(in_cat, W[f"{pfx}.in_msg_mlp.0.weight"], W[f"{pfx}.in_msg_mlp.0.bias"]), W[f"{pfx}.in_msg_mlp.1.weight"], W[f"{pfx}.in_msg_mlp.1.bias"]))
            m_in = _scatter_sum_np(w_in * m_in_v, dst, N)

            out_cat = np.concatenate([h[dst], e], axis=-1)
            w_out = _scatter_softmax_np(_lin_np(out_cat, W[f"{pfx}.out_attn.weight"], W[f"{pfx}.out_attn.bias"]), src, N)
            m_out_v = _silu_np(_ln_np(_lin_np(out_cat, W[f"{pfx}.out_msg_mlp.0.weight"], W[f"{pfx}.out_msg_mlp.0.bias"]), W[f"{pfx}.out_msg_mlp.1.weight"], W[f"{pfx}.out_msg_mlp.1.bias"]))
            m_out = _scatter_sum_np(w_out * m_out_v, src, N)
        else:
            m_in = np.zeros_like(h)
            m_out = np.zeros_like(h)

        if len(sp_src) > 0:
            sp_cat = np.concatenate([h[sp_src], sp_e], axis=-1)
            w_sp = _scatter_softmax_np(_lin_np(sp_cat, W[f"{pfx}.sp_attn.weight"], W[f"{pfx}.sp_attn.bias"]), sp_dst, N)
            m_sp_v = _silu_np(_ln_np(_lin_np(sp_cat, W[f"{pfx}.sp_msg_mlp.0.weight"], W[f"{pfx}.sp_msg_mlp.0.bias"]), W[f"{pfx}.sp_msg_mlp.1.weight"], W[f"{pfx}.sp_msg_mlp.1.bias"]))
            m_sp = _scatter_sum_np(w_sp * m_sp_v, sp_dst, N)
        else:
            m_sp = np.zeros_like(h)

        nu = _silu_np(_ln_np(_lin_np(np.concatenate([h, m_in, m_out, m_sp], axis=-1), W[f"{pfx}.node_upd.0.weight"], W[f"{pfx}.node_upd.0.bias"]), W[f"{pfx}.node_upd.1.weight"], W[f"{pfx}.node_upd.1.bias"]))
        nu = _ln_np(_lin_np(nu, W[f"{pfx}.node_upd.3.weight"], W[f"{pfx}.node_upd.3.bias"]), W[f"{pfx}.node_upd.4.weight"], W[f"{pfx}.node_upd.4.bias"])
        h = _silu_np(h + nu)

        if E > 0:
            e_cat = np.concatenate([e, h[src], h[dst], m_in[src], m_out[dst]], axis=-1)
            eu = _silu_np(_ln_np(_lin_np(e_cat, W[f"{pfx}.edge_upd.0.weight"], W[f"{pfx}.edge_upd.0.bias"]), W[f"{pfx}.edge_upd.1.weight"], W[f"{pfx}.edge_upd.1.bias"]))
            eu = _ln_np(_lin_np(eu, W[f"{pfx}.edge_upd.3.weight"], W[f"{pfx}.edge_upd.3.bias"]), W[f"{pfx}.edge_upd.4.weight"], W[f"{pfx}.edge_upd.4.bias"])
            e = _silu_np(e + eu)

    nh = _silu_np(_lin_np(h, W["mpnn.node_head.0.weight"], W["mpnn.node_head.0.bias"]))
    node_out = _lin_np(nh, W["mpnn.node_head.2.weight"], W["mpnn.node_head.2.bias"])
    if E > 0:
        eh = _silu_np(_lin_np(e, W["mpnn.edge_head.0.weight"], W["mpnn.edge_head.0.bias"]))
        edge_logit = _lin_np(eh, W["mpnn.edge_head.2.weight"], W["mpnn.edge_head.2.bias"]).squeeze(-1)
    else:
        edge_logit = np.zeros(0, dtype=np.float32)
    return node_out, edge_logit


def _traj_dist_np(p_ras, q_ras):
    z0 = max(float(p_ras[0, 2]), float(q_ras[0, 2]))
    z1 = min(float(p_ras[-1, 2]), float(q_ras[-1, 2]))
    if z1 >= z0 + 2.0:
        zs = np.linspace(z0, z1, 12)
    else:
        zs = np.linspace(float(q_ras[0, 2]), float(q_ras[-1, 2]), 12)
    px = np.interp(zs, p_ras[:, 2], p_ras[:, 0])
    py = np.interp(zs, p_ras[:, 2], p_ras[:, 1])
    qx = np.interp(zs, q_ras[:, 2], q_ras[:, 0])
    qy = np.interp(zs, q_ras[:, 2], q_ras[:, 1])
    return float(np.mean(np.hypot(px - qx, py - qy)))


def _softmax_last_np(x):
    mx = np.max(x, axis=-1, keepdims=True)
    ex = np.exp(np.clip(x - mx, -20.0, 20.0))
    return ex / np.maximum(1e-6, np.sum(ex, axis=-1, keepdims=True))


def _forward_tip_bgt_numpy(x_26, adj_feat_8, adj_mask, W):
    """
    Pure-NumPy forward pass of the SynthBrachy3T-pretrained TipAwareBundleGraphTransformer
    (26-D node features, 8-D pairwise bundle features, 2-layer Edge-Biased GAT + Set Transformer).
    Returns [M, 4]: [logit_rec, logit_1to1, logit_long_1to1, pred_dz_tip].
    """
    if "tip_bgt.node_enc.0.weight" not in W:
        return None
    h = _silu_np(_ln_np(_lin_np(x_26, W["tip_bgt.node_enc.0.weight"], W["tip_bgt.node_enc.0.bias"]),
                        W["tip_bgt.node_enc.1.weight"], W["tip_bgt.node_enc.1.bias"]))
    h = _silu_np(_ln_np(_lin_np(h, W["tip_bgt.node_enc.3.weight"], W["tip_bgt.node_enc.3.bias"]),
                        W["tip_bgt.node_enc.4.weight"], W["tip_bgt.node_enc.4.bias"]))
    ef = _silu_np(_lin_np(adj_feat_8, W["tip_bgt.edge_enc.0.weight"], W["tip_bgt.edge_enc.0.bias"]))
    scale = 1.0 / math.sqrt(float(h.shape[-1]))

    # Layer 1: Edge-Biased GAT
    q1 = _lin_np(h, W["tip_bgt.q1.weight"], W["tip_bgt.q1.bias"])
    k1 = _lin_np(h, W["tip_bgt.k1.weight"], W["tip_bgt.k1.bias"])
    v1 = _lin_np(h, W["tip_bgt.v1.weight"], W["tip_bgt.v1.bias"])
    ea1 = _lin_np(ef, W["tip_bgt.e_att1.weight"], W["tip_bgt.e_att1.bias"]).squeeze(-1)
    scores1 = (q1 @ k1.T) * scale + ea1
    scores1 = np.where(adj_mask > 0.0, scores1, -1e4)
    attn1 = _softmax_last_np(scores1) * adj_mask
    attn1 = attn1 / np.maximum(1e-6, np.sum(attn1, axis=-1, keepdims=True))
    msg1 = (attn1 @ v1) + np.sum(attn1[:, :, None] * ef, axis=1)
    h = _ln_np(h + _silu_np(msg1), W["tip_bgt.norm1.weight"], W["tip_bgt.norm1.bias"])

    # Layer 2: Edge-Biased GAT
    q2 = _lin_np(h, W["tip_bgt.q2.weight"], W["tip_bgt.q2.bias"])
    k2 = _lin_np(h, W["tip_bgt.k2.weight"], W["tip_bgt.k2.bias"])
    v2 = _lin_np(h, W["tip_bgt.v2.weight"], W["tip_bgt.v2.bias"])
    ea2 = _lin_np(ef, W["tip_bgt.e_att2.weight"], W["tip_bgt.e_att2.bias"]).squeeze(-1)
    scores2 = (q2 @ k2.T) * scale + ea2
    scores2 = np.where(adj_mask > 0.0, scores2, -1e4)
    attn2 = _softmax_last_np(scores2) * adj_mask
    attn2 = attn2 / np.maximum(1e-6, np.sum(attn2, axis=-1, keepdims=True))
    msg2 = (attn2 @ v2) + np.sum(attn2[:, :, None] * ef, axis=1)
    h = _ln_np(h + _silu_np(msg2), W["tip_bgt.norm2.weight"], W["tip_bgt.norm2.bias"])

    # Layer 3: Global Set Transformer self-attention + FFN
    q3 = _lin_np(h, W["tip_bgt.q3.weight"], W["tip_bgt.q3.bias"])
    k3 = _lin_np(h, W["tip_bgt.k3.weight"], W["tip_bgt.k3.bias"])
    v3 = _lin_np(h, W["tip_bgt.v3.weight"], W["tip_bgt.v3.bias"])
    attn3 = _softmax_last_np((q3 @ k3.T) * scale)
    h = _ln_np(h + (attn3 @ v3), W["tip_bgt.norm3.weight"], W["tip_bgt.norm3.bias"])
    ff = _silu_np(_lin_np(h, W["tip_bgt.ffn.0.weight"], W["tip_bgt.ffn.0.bias"]))
    ff = _lin_np(ff, W["tip_bgt.ffn.2.weight"], W["tip_bgt.ffn.2.bias"])
    h = _ln_np(h + ff, W["tip_bgt.norm4.weight"], W["tip_bgt.norm4.bias"])

    hd = _silu_np(_lin_np(h, W["tip_bgt.head.0.weight"], W["tip_bgt.head.0.bias"]))
    return _lin_np(hd, W["tip_bgt.head.2.weight"], W["tip_bgt.head.2.bias"])


def _conv1d_same_np(x, w, b, dilation=1):
    """Same-padded 1D cross-correlation for odd kernels, matching torch Conv1d."""
    c_out, c_in, kernel = w.shape
    if kernel < 1 or kernel % 2 == 0 or dilation < 1:
        raise ValueError("Expected an odd kernel and positive dilation")
    if x.ndim != 2 or x.shape[0] != c_in:
        raise ValueError("Input channels do not match convolution weights")
    length = x.shape[1]
    pad = (kernel // 2) * dilation
    padded = np.pad(x, ((0, 0), (pad, pad)), mode="constant")
    out = np.zeros((c_out, length), dtype=np.result_type(x, w, b))
    for k in range(kernel):
        offset = k * dilation
        out += np.einsum("oi,il->ol", w[:, :, k], padded[:, offset:offset + length])
    return out + b[:, None]


def _gn_1d_np(x, w, b, num_groups=8, eps=1e-5):
    """Pure-NumPy GroupNorm over [C, L]."""
    C, L = x.shape
    G = min(num_groups, C)
    xg = x.reshape(G, C // G, L)
    mean = np.mean(xg, axis=(1, 2), keepdims=True)
    var = np.var(xg, axis=(1, 2), keepdims=True)
    xg_norm = (xg - mean) / np.sqrt(var + eps)
    return xg_norm.reshape(C, L) * w[:, None] + b[:, None]


def _forward_1d_tip_localizer_numpy(prof_c_l, W):
    """
    Pure-NumPy forward pass of Dedicated1DTipLocalizer(in_channels=24, hidden_dim=48, seq_len=91) (Stage 2C).
    prof_c_l: [24, 91] float32 array -> returns (dz_pred_mm, tip_prob_std_mm, peak_prob).
    """
    if "tip1d.stem.0.weight" not in W:
        return None
    C_expect = W["tip1d.stem.0.weight"].shape[1]
    x = np.asarray(prof_c_l[:C_expect], dtype=np.float32)
    h = _silu_np(_gn_1d_np(_conv1d_same_np(x, W["tip1d.stem.0.weight"], W["tip1d.stem.0.bias"], dilation=1),
                           W["tip1d.stem.1.weight"], W["tip1d.stem.1.bias"]))
    for b_idx, dil in ((1, 1), (2, 2), (3, 4)):
        pfx = f"tip1d.block{b_idx}"
        r = _silu_np(_gn_1d_np(_conv1d_same_np(h, W[f"{pfx}.0.weight"], W[f"{pfx}.0.bias"], dilation=dil),
                               W[f"{pfx}.1.weight"], W[f"{pfx}.1.bias"]))
        r = _gn_1d_np(_conv1d_same_np(r, W[f"{pfx}.3.weight"], W[f"{pfx}.3.bias"], dilation=dil),
                      W[f"{pfx}.4.weight"], W[f"{pfx}.4.bias"])
        h = _silu_np(h + r)

    # 4-head 1D self-attention across L=91
    seq = h.T  # [91, 48]
    L, D = seq.shape
    qkv = _lin_np(seq, W["tip1d.attn.in_proj_weight"], W["tip1d.attn.in_proj_bias"])
    q, k, v = np.split(qkv, 3, axis=-1)
    n_heads = 4
    hd = D // n_heads
    scale = 1.0 / math.sqrt(float(hd))
    qh = q.reshape(L, n_heads, hd).transpose(1, 0, 2)  # [4, 91, 12]
    kh = k.reshape(L, n_heads, hd).transpose(1, 0, 2)
    vh = v.reshape(L, n_heads, hd).transpose(1, 0, 2)
    scores = np.matmul(qh, kh.transpose(0, 2, 1)) * scale
    attn = _softmax_last_np(scores)
    ctx = np.matmul(attn, vh).transpose(1, 0, 2).reshape(L, D)
    attn_out = _lin_np(ctx, W["tip1d.attn.out_proj.weight"], W["tip1d.attn.out_proj.bias"])
    seq = _ln_np(seq + attn_out, W["tip1d.attn_norm.weight"], W["tip1d.attn_norm.bias"])
    h_out = seq.T  # [48, 91]

    sh = _silu_np(_conv1d_same_np(h_out, W["tip1d.step_head.0.weight"], W["tip1d.step_head.0.bias"], dilation=1))
    w_fin = W["tip1d.step_head.2.weight"][:, :, 0]  # [1, 24]
    b_fin = W["tip1d.step_head.2.bias"]             # [1]
    logits = (w_fin @ sh + b_fin[:, None]).squeeze(0)  # [91]
    probs = _softmax_last_np(logits)
    z_grid = W.get("tip1d.dz_grid", np.linspace(-25.0, 20.0, 91, dtype=np.float32))
    dz_soft = float(np.sum(probs * z_grid))
    z_var = float(np.sum(probs * ((z_grid - dz_soft) ** 2)))
    z_std = math.sqrt(max(0.25, z_var))

    h_cat = np.concatenate([h_out[:, 50], np.mean(h_out, axis=-1)], axis=-1)
    rh = _silu_np(_lin_np(h_cat, W["tip1d.reg_head.0.weight"], W["tip1d.reg_head.0.bias"]))
    dz_res = float(_lin_np(rh, W["tip1d.reg_head.2.weight"], W["tip1d.reg_head.2.bias"])[0]) * 10.0
    dz_pred = float(np.clip(0.75 * dz_soft + 0.25 * dz_res, -25.0, 20.0))
    peak_prob = float(np.max(probs))
    return dz_pred, z_std, peak_prob


def _extract_1d_tip_profile_22_fast_np(
    pts_ras, vx_top, vy_top, z_tip_nbr,
    vol_roi, p_mask, p_peak, p_obt, vx_map, vy_map,
    resp_roi, cs_roi, rsym_roi,
    p10, p95, log_r_sc, c_sc,
    k0, i0, j0, sp_i, sp_j, sp_k, ras_to_ijk, center_ras, z_base_ras
):
    """
    Vectorized 91-step (-25.0 mm to +20.0 mm in 0.5 mm steps) 24-channel 1D axial profile extractor
    matching extract_1d_tip_profile_91_24ch channel-for-channel.
    """
    K_sub, H, W = p_mask.shape
    z_top = float(pts_ras[-1, 2])
    s_offsets = np.linspace(-25.0, 20.0, 91, dtype=np.float64)
    z_grid = z_top + s_offsets
    x_grid = np.where(z_grid <= z_top, np.interp(z_grid, pts_ras[:, 2], pts_ras[:, 0]), pts_ras[-1, 0] + vx_top * s_offsets)
    y_grid = np.where(z_grid <= z_top, np.interp(z_grid, pts_ras[:, 2], pts_ras[:, 1]), pts_ras[-1, 1] + vy_top * s_offsets)

    ijks = (ras_to_ijk @ np.column_stack([x_grid, y_grid, z_grid, np.ones_like(z_grid)]).T).T
    i_f = ijks[:, 0] - i0
    j_f = ijks[:, 1] - j0
    k_f = ijks[:, 2] - k0
    r_px = max(1, int(round(1.0 / max(0.3, sp_i))))

    prof = np.zeros((24, 91), dtype=np.float32)
    prof[14, :] = (s_offsets / 25.0).astype(np.float32)
    prof[16, :] = ((z_grid - float(z_tip_nbr)) / 25.0).astype(np.float32)

    for t_idx in range(91):
        fi, fj, fk = float(i_f[t_idx]), float(j_f[t_idx]), float(k_f[t_idx])
        ic, jc = int(round(fi)), int(round(fj))
        if not (r_px <= ic < W - r_px and r_px <= jc < H - r_px and -0.5 <= fk <= K_sub - 0.5):
            continue
        prof[15, t_idx] = 1.0
        fk_c = float(np.clip(fk, 0.0, float(K_sub - 1)))
        k_lo = int(math.floor(fk_c))
        k_hi = min(K_sub - 1, k_lo + 1)
        a_s = float(fk_c - k_lo)
        b_s = 1.0 - a_s
        sl_j = slice(jc - r_px, jc + r_px + 1)
        sl_i = slice(ic - r_px, ic + r_px + 1)

        pm_p = b_s * p_mask[k_lo, sl_j, sl_i] + a_s * p_mask[k_hi, sl_j, sl_i]
        pp_p = b_s * p_peak[k_lo, sl_j, sl_i] + a_s * p_peak[k_hi, sl_j, sl_i]
        po_p = b_s * p_obt[k_lo, sl_j, sl_i] + a_s * p_obt[k_hi, sl_j, sl_i]
        vx_c = float(b_s * vx_map[k_lo, jc, ic] + a_s * vx_map[k_hi, jc, ic])
        vy_c = float(b_s * vy_map[k_lo, jc, ic] + a_s * vy_map[k_hi, jc, ic])
        tan_err = math.hypot(vx_c - vx_top, vy_c - vy_top)

        r_p = b_s * resp_roi[k_lo, sl_j, sl_i] + a_s * resp_roi[k_hi, sl_j, sl_i]
        sym_p = np.clip(np.log1p(np.maximum(0.0, r_p)) / log_r_sc, 0.0, 1.5)
        c_p = np.clip((b_s * cs_roi[k_lo, sl_j, sl_i] + a_s * cs_roi[k_hi, sl_j, sl_i]) / c_sc, 0.0, 1.5)
        v_p = b_s * vol_roi[k_lo, sl_j, sl_i] + a_s * vol_roi[k_hi, sl_j, sl_i]
        rs_p = b_s * rsym_roi[k_lo, sl_j, sl_i] + a_s * rsym_roi[k_hi, sl_j, sl_i]
        dr_p = np.clip((rs_p - v_p) / (rs_p + 15.0), -0.5, 1.0)
        mri_p = np.clip((v_p - p10) / max(1.0, p95 - p10), -0.5, 2.5)
        hull_c = math.hypot((ic - W * 0.5) * sp_i, (jc - H * 0.5) * sp_j) / 50.0

        pm_max = float(np.max(pm_p))
        pm_mean = float(np.mean(pm_p))
        pp_max = float(np.max(pp_p))
        po_max = float(np.max(po_p))
        sym_max = float(np.max(sym_p))
        sym_mean = float(np.mean(sym_p))
        cs_max = float(np.max(c_p))
        dr_max = float(np.max(dr_p))
        t2w_min = float(np.min(mri_p))
        comb_sup = 0.40 * max(pm_max, po_max) + 0.35 * pp_max + 0.25 * min(1.0, sym_max)
        sig_max = float(np.max(0.5 * np.sqrt(np.clip(pm_p * (1.0 - pm_p), 0.0, 0.25))))

        prof[0, t_idx] = pm_max
        prof[1, t_idx] = pm_mean
        prof[2, t_idx] = pp_max
        prof[3, t_idx] = po_max
        prof[4, t_idx] = vx_c
        prof[5, t_idx] = vy_c
        prof[6, t_idx] = tan_err
        prof[7, t_idx] = sym_max
        prof[8, t_idx] = sym_mean
        prof[9, t_idx] = cs_max
        prof[10, t_idx] = dr_max
        prof[11, t_idx] = t2w_min
        prof[12, t_idx] = hull_c
        prof[13, t_idx] = comb_sup
        prof[22, t_idx] = sig_max

    prof[17] = np.gradient(prof[0]) * 2.0
    prof[18] = np.gradient(prof[2]) * 2.0
    prof[19] = np.gradient(prof[7]) * 2.0
    prof[20] = np.gradient(prof[11]) * 2.0
    prof[21] = np.gradient(prof[13]) * 2.0
    prof[23] = np.gradient(prof[22]) * 2.0
    return prof


def _extrapolate_xy_at_z_np(pts_ras, z_query):
    pts = pts_ras[np.argsort(pts_ras[:, 2])]
    z_bot, z_top = float(pts[0, 2]), float(pts[-1, 2])
    if z_bot <= z_query <= z_top:
        return float(np.interp(z_query, pts[:, 2], pts[:, 0])), float(np.interp(z_query, pts[:, 2], pts[:, 1]))
    elif z_query > z_top:
        sub = pts[pts[:, 2] >= max(z_bot, z_top - 25.0)]
        if len(sub) < 2:
            sub = pts[-min(len(pts), 4):]
    else:
        sub = pts[pts[:, 2] <= min(z_top, z_bot + 25.0)]
        if len(sub) < 2:
            sub = pts[:min(len(pts), 4)]
    dz = float(sub[-1, 2] - sub[0, 2])
    if dz < 0.5:
        return float(sub[-1, 0]), float(sub[-1, 1])
    deg = 1 if len(sub) < 6 else 1
    cx = np.polyfit(sub[:, 2], sub[:, 0], deg)
    cy = np.polyfit(sub[:, 2], sub[:, 1], deg)
    return float(np.polyval(cx, z_query)), float(np.polyval(cy, z_query))


def _collinear_traj_dist_np(pts, ref_ras):
    """
    Extrapolation-aware collinear fragment checker:
    Returns (is_duplicate_or_fragment, should_extend_ref, merged_pts).
    """
    pts = pts[np.argsort(pts[:, 2])]
    ref = ref_ras[np.argsort(ref_ras[:, 2])]
    z0 = max(float(pts[0, 2]), float(ref[0, 2]))
    z1 = min(float(pts[-1, 2]), float(ref[-1, 2]))
    ov = z1 - z0

    if ov >= 4.0:
        zs = np.linspace(z0, z1, 12)
        dx = np.interp(zs, pts[:, 2], pts[:, 0]) - np.interp(zs, ref[:, 2], ref[:, 0])
        dy = np.interp(zs, pts[:, 2], pts[:, 1]) - np.interp(zs, ref[:, 2], ref[:, 1])
        d_ov = float(np.mean(np.hypot(dx, dy)))
        if d_ov <= 2.15:
            if float(pts[-1, 2]) > float(ref[-1, 2]) + 2.5:
                upper = pts[pts[:, 2] > float(ref[-1, 2]) + 0.5]
                if len(upper) > 0:
                    merged = _regularize_poly3_np(np.vstack([ref, upper]))
                    return True, True, merged
            return True, False, ref
        return False, False, ref

    gap = -ov
    if gap > 18.0:
        return False, False, ref

    sl_pts = (pts[-1, :2] - pts[0, :2]) / max(1.0, float(pts[-1, 2] - pts[0, 2]))
    sl_ref = (ref[-1, :2] - ref[0, :2]) / max(1.0, float(ref[-1, 2] - ref[0, 2]))
    if float(np.hypot(sl_pts[0] - sl_ref[0], sl_pts[1] - sl_ref[1])) > 0.14:
        return False, False, ref

    zs_probe = np.linspace(float(pts[0, 2]), float(pts[-1, 2]), 7)
    dists = []
    for zq in zs_probe:
        rx, ry = _extrapolate_xy_at_z_np(ref, float(zq))
        px = float(np.interp(zq, pts[:, 2], pts[:, 0]))
        py = float(np.interp(zq, pts[:, 2], pts[:, 1]))
        dists.append(math.hypot(px - rx, py - ry))
    if float(np.mean(dists)) <= 2.30:
        if float(pts[-1, 2]) > float(ref[-1, 2]) + 2.0:
            comb = np.vstack([ref, pts[pts[:, 2] > float(ref[-1, 2]) + 0.5]])
            merged = _regularize_poly3_np(comb)
            return True, True, merged
        return True, False, ref
    return False, False, ref


def _stitch_collinear_pool_fragments_np(pool):
    """
    Collinear Axial Split-Fragment Stitching & Kostoulas 2025 Iterative Multi-Segment Gap-Bridging:
    Bridges candidate fragments (t_low, t_high) separated by mid-shaft dropouts (gap <= 20 mm,
    extrapolated midpoint XY distance <= 2.2 mm, 3D slope difference <= 0.14).
    """
    M0 = len(pool)
    if M0 < 2:
        return pool
    stitched_entries = []
    for i in range(M0):
        p_low = pool[i]
        r_low = p_low["ras"]
        z_bot_l, z_top_l = float(r_low[0, 2]), float(r_low[-1, 2])
        if z_top_l - z_bot_l < 9.0:
            continue
        for j in range(M0):
            if i == j:
                continue
            p_high = pool[j]
            r_high = p_high["ras"]
            z_bot_h, z_top_h = float(r_high[0, 2]), float(r_high[-1, 2])
            if z_top_h - z_bot_h < 8.0 or z_top_h <= z_top_l + 6.0:
                continue
            gap = z_bot_h - z_top_l
            if gap < -6.0 or gap > 20.0:
                continue
            ds = float(np.hypot(p_low["slope"][0] - p_high["slope"][0], p_low["slope"][1] - p_high["slope"][1]))
            if ds > 0.14:
                continue
            z_mid = 0.5 * (z_top_l + z_bot_h)
            xl, yl = _extrapolate_xy_at_z_np(r_low, z_mid)
            xh, yh = _extrapolate_xy_at_z_np(r_high, z_mid)
            if math.hypot(xl - xh, yl - yh) > 2.20:
                continue

            # Combine r_low and r_high and fit smooth Poly3 curve across gap
            z_new = np.linspace(z_bot_l, z_top_h, max(16, int(round((z_top_h - z_bot_l) / 2.0))))
            comb = np.vstack([r_low[r_low[:, 2] <= z_mid], r_high[r_high[:, 2] > z_mid]])
            comb = comb[np.argsort(comb[:, 2])]
            if len(comb) < 5:
                continue
            zn = 2.0 * (comb[:, 2] - z_bot_l) / max(1.0, z_top_h - z_bot_l) - 1.0
            zn_new = 2.0 * (z_new - z_bot_l) / max(1.0, z_top_h - z_bot_l) - 1.0
            deg = 3 if (z_top_h - z_bot_l) >= 28.0 else 2
            cx = np.polyfit(zn, comb[:, 0], deg)
            cy = np.polyfit(zn, comb[:, 1], deg)
            res = float(np.percentile(np.hypot(comb[:, 0] - np.polyval(cx, zn), comb[:, 1] - np.polyval(cy, zn)), 85))
            if res > 1.45:
                continue
            s_ras = np.column_stack([np.polyval(cx, zn_new), np.polyval(cy, zn_new), z_new])
            if any(_traj_dist_np(s_ras, p_ex["ras"]) < 1.35 and abs(float(p_ex["ras"][-1, 2]) - z_top_h) < 3.5 for p_ex in pool + stitched_entries):
                continue

            w_l = z_top_l - z_bot_l
            w_h = z_top_h - z_bot_h
            wt = max(1.0, w_l + w_h)
            diffs = np.diff(s_ras, axis=0)
            len_mm = float(np.sum(np.linalg.norm(diffs, axis=1)))
            sl_new = (s_ras[-1, :2] - s_ras[0, :2]) / max(1.0, z_top_h - z_bot_l)
            conf_new = max(float(p_low["conf_heur"]), (w_l * float(p_low["conf_heur"]) + w_h * float(p_high["conf_heur"])) / wt)
            st_traj = {
                "ras": s_ras,
                "points": s_ras,
                "base_xy": s_ras[0, :2].copy(),
                "slope": sl_new,
                "z_span_tracked": float(z_top_h - z_bot_l),
                "length_mm": len_mm,
                "rmse": res,
                "confidence": float(np.clip(conf_new * 100.0, 50.0, 99.5)),
                "pass": p_low["traj"].get("pass", 1),
                "is_stitched": True,
            }
            stitched_entries.append({
                "traj": st_traj,
                "ras": s_ras,
                "base_xy": s_ras[0, :2].copy(),
                "slope": sl_new,
                "z_span": float(z_top_h - z_bot_l),
                "rmse": res,
                "conf_heur": conf_new,
                "from_mpnn": max(float(p_low["from_mpnn"]), float(p_high["from_mpnn"])),
                "mpnn_pa": (w_l * float(p_low["mpnn_pa"]) + w_h * float(p_high["mpnn_pa"])) / wt,
                "mpnn_pl": max(float(p_low["mpnn_pl"]), float(p_high["mpnn_pl"])),
                "mpnn_pe": (w_l * float(p_low["mpnn_pe"]) + w_h * float(p_high["mpnn_pe"])) / wt,
            })
            if len(stitched_entries) >= 8:
                break
        if len(stitched_entries) >= 8:
            break
    return pool + stitched_entries


def _regularize_poly3_np(pts_ras, weights=None):
    pts = np.asarray(pts_ras, dtype=np.float64)
    order = np.argsort(pts[:, 2])
    pts = pts[order]
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)[order]
    n = len(pts)
    if n < 4:
        return pts
    z = pts[:, 2]
    z_span = float(z[-1] - z[0])
    if z_span < 4.0:
        return pts
    deg = 3 if (n >= 9 and z_span >= 28.0) else (2 if n >= 5 else 1)
    zn = 2.0 * (z - z[0]) / z_span - 1.0
    try:
        cx = np.polyfit(zn, pts[:, 0], deg, w=weights)
        cy = np.polyfit(zn, pts[:, 1], deg, w=weights)
        fx = np.polyval(cx, zn)
        fy = np.polyval(cy, zn)
        if float(np.max(np.hypot(fx - pts[:, 0], fy - pts[:, 1]))) <= 1.85:
            return np.column_stack([fx, fy, z])
    except Exception:
        pass
    return pts


def _apply_tip_adjustment_and_poly3_np(
    pts_ras, dz_tip_mm, vx_top, vy_top,
    p_mask=None, p_peak=None, resp_roi=None, log_r_sc_f=1.0,
    k0=0, i0=0, j0=0, ijk_to_ras_mat=None, ras_to_ijk=None
):
    """
    Applies Stage 2C tip adjustment + Upgrade A (Kim et al. 2026 sub-voxel dark-core centroiding
    & inverse-variance-weighted 3rd-order polynomial shaft regularization).
    """
    pts = np.asarray(pts_ras, dtype=np.float64)
    pts = pts[np.argsort(pts[:, 2])]
    if len(pts) < 3:
        return pts
    dz_clamp = float(np.clip(dz_tip_mm, -11.5, 11.5))
    z_bot = float(pts[0, 2])
    z_top_old = float(pts[-1, 2])
    z_top_new = max(z_bot + 7.5, z_top_old + dz_clamp)

    if z_top_new < z_top_old - 0.5:
        x_top = float(np.interp(z_top_new, pts[:, 2], pts[:, 0]))
        y_top = float(np.interp(z_top_new, pts[:, 2], pts[:, 1]))
        keep = pts[:, 2] < z_top_new - 0.25
        pts_trim = np.vstack([pts[keep], np.array([[x_top, y_top, z_top_new]])])
        if len(pts_trim) >= 3:
            pts = pts_trim
    elif z_top_new > z_top_old + 0.5:
        dz_ext = z_top_new - z_top_old
        x_ext = float(pts[-1, 0]) + vx_top * dz_ext
        y_ext = float(pts[-1, 1]) + vy_top * dz_ext
        pts = np.vstack([pts, np.array([[x_ext, y_ext, z_top_new]])])

    if len(pts) < 4:
        return pts

    # Upgrade A: Kim et al. 2026 sub-voxel dark-core centroiding & precision-weighted Poly3
    weights = None
    if p_mask is not None and p_peak is not None and resp_roi is not None and ras_to_ijk is not None and ijk_to_ras_mat is not None:
        K_sub, H, W = p_mask.shape
        rot_mat = ijk_to_ras_mat[:3, :3]
        trans_vec = ijk_to_ras_mat[:3, 3]
        ijks = (ras_to_ijk @ np.column_stack([pts, np.ones(len(pts))]).T).T
        weights = np.ones(len(pts), dtype=np.float64)
        pts_sub = pts.copy()
        yy_g, xx_g = np.mgrid[-1:2, -1:2]
        for idx_p, row in enumerate(ijks):
            ic = int(round(row[0] - i0))
            jc = int(round(row[1] - j0))
            kc = int(round(row[2] - k0))
            if 1 <= ic < W - 1 and 1 <= jc < H - 1 and 0 <= kc < K_sub:
                pm_patch = p_mask[kc, jc - 1:jc + 2, ic - 1:ic + 2]
                pp_patch = p_peak[kc, jc - 1:jc + 2, ic - 1:ic + 2]
                sym_patch = np.clip(np.log1p(np.maximum(0.0, resp_roi[kc, jc - 1:jc + 2, ic - 1:ic + 2])) / log_r_sc_f, 0.0, 1.5)
                # Single-pass Bernoulli uncertainty proxy sigma_bern = 0.5 * sqrt(p * (1 - p))
                sig_patch = 0.5 * np.sqrt(np.clip(pm_patch * (1.0 - pm_patch), 0.0, 0.25))
                w_patch = np.maximum(0.0, (0.55 * pp_patch + 0.45 * sym_patch) / (1.0 + 2.0 * sig_patch))
                w_sum = float(np.sum(w_patch))
                if w_sum > 0.15:
                    di = float(np.sum(xx_g * w_patch)) / w_sum
                    dj = float(np.sum(yy_g * w_patch)) / w_sum
                    sub_ras = rot_mat @ np.array([ic + di + i0, jc + dj + j0, row[2]]) + trans_vec
                    if math.hypot(sub_ras[0] - pts[idx_p, 0], sub_ras[1] - pts[idx_p, 1]) <= 1.05:
                        pts_sub[idx_p, 0] = 0.72 * pts[idx_p, 0] + 0.28 * float(sub_ras[0])
                        pts_sub[idx_p, 1] = 0.72 * pts[idx_p, 1] + 0.28 * float(sub_ras[1])
                pm_max = float(np.max(pm_patch))
                sym_max = float(np.max(sym_patch))
                sig_c = float(sig_patch[1, 1])
                weights[idx_p] = max(0.15, (pm_max + 0.5 * sym_max) / (0.04 + sig_c * sig_c))
        pts = pts_sub

    pts_poly = _regularize_poly3_np(pts, weights=weights)
    pts_blend = pts.copy()
    pts_blend[:, 0] = 0.55 * pts[:, 0] + 0.45 * pts_poly[:, 0]
    pts_blend[:, 1] = 0.55 * pts[:, 1] + 0.45 * pts_poly[:, 1]
    return pts_blend


def run_mpnn_and_graph_transformer(
    vol, nn_preds, resp_roi, cs_roi, rsym_roi,
    k0, i0, j0, sp_i, sp_j, sp_k,
    ijk_to_ras_mat, ras_to_ijk, candidate_trajs,
    min_confidence=50.0, script=False,
):
    """
    Run Stage 2 (Slicer v2.0 Production Pipeline):
    1. Stage 2A: 3D Spatio-Temporal NeedleSliceMPNN + Collinear Split-Fragment Stitching (Kostoulas 2025).
    2. Stage 2B: Trajectory-Level DenseGraphTransformer + TipAwareBundleGraphTransformer +
       Dai et al. 2020 Axial TV Continuity & Uncertainty Scoring + Extrapolation-Aware Collinear Deduplication.
    3. Stage 2C: Dedicated 91-Step 22-Channel 1D Axial Tip Localizer + Split-Conformal 90% Tip Uncertainty +
       Kim et al. 2026 Sub-Voxel Dark-Core & Inverse-Variance-Weighted Poly3 Shaft Refinement.
    """
    t_s2a_0 = time.perf_counter()
    W_dict = _get_graphnet_weights()
    gt_sess = _get_gt_ort_session()
    if W_dict is None or nn_preds is None or not candidate_trajs:
        LAST_STAGE2_TIMINGS["s2a_mpnn_sec"] = 0.0
        LAST_STAGE2_TIMINGS["s2b_bgt_poly3_sec"] = 0.0
        _reset_stage2_decisions(len(candidate_trajs))
        for r_i, t_c in enumerate(sorted(candidate_trajs, key=lambda t: -float(t.get("confidence", 0.0)))):
            _record_stage2_decision(t_c, REASON_NO_GRAPH_MODEL, float(t_c.get("confidence", 0.0)), r_i + 1)
        return candidate_trajs

    p_mask = nn_preds["p_mask"]
    p_peak = nn_preds["p_peak"]
    vx_map = nn_preds["vx"]
    vy_map = nn_preds["vy"]
    p_obt = nn_preds["p_obt"]
    K_sub, H, W = p_mask.shape

    rot_mat = ijk_to_ras_mat[:3, :3]
    trans_vec = ijk_to_ras_mat[:3, 3]
    center_ras = rot_mat @ np.array([i0 + W * 0.5, j0 + H * 0.5, float(k0)]) + trans_vec
    z_base_ras = float(center_ras[2])

    # 1. Run NeedleSliceMPNN on standard-resolution scans (or strided deep scans)
    mpnn_trajs = []
    p10_v, p95_v = 0.0, 100.0
    try:
        st_xy = 2 if (H > 180 or W > 180) else 1
        st_k = 2 if (K_sub > 68 and sp_k < 2.2) else 1
        pm_s = p_mask[::st_k, ::st_xy, ::st_xy]
        pp_s = p_peak[::st_k, ::st_xy, ::st_xy]
        po_s = p_obt[::st_k, ::st_xy, ::st_xy]
        vx_s = vx_map[::st_k, ::st_xy, ::st_xy]
        vy_s = vy_map[::st_k, ::st_xy, ::st_xy]
        vol_roi = vol if vol.shape == (K_sub, H, W) else vol[k0:k0 + K_sub, j0:j0 + H, i0:i0 + W]
        sub_v = vol_roi[::st_k, ::st_xy, ::st_xy]
        resp_s = resp_roi[::st_k, ::st_xy, ::st_xy]
        cs_s = cs_roi[::st_k, ::st_xy, ::st_xy]
        rsym_s = rsym_roi[::st_k, ::st_xy, ::st_xy]

        Ks, Hs, Ws = pm_s.shape
        v_sub_s = sub_v[::2, ::2, ::2] if Ks > 24 else sub_v
        nz = v_sub_s[v_sub_s > 0]
        p10 = float(np.percentile(nz, 10)) if len(nz) > 100 else 0.0
        p95 = float(np.percentile(nz, 95)) if len(nz) > 100 else 100.0
        p10_v, p95_v = p10, p95
        r_sub_s = resp_s[::2, ::2, ::2] if Ks > 24 else resp_s
        pr = r_sub_s[r_sub_s > 0]
        r_sc = float(np.percentile(pr, 99)) if len(pr) > 20 else 50.0
        log_r_sc = max(1.0, math.log1p(r_sc))
        c_sub_s = cs_s[::2, ::2, ::2] if Ks > 24 else cs_s
        pc = c_sub_s[c_sub_s > 0]
        c_sc = max(1.0, float(np.percentile(pc, 99)) if len(pc) > 20 else 25.0)

        sp_i_e, sp_j_e, sp_k_e = sp_i * st_xy, sp_j * st_xy, sp_k * st_k
        cand_m = (pm_s >= 0.22) | (pp_s >= 0.20) | (po_s >= 0.25)
        ck, cj, ci = np.where(cand_m)
        n_kr, n_ic, n_jc, n_si, n_sj = [], [], [], [], []
        if len(ck) >= 6:
            nms_r = max(1, int(round(1.35 / sp_i_e)))
            jb0 = max(0, int(np.min(cj)) - nms_r - 2)
            jb1 = min(Hs, int(np.max(cj)) + nms_r + 3)
            ib0 = max(0, int(np.min(ci)) - nms_r - 2)
            ib1 = min(Ws, int(np.max(ci)) + nms_r + 3)
            ch1_box = np.clip(np.log1p(np.maximum(0.0, resp_s[:, jb0:jb1, ib0:ib1])) / log_r_sc, 0.0, 1.5).astype(np.float32)
            score_box = (
                0.48 * pp_s[:, jb0:jb1, ib0:ib1]
                + 0.30 * pm_s[:, jb0:jb1, ib0:ib1]
                + 0.14 * po_s[:, jb0:jb1, ib0:ib1]
                + 0.12 * np.clip(ch1_box, 0.0, 1.2)
            )
            loc_max_box = score_box == maximum_filter(score_box, size=(1, 2 * nms_r + 1, 2 * nms_r + 1))
            yy_w, xx_w = np.mgrid[-1:2, -1:2]

            for kr in range(Ks):
                mk_box = loc_max_box[kr] & (score_box[kr] >= 0.20) & cand_m[kr, jb0:jb1, ib0:ib1]
                jj_b, ii_b = np.where(mk_box)
                if len(ii_b) == 0:
                    continue
                jj = jj_b + jb0
                ii = ii_b + ib0
                valid_b = (jj >= 1) & (jj < Hs - 1) & (ii >= 1) & (ii < Ws - 1)
                jj, ii, jj_b, ii_b = jj[valid_b], ii[valid_b], jj_b[valid_b], ii_b[valid_b]
                if len(ii) == 0:
                    continue
                if len(ii) > 32:
                    order = np.argsort(-score_box[kr, jj_b, ii_b])[:32]
                    jj, ii, jj_b, ii_b = jj[order], ii[order], jj_b[order], ii_b[order]
                for jc, ic, jb, ib in zip(jj, ii, jj_b, ii_b):
                    wp = np.maximum(0.0, score_box[kr, jb - 1:jb + 2, ib - 1:ib + 2])
                    ws = float(np.sum(wp))
                    si = float(ic) + float(np.sum(xx_w * wp)) / ws if ws > 1e-6 else float(ic)
                    sj = float(jc) + float(np.sum(yy_w * wp)) / ws if ws > 1e-6 else float(jc)
                    n_kr.append(kr); n_ic.append(ic); n_jc.append(jc); n_si.append(si); n_sj.append(sj)

        N = len(n_kr)
        if N >= 6:
            n_kr = np.asarray(n_kr, dtype=np.int32)
            n_ic = np.asarray(n_ic, dtype=np.int32)
            n_jc = np.asarray(n_jc, dtype=np.int32)
            n_si = np.asarray(n_si, dtype=np.float32)
            n_sj = np.asarray(n_sj, dtype=np.float32)

            ijk_pts = np.column_stack([n_si * st_xy + i0, n_sj * st_xy + j0, n_kr * st_k + k0]).astype(np.float64)
            nodes_ras = (ijk_pts @ rot_mat.T + trans_vec).astype(np.float32)

            pm_v = pm_s[n_kr, n_jc, n_ic]
            pp_v = pp_s[n_kr, n_jc, n_ic]
            po_v = po_s[n_kr, n_jc, n_ic]
            vx_v = vx_s[n_kr, n_jc, n_ic]
            vy_v = vy_s[n_kr, n_jc, n_ic]
            pm_m3 = np.array([float(np.max(pm_s[k_i, j_i - 1:j_i + 2, i_i - 1:i_i + 2])) for k_i, j_i, i_i in zip(n_kr, n_jc, n_ic)], dtype=np.float32)
            pp_m3 = np.array([float(np.max(pp_s[k_i, j_i - 1:j_i + 2, i_i - 1:i_i + 2])) for k_i, j_i, i_i in zip(n_kr, n_jc, n_ic)], dtype=np.float32)
            v_node = sub_v[n_kr, n_jc, n_ic]
            r_node = resp_s[n_kr, n_jc, n_ic]
            c_node = cs_s[n_kr, n_jc, n_ic]
            rs_node = rsym_s[n_kr, n_jc, n_ic]
            ch0_v = np.clip((v_node - p10) / max(1.0, p95 - p10), -0.5, 2.5).astype(np.float32)
            ch1_v = np.clip(np.log1p(np.maximum(0.0, r_node)) / log_r_sc, 0.0, 1.5).astype(np.float32)
            ch2_v = np.clip(c_node / c_sc, 0.0, 1.5).astype(np.float32)
            ch3_v = np.clip((rs_node - v_node) / (rs_node + 15.0), -0.5, 1.0).astype(np.float32)
            r_mm = np.hypot((n_si - Ws * 0.5) * sp_i_e, (n_sj - Hs * 0.5) * sp_j_e) / 50.0

            nodes_feat = np.column_stack([
                (nodes_ras[:, 0] - center_ras[0]) / 45.0,
                (nodes_ras[:, 1] - center_ras[1]) / 45.0,
                (nodes_ras[:, 2] - z_base_ras) / 80.0,
                pm_v, pp_v, po_v, vx_v, vy_v,
                pm_m3, pp_m3, 0.5 * (pm_v + pm_m3),
                ch0_v, ch1_v, ch2_v, ch3_v, r_mm,
            ]).astype(np.float32)

            max_dk = max(4, int(round(24.0 / sp_k_e)))
            dk_mat = n_kr[None, :] - n_kr[:, None]
            cu, cv = np.where((dk_mat >= 1) & (dk_mat <= max_dk))
            if len(cu) > 0:
                ru, rv = nodes_ras[cu], nodes_ras[cv]
                dz = rv[:, 2] - ru[:, 2]
                vz = dz > 0.2
                cu, cv, ru, rv, dz = cu[vz], cv[vz], ru[vz], rv[vz], dz[vz]
                sx = (rv[:, 0] - ru[:, 0]) / dz
                sy = (rv[:, 1] - ru[:, 1]) / dz
                sn = np.hypot(sx, sy)
                vx_u, vy_u = vx_v[cu], vy_v[cu]
                vx_w, vy_w = vx_v[cv], vy_v[cv]
                err_t = np.hypot(rv[:, 0] - (ru[:, 0] + 0.5 * (vx_u + vx_w) * dz), rv[:, 1] - (ru[:, 1] + 0.5 * (vy_u + vy_w) * dz))
                raw_d = np.hypot(rv[:, 0] - ru[:, 0], rv[:, 1] - ru[:, 1])
                keep = (sn <= 0.56) & ((err_t <= 2.8 + 0.14 * dz) | (raw_d <= 3.5 + 0.22 * dz))
                e_src, e_dst = cu[keep], cv[keep]
                sx, sy, sn, dz, err_t = sx[keep], sy[keep], sn[keep], dz[keep], err_t[keep]
                vx_u, vy_u, vx_w, vy_w = vx_u[keep], vy_u[keep], vx_w[keep], vy_w[keep]
                km = np.clip(np.round(0.5 * (n_kr[e_src] + n_kr[e_dst])).astype(np.int32), 0, Ks - 1)
                im = np.clip(np.round(0.5 * (n_si[e_src] + n_si[e_dst])).astype(np.int32), 0, Ws - 1)
                jm = np.clip(np.round(0.5 * (n_sj[e_src] + n_sj[e_dst])).astype(np.int32), 0, Hs - 1)
                edge_feat = np.column_stack([
                    sx, sy, sn, dz / 25.0, (n_kr[e_dst] - n_kr[e_src]).astype(np.float32) / float(max_dk),
                    err_t / 5.0, np.hypot(sx - vx_u, sy - vy_u), np.hypot(sx - vx_w, sy - vy_w),
                    np.hypot(vx_u - vx_w, vy_u - vy_w), pm_s[km, jm, im], pp_s[km, jm, im], np.minimum(pp_v[e_src], pp_v[e_dst])
                ]).astype(np.float32)
            else:
                e_src = np.zeros(0, dtype=np.int64)
                e_dst = np.zeros(0, dtype=np.int64)
                edge_feat = np.zeros((0, 12), dtype=np.float32)

            sp_u, sp_v = np.where((dk_mat == 0) & (np.arange(N)[:, None] != np.arange(N)[None, :]))
            if len(sp_u) > 0:
                dx_s = nodes_ras[sp_v, 0] - nodes_ras[sp_u, 0]
                dy_s = nodes_ras[sp_v, 1] - nodes_ras[sp_u, 1]
                ds_s = np.hypot(dx_s, dy_s)
                ks_p = ds_s <= 22.0
                sp_src, sp_dst = sp_u[ks_p], sp_v[ks_p]
                sp_feat = np.column_stack([
                    dx_s[ks_p] / 22.0, dy_s[ks_p] / 22.0, ds_s[ks_p] / 22.0,
                    vx_v[sp_dst] - vx_v[sp_src], vy_v[sp_dst] - vy_v[sp_src]
                ]).astype(np.float32)
            else:
                sp_src = np.zeros(0, dtype=np.int64)
                sp_dst = np.zeros(0, dtype=np.int64)
                sp_feat = np.zeros((0, 5), dtype=np.float32)

            node_out, edge_logit = _forward_mpnn_numpy(nodes_feat, edge_feat, e_src, e_dst, sp_src, sp_dst, sp_feat, W_dict)
            p_na = 1.0 / (1.0 + np.exp(-np.clip(node_out[:, 0], -20.0, 20.0)))
            p_nl = 1.0 / (1.0 + np.exp(-np.clip(node_out[:, 1], -20.0, 20.0)))
            offs = np.clip(node_out[:, 2:4], -1.5, 1.5)
            p_ed = 1.0 / (1.0 + np.exp(-np.clip(edge_logit, -20.0, 20.0))) if len(edge_logit) > 0 else np.zeros(0, dtype=np.float32)

            nodes_ras[:, 0] += offs[:, 0]
            nodes_ras[:, 1] += offs[:, 1]
            valid_e = np.where((p_ed >= 0.42) & (p_na[e_src] >= 0.32) & (p_na[e_dst] >= 0.32))[0]
            e_sc = p_ed[valid_e] * np.sqrt(p_na[e_src[valid_e]] * p_na[e_dst[valid_e]])
            order_e = valid_e[np.argsort(-e_sc)]

            nxt_n = np.full(N, -1, dtype=np.int32)
            prv_n = np.full(N, -1, dtype=np.int32)
            ep_map = {}
            for ei in order_e:
                u, v = int(e_src[ei]), int(e_dst[ei])
                if nxt_n[u] == -1 and prv_n[v] == -1:
                    if prv_n[u] != -1:
                        h_n = prv_n[u]
                        dz1 = max(0.5, float(nodes_ras[u, 2] - nodes_ras[h_n, 2]))
                        dz2 = max(0.5, float(nodes_ras[v, 2] - nodes_ras[u, 2]))
                        if math.hypot((nodes_ras[u, 0] - nodes_ras[h_n, 0]) / dz1 - (nodes_ras[v, 0] - nodes_ras[u, 0]) / dz2,
                                      (nodes_ras[u, 1] - nodes_ras[h_n, 1]) / dz1 - (nodes_ras[v, 1] - nodes_ras[u, 1]) / dz2) > 0.22:
                            continue
                    nxt_n[u] = v
                    prv_n[v] = u
                    ep_map[(u, v)] = float(p_ed[ei])

            vis_n = np.zeros(N, dtype=bool)
            for u in np.argsort(n_kr):
                if vis_n[u] or prv_n[u] != -1 or nxt_n[u] == -1:
                    continue
                path = [int(u)]
                vis_n[u] = True
                cur = int(u)
                while nxt_n[cur] != -1 and not vis_n[nxt_n[cur]]:
                    nx = int(nxt_n[cur])
                    path.append(nx)
                    vis_n[nx] = True
                    cur = nx
                if len(path) >= 4:
                    pts = nodes_ras[path].astype(np.float64)
                    z_sp = float(pts[-1, 2] - pts[0, 2])
                    mpa = float(np.mean(p_na[path]))
                    mpl = float(np.mean(p_nl[path]))
                    mpe = float(np.mean([ep_map.get((path[i], path[i + 1]), 0.5) for i in range(len(path) - 1)]))
                    if z_sp >= 10.0 and mpa >= 0.42:
                        w_p = np.sqrt(np.maximum(0.1, p_na[path]))
                        px1 = np.polyfit(pts[:, 2], pts[:, 0], 1, w=w_p)
                        py1 = np.polyfit(pts[:, 2], pts[:, 1], 1, w=w_p)
                        fx = np.polyval(px1, pts[:, 2])
                        fy = np.polyval(py1, pts[:, 2])
                        rmse = float(np.sqrt(np.mean((pts[:, 0] - fx) ** 2 + (pts[:, 1] - fy) ** 2)))
                        sl = np.array([float(px1[0]), float(py1[0])])
                        if rmse <= 1.25 and np.linalg.norm(sl) <= 0.45:
                            s_ras = np.column_stack([fx, fy, pts[:, 2]])
                            mpnn_trajs.append({
                                "ras": s_ras,
                                "points": s_ras,
                                "base_xy": np.array([float(fx[0]), float(fy[0])]),
                                "slope": sl,
                                "z_span_tracked": z_sp,
                                "rmse": rmse,
                                "mean_pa": mpa,
                                "mean_pl": mpl,
                                "mean_pe": mpe,
                            })
    except Exception:
        mpnn_trajs = []

    # 2. Enrich candidate_trajs with MPNN graph scores and add pure-MPNN trajectories
    pool = []
    for t in candidate_trajs:
        ras = np.asarray(t["ras"] if "ras" in t else t["points"], dtype=np.float64)
        ras = ras[np.argsort(ras[:, 2])]
        b_xy = np.asarray(t.get("base_xy", ras[0, :2]), dtype=np.float64)
        sl = np.asarray(t.get("slope", (ras[-1, :2] - ras[0, :2]) / max(1.0, ras[-1, 2] - ras[0, 2])), dtype=np.float64)
        z_sp = float(t.get("z_span_tracked", ras[-1, 2] - ras[0, 2]))
        rmse_v = float(t.get("rmse", 0.35))
        conf_h = float(t.get("confidence", 65.0)) / 100.0
        is_s2 = 0.5 if t.get("pass", 1) == 4 else 0.0

        t["ras"] = ras
        t["points"] = ras
        pool.append({
            "traj": t,
            "ras": ras,
            "base_xy": b_xy,
            "slope": sl,
            "z_span": z_sp,
            "rmse": rmse_v,
            "conf_heur": conf_h,
            "from_mpnn": is_s2,
            "mpnn_pa": conf_h if is_s2 > 0 else 0.0,
            "mpnn_pl": 0.5 if is_s2 > 0 else 0.0,
            "mpnn_pe": 0.6 if is_s2 > 0 else 0.0,
        })

    for mt in mpnn_trajs:
        ras = mt["ras"]
        matched_idx = -1
        best_d = 2.05
        for idx_p, p in enumerate(pool):
            d = _traj_dist_np(ras, p["ras"])
            if d < best_d:
                best_d = d
                matched_idx = idx_p
        if matched_idx >= 0:
            pool[matched_idx]["from_mpnn"] = 1.0
            pool[matched_idx]["mpnn_pa"] = mt["mean_pa"]
            pool[matched_idx]["mpnn_pl"] = mt["mean_pl"]
            pool[matched_idx]["mpnn_pe"] = mt["mean_pe"]
        elif not script:
            diffs = np.diff(ras, axis=0)
            len_mm = float(np.sum(np.linalg.norm(diffs, axis=1)))
            mt_traj = {
                "ras": ras,
                "points": ras,
                "base_xy": mt["base_xy"],
                "slope": mt["slope"],
                "z_span_tracked": mt["z_span_tracked"],
                "length_mm": len_mm,
                "rmse": mt["rmse"],
                "confidence": float(100.0 * mt["mean_pa"] * mt["mean_pe"]),
                "pass": 5,
            }
            pool.append({
                "traj": mt_traj,
                "ras": ras,
                "base_xy": mt["base_xy"],
                "slope": mt["slope"],
                "z_span": mt["z_span_tracked"],
                "rmse": mt["rmse"],
                "conf_heur": float(mt["mean_pa"] * mt["mean_pe"]),
                "from_mpnn": 1.0,
                "mpnn_pa": mt["mean_pa"],
                "mpnn_pl": mt["mean_pl"],
                "mpnn_pe": mt["mean_pe"],
            })

    # Collinear Axial Split-Fragment Stitching & Kostoulas 2025 Gap-Bridging
    pool = _stitch_collinear_pool_fragments_np(pool)

    t_s2_split = time.perf_counter()
    LAST_STAGE2_TIMINGS["s2a_mpnn_sec"] = t_s2_split - t_s2a_0
    M = len(pool)
    if M == 0 or gt_sess is None:
        LAST_STAGE2_TIMINGS["s2b_bgt_poly3_sec"] = 0.0
        _reset_stage2_decisions(M)
        kept_fb = []
        for r_i, p_c in enumerate(sorted(pool, key=lambda p: -float(p["traj"].get("confidence", 0.0)))):
            c_fb = float(p_c["traj"].get("confidence", 0.0))
            if c_fb >= float(min_confidence):
                _record_stage2_decision(p_c["traj"], REASON_NO_GRAPH_MODEL, c_fb, r_i + 1); kept_fb.append(p_c["traj"])
            else:
                _record_stage2_decision(p_c["traj"], REASON_BELOW_THRESHOLD, c_fb, r_i + 1)
        return kept_fb

    # 3. Build Trajectory-Level Bundle Graph & Run DenseGraphTransformer + TipAwareBundleGraphTransformer
    bases = np.array([p["base_xy"] for p in pool])
    slopes = np.array([p["slope"] for p in pool])
    med_base = np.median(bases, axis=0)
    med_slope = np.median(slopes, axis=0)

    vol_roi_full = vol if vol.shape == (K_sub, H, W) else vol[k0:k0 + K_sub, j0:j0 + H, i0:i0 + W]
    r_sub_f = resp_roi[::2, ::2, ::2] if K_sub > 40 else resp_roi
    pos_r_f = r_sub_f[r_sub_f > 0]
    r_sc_f = float(np.percentile(pos_r_f, 99)) if len(pos_r_f) > 20 else 50.0
    log_r_sc_f = max(1.0, math.log1p(r_sc_f))
    c_sub_f = cs_roi[::2, ::2, ::2] if K_sub > 40 else cs_roi
    pos_c_f = c_sub_f[c_sub_f > 0]
    c_sc_f = max(1.0, float(np.percentile(pos_c_f, 99)) if len(pos_c_f) > 20 else 25.0)

    z_tops = np.array([float(p["ras"][-1, 2]) for p in pool], dtype=np.float32)
    z_bots = np.array([float(p["ras"][0, 2]) for p in pool], dtype=np.float32)

    prof_meta = []
    for p in pool:
        ras = p["ras"]
        zs = np.linspace(ras[0, 2], ras[-1, 2], 15)
        xs = np.interp(zs, ras[:, 2], ras[:, 0])
        ys = np.interp(zs, ras[:, 2], ras[:, 1])
        ijks = (ras_to_ijk @ np.column_stack([xs, ys, zs, np.ones_like(zs)]).T).T
        pm_l, pp_l, po_l, sym_l, cs_l, rc_l, sig_l = [], [], [], [], [], [], []
        for row in ijks:
            ic = int(round(row[0] - i0))
            jc = int(round(row[1] - j0))
            kc = int(round(row[2] - k0))
            if 1 <= ic < W - 1 and 1 <= jc < H - 1 and 0 <= kc < K_sub:
                pm_v_max = float(np.max(p_mask[kc, jc - 1:jc + 2, ic - 1:ic + 2]))
                pm_l.append(pm_v_max)
                pp_l.append(float(np.max(p_peak[kc, jc - 1:jc + 2, ic - 1:ic + 2])))
                po_l.append(float(np.max(p_obt[kc, jc - 1:jc + 2, ic - 1:ic + 2])))
                r_max = float(np.max(resp_roi[kc, jc - 1:jc + 2, ic - 1:ic + 2]))
                sym_l.append(min(1.5, max(0.0, math.log1p(max(0.0, r_max)) / log_r_sc_f)))
                c_max = float(np.max(cs_roi[kc, jc - 1:jc + 2, ic - 1:ic + 2]))
                cs_l.append(min(1.5, max(0.0, c_max / c_sc_f)))
                rs_p = rsym_roi[kc, jc - 1:jc + 2, ic - 1:ic + 2]
                v_p = vol_roi_full[kc, jc - 1:jc + 2, ic - 1:ic + 2]
                rc_l.append(float(np.clip(np.max((rs_p - v_p) / (rs_p + 15.0)), -0.5, 1.0)))
                sig_l.append(0.5 * math.sqrt(max(0.0, pm_v_max * (1.0 - pm_v_max))))
        mean_pm = float(np.mean(pm_l)) if pm_l else 0.0
        p75_pm = float(np.percentile(pm_l, 75)) if pm_l else 0.0
        mean_pp = float(np.mean(pp_l)) if pp_l else 0.0
        mean_po = float(np.mean(po_l)) if po_l else 0.0
        mean_sym = float(np.mean(sym_l)) if sym_l else 0.0
        p75_sym = float(np.percentile(sym_l, 75)) if sym_l else 0.0
        mean_cs = float(np.mean(cs_l)) if cs_l else 0.0
        mean_rc = float(np.mean(rc_l)) if rc_l else 0.0
        frac_active = float(np.mean([(m > 0.22 or pk > 0.18 or s > 0.25) for m, pk, s in zip(pm_l, pp_l, sym_l)])) if pm_l else 0.0
        # Upgrade B: Dai et al. 2020 Axial TV Continuity & Aleong/Kim Uncertainty ratio
        axial_tv_pm = float(np.mean(np.abs(np.diff(pm_l)))) if len(pm_l) >= 2 else 0.25
        mean_sig = float(np.mean(sig_l)) if sig_l else 0.20
        unc_ratio = mean_sig / max(0.08, mean_pm + 0.05)

        top_pts = ras[-min(6, len(ras)):]
        if len(top_pts) >= 2 and abs(top_pts[-1, 2] - top_pts[0, 2]) > 1.0:
            vx_top = float((top_pts[-1, 0] - top_pts[0, 0]) / (top_pts[-1, 2] - top_pts[0, 2]))
            vy_top = float((top_pts[-1, 1] - top_pts[0, 1]) / (top_pts[-1, 2] - top_pts[0, 2]))
        else:
            vx_top, vy_top = 0.0, 0.0

        z_top_cur = float(ras[-1, 2])
        z_probe = np.linspace(z_top_cur - 15.0, z_top_cur + 15.0, 25)
        x_probe = np.where(z_probe <= z_top_cur, np.interp(z_probe, ras[:, 2], ras[:, 0]), ras[-1, 0] + vx_top * (z_probe - z_top_cur))
        y_probe = np.where(z_probe <= z_top_cur, np.interp(z_probe, ras[:, 2], ras[:, 1]), ras[-1, 1] + vy_top * (z_probe - z_top_cur))
        ijks_p = (ras_to_ijk @ np.column_stack([x_probe, y_probe, z_probe, np.ones_like(z_probe)]).T).T
        support_p = np.zeros(len(z_probe), dtype=np.float32)
        for idx_p, row in enumerate(ijks_p):
            ic, jc, kc = int(round(row[0] - i0)), int(round(row[1] - j0)), int(round(row[2] - k0))
            if 1 <= ic < W - 1 and 1 <= jc < H - 1 and 0 <= kc < K_sub:
                m_v = float(np.max(p_mask[kc, jc - 1:jc + 2, ic - 1:ic + 2]))
                p_v = float(np.max(p_peak[kc, jc - 1:jc + 2, ic - 1:ic + 2]))
                o_v = float(np.max(p_obt[kc, jc - 1:jc + 2, ic - 1:ic + 2]))
                r_max = float(np.max(resp_roi[kc, jc - 1:jc + 2, ic - 1:ic + 2]))
                s_v = min(1.5, max(0.0, math.log1p(max(0.0, r_max)) / log_r_sc_f))
                support_p[idx_p] = 0.40 * max(m_v, o_v) + 0.35 * p_v + 0.25 * min(1.0, s_v)
        valid_sup = np.where(support_p >= 0.24)[0]
        z_prof_tip = float(z_probe[int(valid_sup[-1])]) if len(valid_sup) > 0 else z_top_cur
        tip_sup_cur = float(support_p[12])

        prof_meta.append({
            "mean_pm": mean_pm, "p75_pm": p75_pm, "mean_pp": mean_pp, "mean_po": mean_po,
            "mean_sym": mean_sym, "p75_sym": p75_sym, "mean_cs": mean_cs, "mean_rc": mean_rc,
            "frac_active": frac_active, "z_prof_tip": z_prof_tip, "tip_sup_cur": tip_sup_cur,
            "vx_top": vx_top, "vy_top": vy_top,
            "axial_tv_pm": axial_tv_pm, "unc_ratio": unc_ratio,
        })

    qual_weights = np.array([max(0.05, pr["mean_pm"] * 0.4 + pr["mean_pp"] * 0.3 + pr["mean_sym"] * 0.3) for pr in prof_meta])
    z_tip_nbr = np.zeros(M, dtype=np.float32)
    for u in range(M):
        dists_xy = np.hypot(bases[:, 0] - bases[u, 0], bases[:, 1] - bases[u, 1])
        span_u = z_tops[u] - z_bots[u]
        same_class = np.abs((z_tops - z_bots) - span_u) <= 22.0
        nbr_mask = (dists_xy <= 28.0) & same_class & (qual_weights >= 0.25)
        if np.sum(nbr_mask) >= 2:
            w_n = qual_weights[nbr_mask]
            z_tip_nbr[u] = float(np.sum(z_tops[nbr_mask] * w_n) / np.sum(w_n))
        else:
            z_tip_nbr[u] = float(z_tops[u])

    x_feats = []
    x_feats_26 = []
    for u, p in enumerate(pool):
        ras = p["ras"]
        pr = prof_meta[u]
        d_base = float(np.hypot(p["base_xy"][0] - med_base[0], p["base_xy"][1] - med_base[1])) / 40.0
        d_slope = float(np.hypot(p["slope"][0] - med_slope[0], p["slope"][1] - med_slope[1])) / 0.25
        z_low = float(ras[0, 2] - z_base_ras) / 50.0
        z_high = float(ras[-1, 2] - z_base_ras) / 100.0
        dz_prof = float(pr["z_prof_tip"] - ras[-1, 2]) / 20.0
        dz_nbr = float(z_tip_nbr[u] - ras[-1, 2]) / 20.0

        p["z_prof_tip"] = pr["z_prof_tip"]
        p["z_tip_nbr"] = float(z_tip_nbr[u])
        p["vx_top"] = pr["vx_top"]
        p["vy_top"] = pr["vy_top"]
        p["mean_sym"] = pr["mean_sym"]
        p["frac_active"] = pr["frac_active"]
        p["tip_sup_cur"] = pr["tip_sup_cur"]
        p["axial_tv_pm"] = pr["axial_tv_pm"]
        p["unc_ratio"] = pr["unc_ratio"]

        f18 = [
            p["conf_heur"], p["from_mpnn"], p["mpnn_pa"], p["mpnn_pl"], p["mpnn_pe"],
            pr["mean_pm"], pr["p75_pm"], pr["mean_pp"], pr["mean_po"], max(pr["mean_pm"], pr["mean_po"]),
            p["z_span"] / 60.0, p["rmse"], d_base, d_slope, z_low, z_high,
            float(p["slope"][0]), float(p["slope"][1]),
        ]
        x_feats.append(f18)
        x_feats_26.append(f18 + [
            pr["mean_sym"], pr["p75_sym"], pr["mean_cs"], pr["mean_rc"],
            pr["frac_active"], pr["tip_sup_cur"], dz_prof, dz_nbr,
        ])

    x_arr = np.asarray(x_feats, dtype=np.float32)
    x_arr_26 = np.asarray(x_feats_26, dtype=np.float32)
    adj_feat = np.zeros((M, M, 6), dtype=np.float32)
    adj_feat_8 = np.zeros((M, M, 8), dtype=np.float32)
    adj_mask = np.zeros((M, M), dtype=np.float32)
    for u in range(M):
        for v in range(M):
            if u == v:
                continue
            db = float(np.hypot(pool[u]["base_xy"][0] - pool[v]["base_xy"][0], pool[u]["base_xy"][1] - pool[v]["base_xy"][1]))
            if db <= 35.0:
                ds = float(np.hypot(pool[u]["slope"][0] - pool[v]["slope"][0], pool[u]["slope"][1] - pool[v]["slope"][1]))
                sh_d = _traj_dist_np(pool[u]["ras"], pool[v]["ras"])
                dz_t = float(z_tops[v] - z_tops[u]) / 30.0
                z_ov = max(0.0, min(z_tops[u], z_tops[v]) - max(z_bots[u], z_bots[v])) / max(1.0, max(z_tops[u] - z_bots[u], z_tops[v] - z_bots[v]))
                adj_mask[u, v] = 1.0
                adj_feat[u, v] = [
                    db / 35.0, sh_d / 35.0, ds / 0.30,
                    1.0 if sh_d < 2.5 else 0.0,
                    float(x_arr[v, 5] - x_arr[u, 5]),
                    float(x_arr[v, 0] - x_arr[u, 0]),
                ]
                adj_feat_8[u, v] = [
                    db / 35.0, sh_d / 35.0, ds / 0.30,
                    1.0 if sh_d < 2.2 else 0.0,
                    float(x_arr_26[v, 5] - x_arr_26[u, 5]),
                    float(x_arr_26[v, 0] - x_arr_26[u, 0]),
                    dz_t, z_ov,
                ]

    try:
        logits_base = gt_sess.run(None, {"x": x_arr, "adj_feat": adj_feat, "adj_mask": adj_mask})[0]
        probs_base = 1.0 / (1.0 + np.exp(-np.clip(logits_base, -20.0, 20.0)))
    except Exception:
        _reset_stage2_decisions(M)
        kept_fb = []
        for r_i, p_c in enumerate(sorted(pool, key=lambda p: -float(p["traj"].get("confidence", 0.0)))):
            c_fb = float(p_c["traj"].get("confidence", 0.0))
            if c_fb >= float(min_confidence):
                _record_stage2_decision(p_c["traj"], REASON_NO_GRAPH_MODEL, c_fb, r_i + 1); kept_fb.append(p_c["traj"])
            else:
                _record_stage2_decision(p_c["traj"], REASON_BELOW_THRESHOLD, c_fb, r_i + 1)
        return kept_fb

    tip_out = _forward_tip_bgt_numpy(x_arr_26, adj_feat_8, adj_mask, W_dict)
    if tip_out is not None:
        probs_syn = 1.0 / (1.0 + np.exp(-np.clip(tip_out[:, :3], -20.0, 20.0)))
        probs = 0.42 * probs_base + 0.58 * probs_syn
        dz_pred_net = tip_out[:, 3] * 15.0
    else:
        probs = probs_base
        dz_pred_net = np.zeros(M, dtype=np.float32)

    max_mean_pm = max((pr["mean_pm"] for pr in prof_meta), default=0.0)
    is_clinical_mri = (max_mean_pm >= 0.35)
    conf_arr = np.zeros(M, dtype=np.float64)
    for i, p in enumerate(pool):
        t = p["traj"]
        p_rec = float(probs[i, 0])
        p_1to1 = float(probs[i, 1])
        p_long = float(probs[i, 2])
        if not is_clinical_mri:
            gt_conf_ood = 100.0 * (0.45 * p_rec + 0.55 * p_long)
            phys_blend = 0.65 * float(t.get("confidence", 65.0)) + 0.35 * gt_conf_ood
            if max_mean_pm <= 0.15:
                blended_conf = phys_blend
            else:
                w_in = (max_mean_pm - 0.15) / 0.20
                gt_conf_in = 100.0 * (0.45 * p_rec + 0.55 * p_1to1)
                blended_conf = (1.0 - w_in) * phys_blend + w_in * gt_conf_in
        else:
            blended_conf = 100.0 * (0.45 * p_rec + 0.55 * p_1to1)
            # Upgrade B: Dai et al. 2020 Axial TV Continuity & Aleong/Kim Uncertainty Adjustment
            if p.get("axial_tv_pm", 0.25) <= 0.12 and p.get("unc_ratio", 0.5) <= 0.28 and p.get("frac_active", 0.0) >= 0.70:
                blended_conf += 1.8
            elif p.get("axial_tv_pm", 0.0) >= 0.28 and p.get("unc_ratio", 0.0) >= 0.52 and p.get("mean_sym", 1.0) < 0.22:
                blended_conf -= 5.5
            if t.get("is_stitched", False) and p.get("frac_active", 0.0) >= 0.65:
                blended_conf += 2.2
        t["confidence"] = float(np.clip(blended_conf, 1.0, 99.9))
        t["graph_prob"] = float(0.45 * p_rec + 0.55 * p_1to1)
        conf_arr[i] = t["confidence"]

    # IRGM implant centroid & radius in XY physical space
    high_idx = np.where(conf_arr >= 55.0)[0]
    if len(high_idx) >= 3:
        all_xy = np.concatenate([pool[idx]["ras"][:, :2] for idx in high_idx], axis=0)
        center_xy = np.median(all_xy, axis=0)
        dists_core = np.linalg.norm(all_xy - center_xy[None, :], axis=1)
        irgm_radius_mm = float(np.percentile(dists_core, 98.0)) + 15.5
    else:
        center_xy = None
        irgm_radius_mm = 999.0

    q90_conf = float(W_dict.get("conformal.q90", np.array([1.965]))[0]) if "conformal.q90" in W_dict else 1.965
    out_trajs = []
    out_pool_refs = []
    _reset_stage2_decisions(M)
    for rank_c, idx_c in enumerate(np.argsort(-conf_arr), start=1):
        p_c = pool[idx_c]
        cand = p_c["traj"]
        conf_c = float(conf_arr[idx_c])
        if conf_c < float(min_confidence):
            _record_stage2_decision(cand, REASON_BELOW_THRESHOLD, conf_c, rank_c)
            continue
        pts = np.asarray(cand["ras"], dtype=np.float64)
        if center_xy is not None:
            if float(np.linalg.norm(pts[:, :2].mean(axis=0) - center_xy)) > irgm_radius_mm and conf_c < float(min_confidence) + 20.0:
                _record_stage2_decision(cand, REASON_OUTSIDE_IMPLANT, conf_c, rank_c)
                continue
        if is_clinical_mri and p_c.get("mean_sym", 0.5) < 0.14 and p_c.get("frac_active", 0.5) < 0.40 and conf_c < float(min_confidence) + 15.0:
            _record_stage2_decision(cand, REASON_LOW_SYMMETRY, conf_c, rank_c)
            continue
        dedup_thresh = 1.60 if (conf_c >= 65.0 and p_c.get("mpnn_pa", 0.0) >= 0.50) else 1.95
        if any(_traj_dist_np(pts, s["ras"]) < dedup_thresh for s in out_trajs):
            _record_stage2_decision(cand, REASON_DUPLICATE, conf_c, rank_c)
            continue

        # Extrapolation-aware collinear split-fragment suppression & cranial extension
        is_collinear_dup = False
        merged_into_kept = False
        for idx_s, s_prev in enumerate(out_trajs):
            is_dup, should_ext, merged_pts = _collinear_traj_dist_np(pts, s_prev["ras"])
            if is_dup:
                is_collinear_dup = True
                if should_ext and conf_c >= max(38.0, float(min_confidence) - 4.0):
                    s_prev["ras"] = merged_pts
                    s_prev["points"] = merged_pts
                    s_prev["length_mm"] = float(np.sum(np.linalg.norm(np.diff(merged_pts, axis=0), axis=1)))
                    s_prev["z_span_tracked"] = float(abs(merged_pts[-1, 2] - merged_pts[0, 2]))
                    merged_into_kept = True
                break
        if is_collinear_dup:
            _record_stage2_decision(cand, REASON_COLLINEAR_MERGED if merged_into_kept else REASON_COLLINEAR_FRAGMENT, conf_c, rank_c)
            continue

        _record_stage2_decision(cand, REASON_KEPT, conf_c, rank_c)
        out_trajs.append(cand)
        out_pool_refs.append((idx_c, p_c))

    # Stage 2C: Dedicated 1D Axial Tip Localizer + Split-Conformal Tip Uncertainty + Kim 2026 Weighted Poly3
    if is_clinical_mri:
        for cand, (idx_c, p_c) in zip(out_trajs, out_pool_refs):
            pts = np.asarray(cand["ras"], dtype=np.float64)
            vx_top = float(p_c.get("vx_top", 0.0))
            vy_top = float(p_c.get("vy_top", 0.0))
            dz_prof_mm = float(p_c.get("z_prof_tip", pts[-1, 2]) - pts[-1, 2])
            dz_nbr_mm = float(p_c.get("z_tip_nbr", pts[-1, 2]) - pts[-1, 2])
            dz_s2b_mm = float(dz_pred_net[idx_c]) if tip_out is not None else 0.0

            dz_1d_mm = None
            tip_std_1d = 2.2
            if "tip1d.stem.0.weight" in W_dict:
                try:
                    prof22 = _extract_1d_tip_profile_22_fast_np(
                        pts, vx_top, vy_top, float(p_c.get("z_tip_nbr", pts[-1, 2])),
                        vol_roi_full, p_mask, p_peak, p_obt, vx_map, vy_map,
                        resp_roi, cs_roi, rsym_roi,
                        p10_v, p95_v, log_r_sc_f, c_sc_f,
                        k0, i0, j0, sp_i, sp_j, sp_k, ras_to_ijk, center_ras, z_base_ras
                    )
                    res_1d = _forward_1d_tip_localizer_numpy(prof22, W_dict)
                    if res_1d is not None:
                        dz_1d_mm, tip_std_1d, _ = res_1d
                except Exception:
                    dz_1d_mm = None

            if dz_1d_mm is not None:
                dz_comb_mm = 0.65 * dz_1d_mm + 0.20 * (0.36 * dz_s2b_mm) + 0.15 * (0.36 * dz_prof_mm)
                disagree_mm = abs(dz_1d_mm - dz_prof_mm)
            else:
                dz_comb_mm = 0.36 * (0.65 * dz_s2b_mm + 0.25 * dz_prof_mm + 0.10 * dz_nbr_mm)
                disagree_mm = abs(dz_s2b_mm - dz_prof_mm)

            pts_ref = _apply_tip_adjustment_and_poly3_np(
                pts, dz_comb_mm, vx_top, vy_top,
                p_mask=p_mask, p_peak=p_peak, resp_roi=resp_roi, log_r_sc_f=log_r_sc_f,
                k0=k0, i0=i0, j0=j0, ijk_to_ras_mat=ijk_to_ras_mat, ras_to_ijk=ras_to_ijk
            )
            cand["ras"] = pts_ref
            cand["points"] = pts_ref
            cand["length_mm"] = float(np.sum(np.linalg.norm(np.diff(pts_ref, axis=0), axis=1)))
            cand["z_span_tracked"] = float(abs(pts_ref[-1, 2] - pts_ref[0, 2]))

            # Split-Conformal Tip Uncertainty score sigma_tip (calibrated to median ~1.77 mm, 90% CI radius ~3.5 mm)
            tip_sup = float(p_c.get("tip_sup_cur", 0.5))
            sigma_tip = max(
                0.85,
                0.80
                + 0.14 * min(10.0, tip_std_1d)
                + 0.22 * min(10.0, disagree_mm)
                + 2.4 * max(0.0, 0.45 - tip_sup)
                + 1.2 * float(p_c.get("unc_ratio", 0.20)),
            )
            cand["tip_uncertainty_mm"] = float(round(sigma_tip, 2))
            cand["tip_ci90_mm"] = float(round(q90_conf * sigma_tip, 2))
            cand["needs_tip_review"] = False

        # Flag the k=2 most uncertain tips per implant (Protocol B top-2 uncertainty tip triage)
        if len(out_trajs) > 0:
            ci_vals = np.array([float(t.get("tip_ci90_mm", 0.0)) for t in out_trajs])
            top_unc_idxs = np.argsort(-ci_vals)[:2]
            for idx_u in top_unc_idxs:
                if float(out_trajs[idx_u].get("tip_ci90_mm", 0.0)) >= 4.0:
                    out_trajs[idx_u]["needs_tip_review"] = True

    LAST_STAGE2_TIMINGS["s2b_bgt_poly3_sec"] = time.perf_counter() - t_s2_split
    return out_trajs
