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
import numpy as np
from scipy.ndimage import maximum_filter

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
        sess_opts.intra_op_num_threads = max(1, min(8, (os.cpu_count() or 4)))
        providers = ["CPUExecutionProvider"]
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

    pad_k = (2 - (K_sub % 2)) % 2
    pad_h = (8 - (H % 8)) % 8
    pad_w = (8 - (W % 8)) % 8
    if pad_k > 0 or pad_h > 0 or pad_w > 0:
        X = np.pad(X, ((0, 0), (0, pad_k), (0, pad_h), (0, pad_w)), mode="edge")

    K_pad = X.shape[1]
    if K_pad <= 72:
        out = sess.run(None, {"input": X[None, ...]})[0][0]
    else:
        out = np.zeros((5, K_pad, X.shape[2], X.shape[3]), dtype=np.float32)
        w_acc = np.zeros((1, K_pad, 1, 1), dtype=np.float32)
        chunk = 64
        stride = 52
        for k_s in range(0, K_pad, stride):
            k_e = min(K_pad, k_s + chunk)
            if k_e - k_s < 8:
                k_s = max(0, k_e - chunk)
            sub_x = X[:, k_s:k_e]
            sub_o = sess.run(None, {"input": sub_x[None, ...]})[0][0]
            out[:, k_s:k_e] += sub_o
            w_acc[:, k_s:k_e] += 1.0
            if k_e == K_pad:
                break
        out /= np.maximum(1.0, w_acc)

    out = out[:, :K_sub, :H, :W]
    p_mask_w = 1.0 / (1.0 + np.exp(-np.clip(out[0], -20.0, 20.0)))
    p_peak_w = 1.0 / (1.0 + np.exp(-np.clip(out[1], -20.0, 20.0)))
    vx_w = np.tanh(out[2]) * 0.75
    vy_w = np.tanh(out[3]) * 0.75
    p_obt_w = 1.0 / (1.0 + np.exp(-np.clip(out[4], -20.0, 20.0)))

    def _restore_full(arr_w):
        if H_full > max_h or W_full > max_w:
            arr_c = np.zeros((K_sub, H_full, W_full), dtype=np.float32)
            arr_c[:, hc0:hc1, wc0:wc1] = arr_w
        else:
            arr_c = arr_w.astype(np.float32)
        if stride_k == 2:
            arr_c = _upsample_2x_k(arr_c, K_in)
        if stride_xy == 2:
            arr_c = _upsample_2x_xy(arr_c, H_in, W_in)
        return arr_c

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
    lm = work_peak == maximum_filter(work_peak, size=(1, 2 * nms_px + 1, 2 * nms_px + 1))
    seed_mask = lm & (work_peak >= 0.42) & ((p_mask >= 0.45) | (p_obt >= 0.45))
    sk, sj, si = np.where(seed_mask)
    if len(sk) == 0:
        return []

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
            "confidence": float(np.clip(max(mean_pm, mean_po) * 100.0, 50.0, 99.9)),
            "length_mm": float(smooth_ras[-1, 2] - smooth_ras[0, 2]),
            "num_points": len(ks),
            "pass": 4,
        })

    return new_trajs
