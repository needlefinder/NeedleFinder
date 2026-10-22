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


def run_mpnn_and_graph_transformer(
    vol, nn_preds, resp_roi, cs_roi, rsym_roi,
    k0, i0, j0, sp_i, sp_j, sp_k,
    ijk_to_ras_mat, ras_to_ijk, candidate_trajs,
    min_confidence=50.0, script=False,
):
    """
    Run Stage 2 3D Spatio-Temporal NeedleSliceMPNN + Trajectory-Level DenseGraphTransformer:
    1. Builds the 3D slice detection graph and runs 4-layer Triplet Collinearity MPNN.
    2. Decodes disjoint 3D needle trajectories from MPNN edge probabilities and merges them
       with the candidate trajectory pool.
    3. Runs the Trajectory-Level GATv2 + Set Transformer (`needle_graph_transformer.onnx`)
       to assign calibrated confidence scores and suppress duplicate/false-positive tracks.
    """
    W_dict = _get_graphnet_weights()
    gt_sess = _get_gt_ort_session()
    if W_dict is None or nn_preds is None or not candidate_trajs:
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
        nz = sub_v[sub_v > 0]
        p10 = float(np.percentile(nz, 10)) if len(nz) > 100 else 0.0
        p95 = float(np.percentile(nz, 95)) if len(nz) > 100 else 100.0
        ch0 = np.clip((sub_v - p10) / max(1.0, p95 - p10), -0.5, 2.5).astype(np.float32)
        pr = resp_s[resp_s > 0]
        r_sc = float(np.percentile(pr, 99)) if len(pr) > 20 else 50.0
        ch1 = np.clip(np.log1p(np.maximum(0.0, resp_s)) / max(1.0, math.log1p(r_sc)), 0.0, 1.5).astype(np.float32)
        pc = cs_s[cs_s > 0]
        c_sc = float(np.percentile(pc, 99)) if len(pc) > 20 else 25.0
        ch2 = np.clip(cs_s / max(1.0, c_sc), 0.0, 1.5).astype(np.float32)
        ch3 = np.clip((rsym_s - sub_v) / (rsym_s + 15.0), -0.5, 1.0).astype(np.float32)

        sp_i_e, sp_j_e, sp_k_e = sp_i * st_xy, sp_j * st_xy, sp_k * st_k
        score_vol = 0.48 * pp_s + 0.30 * pm_s + 0.14 * po_s + 0.12 * np.clip(ch1, 0.0, 1.2)
        nms_r = max(1, int(round(1.35 / sp_i_e)))
        loc_max = score_vol == maximum_filter(score_vol, size=(1, 2 * nms_r + 1, 2 * nms_r + 1))
        pm_max3_vol = maximum_filter(pm_s, size=(1, 3, 3))
        pp_max3_vol = maximum_filter(pp_s, size=(1, 3, 3))
        yy_w, xx_w = np.mgrid[-1:2, -1:2]

        n_kr, n_ic, n_jc, n_si, n_sj = [], [], [], [], []
        for kr in range(Ks):
            mk = loc_max[kr] & (score_vol[kr] >= 0.20) & ((pm_s[kr] >= 0.22) | (pp_s[kr] >= 0.20) | (po_s[kr] >= 0.25))
            mk[0, :] = False; mk[-1, :] = False; mk[:, 0] = False; mk[:, -1] = False
            jj, ii = np.where(mk)
            if len(ii) == 0:
                continue
            if len(ii) > 32:
                order = np.argsort(-score_vol[kr, jj, ii])[:32]
                jj, ii = jj[order], ii[order]
            for jc, ic in zip(jj, ii):
                wp = np.maximum(0.0, score_vol[kr, jc - 1:jc + 2, ic - 1:ic + 2])
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
            pm_m3 = pm_max3_vol[n_kr, n_jc, n_ic]
            pp_m3 = pp_max3_vol[n_kr, n_jc, n_ic]
            r_mm = np.hypot((n_si - Ws * 0.5) * sp_i_e, (n_sj - Hs * 0.5) * sp_j_e) / 50.0

            nodes_feat = np.column_stack([
                (nodes_ras[:, 0] - center_ras[0]) / 45.0,
                (nodes_ras[:, 1] - center_ras[1]) / 45.0,
                (nodes_ras[:, 2] - z_base_ras) / 80.0,
                pm_v, pp_v, po_v, vx_v, vy_v,
                pm_m3, pp_m3, 0.5 * (pm_v + pm_m3),
                ch0[n_kr, n_jc, n_ic],
                ch1[n_kr, n_jc, n_ic],
                ch2[n_kr, n_jc, n_ic],
                ch3[n_kr, n_jc, n_ic],
                r_mm,
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

    M = len(pool)
    if M == 0 or gt_sess is None:
        return [p["traj"] for p in pool if p["traj"].get("confidence", 0.0) >= float(min_confidence)]

    # 3. Build Trajectory-Level Bundle Graph & Run DenseGraphTransformer
    bases = np.array([p["base_xy"] for p in pool])
    slopes = np.array([p["slope"] for p in pool])
    med_base = np.median(bases, axis=0)
    med_slope = np.median(slopes, axis=0)

    x_feats = []
    for p in pool:
        ras = p["ras"]
        zs = np.linspace(ras[0, 2], ras[-1, 2], 15)
        xs = np.interp(zs, ras[:, 2], ras[:, 0])
        ys = np.interp(zs, ras[:, 2], ras[:, 1])
        ijks = (ras_to_ijk @ np.column_stack([xs, ys, zs, np.ones_like(zs)]).T).T
        pm_l, pp_l, po_l = [], [], []
        for row in ijks:
            ic = int(round(row[0] - i0))
            jc = int(round(row[1] - j0))
            kc = int(round(row[2] - k0))
            if 1 <= ic < W - 1 and 1 <= jc < H - 1 and 0 <= kc < K_sub:
                pm_l.append(float(np.max(p_mask[kc, jc - 1:jc + 2, ic - 1:ic + 2])))
                pp_l.append(float(np.max(p_peak[kc, jc - 1:jc + 2, ic - 1:ic + 2])))
                po_l.append(float(np.max(p_obt[kc, jc - 1:jc + 2, ic - 1:ic + 2])))
        mean_pm = float(np.mean(pm_l)) if pm_l else 0.0
        p75_pm = float(np.percentile(pm_l, 75)) if pm_l else 0.0
        mean_pp = float(np.mean(pp_l)) if pp_l else 0.0
        mean_po = float(np.mean(po_l)) if po_l else 0.0

        d_base = float(np.hypot(p["base_xy"][0] - med_base[0], p["base_xy"][1] - med_base[1])) / 40.0
        d_slope = float(np.hypot(p["slope"][0] - med_slope[0], p["slope"][1] - med_slope[1])) / 0.25
        z_low = float(ras[0, 2] - z_base_ras) / 50.0
        z_high = float(ras[-1, 2] - z_base_ras) / 100.0

        x_feats.append([
            p["conf_heur"], p["from_mpnn"], p["mpnn_pa"], p["mpnn_pl"], p["mpnn_pe"],
            mean_pm, p75_pm, mean_pp, mean_po, max(mean_pm, mean_po),
            p["z_span"] / 60.0, p["rmse"], d_base, d_slope, z_low, z_high,
            float(p["slope"][0]), float(p["slope"][1]),
        ])

    x_arr = np.asarray(x_feats, dtype=np.float32)
    adj_feat = np.zeros((M, M, 6), dtype=np.float32)
    adj_mask = np.zeros((M, M), dtype=np.float32)
    for u in range(M):
        for v in range(M):
            if u == v:
                continue
            db = float(np.hypot(pool[u]["base_xy"][0] - pool[v]["base_xy"][0], pool[u]["base_xy"][1] - pool[v]["base_xy"][1]))
            if db <= 35.0:
                ds = float(np.hypot(pool[u]["slope"][0] - pool[v]["slope"][0], pool[u]["slope"][1] - pool[v]["slope"][1]))
                sh_d = _traj_dist_np(pool[u]["ras"], pool[v]["ras"])
                adj_mask[u, v] = 1.0
                adj_feat[u, v] = [
                    db / 35.0, sh_d / 35.0, ds / 0.30,
                    1.0 if sh_d < 2.5 else 0.0,
                    float(x_arr[v, 5] - x_arr[u, 5]),
                    float(x_arr[v, 0] - x_arr[u, 0]),
                ]

    try:
        logits = gt_sess.run(None, {"x": x_arr, "adj_feat": adj_feat, "adj_mask": adj_mask})[0]
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
    except Exception:
        return [p["traj"] for p in pool if p["traj"].get("confidence", 0.0) >= float(min_confidence)]

    conf_arr = np.zeros(M, dtype=np.float64)
    for i, p in enumerate(pool):
        t = p["traj"]
        p_rec = float(probs[i, 0])
        p_1to1 = float(probs[i, 1])
        p_long = float(probs[i, 2])
        if script:
            gt_conf = 100.0 * (0.45 * p_rec + 0.55 * p_long)
            blended_conf = 0.65 * float(t.get("confidence", 65.0)) + 0.35 * gt_conf
        else:
            blended_conf = 100.0 * (0.45 * p_rec + 0.55 * p_1to1)
        t["confidence"] = float(np.clip(blended_conf, 1.0, 99.9))
        t["graph_prob"] = float(0.45 * p_rec + 0.55 * p_1to1)
        conf_arr[i] = t["confidence"]

    out_trajs = []
    for idx_c in np.argsort(-conf_arr):
        if conf_arr[idx_c] < float(min_confidence):
            continue
        cand = pool[idx_c]["traj"]
        if any(_traj_dist_np(cand["ras"], s["ras"]) < 1.95 for s in out_trajs):
            continue
        out_trajs.append(cand)

    return out_trajs

