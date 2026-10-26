#!/usr/bin/env python3
"""
Prepare and verify the multi-observer 3D Needle Segmentation dataset on edelweiss:
- Combines Manual_CY (long perineal needles) + Manual-shortOneedle (Hongtao's short obturator needles) + Manual_LB
- Extracts 3D input features (Normalized MRI, Rust 3D Radial Symmetry, Ring Contrast, Darkness Ratio, Z-coord)
- Generates dense 3D ground-truth targets:
  1. Binary/soft needle labelmap mask (r <= 1.25 mm)
  2. Sharp sub-voxel centerline Gaussian peak map (sigma = 0.75 mm) for instance separation
  3. 3D unit tangent / slope field (dx/dz, dy/dz)
  4. Needle tip & base keypoint heatmaps
"""

from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from reproducibility.paths import resolve_path

import glob
import gzip
import math
import os
import pickle
import re
import sys
import time
import types
import urllib.parse
import xml.etree.ElementTree as ET
import numpy as np

for mod in ("vtk", "vtk.util", "vtk.util.numpy_support", "slicer", "slicer.util", "qt", "ctk", "SimpleITK", "sitkUtils"):
    sys.modules.setdefault(mod, types.ModuleType(mod))

sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder"))
from NeedleFinderLib.RustCore import rust_compute_3d_sym_response, is_rust_available


def read_nrrd_local(path, acsv_path=None):
    with open(path, "rb") as f:
        raw = f.read()
    sep = raw.find(b"\n\n")
    if sep < 0:
        sep = raw.find(b"\r\n\r\n")
        data = raw[sep + 4:]
    else:
        data = raw[sep + 2:]
    hdr = raw[:sep].decode("ascii", errors="ignore")
    meta = {}
    for line in hdr.splitlines():
        if ":" in line and not line.startswith("#"):
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip()
    sizes = [int(x) for x in meta["sizes"].split()]  # I, J, K
    enc = meta.get("encoding", "raw").lower()
    if "gzip" in enc or "gz" in enc:
        data = gzip.decompress(data)
    tp = meta["type"].lower()
    dt = {
        "short": "<i2", "int16": "<i2", "signed short": "<i2",
        "unsigned short": "<u2", "ushort": "<u2", "uint16": "<u2",
        "int": "<i4", "int32": "<i4", "unsigned int": "<u4", "uint32": "<u4",
        "float": "<f4", "double": "<f8",
        "unsigned char": "<u1", "uchar": "<u1", "uint8": "<u1",
        "char": "<i1", "int8": "<i1",
    }[tp]
    if meta.get("endian", "little").lower() == "big" and len(dt) > 2:
        dt = ">" + dt[1:]
    arr = np.ascontiguousarray(
        np.frombuffer(data, dtype=dt).reshape((sizes[2], sizes[1], sizes[0])),
        dtype=np.float64,
    )
    dirs = [
        [float(x) for x in p.strip("()").split(",")]
        for p in meta["space directions"].split()
        if "(" in p
    ]
    dmat_lps = np.array(dirs, dtype=np.float64).T
    sp = np.linalg.norm(dmat_lps, axis=0)
    org_lps = np.array(
        [float(x) for x in meta["space origin"].strip("()").split(",")],
        dtype=np.float64,
    )
    space_name = meta.get("space", "left-posterior-superior").lower()
    if "right-anterior-superior" in space_name or space_name == "ras":
        lps_to_ras = np.eye(3, dtype=np.float64)
    else:
        lps_to_ras = np.diag([-1.0, -1.0, 1.0])
    ijk_to_ras = np.eye(4, dtype=np.float64)
    ijk_to_ras[:3, :3] = lps_to_ras @ dmat_lps
    ijk_to_ras[:3, 3] = lps_to_ras @ org_lps
    ras_to_ijk = np.linalg.inv(ijk_to_ras)

    k_lim_acsv = None
    z_lim_ras = None
    if acsv_path and os.path.exists(acsv_path):
        try:
            with open(acsv_path, errors="ignore") as f:
                for line in f:
                    if line.startswith("point|"):
                        parts = line.split("|")
                        x_r, y_r, z_r = float(parts[1]), float(parts[2]), float(parts[3])
                        z_lim_ras = z_r
                        k_val = (ras_to_ijk @ np.array([x_r, y_r, z_r, 1.0]))[2]
                        k_lim_acsv = int(round(float(k_val)))
        except Exception:
            pass
    return arr, sp, ijk_to_ras, k_lim_acsv, z_lim_ras


def read_vtk_centerline_local(fp, z_lim_ras=None):
    try:
        with open(fp, "rb") as f:
            raw = f.read()
        m = re.search(rb"POINTS\s+(\d+)\s+(\w+)\r?\n", raw)
        if not m:
            return None
        npts = int(m.group(1))
        if npts < 100:
            return None
        is_bin = b"BINARY" in raw[:120]
        start = m.end()
        if is_bin:
            pts = np.frombuffer(raw[start:start + npts * 12], dtype=">f4").reshape(npts, 3).astype(np.float64)
        else:
            txt = raw[start:].decode("ascii", errors="ignore").split()
            pts = np.array([float(x) for x in txt[:npts * 3]], dtype=np.float64).reshape(npts, 3)
        idx1 = np.minimum(np.arange(50) * 50, npts - 1)
        idx2 = np.minimum(np.arange(50) * 50 + 25, npts - 1)
        cl = 0.5 * (pts[idx1] + pts[idx2])
        cl = cl[np.argsort(cl[:, 2])]
        if z_lim_ras is not None:
            cl_above = cl[cl[:, 2] >= z_lim_ras - 2.0]
            if len(cl_above) >= 2:
                cl = cl_above
        if np.linalg.norm(cl[-1] - cl[0]) < 3.0:
            return None
        return cl
    except Exception:
        return None


def dedup_needle_list(needles, dist_thresh=2.4):
    """Deduplicate needles across observers (CY, HT short obturator, LB) while keeping distinct segments."""
    out = []
    for cand in needles:
        cl_c = cand["pts"]
        is_dup = False
        for ex in out:
            cl_e = ex["pts"]
            z0 = max(float(cl_c[0, 2]), float(cl_e[0, 2]))
            z1 = min(float(cl_c[-1, 2]), float(cl_e[-1, 2]))
            if z1 > z0 + 2.0:
                zs = np.linspace(z0, z1, 12)
                dx = np.interp(zs, cl_c[:, 2], cl_c[:, 0]) - np.interp(zs, cl_e[:, 2], cl_e[:, 0])
                dy = np.interp(zs, cl_c[:, 2], cl_c[:, 1]) - np.interp(zs, cl_e[:, 2], cl_e[:, 1])
                if float(np.mean(np.hypot(dx, dy))) < dist_thresh:
                    is_dup = True
                    break
        if not is_dup:
            out.append(cand)
    return out


def load_all_dataset_cases(gyn_base=resolve_path("/home/guillaume/Dropbox/GYN Cases")):
    with open(os.path.join(gyn_base, "760CYScenes.txt")) as f:
        cy_lines = [l.strip() for l in f if l.strip()]

    cases = []
    for l in cy_lines:
        p = os.path.join(gyn_base, l.lstrip("./"))
        for a, b in [("NRRD-MRI", "NRRD"), ("Maunal_CY", "Manual_CY"), ("Manual CY", "Manual_CY")]:
            if not os.path.exists(p):
                p2 = p.replace(a, b)
                if os.path.exists(p2):
                    p = p2
        d = os.path.dirname(p)
        case_dir = os.path.dirname(os.path.dirname(d))
        case_num = int(re.search(r"(\d+)", os.path.basename(case_dir)).group(1))
        tree = ET.parse(p)
        root = tree.getroot()
        nrrd_path = None
        for vs in root.findall(".//VolumeArchetypeStorage"):
            fn = urllib.parse.unquote(vs.attrib.get("fileName", ""))
            if fn and "label" not in fn.lower() and "pelvis" not in fn.lower() and not fn.endswith(".png") and not fn.endswith(".tiff"):
                cand = os.path.join(d, fn)
                if os.path.exists(cand):
                    nrrd_path = cand
                    break
        if nrrd_path is None:
            cands = [
                x for x in glob.glob(os.path.join(case_dir, "**", "*.nrrd"), recursive=True)
                if "label" not in x.lower() and "pelvis" not in x.lower() and "crop" not in x.lower()
            ]
            nrrd_path = cands[0]

        acsv_cands = (
            glob.glob(os.path.join(d, "markers", "template slice position.acsv"))
            + glob.glob(os.path.join(d, "Markers*", "template slice position.acsv"))
            + glob.glob(os.path.join(case_dir, "*", "Manual_LB", "markers", "template slice position.acsv"))
            + glob.glob(os.path.join(case_dir, "*", "Manual_LB", "template slice position.acsv"))
            + [os.path.join(d, "template slice position.acsv")]
        )
        acsv_path = next((ap for ap in acsv_cands if os.path.exists(ap)), None)
        vol, sp_ras, ijk_to_ras, k_lim_gt, z_lim_ras = read_nrrd_local(nrrd_path, acsv_path)
        K, J, I = vol.shape

        # 1. Load official CY MRML needles
        st_map = {ms.attrib["id"]: urllib.parse.unquote(ms.attrib.get("fileName", "")) for ms in root.findall(".//ModelStorage")}
        long_vtks = {}
        for m in root.findall(".//Model"):
            name = m.attrib.get("name", "")
            attrs = m.attrib.get("attributes", "")
            if "type:Validation" in attrs and len(name.split("_")) == 2 and name.startswith("manual-seg_"):
                sref = m.attrib.get("storageNodeRef", "")
                fn = st_map.get(sref, name + ".vtk")
                fp = os.path.join(d, fn)
                if os.path.exists(fp):
                    long_vtks[name] = fp
        gts_cy = []
        for name in sorted(long_vtks.keys(), key=lambda s: int(s.split("_")[1])):
            cl = read_vtk_centerline_local(long_vtks[name], z_lim_ras=z_lim_ras)
            if cl is not None:
                gts_cy.append({"name": name, "pts": cl, "base": cl[0], "tip": cl[-1], "type": "cy_long"})

        # 2. Load Hongtao's short obturator needles (Manual-shortOneedle)
        ht_vtks = sorted(glob.glob(os.path.join(case_dir, "*", "Manual-shortOneedle", "manual-seg_*.vtk")))
        gts_ht = []
        for fp in ht_vtks:
            bn = os.path.basename(fp)
            if len(bn.split("_")) == 2 and not bn.endswith(".incorrect"):
                cl = read_vtk_centerline_local(fp, z_lim_ras=z_lim_ras)
                if cl is not None:
                    gts_ht.append({"name": "ht_" + bn[:-4], "pts": cl, "base": cl[0], "tip": cl[-1], "type": "ht_obturator"})

        # 3. Combine CY + HT (deduplicated) for complete ground truth
        gts_all = dedup_needle_list(gts_cy + gts_ht, dist_thresh=2.4)

        if k_lim_gt is None or not (0 <= k_lim_gt < K):
            ras_to_ijk = np.linalg.inv(ijk_to_ras)
            gt_base_ks = [float((ras_to_ijk @ np.append(g["base"], 1.0))[2]) for g in gts_cy]
            k_lim_gt = int(np.clip(round(float(np.median(gt_base_ks))), 0, K - 2))

        cases.append({
            "case_num": case_num,
            "case_id": f"Case {case_num:03d}",
            "nrrd_path": nrrd_path,
            "vol": vol,
            "sp_ras": sp_ras,
            "ijk_to_ras": ijk_to_ras,
            "k_lim_gt": k_lim_gt,
            "z_lim_ras": z_lim_ras,
            "gts_long": gts_cy,
            "gts_ht": [g for g in gts_all if g["type"] == "ht_obturator"],
            "gts_all": gts_all,
        })
    return cases


if __name__ == "__main__":
    t0 = time.perf_counter()
    print("Rust available:", is_rust_available())
    cases = load_all_dataset_cases()
    n_cy = sum(len(c["gts_long"]) for c in cases)
    n_ht = sum(len(c["gts_ht"]) for c in cases)
    n_all = sum(len(c["gts_all"]) for c in cases)
    print(f"Loaded {len(cases)} cases in {time.perf_counter()-t0:.2f}s:")
    print(f"  - CY Long Needles:      {n_cy}")
    print(f"  - HT Obturator Needles: {n_ht}")
    print(f"  - Total True Needles:   {n_all}")
