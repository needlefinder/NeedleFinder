"""Python ctypes bridge to the `needlefinder_rust` native acceleration library.

Automatically locates (or builds via `cargo build --release` if needed) the
platform-native shared library (`libneedlefinder_rust.so` / `.dylib` / `.dll`)
and exposes fast Python wrappers for the 3D cone-search, tip-finding, Bezier,
and center-of-mass kernels. Gracefully reports availability via `is_rust_available()`.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import numpy as np
import vtk
import vtk.util.numpy_support

_DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)
_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
_INT_PTR = ctypes.POINTER(ctypes.c_int32)
_SIZE_PTR = ctypes.POINTER(ctypes.c_size_t)
_SSIZE_PTR = ctypes.POINTER(ctypes.c_ssize_t)

_LIB = None
_LIB_CHECKED = False


def _shared_lib_filename() -> str:
    sys_name = platform.system().lower()
    if sys_name == "darwin":
        return "libneedlefinder_rust.dylib"
    if sys_name == "windows":
        return "needlefinder_rust.dll"
    return "libneedlefinder_rust.so"


def _select_macos_sdk() -> str | None:
    """Return a compatible macOS SDK path if the default CommandLineTools SDK has TAPI issues."""
    sdk_dir = "/Library/Developer/CommandLineTools/SDKs"
    if not os.path.isdir(sdk_dir):
        return None
    for candidate in ("MacOSX14.4.sdk", "MacOSX14.sdk", "MacOSX15.sdk", "MacOSX13.3.sdk", "MacOSX13.sdk"):
        full = os.path.join(sdk_dir, candidate)
        if os.path.exists(full):
            return full
    return None


def _build_cpp_fallback(crate_dir: str, out_lib_path: str, env: dict) -> bool:
    """Compile `src/needlefinder_core.cpp` (and combine with Rust arm64 via `lipo` on macOS if present)."""
    cpp_src = os.path.join(crate_dir, "src", "needlefinder_core.cpp")
    if not os.path.exists(cpp_src):
        return False
    os.makedirs(os.path.dirname(out_lib_path), exist_ok=True)
    sys_name = platform.system().lower()
    if sys_name == "darwin":
        sdk = env.get("SDKROOT") or _select_macos_sdk()
        sdk_args = ["-isysroot", sdk] if sdk else []
        # If cargo already built an arm64 Rust dylib, build x86_64 and lipo them into a Universal dylib
        if os.path.exists(out_lib_path):
            arm64_rust_path = out_lib_path + ".rust_arm64"
            x86_cpp_path = out_lib_path + ".cpp_x86_64"
            try:
                os.replace(out_lib_path, arm64_rust_path)
                subprocess.run(
                    [
                        "clang++",
                        "-O3",
                        "-std=c++17",
                        "-dynamiclib",
                        "-arch",
                        "x86_64",
                        "-mmacosx-version-min=11.0",
                        *sdk_args,
                        cpp_src,
                        "-o",
                        x86_cpp_path,
                    ],
                    cwd=crate_dir,
                    env=env,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=60,
                )
                subprocess.run(
                    ["lipo", "-create", arm64_rust_path, x86_cpp_path, "-output", out_lib_path],
                    cwd=crate_dir,
                    env=env,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
                return os.path.exists(out_lib_path)
            except Exception:
                pass
        cmd = [
            "clang++",
            "-O3",
            "-std=c++17",
            "-dynamiclib",
            "-arch",
            "x86_64",
            "-arch",
            "arm64",
            "-mmacosx-version-min=11.0",
            *sdk_args,
            cpp_src,
            "-o",
            out_lib_path,
        ]
    else:
        cmd = ["g++", "-O3", "-std=c++17", "-shared", "-fPIC", cpp_src, "-o", out_lib_path]

    try:
        subprocess.run(
            cmd,
            cwd=crate_dir,
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
        return os.path.exists(out_lib_path)
    except Exception:
        return False


def _find_or_build_rust_lib():
    """Locate or compile the Rust/C++ native shared library for the current OS and Python ABI."""
    crate_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "NeedleFinderRust")
    )
    if not os.path.isdir(crate_dir):
        alt_crate_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "Resources", "NeedleFinderRust")
        )
        if os.path.isdir(alt_crate_dir):
            crate_dir = alt_crate_dir
    lib_name = _shared_lib_filename()
    release_dir = os.path.join(crate_dir, "target", "release")
    lib_path = os.path.join(release_dir, lib_name)
    rs_src = os.path.join(crate_dir, "src", "lib.rs")
    cpp_src = os.path.join(crate_dir, "src", "needlefinder_core.cpp")

    env = os.environ.copy()
    extra_paths = [
        os.path.expanduser("~/.cargo/bin"),
        "/opt/homebrew/bin",
        "/usr/local/bin",
    ]
    env["PATH"] = os.pathsep.join(extra_paths + [env.get("PATH", "")])
    if platform.system().lower() == "darwin" and "SDKROOT" not in env:
        sdk = _select_macos_sdk()
        if sdk:
            env["SDKROOT"] = sdk

    needs_build = not os.path.exists(lib_path)
    if not needs_build:
        try:
            lib_mtime = os.path.getmtime(lib_path)
            if (os.path.exists(rs_src) and os.path.getmtime(rs_src) > lib_mtime) or (
                os.path.exists(cpp_src) and os.path.getmtime(cpp_src) > lib_mtime
            ):
                needs_build = True
        except OSError:
            pass

    if not needs_build:
        try:
            lib = ctypes.CDLL(lib_path)
            _bind_signatures(lib)
            return lib
        except Exception:
            needs_build = True

    if needs_build and os.path.exists(os.path.join(crate_dir, "Cargo.toml")):
        try:
            subprocess.run(
                ["cargo", "build", "--release"],
                cwd=crate_dir,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
            )
        except Exception:
            pass

        if os.path.exists(lib_path):
            try:
                lib = ctypes.CDLL(lib_path)
                _bind_signatures(lib)
                return lib
            except Exception:
                # Architecture mismatch (e.g. Rosetta x86_64 Slicer with arm64 cargo dylib)
                pass

    # Build universal / native C++17 fallback if cargo wasn't available or target arch differed
    if _build_cpp_fallback(crate_dir, lib_path, env):
        try:
            lib = ctypes.CDLL(lib_path)
            _bind_signatures(lib)
            return lib
        except Exception as exc:
            print(f"[NeedleFinder] Could not load native acceleration library: {exc}")

    return None


def _bind_signatures(lib: ctypes.CDLL) -> None:
    lib.nf_search_cone_15_1.argtypes = [
        _DOUBLE_PTR,        # vol_ptr
        _DOUBLE_PTR,        # label_ptr
        ctypes.c_ssize_t,   # dim_x
        ctypes.c_ssize_t,   # dim_y
        ctypes.c_ssize_t,   # dim_z
        _DOUBLE_PTR,        # ijk_a_ptr
        _DOUBLE_PTR,        # ijk_b_ptr
        _DOUBLE_PTR,        # ijk_a_prev_ptr
        ctypes.c_double,    # r_max
        ctypes.c_double,    # n_r_iter
        ctypes.c_int32,     # n_rotating_its
        ctypes.c_double,    # n_t_iter
        ctypes.c_int32,     # b_gradient
        ctypes.c_ssize_t,   # radius_needle
        ctypes.c_ssize_t,   # radius_needle_corner
        ctypes.c_double,    # gradient_ponderation
        ctypes.c_int32,     # b_gaussian_attenuation
        ctypes.c_double,    # sigma_value
        _DOUBLE_PTR,        # ijk_to_ras_ptr
        _DOUBLE_PTR,        # related_pts_ptr
        _SIZE_PTR,          # related_offsets_ptr
        ctypes.c_size_t,    # num_related
        ctypes.c_double,    # radius_needle_mm
        ctypes.c_double,    # punish
        _DOUBLE_PTR,        # out_c_best_ptr
        _DOUBLE_PTR,        # out_min_total_ptr
        _DOUBLE_PTR,        # out_last_total_ptr
    ]
    lib.nf_search_cone_15_1.restype = ctypes.c_int32

    lib.nf_search_cone_13.argtypes = [
        _DOUBLE_PTR,        # vol_ptr
        _DOUBLE_PTR,        # label_ptr
        ctypes.c_ssize_t,   # dim_x
        ctypes.c_ssize_t,   # dim_y
        ctypes.c_ssize_t,   # dim_z
        _DOUBLE_PTR,        # ijk_a_ptr
        _DOUBLE_PTR,        # ijk_c0_ptr
        _DOUBLE_PTR,        # ijk_tip0_ptr
        ctypes.c_double,    # r_max
        ctypes.c_double,    # r_iter
        ctypes.c_int32,     # nb_rotating_step
        ctypes.c_double,    # t_iter
        ctypes.c_int32,     # b_gradient
        ctypes.c_ssize_t,   # radius_needle
        ctypes.c_ssize_t,   # radius_needle_corner
        ctypes.c_double,    # gradient_ponderation
        ctypes.c_int32,     # b_gaussian_attenuation
        ctypes.c_double,    # sigma_value
        _DOUBLE_PTR,        # ijk_to_ras_ptr
        _DOUBLE_PTR,        # related_pts_ptr
        _SIZE_PTR,          # related_offsets_ptr
        ctypes.c_size_t,    # num_related
        ctypes.c_double,    # radius_needle_mm
        ctypes.c_double,    # punish
        _DOUBLE_PTR,        # in_out_best_point_ptr
        _DOUBLE_PTR,        # out_min_estimator_ptr
    ]
    lib.nf_search_cone_13.restype = ctypes.c_int32

    lib.nf_search_cone_current_dev.argtypes = [
        _DOUBLE_PTR,        # vol_ptr
        _DOUBLE_PTR,        # label_ptr
        ctypes.c_ssize_t,   # dim_x
        ctypes.c_ssize_t,   # dim_y
        ctypes.c_ssize_t,   # dim_z
        _DOUBLE_PTR,        # ijk_a_ptr
        _DOUBLE_PTR,        # ijk_c0_ptr
        ctypes.c_double,    # r_max
        ctypes.c_int32,     # r_iter
        ctypes.c_int32,     # nb_rotating_step
        ctypes.c_int32,     # t_iter
        ctypes.c_double,    # exponent
        ctypes.c_int32,     # b_gradient
        ctypes.c_int32,     # mode_flag
        ctypes.c_ssize_t,   # radius_needle
        ctypes.c_ssize_t,   # radius_needle_corner
        ctypes.c_double,    # gradient_ponderation
        ctypes.c_int32,     # b_gaussian_attenuation
        ctypes.c_int32,     # vz_non_zero
        ctypes.c_double,    # sigma_value
        _DOUBLE_PTR,        # in_out_best_point_ptr
        _DOUBLE_PTR,        # out_min_estimator_ptr
    ]
    lib.nf_search_cone_current_dev.restype = ctypes.c_int32

    lib.nf_find_tip.argtypes = [
        _DOUBLE_PTR,        # vol_ptr
        ctypes.c_ssize_t,   # dim_x
        ctypes.c_ssize_t,   # dim_y
        ctypes.c_ssize_t,   # dim_z
        ctypes.c_ssize_t,   # a_x
        ctypes.c_ssize_t,   # a_y
        ctypes.c_ssize_t,   # a_z
        ctypes.c_ssize_t,   # radius_needle
        ctypes.c_double,    # coeff
        ctypes.c_double,    # sigma_value
        ctypes.c_double,    # gradient_ponderation
        ctypes.c_ssize_t,   # search_x
        ctypes.c_ssize_t,   # search_y
        _SSIZE_PTR,         # out_best_ijk_ptr
    ]
    lib.nf_find_tip.restype = ctypes.c_int32

    lib.nf_bezier_curve.argtypes = [
        _DOUBLE_PTR,        # ctrl_pts_ptr
        ctypes.c_size_t,    # num_ctrl_pts
        ctypes.c_size_t,    # num_output_pts
        _DOUBLE_PTR,        # out_curve_ptr
    ]
    lib.nf_bezier_curve.restype = ctypes.c_int32

    lib.nf_center_of_mass_ijk.argtypes = [
        _DOUBLE_PTR,        # vol_ptr
        ctypes.c_size_t,    # dim_x
        ctypes.c_size_t,    # dim_y
        ctypes.c_size_t,    # dim_z
        _DOUBLE_PTR,        # out_com_ptr
    ]
    lib.nf_center_of_mass_ijk.restype = ctypes.c_int32

    lib.nf_compute_3d_sym_response.argtypes = [
        _DOUBLE_PTR,        # vol_ptr
        ctypes.c_size_t,    # dim_x
        ctypes.c_size_t,    # dim_y
        ctypes.c_size_t,    # dim_z
        ctypes.c_double,    # sp_x
        ctypes.c_double,    # sp_y
        ctypes.c_size_t,    # i0
        ctypes.c_size_t,    # i1
        ctypes.c_size_t,    # j0
        ctypes.c_size_t,    # j1
        ctypes.c_size_t,    # k0
        ctypes.c_size_t,    # k1
        ctypes.c_double,    # body_thresh
        _FLOAT_PTR,         # out_resp_ptr
        _FLOAT_PTR,         # out_csmooth_ptr
        _FLOAT_PTR,         # out_ringsym_ptr
    ]
    lib.nf_compute_3d_sym_response.restype = ctypes.c_int32

    if hasattr(lib, "nf_find_2d_nms_peaks"):
        lib.nf_find_2d_nms_peaks.argtypes = [
            _FLOAT_PTR,         # slice_ptr
            ctypes.c_size_t,    # dim_x
            ctypes.c_size_t,    # dim_y
            ctypes.c_size_t,    # nms_px
            ctypes.c_float,     # thresh
            _INT_PTR,           # out_i
            _INT_PTR,           # out_j
            _FLOAT_PTR,         # out_val
            ctypes.c_size_t,    # max_out
            _SIZE_PTR,          # out_count
        ]
        lib.nf_find_2d_nms_peaks.restype = ctypes.c_int32

    if hasattr(lib, "nf_trace_seed_3d"):
        lib.nf_trace_seed_3d.argtypes = [
            _FLOAT_PTR,         # resp_sub
            _FLOAT_PTR,         # c_smooth
            _FLOAT_PTR,         # ring_sym
            ctypes.c_size_t,    # dim_x
            ctypes.c_size_t,    # dim_y
            ctypes.c_size_t,    # dim_z
            ctypes.c_size_t,    # k_start
            ctypes.c_double,    # i_start
            ctypes.c_double,    # j_start
            ctypes.c_double,    # s_val
            ctypes.c_double,    # cand_thresh
            ctypes.c_double,    # max_step_px
            ctypes.c_size_t,    # search_r_px
            ctypes.c_size_t,    # max_gap_slices
            ctypes.c_size_t,    # min_pts_slices
            ctypes.c_int32,     # pass_idx
            ctypes.c_int32,     # has_prior_slope
            ctypes.c_double,    # prior_slope_i
            ctypes.c_double,    # prior_slope_j
            _DOUBLE_PTR,        # gauss_base
            _INT_PTR,           # out_ks
            _DOUBLE_PTR,        # out_i
            _DOUBLE_PTR,        # out_j
            _DOUBLE_PTR,        # out_r
            _SIZE_PTR,          # out_n
            _DOUBLE_PTR,        # out_shaft_p75
        ]
        lib.nf_trace_seed_3d.restype = ctypes.c_int32

    if hasattr(lib, "nf_trim_trajectory_inside_out"):
        lib.nf_trim_trajectory_inside_out.argtypes = [
            _DOUBLE_PTR,        # ras_pts (n, 3)
            _DOUBLE_PTR,        # arr_pts (n, 4)
            ctypes.c_size_t,    # n
            ctypes.c_double,    # shaft_p75
            ctypes.c_int32,     # check_shallow
            _SIZE_PTR,          # out_start_idx
            _SIZE_PTR,          # out_end_idx
        ]
        lib.nf_trim_trajectory_inside_out.restype = ctypes.c_int32


def get_rust_lib():
    """Return the loaded `ctypes.CDLL` instance or `None` if unavailable."""
    global _LIB, _LIB_CHECKED
    if not _LIB_CHECKED:
        _LIB_CHECKED = True
        _LIB = _find_or_build_rust_lib()
    return _LIB


def is_rust_available() -> bool:
    """Return `True` if the native Rust library is loaded and ready."""
    return get_rust_lib() is not None


def get_volume_f64_buffer(image_data):
    """Return `(arr_f64, (dim_x, dim_y, dim_z))` cached by VTK scalar `MTime`."""
    if image_data is None:
        return None, (0, 0, 0)
    dims = image_data.GetDimensions()
    scalars = image_data.GetPointData().GetScalars() if image_data.GetPointData() else None
    if scalars is None:
        return None, dims
    mtime = (image_data.GetMTime(), scalars.GetMTime())
    cache = getattr(image_data, "_nf_f64_cache", None)
    if cache is not None and cache[0] == mtime:
        return cache[1], dims

    raw = vtk.util.numpy_support.vtk_to_numpy(scalars)
    num_comp = scalars.GetNumberOfComponents()
    if num_comp > 1:
        raw = raw.reshape(dims[2], dims[1], dims[0], num_comp)[..., 0]
    else:
        raw = raw.reshape(dims[2], dims[1], dims[0])
    arr_f64 = np.ascontiguousarray(raw, dtype=np.float64)
    try:
        image_data._nf_f64_cache = (mtime, arr_f64)
    except AttributeError:
        pass
    return arr_f64, dims


def _pack_feedback_args(whether_feedback, control_points_package, related_indices, ijk_to_ras_mat):
    """Pack related needle trajectories into flat contiguous arrays for Rust."""
    if not whether_feedback or not related_indices or control_points_package is None or ijk_to_ras_mat is None:
        return (
            ctypes.cast(None, _DOUBLE_PTR),
            ctypes.cast(None, _DOUBLE_PTR),
            ctypes.cast(None, _SIZE_PTR),
            0,
            None,
        )
    pts_list = []
    offsets = [0]
    for idx in related_indices:
        needle_pts = control_points_package[idx]
        for pt in needle_pts:
            pts_list.append((float(pt[0]), float(pt[1]), float(pt[2])))
        offsets.append(len(pts_list))
    if not pts_list:
        return (
            ctypes.cast(None, _DOUBLE_PTR),
            ctypes.cast(None, _DOUBLE_PTR),
            ctypes.cast(None, _SIZE_PTR),
            0,
            None,
        )
    mat_arr = np.ascontiguousarray(ijk_to_ras_mat, dtype=np.float64)
    pts_arr = np.ascontiguousarray(pts_list, dtype=np.float64)
    offs_arr = np.ascontiguousarray(offsets, dtype=np.uintp)
    keep_alive = (mat_arr, pts_arr, offs_arr)
    return (
        mat_arr.ctypes.data_as(_DOUBLE_PTR),
        pts_arr.ctypes.data_as(_DOUBLE_PTR),
        offs_arr.ctypes.data_as(_SIZE_PTR),
        len(related_indices),
        keep_alive,
    )


def _get_nonempty_label_buffer(img_label_data):
    """Return `f64` buffer for `img_label_data` only if it contains non-zero labels."""
    if img_label_data is None:
        return None
    scalars = img_label_data.GetPointData().GetScalars() if img_label_data.GetPointData() else None
    if scalars is None:
        return None
    r_min, r_max = scalars.GetRange()
    if r_min == 0.0 and r_max == 0.0:
        return None
    lbl_arr, _ = get_volume_f64_buffer(img_label_data)
    return lbl_arr


def rust_search_cone_15_1(
    img_data,
    img_label_data,
    ijk_a,
    ijk_b,
    ijk_a_prev,
    r_max: float,
    n_r_iter: float,
    n_rotating_its: int,
    n_t_iter: float,
    b_gradient: int,
    radius_needle: int,
    radius_needle_corner: int,
    gradient_ponderation: float,
    b_gaussian_attenuation: int,
    sigma_value: float,
    whether_feedback: bool = False,
    control_points_package=None,
    related_indices=None,
    ijk_to_ras_mat=None,
    radius_needle_mm: float = 0.0,
    punish: float = 0.0,
):
    """Execute the 3D cone search for `needleDetectionThread15_1` in Rust."""
    lib = get_rust_lib()
    if lib is None:
        return None
    vol_arr, dims = get_volume_f64_buffer(img_data)
    if vol_arr is None:
        return None
    lbl_arr = _get_nonempty_label_buffer(img_label_data)

    a_arr = np.ascontiguousarray([ijk_a[0], ijk_a[1], ijk_a[2]], dtype=np.float64)
    b_arr = np.ascontiguousarray([ijk_b[0], ijk_b[1], ijk_b[2]], dtype=np.float64)
    prev_arr = (
        np.ascontiguousarray([ijk_a_prev[0], ijk_a_prev[1], ijk_a_prev[2]], dtype=np.float64)
        if ijk_a_prev is not None
        else None
    )
    out_c_best = np.full(3, np.nan, dtype=np.float64)
    out_min_total = ctypes.c_double(float("inf"))
    out_last_total = ctypes.c_double(0.0)

    mat_ptr, pts_ptr, offs_ptr, num_rel, _keep = _pack_feedback_args(
        whether_feedback, control_points_package, related_indices, ijk_to_ras_mat
    )

    rc = lib.nf_search_cone_15_1(
        vol_arr.ctypes.data_as(_DOUBLE_PTR),
        lbl_arr.ctypes.data_as(_DOUBLE_PTR) if lbl_arr is not None else ctypes.cast(None, _DOUBLE_PTR),
        int(dims[0]),
        int(dims[1]),
        int(dims[2]),
        a_arr.ctypes.data_as(_DOUBLE_PTR),
        b_arr.ctypes.data_as(_DOUBLE_PTR),
        prev_arr.ctypes.data_as(_DOUBLE_PTR) if prev_arr is not None else ctypes.cast(None, _DOUBLE_PTR),
        float(r_max),
        float(n_r_iter),
        int(n_rotating_its),
        float(n_t_iter),
        int(b_gradient),
        int(radius_needle),
        int(radius_needle_corner),
        float(gradient_ponderation),
        int(b_gaussian_attenuation),
        float(sigma_value),
        mat_ptr,
        pts_ptr,
        offs_ptr,
        int(num_rel),
        float(radius_needle_mm),
        float(punish if punish is not None else 0.0),
        out_c_best.ctypes.data_as(_DOUBLE_PTR),
        ctypes.byref(out_min_total),
        ctypes.byref(out_last_total),
    )
    if rc != 0:
        return None
    return [float(out_c_best[0]), float(out_c_best[1]), float(out_c_best[2])], float(out_min_total.value), float(out_last_total.value)


def rust_search_cone_13(
    img_data,
    img_label_data,
    ijk_a,
    ijk_c0,
    ijk_tip0,
    r_max: float,
    r_iter: float,
    nb_rotating_step: int,
    t_iter: float,
    b_gradient: int,
    radius_needle: int,
    radius_needle_corner: int,
    gradient_ponderation: float,
    b_gaussian_attenuation: int,
    sigma_value: float,
    best_point_init,
    whether_feedback: bool = False,
    control_points_package=None,
    related_indices=None,
    ijk_to_ras_mat=None,
    radius_needle_mm: float = 0.0,
    punish: float = 0.0,
):
    """Execute the 3D cone search for `needleDetectionThread13_1` / `13_2` in Rust."""
    lib = get_rust_lib()
    if lib is None:
        return None
    vol_arr, dims = get_volume_f64_buffer(img_data)
    if vol_arr is None:
        return None
    lbl_arr = _get_nonempty_label_buffer(img_label_data)

    a_arr = np.ascontiguousarray([ijk_a[0], ijk_a[1], ijk_a[2]], dtype=np.float64)
    c0_arr = np.ascontiguousarray([ijk_c0[0], ijk_c0[1], ijk_c0[2]], dtype=np.float64)
    tip0_arr = (
        np.ascontiguousarray([ijk_tip0[0], ijk_tip0[1], ijk_tip0[2]], dtype=np.float64)
        if ijk_tip0 is not None
        else None
    )
    best_pt_arr = np.ascontiguousarray(best_point_init, dtype=np.float64)
    out_min_est = ctypes.c_double(0.0)

    mat_ptr, pts_ptr, offs_ptr, num_rel, _keep = _pack_feedback_args(
        whether_feedback, control_points_package, related_indices, ijk_to_ras_mat
    )

    rc = lib.nf_search_cone_13(
        vol_arr.ctypes.data_as(_DOUBLE_PTR),
        lbl_arr.ctypes.data_as(_DOUBLE_PTR) if lbl_arr is not None else ctypes.cast(None, _DOUBLE_PTR),
        int(dims[0]),
        int(dims[1]),
        int(dims[2]),
        a_arr.ctypes.data_as(_DOUBLE_PTR),
        c0_arr.ctypes.data_as(_DOUBLE_PTR),
        tip0_arr.ctypes.data_as(_DOUBLE_PTR) if tip0_arr is not None else ctypes.cast(None, _DOUBLE_PTR),
        float(r_max),
        float(r_iter),
        int(nb_rotating_step),
        float(t_iter),
        int(b_gradient),
        int(radius_needle),
        int(radius_needle_corner),
        float(gradient_ponderation),
        int(b_gaussian_attenuation),
        float(sigma_value),
        mat_ptr,
        pts_ptr,
        offs_ptr,
        int(num_rel),
        float(radius_needle_mm),
        float(punish if punish is not None else 0.0),
        best_pt_arr.ctypes.data_as(_DOUBLE_PTR),
        ctypes.byref(out_min_est),
    )
    if rc != 0:
        return None
    return [float(best_pt_arr[0]), float(best_pt_arr[1]), float(best_pt_arr[2])], float(out_min_est.value)


def rust_search_cone_current_dev(
    img_data,
    img_label_data,
    ijk_a,
    ijk_c0,
    r_max: float,
    r_iter: int,
    nb_rotating_step: int,
    t_iter: int,
    exponent: float,
    b_gradient: int,
    mode: str,
    radius_needle: int,
    radius_needle_corner: int,
    gradient_ponderation: float,
    b_gaussian_attenuation: int,
    vz_non_zero: bool,
    sigma_value: float,
    best_point_init,
):
    """Execute the 3D cone search for `needleDetectionThreadCurrentDev` in Rust."""
    lib = get_rust_lib()
    if lib is None:
        return None
    vol_arr, dims = get_volume_f64_buffer(img_data)
    if vol_arr is None:
        return None
    lbl_arr, _ = get_volume_f64_buffer(img_label_data) if img_label_data is not None else (None, None)

    a_arr = np.ascontiguousarray([ijk_a[0], ijk_a[1], ijk_a[2]], dtype=np.float64)
    c0_arr = np.ascontiguousarray([ijk_c0[0], ijk_c0[1], ijk_c0[2]], dtype=np.float64)
    best_pt_arr = np.ascontiguousarray(best_point_init, dtype=np.float64)
    out_min_est = ctypes.c_double(0.0)
    mode_flag = 0 if mode == "circle" else (1 if mode == "square" else -1)

    rc = lib.nf_search_cone_current_dev(
        vol_arr.ctypes.data_as(_DOUBLE_PTR),
        lbl_arr.ctypes.data_as(_DOUBLE_PTR) if lbl_arr is not None else ctypes.cast(None, _DOUBLE_PTR),
        int(dims[0]),
        int(dims[1]),
        int(dims[2]),
        a_arr.ctypes.data_as(_DOUBLE_PTR),
        c0_arr.ctypes.data_as(_DOUBLE_PTR),
        float(r_max),
        int(r_iter),
        int(nb_rotating_step),
        int(t_iter),
        float(exponent),
        int(b_gradient),
        int(mode_flag),
        int(radius_needle),
        int(radius_needle_corner),
        float(gradient_ponderation),
        int(b_gaussian_attenuation),
        1 if vz_non_zero else 0,
        float(sigma_value),
        best_pt_arr.ctypes.data_as(_DOUBLE_PTR),
        ctypes.byref(out_min_est),
    )
    if rc != 0:
        return None
    return [float(best_pt_arr[0]), float(best_pt_arr[1]), float(best_pt_arr[2])], float(out_min_est.value)


def rust_find_tip(
    img_data,
    a_ijk,
    radius_needle: int,
    coeff: float,
    sigma_value: float,
    gradient_ponderation: float,
    search_x: int,
    search_y: int,
):
    """Execute `findTip` neighborhood search in Rust."""
    lib = get_rust_lib()
    if lib is None:
        return None
    vol_arr, dims = get_volume_f64_buffer(img_data)
    if vol_arr is None:
        return None
    out_best = (ctypes.c_ssize_t * 3)()
    rc = lib.nf_find_tip(
        vol_arr.ctypes.data_as(_DOUBLE_PTR),
        int(dims[0]),
        int(dims[1]),
        int(dims[2]),
        int(a_ijk[0]),
        int(a_ijk[1]),
        int(a_ijk[2]),
        int(radius_needle),
        float(coeff),
        float(sigma_value),
        float(gradient_ponderation),
        int(search_x),
        int(search_y),
        out_best,
    )
    if rc != 0:
        return None
    return [int(out_best[0]), int(out_best[1]), int(out_best[2])]


def rust_bezier_curve(control_points, num_eval_points: int = 50):
    """Evaluate Bernstein Bezier curve in Rust (`num_eval_points + 1` points)."""
    lib = get_rust_lib()
    if lib is None or len(control_points) == 0 or num_eval_points <= 0:
        return None
    ctrl_arr = np.ascontiguousarray(control_points, dtype=np.float64)
    out_arr = np.empty((num_eval_points + 1, 3), dtype=np.float64)
    rc = lib.nf_bezier_curve(
        ctrl_arr.ctypes.data_as(_DOUBLE_PTR),
        int(len(control_points)),
        int(num_eval_points),
        out_arr.ctypes.data_as(_DOUBLE_PTR),
    )
    if rc != 0:
        return None
    return out_arr.tolist()


def rust_center_of_mass_ijk(image_data):
    """Compute 3D center of mass in IJK coordinates in Rust."""
    lib = get_rust_lib()
    if lib is None:
        return None
    vol_arr, dims = get_volume_f64_buffer(image_data)
    if vol_arr is None:
        return None
    out_com = (ctypes.c_double * 3)()
    rc = lib.nf_center_of_mass_ijk(
        vol_arr.ctypes.data_as(_DOUBLE_PTR),
        int(dims[0]),
        int(dims[1]),
        int(dims[2]),
        out_com,
    )
    if rc != 0:
        return None
    return [float(out_com[0]), float(out_com[1]), float(out_com[2])]


_SYM_BUF_CACHE = {}


def rust_compute_3d_sym_response(
    vol_arr: np.ndarray,
    sp_x: float,
    sp_y: float,
    i0: int,
    i1: int,
    j0: int,
    j1: int,
    k0: int,
    k1: int,
    body_thresh: float,
):
    """Compute 3D symmetric diameter-pair dark-tube response `(resp, csmooth, ringsym)` in Rust/C++."""
    lib = get_rust_lib()
    if lib is None or vol_arr is None or vol_arr.ndim != 3:
        return None
    if vol_arr.dtype == np.float64 and vol_arr.flags.c_contiguous:
        vol_f64 = vol_arr
    else:
        vol_f64 = np.ascontiguousarray(vol_arr, dtype=np.float64)
    dim_z, dim_y, dim_x = vol_f64.shape
    k_lo = max(0, int(k0))
    k_hi = min(dim_z, int(k1))
    shape_key = (dim_z, dim_y, dim_x)
    bufs = _SYM_BUF_CACHE.get(shape_key)
    if bufs is None:
        out_resp = np.zeros(shape_key, dtype=np.float32)
        out_cs = np.zeros(shape_key, dtype=np.float32)
        out_rsym = np.zeros(shape_key, dtype=np.float32)
        _SYM_BUF_CACHE.clear()
        _SYM_BUF_CACHE[shape_key] = (out_resp, out_cs, out_rsym)
    else:
        out_resp, out_cs, out_rsym = bufs
    rc = lib.nf_compute_3d_sym_response(
        vol_f64.ctypes.data_as(_DOUBLE_PTR),
        int(dim_x),
        int(dim_y),
        int(dim_z),
        float(sp_x),
        float(sp_y),
        int(max(0, i0)),
        int(min(dim_x, i1)),
        int(max(0, j0)),
        int(min(dim_y, j1)),
        k_lo,
        k_hi,
        float(body_thresh),
        out_resp.ctypes.data_as(_FLOAT_PTR),
        out_cs.ctypes.data_as(_FLOAT_PTR),
        out_rsym.ctypes.data_as(_FLOAT_PTR),
    )
    if rc != 0:
        return None
    return out_resp, out_cs, out_rsym


_NMS_MAX_OUT = 16384
_NMS_OUT_I = np.empty(_NMS_MAX_OUT, dtype=np.int32)
_NMS_OUT_J = np.empty(_NMS_MAX_OUT, dtype=np.int32)
_NMS_OUT_VAL = np.empty(_NMS_MAX_OUT, dtype=np.float32)
_NMS_I_PTR = _NMS_OUT_I.ctypes.data_as(_INT_PTR)
_NMS_J_PTR = _NMS_OUT_J.ctypes.data_as(_INT_PTR)
_NMS_VAL_PTR = _NMS_OUT_VAL.ctypes.data_as(_FLOAT_PTR)

_TRACE_MAX_Z = 512
_TRACE_OUT_KS = np.empty(_TRACE_MAX_Z, dtype=np.int32)
_TRACE_OUT_I = np.empty(_TRACE_MAX_Z, dtype=np.float64)
_TRACE_OUT_J = np.empty(_TRACE_MAX_Z, dtype=np.float64)
_TRACE_OUT_R = np.empty(_TRACE_MAX_Z, dtype=np.float64)
_TRACE_KS_PTR = _TRACE_OUT_KS.ctypes.data_as(_INT_PTR)
_TRACE_I_PTR = _TRACE_OUT_I.ctypes.data_as(_DOUBLE_PTR)
_TRACE_J_PTR = _TRACE_OUT_J.ctypes.data_as(_DOUBLE_PTR)
_TRACE_R_PTR = _TRACE_OUT_R.ctypes.data_as(_DOUBLE_PTR)


def rust_find_2d_nms_peaks(slice_arr: np.ndarray, nms_px: int, thresh: float):
    """Find 2D local maxima >= `thresh` within a `(2*nms_px+1)^2` window in Rust/C++."""
    lib = get_rust_lib()
    if lib is None or not hasattr(lib, "nf_find_2d_nms_peaks"):
        return None
    if slice_arr.dtype == np.float32 and slice_arr.flags.c_contiguous:
        sl = slice_arr
    else:
        sl = np.ascontiguousarray(slice_arr, dtype=np.float32)
    dim_y, dim_x = sl.shape
    out_count = ctypes.c_size_t(0)
    rc = lib.nf_find_2d_nms_peaks(
        sl.ctypes.data_as(_FLOAT_PTR),
        int(dim_x),
        int(dim_y),
        int(nms_px),
        float(thresh),
        _NMS_I_PTR,
        _NMS_J_PTR,
        _NMS_VAL_PTR,
        _NMS_MAX_OUT,
        ctypes.byref(out_count),
    )
    if rc != 0:
        return None
    n = int(out_count.value)
    return _NMS_OUT_J[:n], _NMS_OUT_I[:n], _NMS_OUT_VAL[:n]


def rust_trace_seed_3d(
    resp_ptr,
    cs_ptr,
    sym_ptr,
    dim_x: int,
    dim_y: int,
    dim_z: int,
    k_start: int,
    i_start: float,
    j_start: float,
    s_val: float,
    cand_thresh: float,
    max_step_px: float,
    search_r_px: int,
    max_gap_slices: int,
    min_pts_slices: int,
    pass_idx: int,
    has_prior_slope: int,
    prior_slope_i: float,
    prior_slope_j: float,
    gauss_ptr,
):
    """Trace a 3D candidate seed bidirectionally along Z in Rust/C++."""
    lib = get_rust_lib()
    if lib is None or not hasattr(lib, "nf_trace_seed_3d") or dim_z > _TRACE_MAX_Z:
        return None
    out_n = ctypes.c_size_t(0)
    out_p75 = ctypes.c_double(0.0)
    rc = lib.nf_trace_seed_3d(
        resp_ptr,
        cs_ptr,
        sym_ptr,
        int(dim_x),
        int(dim_y),
        int(dim_z),
        int(k_start),
        float(i_start),
        float(j_start),
        float(s_val),
        float(cand_thresh),
        float(max_step_px),
        int(search_r_px),
        int(max_gap_slices),
        int(min_pts_slices),
        int(pass_idx),
        int(has_prior_slope),
        float(prior_slope_i),
        float(prior_slope_j),
        gauss_ptr,
        _TRACE_KS_PTR,
        _TRACE_I_PTR,
        _TRACE_J_PTR,
        _TRACE_R_PTR,
        ctypes.byref(out_n),
        ctypes.byref(out_p75),
    )
    if rc != 0:
        return None
    n = int(out_n.value)
    if n == 0:
        return (), (), (), (), 0.0
    return _TRACE_OUT_KS[:n], _TRACE_OUT_I[:n], _TRACE_OUT_J[:n], _TRACE_OUT_R[:n], float(out_p75.value)


def rust_trim_trajectory_inside_out(
    ras_pts: np.ndarray,
    arr_pts: np.ndarray,
    shaft_p75: float,
    check_shallow: int = 0,
):
    """Run inside-out trajectory trimming in Rust/C++, returning `(start_idx, end_idx)`."""
    lib = get_rust_lib()
    if lib is None or not hasattr(lib, "nf_trim_trajectory_inside_out"):
        return None
    if not (ras_pts.dtype == np.float64 and ras_pts.flags.c_contiguous):
        ras_pts = np.ascontiguousarray(ras_pts, dtype=np.float64)
    if not (arr_pts.dtype == np.float64 and arr_pts.flags.c_contiguous):
        arr_pts = np.ascontiguousarray(arr_pts, dtype=np.float64)
    n = len(ras_pts)
    out_s = ctypes.c_size_t(0)
    out_e = ctypes.c_size_t(n)
    rc = lib.nf_trim_trajectory_inside_out(
        ras_pts.ctypes.data_as(_DOUBLE_PTR),
        arr_pts.ctypes.data_as(_DOUBLE_PTR),
        int(n),
        float(shaft_p75),
        int(check_shallow),
        ctypes.byref(out_s),
        ctypes.byref(out_e),
    )
    if rc != 0:
        return None
    return int(out_s.value), int(out_e.value)

