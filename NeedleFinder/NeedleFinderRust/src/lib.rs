//! High-performance Rust native acceleration kernels for 3D Slicer NeedleFinder.
//!
//! Exposes a zero-dependency C-ABI (`#[no_mangle] pub extern "C"`) loaded via Python `ctypes`
//! so that it works across any Python 3.x / 3D Slicer build on Linux, macOS, and Windows
//! without linking against a specific `libpython` or VTK C++ ABI.

use std::slice;
use std::os::raw::{c_double, c_float, c_int};

#[inline(always)]
fn get_scalar(vol: &[f64], dim_x: isize, dim_y: isize, dim_z: isize, ix: isize, iy: isize, iz: isize) -> f64 {
    if ix >= 0 && ix < dim_x && iy >= 0 && iy < dim_y && iz >= 0 && iz < dim_z {
        let idx = ((iz * dim_y + iy) * dim_x + ix) as usize;
        unsafe { *vol.get_unchecked(idx) }
    } else {
        0.0
    }
}

#[inline(always)]
fn compute_feedback(
    ijk_x: f64,
    ijk_y: f64,
    ijk_z: f64,
    ijk_to_ras: &[f64],
    related_pts: &[f64],
    related_offsets: &[usize],
    num_related: usize,
    radius_needle_mm: f64,
    punish: f64,
) -> f64 {
    let ras_x = ijk_to_ras[0] * ijk_x + ijk_to_ras[1] * ijk_y + ijk_to_ras[2] * ijk_z + ijk_to_ras[3];
    let ras_y = ijk_to_ras[4] * ijk_x + ijk_to_ras[5] * ijk_y + ijk_to_ras[6] * ijk_z + ijk_to_ras[7];
    let ras_z = ijk_to_ras[8] * ijk_x + ijk_to_ras[9] * ijk_y + ijk_to_ras[10] * ijk_z + ijk_to_ras[11];

    let mut neighbour_attenuation = 0.0;
    let thresh = radius_needle_mm * punish;

    for w in 0..num_related {
        let start = related_offsets[w];
        let end = related_offsets[w + 1];
        if end <= start {
            continue;
        }
        let mut distance = 9999.0;
        let first_z = related_pts[start * 3 + 2];
        let last_z = related_pts[(end - 1) * 3 + 2];

        if ras_z >= first_z {
            let dz = ras_z - first_z;
            distance = (3.0 * dz * dz).sqrt();
        } else if ras_z <= last_z {
            let dz = ras_z - last_z;
            distance = (3.0 * dz * dz).sqrt();
        } else {
            for q in start..(end - 1) {
                let pax = related_pts[q * 3];
                let pay = related_pts[q * 3 + 1];
                let paz = related_pts[q * 3 + 2];
                let pbx = related_pts[(q + 1) * 3];
                let pby = related_pts[(q + 1) * 3 + 1];
                let pbz = related_pts[(q + 1) * 3 + 2];

                let h = if paz <= ras_z {
                    ((ras_x - pax).powi(2) + (ras_y - pay).powi(2) + (ras_z - paz).powi(2)).sqrt()
                } else if pbz >= ras_z {
                    ((ras_x - pbx).powi(2) + (ras_y - pby).powi(2) + (ras_z - pbz).powi(2)).sqrt()
                } else {
                    let a = ((pax - ras_x).powi(2) + (pay - ras_y).powi(2) + (paz - ras_z).powi(2)).sqrt();
                    let b = ((pbx - ras_x).powi(2) + (pby - ras_y).powi(2) + (pbz - ras_z).powi(2)).sqrt();
                    let c = ((pax - pbx).powi(2) + (pay - pby).powi(2) + (paz - pbz).powi(2)).sqrt();
                    let hc = (a + b + c) / 2.0;
                    (hc * (hc - a) * (hc - b) * (hc - c)).max(0.0).sqrt() * 2.0 / c
                };
                if h < distance {
                    distance = h;
                }
            }
        }

        if distance == 0.0 {
            neighbour_attenuation += 3000.0;
        } else if distance < thresh {
            neighbour_attenuation += 1000.0 * (radius_needle_mm / distance);
        }
    }
    neighbour_attenuation
}

/// 3D Cone ray-integral search for `needleDetectionThread15_1` (MICCAI 2015 algorithm).
///
/// Evaluates `(n_r_iter + 1) * n_rotating_its * (n_t_iter + 1)` points in 3D IJK space
/// with optional 8-neighbor radial gradient, Gaussian deflection attenuation, and
/// multi-needle rationality feedback.
#[no_mangle]
pub unsafe extern "C" fn nf_search_cone_15_1(
    vol_ptr: *const f64,
    label_ptr: *const f64,
    dim_x: isize,
    dim_y: isize,
    dim_z: isize,
    ijk_a_ptr: *const f64,
    ijk_b_ptr: *const f64,
    ijk_a_prev_ptr: *const f64,
    r_max: f64,
    n_r_iter: f64,
    n_rotating_its: i32,
    n_t_iter: f64,
    b_gradient: i32,
    radius_needle: isize,
    radius_needle_corner: isize,
    gradient_ponderation: f64,
    b_gaussian_attenuation: i32,
    sigma_value: f64,
    ijk_to_ras_ptr: *const f64,
    related_pts_ptr: *const f64,
    related_offsets_ptr: *const usize,
    num_related: usize,
    radius_needle_mm: f64,
    punish: f64,
    out_c_best_ptr: *mut f64,
    out_min_total_ptr: *mut f64,
    out_last_total_ptr: *mut f64,
) -> i32 {
    if vol_ptr.is_null() || ijk_a_ptr.is_null() || ijk_b_ptr.is_null() || out_c_best_ptr.is_null() {
        return -1;
    }
    let total_voxels = (dim_x * dim_y * dim_z) as usize;
    let vol = slice::from_raw_parts(vol_ptr, total_voxels);
    let label = if !label_ptr.is_null() {
        Some(slice::from_raw_parts(label_ptr, total_voxels))
    } else {
        None
    };

    let has_feedback = num_related > 0 && !ijk_to_ras_ptr.is_null() && !related_pts_ptr.is_null() && !related_offsets_ptr.is_null();
    let (ijk_to_ras, related_pts, related_offsets) = if has_feedback {
        let offs = slice::from_raw_parts(related_offsets_ptr, num_related + 1);
        let total_pts = offs[num_related];
        (
            slice::from_raw_parts(ijk_to_ras_ptr, 16),
            slice::from_raw_parts(related_pts_ptr, total_pts * 3),
            offs,
        )
    } else {
        (&[][..], &[][..], &[][..])
    };

    let ijk_a = slice::from_raw_parts(ijk_a_ptr, 3);
    let ijk_b = slice::from_raw_parts(ijk_b_ptr, 3);
    let (ax, ay, az) = (ijk_a[0], ijk_a[1], ijk_a[2]);
    let (bx, by, bz) = (ijk_b[0], ijk_b[1], ijk_b[2]);

    let mut has_gauss = false;
    let (mut x_att_x, mut x_att_y, mut x_att_z) = (0.0, 0.0, 0.0);
    let mut gauss_denom = 1.0;

    if b_gaussian_attenuation == 1 && !ijk_a_prev_ptr.is_null() {
        let ijk_a_prev = slice::from_raw_parts(ijk_a_prev_ptr, 3);
        let dz_prev = ijk_a_prev[2] - az;
        if dz_prev != 0.0 && r_max != 0.0 && sigma_value != 0.0 {
            has_gauss = true;
            let f_k_att = (az - bz) / dz_prev;
            x_att_x = ax + f_k_att * (ax - ijk_a_prev[0]);
            x_att_y = ay + f_k_att * (ay - ijk_a_prev[1]);
            x_att_z = az + f_k_att * (az - ijk_a_prev[2]);
            let s = 2.0 * (sigma_value / 10.0);
            gauss_denom = s * s;
        }
    }

    let mut f_min_total = f64::INFINITY;
    let mut c_best = [f64::NAN, f64::NAN, f64::NAN];
    let mut f_last_total = 0.0;

    let n_r_steps = n_r_iter as i32;
    let n_t_steps = n_t_iter as i32;
    let n_rot_f64 = n_rotating_its as f64;

    let mut cos_sin = Vec::with_capacity(n_rotating_its as usize);
    for i_theta in 0..n_rotating_its {
        let f_angle_deg = ((i_theta as f64) * 360.0) / n_rot_f64;
        let f_theta_rad = f_angle_deg.to_radians();
        cos_sin.push((f_theta_rad.cos(), f_theta_rad.sin()));
    }

    for i_r in 0..=n_r_steps {
        let f_r = (i_r as f64) * (r_max / n_r_iter);
        for &(cos_t, sin_t) in &cos_sin {
            let cx = bx + f_r * cos_t;
            let cy = by + f_r * sin_t;
            let cz = bz;

            let mut f_total = 0.0;
            for i_t in 0..=n_t_steps {
                let f_t = (i_t as f64) / n_t_iter;
                let one_minus_t = 1.0 - f_t;
                let mx = one_minus_t * ax + f_t * cx;
                let my = one_minus_t * ay + f_t * cy;
                let mz = one_minus_t * az + f_t * cz;

                let ix = mx.round_ties_even() as isize;
                let iy = my.round_ties_even() as isize;
                let iz = mz.round_ties_even() as isize;

                if ix > 0 && ix < dim_x && iy > 0 && iy < dim_y && iz > 0 && iz < dim_z {
                    let d_center = get_scalar(vol, dim_x, dim_y, dim_z, ix, iy, iz);
                    f_total += d_center;

                    if has_feedback {
                        f_total += compute_feedback(
                            mx,
                            my,
                            mz,
                            ijk_to_ras,
                            related_pts,
                            related_offsets,
                            num_related,
                            radius_needle_mm,
                            punish,
                        );
                    }

                    if b_gradient == 1 {
                        let g1 = get_scalar(vol, dim_x, dim_y, dim_z, ix + radius_needle, iy, iz);
                        let g2 = get_scalar(vol, dim_x, dim_y, dim_z, ix - radius_needle, iy, iz);
                        let g3 = get_scalar(vol, dim_x, dim_y, dim_z, ix, iy + radius_needle, iz);
                        let g4 = get_scalar(vol, dim_x, dim_y, dim_z, ix, iy - radius_needle, iz);
                        let g5 = get_scalar(vol, dim_x, dim_y, dim_z, ix + radius_needle_corner, iy + radius_needle_corner, iz);
                        let g6 = get_scalar(vol, dim_x, dim_y, dim_z, ix - radius_needle_corner, iy - radius_needle_corner, iz);
                        let g7 = get_scalar(vol, dim_x, dim_y, dim_z, ix - radius_needle_corner, iy + radius_needle_corner, iz);
                        let g8 = get_scalar(vol, dim_x, dim_y, dim_z, ix + radius_needle_corner, iy - radius_needle_corner, iz);
                        f_total += 8.0 * d_center - ((g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / 8.0) * gradient_ponderation;
                    }

                    if let Some(lbl) = label {
                        let f_label = get_scalar(lbl, dim_x, dim_y, dim_z, ix, iy, iz);
                        if f_label != 0.0 && f_label < 300.0 {
                            f_total -= 10000.0;
                        }
                    }
                }
            }

            if has_gauss {
                let dx = cx - x_att_x;
                let dy = cy - x_att_y;
                let dz = cz - x_att_z;
                let f_rgauss = (dx * dx + dy * dy + dz * dz).sqrt();
                let ratio = f_rgauss / r_max;
                let f_gauss = (-(ratio * ratio) / gauss_denom).exp();
                f_total *= f_gauss;
            }

            f_last_total = f_total;
            if f_total < f_min_total {
                f_min_total = f_total;
                c_best = [cx, cy, cz];
            }
        }
    }

    let out_c_best = slice::from_raw_parts_mut(out_c_best_ptr, 3);
    out_c_best[0] = c_best[0];
    out_c_best[1] = c_best[1];
    out_c_best[2] = c_best[2];
    if !out_min_total_ptr.is_null() {
        *out_min_total_ptr = f_min_total;
    }
    if !out_last_total_ptr.is_null() {
        *out_last_total_ptr = f_last_total;
    }
    0
}

/// 3D Cone ray-integral search for `needleDetectionThread13_1` and `needleDetectionThread13_2` (MICCAI 2013).
#[no_mangle]
pub unsafe extern "C" fn nf_search_cone_13(
    vol_ptr: *const f64,
    label_ptr: *const f64,
    dim_x: isize,
    dim_y: isize,
    dim_z: isize,
    ijk_a_ptr: *const f64,
    ijk_c0_ptr: *const f64,
    ijk_tip0_ptr: *const f64,
    r_max: f64,
    r_iter: f64,
    nb_rotating_step: i32,
    t_iter: f64,
    b_gradient: i32,
    radius_needle: isize,
    radius_needle_corner: isize,
    gradient_ponderation: f64,
    b_gaussian_attenuation: i32,
    sigma_value: f64,
    ijk_to_ras_ptr: *const f64,
    related_pts_ptr: *const f64,
    related_offsets_ptr: *const usize,
    num_related: usize,
    radius_needle_mm: f64,
    punish: f64,
    in_out_best_point_ptr: *mut f64,
    out_min_estimator_ptr: *mut f64,
) -> i32 {
    if vol_ptr.is_null() || ijk_a_ptr.is_null() || ijk_c0_ptr.is_null() || in_out_best_point_ptr.is_null() {
        return -1;
    }
    let total_voxels = (dim_x * dim_y * dim_z) as usize;
    let vol = slice::from_raw_parts(vol_ptr, total_voxels);
    let label = if !label_ptr.is_null() {
        Some(slice::from_raw_parts(label_ptr, total_voxels))
    } else {
        None
    };

    let has_feedback = num_related > 0 && !ijk_to_ras_ptr.is_null() && !related_pts_ptr.is_null() && !related_offsets_ptr.is_null();
    let (ijk_to_ras, related_pts, related_offsets) = if has_feedback {
        let offs = slice::from_raw_parts(related_offsets_ptr, num_related + 1);
        let total_pts = offs[num_related];
        (
            slice::from_raw_parts(ijk_to_ras_ptr, 16),
            slice::from_raw_parts(related_pts_ptr, total_pts * 3),
            offs,
        )
    } else {
        (&[][..], &[][..], &[][..])
    };

    let a = slice::from_raw_parts(ijk_a_ptr, 3);
    let c0 = slice::from_raw_parts(ijk_c0_ptr, 3);
    let best_point = slice::from_raw_parts_mut(in_out_best_point_ptr, 3);
    let (ax, ay, az) = (a[0], a[1], a[2]);
    let (c0x, c0y, c0z) = (c0[0], c0[1], c0[2]);

    let mut has_gauss = false;
    let (mut x_att_x, mut x_att_y, mut x_att_z) = (0.0, 0.0, 0.0);
    let mut gauss_denom = 1.0;

    if b_gaussian_attenuation == 1 && !ijk_tip0_ptr.is_null() {
        let tip0 = slice::from_raw_parts(ijk_tip0_ptr, 3);
        let dz_prev = tip0[2] - az;
        if dz_prev != 0.0 && r_max != 0.0 && sigma_value != 0.0 {
            has_gauss = true;
            let step_size = az - c0z;
            let k = step_size / dz_prev;
            x_att_x = ax + k * (ax - tip0[0]);
            x_att_y = ay + k * (ay - tip0[1]);
            x_att_z = az + k * (az - tip0[2]);
            let s = 2.0 * (sigma_value / 10.0);
            gauss_denom = s * s;
        }
    }

    let mut initial_intensity = 0.0;
    let mut min_estimator = 0.0;

    let r_steps = r_iter as i32;
    let t_steps = t_iter as i32;
    let nb_rot_f64 = nb_rotating_step as f64;

    let mut cos_sin = Vec::with_capacity(nb_rotating_step as usize);
    for theta_step in 0..nb_rotating_step {
        let angle_deg = ((theta_step as f64) * 360.0) / nb_rot_f64;
        let theta = angle_deg.to_radians();
        cos_sin.push((theta.cos(), theta.sin()));
    }

    for r_idx in 0..=r_steps {
        let r = (r_idx as f64) * (r_max / r_iter);
        for &(cos_t, sin_t) in &cos_sin {
            let cx = c0x + r * cos_t;
            let cy = c0y + r * sin_t;
            let cz = c0z;

            let mut total = 0.0;
            for t in 0..=t_steps {
                let tt = (t as f64) / t_iter;
                let one_minus_tt = 1.0 - tt;
                let mx = one_minus_tt * ax + tt * cx;
                let my = one_minus_tt * ay + tt * cy;
                let mz = one_minus_tt * az + tt * cz;

                let ix = mx.round_ties_even() as isize;
                let iy = my.round_ties_even() as isize;
                let iz = mz.round_ties_even() as isize;

                if ix > 0 && ix < dim_x && iy > 0 && iy < dim_y && iz > 0 && iz < dim_z {
                    let center = get_scalar(vol, dim_x, dim_y, dim_z, ix, iy, iz);
                    total += center;

                    if has_feedback {
                        total += compute_feedback(
                            mx,
                            my,
                            mz,
                            ijk_to_ras,
                            related_pts,
                            related_offsets,
                            num_related,
                            radius_needle_mm,
                            punish,
                        );
                    }

                    if b_gradient == 1 {
                        let g1 = get_scalar(vol, dim_x, dim_y, dim_z, ix + radius_needle, iy, iz);
                        let g2 = get_scalar(vol, dim_x, dim_y, dim_z, ix - radius_needle, iy, iz);
                        let g3 = get_scalar(vol, dim_x, dim_y, dim_z, ix, iy + radius_needle, iz);
                        let g4 = get_scalar(vol, dim_x, dim_y, dim_z, ix, iy - radius_needle, iz);
                        let g5 = get_scalar(vol, dim_x, dim_y, dim_z, ix + radius_needle_corner, iy + radius_needle_corner, iz);
                        let g6 = get_scalar(vol, dim_x, dim_y, dim_z, ix - radius_needle_corner, iy - radius_needle_corner, iz);
                        let g7 = get_scalar(vol, dim_x, dim_y, dim_z, ix - radius_needle_corner, iy + radius_needle_corner, iz);
                        let g8 = get_scalar(vol, dim_x, dim_y, dim_z, ix + radius_needle_corner, iy - radius_needle_corner, iz);
                        total += 8.0 * center - ((g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / 8.0) * gradient_ponderation;
                    }

                    if let Some(lbl) = label {
                        let f_label = get_scalar(lbl, dim_x, dim_y, dim_z, ix, iy, iz);
                        if f_label != 0.0 && f_label < 300.0 {
                            total -= 10000.0;
                        }
                    }
                }
            }

            if r_idx == 0 {
                initial_intensity = total;
            }

            let estimator = if has_gauss {
                let dx = cx - x_att_x;
                let dy = cy - x_att_y;
                let dz = cz - x_att_z;
                let rgauss = (dx * dx + dy * dy + dz * dz).sqrt();
                let ratio = rgauss / r_max;
                let gauss = (-(ratio * ratio) / gauss_denom).exp();
                total * gauss
            } else {
                total
            };

            if estimator < initial_intensity {
                if estimator < min_estimator || min_estimator == 0.0 {
                    min_estimator = estimator;
                    if min_estimator != 0.0 {
                        best_point[0] = cx;
                        best_point[1] = cy;
                        best_point[2] = cz;
                    }
                }
            }
        }
    }

    if !out_min_estimator_ptr.is_null() {
        *out_min_estimator_ptr = min_estimator;
    }
    0
}

/// 3D Cone ray-integral search for `needleDetectionThreadCurrentDev` (Algorithm 0).
/// `mode_flag`: 0 for "circle", 1 for "square".
#[no_mangle]
pub unsafe extern "C" fn nf_search_cone_current_dev(
    vol_ptr: *const f64,
    label_ptr: *const f64,
    dim_x: isize,
    dim_y: isize,
    dim_z: isize,
    ijk_a_ptr: *const f64,
    ijk_c0_ptr: *const f64,
    r_max: f64,
    r_iter: i32,
    nb_rotating_step: i32,
    t_iter: i32,
    exponent: f64,
    b_gradient: i32,
    mode_flag: i32,
    radius_needle: isize,
    radius_needle_corner: isize,
    gradient_ponderation: f64,
    b_gaussian_attenuation: i32,
    vz_non_zero: i32,
    sigma_value: f64,
    in_out_best_point_ptr: *mut f64,
    out_min_estimator_ptr: *mut f64,
) -> i32 {
    if vol_ptr.is_null() || ijk_a_ptr.is_null() || ijk_c0_ptr.is_null() || in_out_best_point_ptr.is_null() {
        return -1;
    }
    let total_voxels = (dim_x * dim_y * dim_z) as usize;
    let vol = slice::from_raw_parts(vol_ptr, total_voxels);
    let label = if !label_ptr.is_null() {
        Some(slice::from_raw_parts(label_ptr, total_voxels))
    } else {
        None
    };

    let a = slice::from_raw_parts(ijk_a_ptr, 3);
    let c0 = slice::from_raw_parts(ijk_c0_ptr, 3);
    let best_point = slice::from_raw_parts_mut(in_out_best_point_ptr, 3);
    let (ax, ay, az) = (a[0], a[1], a[2]);
    let (c0x, c0y, c0z) = (c0[0], c0[1], c0[2]);

    let r_iter_f64 = r_iter as f64;
    let nb_rot_f64 = nb_rotating_step as f64;
    let t_iter_f64 = t_iter as f64;

    let has_gauss = b_gaussian_attenuation == 1 && vz_non_zero != 0 && r_max != 0.0 && sigma_value != 0.0;
    let gauss_denom = if has_gauss {
        let s = 2.0 * (sigma_value / 10.0);
        s * s
    } else {
        1.0
    };

    let mut initial_intensity = 0.0;
    let mut min_estimator = 0.0;

    let mut cos_sin = Vec::with_capacity(nb_rotating_step as usize);
    for theta_step in 0..nb_rotating_step {
        let angle_deg = ((theta_step as f64) * 360.0) / nb_rot_f64;
        let theta = angle_deg.to_radians();
        cos_sin.push((theta.cos(), theta.sin()));
    }

    for r_idx in 0..=r_iter {
        let r = (r_idx as f64) * (r_max / r_iter_f64);
        for &(cos_t, sin_t) in &cos_sin {
            let cx = c0x + r * cos_t;
            let cy = c0y + r * sin_t;
            let cz = c0z;

            let mut total = 0.0;
            for t in 0..=t_iter {
                let tt = (t as f64) / t_iter_f64;
                let one_minus_tt = 1.0 - tt;
                let mx = one_minus_tt * ax + tt * cx;
                let my = one_minus_tt * ay + tt * cy;
                let mz = one_minus_tt * az + tt * cz;

                let ix = mx.round_ties_even() as isize;
                let iy = my.round_ties_even() as isize;
                let iz = mz.round_ties_even() as isize;

                if ix > 0 && ix < dim_x && iy > 0 && iy < dim_y && iz > 0 && iz < dim_z {
                    let center = get_scalar(vol, dim_x, dim_y, dim_z, ix, iy, iz)
                        + get_scalar(vol, dim_x, dim_y, dim_z, ix + 1, iy, iz)
                        + get_scalar(vol, dim_x, dim_y, dim_z, ix - 1, iy, iz)
                        + get_scalar(vol, dim_x, dim_y, dim_z, ix, iy + 1, iz)
                        + get_scalar(vol, dim_x, dim_y, dim_z, ix, iy - 1, iz)
                        + get_scalar(vol, dim_x, dim_y, dim_z, ix + 1, iy + 1, iz)
                        + get_scalar(vol, dim_x, dim_y, dim_z, ix + 1, iy - 1, iz)
                        + get_scalar(vol, dim_x, dim_y, dim_z, ix - 1, iy + 1, iz)
                        + get_scalar(vol, dim_x, dim_y, dim_z, ix - 1, iy - 1, iz);

                    total += center.powf(exponent);

                    if b_gradient == 1 && mode_flag == 0 {
                        let g = [
                            get_scalar(vol, dim_x, dim_y, dim_z, ix + radius_needle, iy, iz),
                            get_scalar(vol, dim_x, dim_y, dim_z, ix - radius_needle, iy, iz),
                            get_scalar(vol, dim_x, dim_y, dim_z, ix, iy + radius_needle, iz),
                            get_scalar(vol, dim_x, dim_y, dim_z, ix, iy - radius_needle, iz),
                            get_scalar(vol, dim_x, dim_y, dim_z, ix + radius_needle_corner, iy + radius_needle_corner, iz),
                            get_scalar(vol, dim_x, dim_y, dim_z, ix - radius_needle_corner, iy - radius_needle_corner, iz),
                            get_scalar(vol, dim_x, dim_y, dim_z, ix - radius_needle_corner, iy + radius_needle_corner, iz),
                            get_scalar(vol, dim_x, dim_y, dim_z, ix + radius_needle_corner, iy - radius_needle_corner, iz),
                        ];
                        for gv in g {
                            total += (center - gv).powf(gradient_ponderation) / t_iter_f64;
                        }
                    } else if b_gradient == 1 && mode_flag == 1 {
                        let g1 = get_scalar(vol, dim_x, dim_y, dim_z, ix + radius_needle, iy, iz);
                        let g2 = get_scalar(vol, dim_x, dim_y, dim_z, ix - radius_needle, iy, iz);
                        let g3 = get_scalar(vol, dim_x, dim_y, dim_z, ix, iy + radius_needle, iz);
                        let g4 = get_scalar(vol, dim_x, dim_y, dim_z, ix, iy - radius_needle, iz);
                        total += (8.0 * center - ((g1 + g2 + g3 + g4) / 4.0) * gradient_ponderation) / t_iter_f64;
                    }

                    if let Some(lbl) = label {
                        let f_label = get_scalar(lbl, dim_x, dim_y, dim_z, ix, iy, iz);
                        if f_label != 0.0 && f_label < 300.0 {
                            total -= 10000.0;
                        }
                    }
                }
            }

            if r_idx == 0 {
                initial_intensity = total;
            }

            let estimator = if has_gauss {
                let dx = cx - c0x;
                let dy = cy - c0y;
                let dz = cz - c0z;
                let rgauss = (dx * dx + dy * dy + dz * dz).sqrt();
                let ratio = rgauss / r_max;
                let gauss = (-(ratio * ratio) / gauss_denom).exp();
                total * gauss
            } else {
                total
            };

            if estimator < initial_intensity {
                if estimator < min_estimator || min_estimator == 0.0 {
                    min_estimator = estimator;
                    if min_estimator != 0.0 {
                        best_point[0] = cx;
                        best_point[1] = cy;
                        best_point[2] = cz;
                    }
                }
            }
        }
    }

    if !out_min_estimator_ptr.is_null() {
        *out_min_estimator_ptr = min_estimator;
    }
    0
}

/// Neighborhood needle-tip refinement (`findTip`).
#[no_mangle]
pub unsafe extern "C" fn nf_find_tip(
    vol_ptr: *const f64,
    dim_x: isize,
    dim_y: isize,
    dim_z: isize,
    a_x: isize,
    a_y: isize,
    a_z: isize,
    radius_needle: isize,
    coeff: f64,
    sigma_value: f64,
    gradient_ponderation: f64,
    search_x: isize,
    search_y: isize,
    out_best_ijk_ptr: *mut isize,
) -> i32 {
    if vol_ptr.is_null() || out_best_ijk_ptr.is_null() {
        return -1;
    }
    let total_voxels = (dim_x * dim_y * dim_z) as usize;
    let vol = slice::from_raw_parts(vol_ptr, total_voxels);
    let mut min_total_tip = 0.0;
    let mut i_best = a_x;
    let mut j_best = a_y;
    let mut k_best = a_z;
    let denom = 2.0 * (sigma_value / 10.0) * (sigma_value / 10.0);

    for cap_i in -search_x..search_x {
        let i = ((cap_i as f64) / coeff) as isize;
        for cap_j in -search_y..search_y {
            let j = ((cap_j as f64) / coeff) as isize;
            let k: isize = 0;
            let mut total_tip = 0.0;
            let ax = a_x + i;
            let ay = a_y + j;
            for l in -3..1 {
                let az = a_z + k + l;
                let v0 = 8.0 * get_scalar(vol, dim_x, dim_y, dim_z, ax, ay, az);
                let v1 = get_scalar(vol, dim_x, dim_y, dim_z, ax + radius_needle, ay, az);
                let v2 = get_scalar(vol, dim_x, dim_y, dim_z, ax - radius_needle, ay, az);
                let v3 = get_scalar(vol, dim_x, dim_y, dim_z, ax, ay + radius_needle, az);
                let v4 = get_scalar(vol, dim_x, dim_y, dim_z, ax, ay - radius_needle, az);
                total_tip += v0 - ((v1 + v2 + v3 + v4) / 4.0) * gradient_ponderation;
            }
            let rgauss = ((i * i + j * j + k * k) as f64).sqrt();
            let gauss = (-((rgauss / 10.0) * (rgauss / 10.0)) / denom).exp();
            total_tip *= gauss;

            if total_tip < min_total_tip || min_total_tip == 0.0 {
                min_total_tip = total_tip;
                i_best = ax;
                j_best = ay;
                k_best = a_z + k;
            }
        }
    }

    let out = slice::from_raw_parts_mut(out_best_ijk_ptr, 3);
    out[0] = i_best;
    out[1] = j_best;
    out[2] = k_best;
    0
}

/// Fast Bernstein polynomial Bezier curve evaluation (`compute_bezier_curve`).
#[no_mangle]
pub unsafe extern "C" fn nf_bezier_curve(
    ctrl_pts_ptr: *const f64,
    num_ctrl_pts: usize,
    num_eval_pts: usize,
    out_curve_ptr: *mut f64,
) -> i32 {
    if ctrl_pts_ptr.is_null() || out_curve_ptr.is_null() || num_ctrl_pts == 0 || num_eval_pts == 0 {
        return -1;
    }
    let total_out = num_eval_pts + 1;
    let ctrl = slice::from_raw_parts(ctrl_pts_ptr, num_ctrl_pts * 3);
    let out = slice::from_raw_parts_mut(out_curve_ptr, total_out * 3);
    let n = num_ctrl_pts - 1;

    let mut coeffs = vec![1.0_f64; num_ctrl_pts];
    for i in 1..=n {
        coeffs[i] = coeffs[i - 1] * ((n - i + 1) as f64) / (i as f64);
    }

    let denom = num_eval_pts as f64;
    for t_idx in 0..total_out {
        let tt = (t_idx as f64) / denom;
        let one_minus_tt = 1.0 - tt;
        let mut x = 0.0;
        let mut y = 0.0;
        let mut z = 0.0;
        for i in 0..=n {
            let w = coeffs[i] * one_minus_tt.powi((n - i) as i32) * tt.powi(i as i32);
            x += w * ctrl[i * 3];
            y += w * ctrl[i * 3 + 1];
            z += w * ctrl[i * 3 + 2];
        }
        out[t_idx * 3] = x;
        out[t_idx * 3 + 1] = y;
        out[t_idx * 3 + 2] = z;
    }
    0
}

/// Fast 3D center of mass in IJK coordinates (`compute_center_of_mass_ijk`).
#[no_mangle]
pub unsafe extern "C" fn nf_center_of_mass_ijk(
    vol_ptr: *const f64,
    dim_x: usize,
    dim_y: usize,
    dim_z: usize,
    out_com_ptr: *mut f64,
) -> i32 {
    if vol_ptr.is_null() || out_com_ptr.is_null() {
        return -1;
    }
    let vol = slice::from_raw_parts(vol_ptr, dim_x * dim_y * dim_z);
    let mut sum_w = 0.0_f64;
    let mut sum_i = 0.0_f64;
    let mut sum_j = 0.0_f64;
    let mut sum_k = 0.0_f64;

    for k in 0..dim_z {
        let kf = k as f64;
        let k_off = k * dim_y;
        for j in 0..dim_y {
            let jf = j as f64;
            let row_off = (k_off + j) * dim_x;
            for i in 0..dim_x {
                let w = *vol.get_unchecked(row_off + i);
                if w != 0.0 {
                    sum_w += w;
                    sum_i += w * (i as f64);
                    sum_j += w * jf;
                    sum_k += w * kf;
                }
            }
        }
    }

    let out = slice::from_raw_parts_mut(out_com_ptr, 3);
    if sum_w == 0.0 {
        out[0] = 0.0;
        out[1] = 0.0;
        out[2] = 0.0;
    } else {
        out[0] = sum_i / sum_w;
        out[1] = sum_j / sum_w;
        out[2] = sum_k / sum_w;
    }
    0
}

/// Multi-threaded 3D symmetric diameter-pair tubularity filter for automatic needle detection.
///
/// Computes per-slice 3x3 Gaussian-smoothed intensity `out_csmooth`, multi-radius (1.7, 2.5, 3.4 mm)
/// 6-diameter symmetric outer ring intensity `out_ringsym`, and dark-tube response `out_resp`.
#[no_mangle]
pub unsafe extern "C" fn nf_compute_3d_sym_response(
    vol_ptr: *const f64,
    dim_x: usize,
    dim_y: usize,
    dim_z: usize,
    sp_x: f64,
    sp_y: f64,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    k0: usize,
    k1: usize,
    body_thresh: f64,
    out_resp_ptr: *mut f32,
    out_csmooth_ptr: *mut f32,
    out_ringsym_ptr: *mut f32,
) -> i32 {
    if vol_ptr.is_null()
        || out_resp_ptr.is_null()
        || out_csmooth_ptr.is_null()
        || out_ringsym_ptr.is_null()
        || dim_x < 25
        || dim_y < 25
        || dim_z == 0
    {
        return -1;
    }
    let k_start = k0.min(dim_z);
    let k_end = k1.min(dim_z);
    if k_end <= k_start {
        return 0;
    }

    let slice_elems = dim_x * dim_y;
    let total_elems = slice_elems * dim_z;
    let vol = slice::from_raw_parts(vol_ptr, total_elems);
    let out_resp = slice::from_raw_parts_mut(out_resp_ptr, total_elems);
    let out_cs = slice::from_raw_parts_mut(out_csmooth_ptr, total_elems);
    let out_rsym = slice::from_raw_parts_mut(out_ringsym_ptr, total_elems);

    let radii = [1.35_f64, 1.85_f64, 2.5_f64, 3.3_f64];
    let mut di_tab = [[0_isize; 6]; 4];
    let mut dj_tab = [[0_isize; 6]; 4];
    for r in 0..4 {
        for a in 0..6 {
            let th = (a as f64) * (std::f64::consts::PI / 6.0);
            di_tab[r][a] = ((radii[r] * th.cos() / sp_x).round() as isize).clamp(-11, 11);
            dj_tab[r][a] = ((radii[r] * th.sin() / sp_y).round() as isize).clamp(-11, 11);
        }
    }

    let margin = 12_usize;
    let ja_s = j0.saturating_sub(margin).max(1);
    let jb_s = (j1 + margin).min(dim_y - 1);
    let ia_s = i0.saturating_sub(margin).max(1);
    let ib_s = (i1 + margin).min(dim_x - 1);

    let ja = j0.max(margin);
    let jb = j1.min(dim_y - margin);
    let ia = i0.max(margin);
    let ib = i1.min(dim_x - margin);
    let min_tissue = (body_thresh as f32 * 0.45_f32).max(12.0_f32);

    let num_slices = k_end - k_start;
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(num_slices)
        .max(1);
    let chunk_slices = (num_slices + n_threads - 1) / n_threads;

    let start_offset = k_start * slice_elems;
    let active_len = num_slices * slice_elems;
    let cs_sub = &mut out_cs[start_offset..start_offset + active_len];
    let resp_sub = &mut out_resp[start_offset..start_offset + active_len];
    let rsym_sub = &mut out_rsym[start_offset..start_offset + active_len];

    let chunk_elems = chunk_slices * slice_elems;
    std::thread::scope(|s| {
        for (chunk_idx, ((cs_chunk, resp_chunk), rsym_chunk)) in cs_sub
            .chunks_mut(chunk_elems)
            .zip(resp_sub.chunks_mut(chunk_elems))
            .zip(rsym_sub.chunks_mut(chunk_elems))
            .enumerate()
        {
            let k_chunk_start = k_start + chunk_idx * chunk_slices;
            let slices_in_chunk = cs_chunk.len() / slice_elems;
            s.spawn(move || {
                for s_idx in 0..slices_in_chunk {
                    let k = k_chunk_start + s_idx;
                    let sl = &vol[k * slice_elems..(k + 1) * slice_elems];
                    let cs = &mut cs_chunk[s_idx * slice_elems..(s_idx + 1) * slice_elems];
                    let resp = &mut resp_chunk[s_idx * slice_elems..(s_idx + 1) * slice_elems];
                    let rsym = &mut rsym_chunk[s_idx * slice_elems..(s_idx + 1) * slice_elems];

                    for j in ja_s..jb_s {
                        let row = j * dim_x;
                        let row_m = (j - 1) * dim_x;
                        let row_p = (j + 1) * dim_x;
                        for i in ia_s..ib_s {
                            let sum = 8.0 * sl[row + i]
                                + 2.0 * (sl[row + i - 1] + sl[row + i + 1] + sl[row_m + i] + sl[row_p + i])
                                + (sl[row_m + i - 1] + sl[row_m + i + 1] + sl[row_p + i - 1] + sl[row_p + i + 1]);
                            cs[row + i] = (sum * 0.05) as f32;
                        }
                    }

                    for j in ja..jb {
                        let row = j * dim_x;
                        for i in ia..ib {
                            let c = cs[row + i];
                            let mut best_sym = 0.0_f32;
                            let mut best_med = 0.0_f32;
                            for r in 0..4 {
                                let mut pmins = [0.0_f32; 6];
                                let mut all12 = [0.0_f32; 12];
                                for a in 0..6 {
                                    let di = di_tab[r][a];
                                    let dj = dj_tab[r][a];
                                    let idx_pos = ((j as isize + dj) as usize) * dim_x + ((i as isize + di) as usize);
                                    let idx_neg = ((j as isize - dj) as usize) * dim_x + ((i as isize - di) as usize);
                                    let v_pos = cs[idx_pos];
                                    let v_neg = cs[idx_neg];
                                    pmins[a] = v_pos.min(v_neg);
                                    all12[2 * a] = v_pos;
                                    all12[2 * a + 1] = v_neg;
                                }
                                pmins.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
                                all12.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
                                let sym_r = 0.45_f32 * pmins[1] + 0.55_f32 * pmins[2];
                                if sym_r > best_sym {
                                    best_sym = sym_r;
                                }
                                if all12[6] > best_med {
                                    best_med = all12[6];
                                }
                            }
                            rsym[row + i] = best_sym;
                            if best_med > min_tissue && best_sym > c {
                                let drop = best_sym - c;
                                let rel_drop = drop / (best_med + 12.0_f32);
                                let dark_ratio = 1.0_f32 - c / (best_sym + 1e-3_f32);
                                resp[row + i] = drop * rel_drop * rel_drop.sqrt() * (dark_ratio * dark_ratio);
                            } else {
                                resp[row + i] = 0.0_f32;
                            }
                        }
                    }
                }
            });
        }
    });

    0
}

#[no_mangle]
pub unsafe extern "C" fn nf_find_2d_nms_peaks(
    slice_ptr: *const c_float,
    dim_x: usize,
    dim_y: usize,
    nms_px: usize,
    thresh: c_float,
    out_i: *mut c_int,
    out_j: *mut c_int,
    out_val: *mut c_float,
    max_out: usize,
    out_count: *mut usize,
) -> c_int {
    if slice_ptr.is_null()
        || out_i.is_null()
        || out_j.is_null()
        || out_val.is_null()
        || out_count.is_null()
        || dim_x == 0
        || dim_y == 0
    {
        return -1;
    }
    let slice = std::slice::from_raw_parts(slice_ptr, dim_x * dim_y);
    let mut count = 0_usize;
    for j in 0..dim_y {
        let j_lo = j.saturating_sub(nms_px);
        let j_hi = (j + nms_px).min(dim_y - 1);
        let row_off = j * dim_x;
        for i in 0..dim_x {
            let val = slice[row_off + i];
            if !(val >= thresh) {
                continue;
            }
            let i_lo = i.saturating_sub(nms_px);
            let i_hi = (i + nms_px).min(dim_x - 1);
            let mut is_max = true;
            'outer: for jj in j_lo..=j_hi {
                let w_off = jj * dim_x;
                for ii in i_lo..=i_hi {
                    if slice[w_off + ii] > val {
                        is_max = false;
                        break 'outer;
                    }
                }
            }
            if is_max && count < max_out {
                *out_i.add(count) = i as c_int;
                *out_j.add(count) = j as c_int;
                *out_val.add(count) = val;
                count += 1;
            }
        }
    }
    *out_count = count;
    0
}

#[no_mangle]
pub unsafe extern "C" fn nf_trace_seed_3d(
    resp_sub: *const c_float,
    c_smooth: *const c_float,
    ring_sym: *const c_float,
    dim_x: usize,
    dim_y: usize,
    dim_z: usize,
    k_start: usize,
    i_start: c_double,
    j_start: c_double,
    s_val: c_double,
    cand_thresh: c_double,
    max_step_px: c_double,
    search_r_px: usize,
    max_gap_slices: usize,
    min_pts_slices: usize,
    pass_idx: c_int,
    has_prior_slope: c_int,
    prior_slope_i: c_double,
    prior_slope_j: c_double,
    gauss_base: *const c_double,
    out_ks: *mut c_int,
    out_i: *mut c_double,
    out_j: *mut c_double,
    out_r: *mut c_double,
    out_n: *mut usize,
    out_shaft_p75: *mut c_double,
) -> c_int {
    if resp_sub.is_null()
        || c_smooth.is_null()
        || ring_sym.is_null()
        || gauss_base.is_null()
        || out_ks.is_null()
        || out_i.is_null()
        || out_j.is_null()
        || out_r.is_null()
        || out_n.is_null()
        || out_shaft_p75.is_null()
    {
        return -1;
    }
    if k_start >= dim_z || dim_x <= 2 * search_r_px || dim_y <= 2 * search_r_px {
        *out_n = 0;
        return 0;
    }

    let patch_w = 2 * search_r_px + 1;
    let slice_stride = dim_y * dim_x;
    let total_vox = dim_z * slice_stride;
    let resp = std::slice::from_raw_parts(resp_sub, total_vox);
    let cs = std::slice::from_raw_parts(c_smooth, total_vox);
    let sym = std::slice::from_raw_parts(ring_sym, total_vox);
    let g_base = std::slice::from_raw_parts(gauss_base, patch_w * patch_w);

    let mut peak_r = s_val.min(cand_thresh * 40.0);
    let base_min_r = cand_thresh * if pass_idx == 1 { 0.28 } else { 0.20 };

    let mut fwd_k = Vec::with_capacity(dim_z);
    let mut fwd_i = Vec::with_capacity(dim_z);
    let mut fwd_j = Vec::with_capacity(dim_z);
    let mut fwd_r = Vec::with_capacity(dim_z);
    let mut bwd_k = Vec::with_capacity(dim_z);
    let mut bwd_i = Vec::with_capacity(dim_z);
    let mut bwd_j = Vec::with_capacity(dim_z);
    let mut bwd_r = Vec::with_capacity(dim_z);

    for &direction in &[1_isize, -1_isize] {
        let mut curr_i = i_start;
        let mut curr_j = j_start;
        let mut vel_i = if has_prior_slope != 0 {
            prior_slope_i * (direction as f64)
        } else {
            0.0
        };
        let mut vel_j = if has_prior_slope != 0 {
            prior_slope_j * (direction as f64)
        } else {
            0.0
        };
        let mut gap = 0_usize;
        let mut k_cur = (k_start as isize) + direction;

        while k_cur >= 0 && (k_cur as usize) < dim_z {
            let pred_i = curr_i + vel_i;
            let pred_j = curr_j + vel_j;
            let ic = pred_i.round_ties_even() as isize;
            let jc = pred_j.round_ties_even() as isize;
            if ic < (search_r_px as isize)
                || ic >= ((dim_x - search_r_px) as isize)
                || jc < (search_r_px as isize)
                || jc >= ((dim_y - search_r_px) as isize)
            {
                break;
            }

            let k_idx = k_cur as usize;
            let sl_off = k_idx * slice_stride;
            let j0_p = (jc as usize) - search_r_px;
            let i0_p = (ic as usize) - search_r_px;

            let mut best_eff = -1.0e300_f64;
            let mut best_dy = 0_usize;
            let mut best_dx = 0_usize;

            for dy in 0..patch_w {
                let r_off = sl_off + (j0_p + dy) * dim_x + i0_p;
                let g_off = dy * patch_w;
                for dx in 0..patch_w {
                    let eff = (resp[r_off + dx] as f64) * g_base[g_off + dx];
                    if eff > best_eff {
                        best_eff = eff;
                        best_dy = dy;
                        best_dx = dx;
                    }
                }
            }

            let best_j_idx = j0_p + best_dy;
            let best_i_idx = i0_p + best_dx;
            let v_idx = sl_off + best_j_idx * dim_x + best_i_idx;
            let best_r = resp[v_idx] as f64;
            let best_c = cs[v_idx] as f64;
            let best_sym = sym[v_idx] as f64;

            let mut ni = best_i_idx as f64;
            let mut nj = best_j_idx as f64;
            let ii0 = best_i_idx;
            let jj0 = best_j_idx;

            if ii0 >= 1 && ii0 + 1 < dim_x && jj0 >= 1 && jj0 + 1 < dim_y && best_r > 0.5 {
                let mut w_sum = 0.0_f64;
                let mut wx_sum = 0.0_f64;
                let mut wy_sum = 0.0_f64;
                for dj in -1_isize..=1_isize {
                    let w_row = sl_off + ((jj0 as isize + dj) as usize) * dim_x;
                    for di in -1_isize..=1_isize {
                        let rv = resp[w_row + ((ii0 as isize + di) as usize)];
                        if rv > 0.0 {
                            let w = rv as f64;
                            w_sum += w;
                            wx_sum += (di as f64) * w;
                            wy_sum += (dj as f64) * w;
                        }
                    }
                }
                if w_sum > 0.0 {
                    ni = (ii0 as f64) + wx_sum / w_sum;
                    nj = (jj0 as f64) + wy_sum / w_sum;
                }
            }

            let min_step_r = base_min_r.max(0.045 * peak_r);
            let is_dark = (best_r >= min_step_r)
                || (best_sym > 14.0 && best_c < 0.56 * best_sym && best_r >= 0.25 * min_step_r);
            if is_dark {
                if direction == 1 {
                    fwd_k.push(k_cur as c_int);
                    fwd_i.push(ni);
                    fwd_j.push(nj);
                    fwd_r.push(best_r);
                } else {
                    bwd_k.push(k_cur as c_int);
                    bwd_i.push(ni);
                    bwd_j.push(nj);
                    bwd_r.push(best_r);
                }
                peak_r = peak_r.max(best_r.min(cand_thresh * 42.0));
                let step_i = (ni - curr_i).clamp(-max_step_px, max_step_px);
                let step_j = (nj - curr_j).clamp(-max_step_px, max_step_px);
                vel_i = 0.65 * vel_i + 0.35 * step_i;
                vel_j = 0.65 * vel_j + 0.35 * step_j;
                curr_i = ni;
                curr_j = nj;
                gap = 0;
            } else {
                gap += 1;
                if gap > max_gap_slices {
                    break;
                }
                curr_i = pred_i;
                curr_j = pred_j;
            }
            k_cur += direction;
        }
    }

    let total_raw = bwd_k.len() + 1 + fwd_k.len();
    if total_raw < min_pts_slices {
        *out_n = 0;
        *out_shaft_p75 = 0.0;
        return 0;
    }

    let mut idx = 0_usize;
    for b in (0..bwd_k.len()).rev() {
        *out_ks.add(idx) = bwd_k[b];
        *out_i.add(idx) = bwd_i[b];
        *out_j.add(idx) = bwd_j[b];
        *out_r.add(idx) = bwd_r[b];
        idx += 1;
    }
    *out_ks.add(idx) = k_start as c_int;
    *out_i.add(idx) = i_start;
    *out_j.add(idx) = j_start;
    *out_r.add(idx) = s_val;
    idx += 1;
    for f in 0..fwd_k.len() {
        *out_ks.add(idx) = fwd_k[f];
        *out_i.add(idx) = fwd_i[f];
        *out_j.add(idx) = fwd_j[f];
        *out_r.add(idx) = fwd_r[f];
        idx += 1;
    }

    let r_slice = std::slice::from_raw_parts_mut(out_r, idx);
    let mut r_sorted = r_slice.to_vec();
    r_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pos = 0.75 * ((idx - 1) as f64);
    let lo = pos as usize;
    let hi = (lo + 1).min(idx - 1);
    let shaft_p75 = r_sorted[lo] + (pos - (lo as f64)) * (r_sorted[hi] - r_sorted[lo]);
    *out_shaft_p75 = shaft_p75;

    let tip_cut = (cand_thresh * 0.35).max(0.08 * shaft_p75);
    let base_cut = (cand_thresh * 0.22).max(0.05 * shaft_p75);
    let mut i_lo = 0_usize;
    let mut i_hi = idx;
    while (i_hi - i_lo) > 2 && r_slice[i_hi - 1] < tip_cut {
        i_hi -= 1;
    }
    while (i_hi - i_lo) > 2 && r_slice[i_lo] < base_cut {
        i_lo += 1;
    }
    let n_trim = i_hi - i_lo;
    if n_trim < min_pts_slices {
        *out_n = 0;
        return 0;
    }
    if i_lo > 0 {
        for k in 0..n_trim {
            *out_ks.add(k) = *out_ks.add(i_lo + k);
            *out_i.add(k) = *out_i.add(i_lo + k);
            *out_j.add(k) = *out_j.add(i_lo + k);
            *out_r.add(k) = *out_r.add(i_lo + k);
        }
    }
    *out_n = n_trim;
    0
}

#[inline]
fn fast_line_fit_strided_rs(ras: &[f64], n: usize) -> (f64, f64, f64, f64) {
    let inv_n = 1.0 / (n as f64);
    let mut zm = 0.0_f64;
    let mut xm = 0.0_f64;
    let mut ym = 0.0_f64;
    for i in 0..n {
        xm += ras[3 * i];
        ym += ras[3 * i + 1];
        zm += ras[3 * i + 2];
    }
    zm *= inv_n;
    xm *= inv_n;
    ym *= inv_n;
    let mut denom = 0.0_f64;
    let mut num_x = 0.0_f64;
    let mut num_y = 0.0_f64;
    for i in 0..n {
        let dz = ras[3 * i + 2] - zm;
        denom += dz * dz;
        num_x += dz * (ras[3 * i] - xm);
        num_y += dz * (ras[3 * i + 1] - ym);
    }
    if denom < 1e-6 {
        (0.0, xm, 0.0, ym)
    } else {
        let sx = num_x / denom;
        let sy = num_y / denom;
        (sx, xm - sx * zm, sy, ym - sy * zm)
    }
}

#[no_mangle]
pub unsafe extern "C" fn nf_trim_trajectory_inside_out(
    ras_pts: *const c_double,
    arr_pts: *const c_double,
    n: usize,
    shaft_p75: c_double,
    check_shallow: c_int,
    out_start_idx: *mut usize,
    out_end_idx: *mut usize,
) -> c_int {
    if ras_pts.is_null()
        || arr_pts.is_null()
        || out_start_idx.is_null()
        || out_end_idx.is_null()
    {
        return -1;
    }
    if n < 6 {
        *out_start_idx = 0;
        *out_end_idx = n;
        return 0;
    }
    let ras = std::slice::from_raw_parts(ras_pts, 3 * n);
    let arr = std::slice::from_raw_parts(arr_pts, 4 * n);

    if check_shallow != 0 {
        let inv_n = 1.0 / (n as f64);
        let mut zm = 0.0_f64;
        let mut xm = 0.0_f64;
        let mut ym = 0.0_f64;
        for i in 0..n {
            xm += ras[3 * i];
            ym += ras[3 * i + 1];
            zm += ras[3 * i + 2];
        }
        zm *= inv_n;
        xm *= inv_n;
        ym *= inv_n;
        let mut s2 = 0.0_f64;
        let mut s3 = 0.0_f64;
        let mut s4 = 0.0_f64;
        let mut rx1 = 0.0_f64;
        let mut rx2 = 0.0_f64;
        let mut ry1 = 0.0_f64;
        let mut ry2 = 0.0_f64;
        for i in 0..n {
            let u = ras[3 * i + 2] - zm;
            let u2 = u * u;
            let dx = ras[3 * i] - xm;
            let dy = ras[3 * i + 1] - ym;
            s2 += u2;
            s3 += u2 * u;
            s4 += u2 * u2;
            rx1 += u2 * dx;
            rx2 += u * dx;
            ry1 += u2 * dy;
            ry2 += u * dy;
        }
        let mut ax = 0.0_f64;
        let mut bx = 0.0_f64;
        let mut cx = 0.0_f64;
        let mut ay = 0.0_f64;
        let mut by = 0.0_f64;
        let mut cy = 0.0_f64;
        if n >= 7 && s2 > 1e-12 {
            let a11 = s4 - (s2 * s2) * inv_n;
            let a12 = s3;
            let a22 = s2;
            let det = a11 * a22 - a12 * a12;
            if det.abs() > 1e-12 {
                let inv_det = 1.0 / det;
                ax = (rx1 * a22 - rx2 * a12) * inv_det;
                bx = (a11 * rx2 - a12 * rx1) * inv_det;
                cx = -s2 * inv_n * ax;
                ay = (ry1 * a22 - ry2 * a12) * inv_det;
                by = (a11 * ry2 - a12 * ry1) * inv_det;
                cy = -s2 * inv_n * ay;
            } else {
                bx = rx2 / s2;
                by = ry2 / s2;
            }
        } else if s2 > 1e-6 {
            bx = rx2 / s2;
            by = ry2 / s2;
        }
        let mut sq_err = 0.0_f64;
        for i in 0..n {
            let u = ras[3 * i + 2] - zm;
            let fx = xm + ax * u * u + bx * u + cx;
            let fy = ym + ay * u * u + by * u + cy;
            let dx = ras[3 * i] - fx;
            let dy = ras[3 * i + 1] - fy;
            sq_err += dx * dx + dy * dy;
        }
        if (sq_err * inv_n).sqrt() <= 1.30 {
            *out_start_idx = 0;
            *out_end_idx = n;
            return 0;
        }
    }

    let win_n = n.min(5_usize.max(16_usize.min((0.55 * (n as f64)).round_ties_even() as usize)));
    let max_start = 1_usize.max(((0.50 * ((n - win_n) as f64)).round_ties_even() as usize) + 1);

    let mut best_rmse = 1e9_f64;
    let mut best_i0 = 0_usize;
    let mut best_fit: Option<(f64, f64, f64, f64)> = None;

    for s_i in 0..max_start {
        let e_i = s_i + win_n;
        if (ras[3 * (e_i - 1) + 2] - ras[3 * s_i + 2]).abs() < 3.5 {
            continue;
        }
        let (sx, bx, sy, by) = fast_line_fit_strided_rs(&ras[(3 * s_i)..(3 * e_i)], win_n);
        let mut sq_sum = 0.0_f64;
        for k in s_i..e_i {
            let zk = ras[3 * k + 2];
            let dx = ras[3 * k] - (sx * zk + bx);
            let dy = ras[3 * k + 1] - (sy * zk + by);
            sq_sum += dx * dx + dy * dy;
        }
        let err = (sq_sum / (win_n as f64)).sqrt();
        if err < best_rmse {
            best_rmse = err;
            best_i0 = s_i;
            best_fit = Some((sx, bx, sy, by));
        }
    }

    let (gsx, gbx, gsy, gby) = match best_fit {
        Some(f) => f,
        None => {
            *out_start_idx = 0;
            *out_end_idx = n;
            return 0;
        }
    };

    let mut end_idx = n;
    let mut bad_streak = 0_usize;
    for idx in (best_i0 + win_n)..n {
        let r0 = best_i0.max(idx.saturating_sub(14));
        let (lsx, lbx, lsy, lby) = fast_line_fit_strided_rs(&ras[(3 * r0)..(3 * idx)], idx - r0);
        let xi = ras[3 * idx];
        let yi = ras[3 * idx + 1];
        let zi = ras[3 * idx + 2];
        let dev_loc = (xi - (lsx * zi + lbx)).hypot(yi - (lsy * zi + lby));
        let dev_glob = (xi - (gsx * zi + gbx)).hypot(yi - (gsy * zi + gby));
        let max_glob = 3.25_f64.max(0.048 * (zi - ras[3 * best_i0 + 2]).abs());
        if dev_loc > 1.85 || dev_glob > max_glob || (dev_loc > 1.35 && arr[4 * idx + 3] < 0.35 * shaft_p75) {
            bad_streak += 1;
            if bad_streak >= 2 || dev_loc > 2.35 || idx == n - 1 {
                end_idx = idx + 1 - bad_streak;
                break;
            }
        } else {
            bad_streak = 0;
        }
    }

    let mut start_idx = 0_usize;
    bad_streak = 0;
    for idx in (0..best_i0).rev() {
        let r1 = end_idx.min(idx + 15);
        let (lsx, lbx, lsy, lby) = fast_line_fit_strided_rs(&ras[(3 * (idx + 1))..(3 * r1)], r1 - (idx + 1));
        let xi = ras[3 * idx];
        let yi = ras[3 * idx + 1];
        let zi = ras[3 * idx + 2];
        let dev_loc = (xi - (lsx * zi + lbx)).hypot(yi - (lsy * zi + lby));
        let dev_glob = (xi - (gsx * zi + gbx)).hypot(yi - (gsy * zi + gby));
        let max_glob = 3.25_f64.max(0.048 * (ras[3 * (best_i0 + win_n - 1) + 2] - zi).abs());
        if dev_loc > 1.95 || dev_glob > max_glob || (dev_loc > 1.45 && arr[4 * idx + 3] < 0.30 * shaft_p75) {
            bad_streak += 1;
            if bad_streak >= 2 || dev_loc > 2.45 || idx == 0 {
                start_idx = idx + bad_streak;
                break;
            }
        } else {
            bad_streak = 0;
        }
    }

    *out_start_idx = start_idx;
    *out_end_idx = end_idx;
    0
}
