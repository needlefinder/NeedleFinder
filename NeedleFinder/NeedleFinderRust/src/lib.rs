//! High-performance Rust native acceleration kernels for 3D Slicer NeedleFinder.
//!
//! Exposes a zero-dependency C-ABI (`#[no_mangle] pub extern "C"`) loaded via Python `ctypes`
//! so that it works across any Python 3.x / 3D Slicer build on Linux, macOS, and Windows
//! without linking against a specific `libpython` or VTK C++ ABI.

use std::slice;

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
