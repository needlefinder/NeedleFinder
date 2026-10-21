// High-performance C++17 native acceleration kernels for 3D Slicer NeedleFinder.
// Exposes the exact same C-ABI as `src/lib.rs` so `RustCore.py` can build/load
// either the Rust `cdylib` or a multi-arch C++17 shared library (e.g. when Slicer
// runs under Rosetta x86_64 on Apple Silicon without an x86_64 Rust stdlib).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <algorithm>
#include <thread>

namespace {

inline double get_scalar(
    const double* vol,
    ptrdiff_t dim_x,
    ptrdiff_t dim_y,
    ptrdiff_t dim_z,
    ptrdiff_t ix,
    ptrdiff_t iy,
    ptrdiff_t iz
) noexcept {
    if (ix >= 0 && ix < dim_x && iy >= 0 && iy < dim_y && iz >= 0 && iz < dim_z) {
        return vol[(iz * dim_y + iy) * dim_x + ix];
    }
    return 0.0;
}

inline double compute_feedback(
    double ijk_x,
    double ijk_y,
    double ijk_z,
    const double* ijk_to_ras,
    const double* related_pts,
    const size_t* related_offsets,
    size_t num_related,
    double radius_needle_mm,
    double punish
) noexcept {
    const double ras_x = ijk_to_ras[0] * ijk_x + ijk_to_ras[1] * ijk_y + ijk_to_ras[2] * ijk_z + ijk_to_ras[3];
    const double ras_y = ijk_to_ras[4] * ijk_x + ijk_to_ras[5] * ijk_y + ijk_to_ras[6] * ijk_z + ijk_to_ras[7];
    const double ras_z = ijk_to_ras[8] * ijk_x + ijk_to_ras[9] * ijk_y + ijk_to_ras[10] * ijk_z + ijk_to_ras[11];

    double neighbour_attenuation = 0.0;
    const double thresh = radius_needle_mm * punish;

    for (size_t w = 0; w < num_related; ++w) {
        const size_t start = related_offsets[w];
        const size_t end = related_offsets[w + 1];
        if (end <= start) continue;

        double distance = 9999.0;
        const double first_z = related_pts[start * 3 + 2];
        const double last_z = related_pts[(end - 1) * 3 + 2];

        if (ras_z >= first_z) {
            const double dz = ras_z - first_z;
            distance = std::sqrt(3.0 * dz * dz);
        } else if (ras_z <= last_z) {
            const double dz = ras_z - last_z;
            distance = std::sqrt(3.0 * dz * dz);
        } else {
            for (size_t q = start; q + 1 < end; ++q) {
                const double pax = related_pts[q * 3];
                const double pay = related_pts[q * 3 + 1];
                const double paz = related_pts[q * 3 + 2];
                const double pbx = related_pts[(q + 1) * 3];
                const double pby = related_pts[(q + 1) * 3 + 1];
                const double pbz = related_pts[(q + 1) * 3 + 2];

                double h = 0.0;
                if (paz <= ras_z) {
                    const double dx = ras_x - pax, dy = ras_y - pay, dz = ras_z - paz;
                    h = std::sqrt(dx * dx + dy * dy + dz * dz);
                } else if (pbz >= ras_z) {
                    const double dx = ras_x - pbx, dy = ras_y - pby, dz = ras_z - pbz;
                    h = std::sqrt(dx * dx + dy * dy + dz * dz);
                } else {
                    const double dax = pax - ras_x, day = pay - ras_y, daz = paz - ras_z;
                    const double dbx = pbx - ras_x, dby = pby - ras_y, dbz = pbz - ras_z;
                    const double dcx = pax - pbx, dcy = pay - pby, dcz = paz - pbz;
                    const double a = std::sqrt(dax * dax + day * day + daz * daz);
                    const double b = std::sqrt(dbx * dbx + dby * dby + dbz * dbz);
                    const double c = std::sqrt(dcx * dcx + dcy * dcy + dcz * dcz);
                    const double hc = (a + b + c) * 0.5;
                    const double prod = std::max(0.0, hc * (hc - a) * (hc - b) * (hc - c));
                    h = std::sqrt(prod) * 2.0 / c;
                }
                if (h < distance) {
                    distance = h;
                }
            }
        }

        if (distance == 0.0) {
            neighbour_attenuation += 3000.0;
        } else if (distance < thresh) {
            neighbour_attenuation += 1000.0 * (radius_needle_mm / distance);
        }
    }
    return neighbour_attenuation;
}

constexpr double kPi = 3.14159265358979323846;

} // namespace

extern "C" {

int32_t nf_search_cone_15_1(
    const double* vol_ptr,
    const double* label_ptr,
    ptrdiff_t dim_x,
    ptrdiff_t dim_y,
    ptrdiff_t dim_z,
    const double* ijk_a_ptr,
    const double* ijk_b_ptr,
    const double* ijk_a_prev_ptr,
    double r_max,
    double n_r_iter,
    int32_t n_rotating_its,
    double n_t_iter,
    int32_t b_gradient,
    ptrdiff_t radius_needle,
    ptrdiff_t radius_needle_corner,
    double gradient_ponderation,
    int32_t b_gaussian_attenuation,
    double sigma_value,
    const double* ijk_to_ras_ptr,
    const double* related_pts_ptr,
    const size_t* related_offsets_ptr,
    size_t num_related,
    double radius_needle_mm,
    double punish,
    double* out_c_best_ptr,
    double* out_min_total_ptr,
    double* out_last_total_ptr
) {
    if (!vol_ptr || !ijk_a_ptr || !ijk_b_ptr || !out_c_best_ptr) return -1;

    const bool has_feedback = (num_related > 0 && ijk_to_ras_ptr && related_pts_ptr && related_offsets_ptr);
    const double ax = ijk_a_ptr[0], ay = ijk_a_ptr[1], az = ijk_a_ptr[2];
    const double bx = ijk_b_ptr[0], by = ijk_b_ptr[1], bz = ijk_b_ptr[2];

    bool has_gauss = false;
    double x_att_x = 0.0, x_att_y = 0.0, x_att_z = 0.0, gauss_denom = 1.0;
    if (b_gaussian_attenuation == 1 && ijk_a_prev_ptr) {
        const double dz_prev = ijk_a_prev_ptr[2] - az;
        if (dz_prev != 0.0 && r_max != 0.0 && sigma_value != 0.0) {
            has_gauss = true;
            const double f_k_att = (az - bz) / dz_prev;
            x_att_x = ax + f_k_att * (ax - ijk_a_prev_ptr[0]);
            x_att_y = ay + f_k_att * (ay - ijk_a_prev_ptr[1]);
            x_att_z = az + f_k_att * (az - ijk_a_prev_ptr[2]);
            const double s = 2.0 * (sigma_value / 10.0);
            gauss_denom = s * s;
        }
    }

    double f_min_total = std::numeric_limits<double>::infinity();
    double c_best[3] = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()
    };
    double f_last_total = 0.0;

    const int32_t n_r_steps = static_cast<int32_t>(n_r_iter);
    const int32_t n_t_steps = static_cast<int32_t>(n_t_iter);
    const double n_rot_f64 = static_cast<double>(n_rotating_its);

    std::vector<double> cos_tab(n_rotating_its), sin_tab(n_rotating_its);
    for (int32_t i_theta = 0; i_theta < n_rotating_its; ++i_theta) {
        const double rad = ((static_cast<double>(i_theta) * 360.0) / n_rot_f64) * (kPi / 180.0);
        cos_tab[i_theta] = std::cos(rad);
        sin_tab[i_theta] = std::sin(rad);
    }

    for (int32_t i_r = 0; i_r <= n_r_steps; ++i_r) {
        const double f_r = static_cast<double>(i_r) * (r_max / n_r_iter);
        for (int32_t i_theta = 0; i_theta < n_rotating_its; ++i_theta) {
            const double cx = bx + f_r * cos_tab[i_theta];
            const double cy = by + f_r * sin_tab[i_theta];
            const double cz = bz;

            double f_total = 0.0;
            for (int32_t i_t = 0; i_t <= n_t_steps; ++i_t) {
                const double f_t = static_cast<double>(i_t) / n_t_iter;
                const double one_minus_t = 1.0 - f_t;
                const double mx = one_minus_t * ax + f_t * cx;
                const double my = one_minus_t * ay + f_t * cy;
                const double mz = one_minus_t * az + f_t * cz;

                const ptrdiff_t ix = static_cast<ptrdiff_t>(std::rint(mx));
                const ptrdiff_t iy = static_cast<ptrdiff_t>(std::rint(my));
                const ptrdiff_t iz = static_cast<ptrdiff_t>(std::rint(mz));

                if (ix > 0 && ix < dim_x && iy > 0 && iy < dim_y && iz > 0 && iz < dim_z) {
                    const double d_center = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix, iy, iz);
                    f_total += d_center;

                    if (has_feedback) {
                        f_total += compute_feedback(
                            mx, my, mz, ijk_to_ras_ptr, related_pts_ptr,
                            related_offsets_ptr, num_related, radius_needle_mm, punish
                        );
                    }

                    if (b_gradient == 1) {
                        const double g1 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix + radius_needle, iy, iz);
                        const double g2 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix - radius_needle, iy, iz);
                        const double g3 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix, iy + radius_needle, iz);
                        const double g4 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix, iy - radius_needle, iz);
                        const double g5 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix + radius_needle_corner, iy + radius_needle_corner, iz);
                        const double g6 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix - radius_needle_corner, iy - radius_needle_corner, iz);
                        const double g7 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix - radius_needle_corner, iy + radius_needle_corner, iz);
                        const double g8 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix + radius_needle_corner, iy - radius_needle_corner, iz);
                        f_total += 8.0 * d_center - ((g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / 8.0) * gradient_ponderation;
                    }

                    if (label_ptr) {
                        const double f_label = get_scalar(label_ptr, dim_x, dim_y, dim_z, ix, iy, iz);
                        if (f_label != 0.0 && f_label < 300.0) {
                            f_total -= 10000.0;
                        }
                    }
                }
            }

            if (has_gauss) {
                const double dx = cx - x_att_x, dy = cy - x_att_y, dz = cz - x_att_z;
                const double ratio = std::sqrt(dx * dx + dy * dy + dz * dz) / r_max;
                f_total *= std::exp(-(ratio * ratio) / gauss_denom);
            }

            f_last_total = f_total;
            if (f_total < f_min_total) {
                f_min_total = f_total;
                c_best[0] = cx;
                c_best[1] = cy;
                c_best[2] = cz;
            }
        }
    }

    out_c_best_ptr[0] = c_best[0];
    out_c_best_ptr[1] = c_best[1];
    out_c_best_ptr[2] = c_best[2];
    if (out_min_total_ptr) *out_min_total_ptr = f_min_total;
    if (out_last_total_ptr) *out_last_total_ptr = f_last_total;
    return 0;
}

int32_t nf_search_cone_13(
    const double* vol_ptr,
    const double* label_ptr,
    ptrdiff_t dim_x,
    ptrdiff_t dim_y,
    ptrdiff_t dim_z,
    const double* ijk_a_ptr,
    const double* ijk_c0_ptr,
    const double* ijk_tip0_ptr,
    double r_max,
    double r_iter,
    int32_t nb_rotating_step,
    double t_iter,
    int32_t b_gradient,
    ptrdiff_t radius_needle,
    ptrdiff_t radius_needle_corner,
    double gradient_ponderation,
    int32_t b_gaussian_attenuation,
    double sigma_value,
    const double* ijk_to_ras_ptr,
    const double* related_pts_ptr,
    const size_t* related_offsets_ptr,
    size_t num_related,
    double radius_needle_mm,
    double punish,
    double* in_out_best_point_ptr,
    double* out_min_estimator_ptr
) {
    if (!vol_ptr || !ijk_a_ptr || !ijk_c0_ptr || !in_out_best_point_ptr) return -1;

    const bool has_feedback = (num_related > 0 && ijk_to_ras_ptr && related_pts_ptr && related_offsets_ptr);
    const double ax = ijk_a_ptr[0], ay = ijk_a_ptr[1], az = ijk_a_ptr[2];
    const double c0x = ijk_c0_ptr[0], c0y = ijk_c0_ptr[1], c0z = ijk_c0_ptr[2];

    bool has_gauss = false;
    double x_att_x = 0.0, x_att_y = 0.0, x_att_z = 0.0, gauss_denom = 1.0;
    if (b_gaussian_attenuation == 1 && ijk_tip0_ptr) {
        const double dz_prev = ijk_tip0_ptr[2] - az;
        if (dz_prev != 0.0 && r_max != 0.0 && sigma_value != 0.0) {
            has_gauss = true;
            const double k = (az - c0z) / dz_prev;
            x_att_x = ax + k * (ax - ijk_tip0_ptr[0]);
            x_att_y = ay + k * (ay - ijk_tip0_ptr[1]);
            x_att_z = az + k * (az - ijk_tip0_ptr[2]);
            const double s = 2.0 * (sigma_value / 10.0);
            gauss_denom = s * s;
        }
    }

    double initial_intensity = 0.0;
    double min_estimator = 0.0;

    const int32_t r_steps = static_cast<int32_t>(r_iter);
    const int32_t t_steps = static_cast<int32_t>(t_iter);
    const double nb_rot_f64 = static_cast<double>(nb_rotating_step);

    std::vector<double> cos_tab(nb_rotating_step), sin_tab(nb_rotating_step);
    for (int32_t theta_step = 0; theta_step < nb_rotating_step; ++theta_step) {
        const double rad = ((static_cast<double>(theta_step) * 360.0) / nb_rot_f64) * (kPi / 180.0);
        cos_tab[theta_step] = std::cos(rad);
        sin_tab[theta_step] = std::sin(rad);
    }

    for (int32_t r_idx = 0; r_idx <= r_steps; ++r_idx) {
        const double r = static_cast<double>(r_idx) * (r_max / r_iter);
        for (int32_t theta_step = 0; theta_step < nb_rotating_step; ++theta_step) {
            const double cx = c0x + r * cos_tab[theta_step];
            const double cy = c0y + r * sin_tab[theta_step];
            const double cz = c0z;

            double total = 0.0;
            for (int32_t t = 0; t <= t_steps; ++t) {
                const double tt = static_cast<double>(t) / t_iter;
                const double one_minus_tt = 1.0 - tt;
                const double mx = one_minus_tt * ax + tt * cx;
                const double my = one_minus_tt * ay + tt * cy;
                const double mz = one_minus_tt * az + tt * cz;

                const ptrdiff_t ix = static_cast<ptrdiff_t>(std::rint(mx));
                const ptrdiff_t iy = static_cast<ptrdiff_t>(std::rint(my));
                const ptrdiff_t iz = static_cast<ptrdiff_t>(std::rint(mz));

                if (ix > 0 && ix < dim_x && iy > 0 && iy < dim_y && iz > 0 && iz < dim_z) {
                    const double center = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix, iy, iz);
                    total += center;

                    if (has_feedback) {
                        total += compute_feedback(
                            mx, my, mz, ijk_to_ras_ptr, related_pts_ptr,
                            related_offsets_ptr, num_related, radius_needle_mm, punish
                        );
                    }

                    if (b_gradient == 1) {
                        const double g1 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix + radius_needle, iy, iz);
                        const double g2 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix - radius_needle, iy, iz);
                        const double g3 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix, iy + radius_needle, iz);
                        const double g4 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix, iy - radius_needle, iz);
                        const double g5 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix + radius_needle_corner, iy + radius_needle_corner, iz);
                        const double g6 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix - radius_needle_corner, iy - radius_needle_corner, iz);
                        const double g7 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix - radius_needle_corner, iy + radius_needle_corner, iz);
                        const double g8 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix + radius_needle_corner, iy - radius_needle_corner, iz);
                        total += 8.0 * center - ((g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / 8.0) * gradient_ponderation;
                    }

                    if (label_ptr) {
                        const double f_label = get_scalar(label_ptr, dim_x, dim_y, dim_z, ix, iy, iz);
                        if (f_label != 0.0 && f_label < 300.0) {
                            total -= 10000.0;
                        }
                    }
                }
            }

            if (r_idx == 0) {
                initial_intensity = total;
            }

            double estimator = total;
            if (has_gauss) {
                const double dx = cx - x_att_x, dy = cy - x_att_y, dz = cz - x_att_z;
                const double ratio = std::sqrt(dx * dx + dy * dy + dz * dz) / r_max;
                estimator = total * std::exp(-(ratio * ratio) / gauss_denom);
            }

            if (estimator < initial_intensity) {
                if (estimator < min_estimator || min_estimator == 0.0) {
                    min_estimator = estimator;
                    if (min_estimator != 0.0) {
                        in_out_best_point_ptr[0] = cx;
                        in_out_best_point_ptr[1] = cy;
                        in_out_best_point_ptr[2] = cz;
                    }
                }
            }
        }
    }

    if (out_min_estimator_ptr) *out_min_estimator_ptr = min_estimator;
    return 0;
}

int32_t nf_search_cone_current_dev(
    const double* vol_ptr,
    const double* label_ptr,
    ptrdiff_t dim_x,
    ptrdiff_t dim_y,
    ptrdiff_t dim_z,
    const double* ijk_a_ptr,
    const double* ijk_c0_ptr,
    double r_max,
    int32_t r_iter,
    int32_t nb_rotating_step,
    int32_t t_iter,
    double exponent,
    int32_t b_gradient,
    int32_t mode_flag,
    ptrdiff_t radius_needle,
    ptrdiff_t radius_needle_corner,
    double gradient_ponderation,
    int32_t b_gaussian_attenuation,
    int32_t vz_non_zero,
    double sigma_value,
    double* in_out_best_point_ptr,
    double* out_min_estimator_ptr
) {
    if (!vol_ptr || !ijk_a_ptr || !ijk_c0_ptr || !in_out_best_point_ptr) return -1;

    const double ax = ijk_a_ptr[0], ay = ijk_a_ptr[1], az = ijk_a_ptr[2];
    const double c0x = ijk_c0_ptr[0], c0y = ijk_c0_ptr[1], c0z = ijk_c0_ptr[2];

    const double r_iter_f64 = static_cast<double>(r_iter);
    const double nb_rot_f64 = static_cast<double>(nb_rotating_step);
    const double t_iter_f64 = static_cast<double>(t_iter);

    const bool has_gauss = (b_gaussian_attenuation == 1 && vz_non_zero != 0 && r_max != 0.0 && sigma_value != 0.0);
    const double gauss_denom = has_gauss ? (2.0 * (sigma_value / 10.0)) * (2.0 * (sigma_value / 10.0)) : 1.0;

    double initial_intensity = 0.0;
    double min_estimator = 0.0;

    std::vector<double> cos_tab(nb_rotating_step), sin_tab(nb_rotating_step);
    for (int32_t theta_step = 0; theta_step < nb_rotating_step; ++theta_step) {
        const double rad = ((static_cast<double>(theta_step) * 360.0) / nb_rot_f64) * (kPi / 180.0);
        cos_tab[theta_step] = std::cos(rad);
        sin_tab[theta_step] = std::sin(rad);
    }

    for (int32_t r_idx = 0; r_idx <= r_iter; ++r_idx) {
        const double r = static_cast<double>(r_idx) * (r_max / r_iter_f64);
        for (int32_t theta_step = 0; theta_step < nb_rotating_step; ++theta_step) {
            const double cx = c0x + r * cos_tab[theta_step];
            const double cy = c0y + r * sin_tab[theta_step];
            const double cz = c0z;

            double total = 0.0;
            for (int32_t t = 0; t <= t_iter; ++t) {
                const double tt = static_cast<double>(t) / t_iter_f64;
                const double one_minus_tt = 1.0 - tt;
                const double mx = one_minus_tt * ax + tt * cx;
                const double my = one_minus_tt * ay + tt * cy;
                const double mz = one_minus_tt * az + tt * cz;

                const ptrdiff_t ix = static_cast<ptrdiff_t>(std::rint(mx));
                const ptrdiff_t iy = static_cast<ptrdiff_t>(std::rint(my));
                const ptrdiff_t iz = static_cast<ptrdiff_t>(std::rint(mz));

                if (ix > 0 && ix < dim_x && iy > 0 && iy < dim_y && iz > 0 && iz < dim_z) {
                    const double center =
                        get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix, iy, iz) +
                        get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix + 1, iy, iz) +
                        get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix - 1, iy, iz) +
                        get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix, iy + 1, iz) +
                        get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix, iy - 1, iz) +
                        get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix + 1, iy + 1, iz) +
                        get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix + 1, iy - 1, iz) +
                        get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix - 1, iy + 1, iz) +
                        get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix - 1, iy - 1, iz);

                    total += std::pow(center, exponent);

                    if (b_gradient == 1 && mode_flag == 0) {
                        const double g[8] = {
                            get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix + radius_needle, iy, iz),
                            get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix - radius_needle, iy, iz),
                            get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix, iy + radius_needle, iz),
                            get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix, iy - radius_needle, iz),
                            get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix + radius_needle_corner, iy + radius_needle_corner, iz),
                            get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix - radius_needle_corner, iy - radius_needle_corner, iz),
                            get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix - radius_needle_corner, iy + radius_needle_corner, iz),
                            get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix + radius_needle_corner, iy - radius_needle_corner, iz),
                        };
                        for (double gv : g) {
                            total += std::pow(center - gv, gradient_ponderation) / t_iter_f64;
                        }
                    } else if (b_gradient == 1 && mode_flag == 1) {
                        const double g1 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix + radius_needle, iy, iz);
                        const double g2 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix - radius_needle, iy, iz);
                        const double g3 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix, iy + radius_needle, iz);
                        const double g4 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ix, iy - radius_needle, iz);
                        total += (8.0 * center - ((g1 + g2 + g3 + g4) / 4.0) * gradient_ponderation) / t_iter_f64;
                    }

                    if (label_ptr) {
                        const double f_label = get_scalar(label_ptr, dim_x, dim_y, dim_z, ix, iy, iz);
                        if (f_label != 0.0 && f_label < 300.0) {
                            total -= 10000.0;
                        }
                    }
                }
            }

            if (r_idx == 0) {
                initial_intensity = total;
            }

            double estimator = total;
            if (has_gauss) {
                const double dx = cx - c0x, dy = cy - c0y, dz = cz - c0z;
                const double ratio = std::sqrt(dx * dx + dy * dy + dz * dz) / r_max;
                estimator = total * std::exp(-(ratio * ratio) / gauss_denom);
            }

            if (estimator < initial_intensity) {
                if (estimator < min_estimator || min_estimator == 0.0) {
                    min_estimator = estimator;
                    if (min_estimator != 0.0) {
                        in_out_best_point_ptr[0] = cx;
                        in_out_best_point_ptr[1] = cy;
                        in_out_best_point_ptr[2] = cz;
                    }
                }
            }
        }
    }

    if (out_min_estimator_ptr) *out_min_estimator_ptr = min_estimator;
    return 0;
}

int32_t nf_find_tip(
    const double* vol_ptr,
    ptrdiff_t dim_x,
    ptrdiff_t dim_y,
    ptrdiff_t dim_z,
    ptrdiff_t a_x,
    ptrdiff_t a_y,
    ptrdiff_t a_z,
    ptrdiff_t radius_needle,
    double coeff,
    double sigma_value,
    double gradient_ponderation,
    ptrdiff_t search_x,
    ptrdiff_t search_y,
    ptrdiff_t* out_best_ijk_ptr
) {
    if (!vol_ptr || !out_best_ijk_ptr) return -1;
    double min_total_tip = 0.0;
    ptrdiff_t i_best = a_x, j_best = a_y, k_best = a_z;
    const double denom = 2.0 * (sigma_value / 10.0) * (sigma_value / 10.0);

    for (ptrdiff_t cap_i = -search_x; cap_i < search_x; ++cap_i) {
        const ptrdiff_t i = static_cast<ptrdiff_t>(static_cast<double>(cap_i) / coeff);
        for (ptrdiff_t cap_j = -search_y; cap_j < search_y; ++cap_j) {
            const ptrdiff_t j = static_cast<ptrdiff_t>(static_cast<double>(cap_j) / coeff);
            const ptrdiff_t k = 0;
            double total_tip = 0.0;
            const ptrdiff_t ax = a_x + i;
            const ptrdiff_t ay = a_y + j;
            for (ptrdiff_t l = -3; l < 1; ++l) {
                const ptrdiff_t az = a_z + k + l;
                const double v0 = 8.0 * get_scalar(vol_ptr, dim_x, dim_y, dim_z, ax, ay, az);
                const double v1 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ax + radius_needle, ay, az);
                const double v2 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ax - radius_needle, ay, az);
                const double v3 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ax, ay + radius_needle, az);
                const double v4 = get_scalar(vol_ptr, dim_x, dim_y, dim_z, ax, ay - radius_needle, az);
                total_tip += v0 - ((v1 + v2 + v3 + v4) / 4.0) * gradient_ponderation;
            }
            const double rgauss = std::sqrt(static_cast<double>(i * i + j * j + k * k));
            const double gauss = std::exp(-((rgauss / 10.0) * (rgauss / 10.0)) / denom);
            total_tip *= gauss;

            if (total_tip < min_total_tip || min_total_tip == 0.0) {
                min_total_tip = total_tip;
                i_best = ax;
                j_best = ay;
                k_best = a_z + k;
            }
        }
    }

    out_best_ijk_ptr[0] = i_best;
    out_best_ijk_ptr[1] = j_best;
    out_best_ijk_ptr[2] = k_best;
    return 0;
}

int32_t nf_bezier_curve(
    const double* ctrl_pts_ptr,
    size_t num_ctrl_pts,
    size_t num_eval_pts,
    double* out_curve_ptr
) {
    if (!ctrl_pts_ptr || !out_curve_ptr || num_ctrl_pts == 0 || num_eval_pts == 0) return -1;
    const size_t total_out = num_eval_pts + 1;
    const size_t n = num_ctrl_pts - 1;

    std::vector<double> coeffs(num_ctrl_pts, 1.0);
    for (size_t i = 1; i <= n; ++i) {
        coeffs[i] = coeffs[i - 1] * static_cast<double>(n - i + 1) / static_cast<double>(i);
    }

    const double denom = static_cast<double>(num_eval_pts);
    for (size_t t_idx = 0; t_idx < total_out; ++t_idx) {
        const double tt = static_cast<double>(t_idx) / denom;
        const double one_minus_tt = 1.0 - tt;
        double x = 0.0, y = 0.0, z = 0.0;
        for (size_t i = 0; i <= n; ++i) {
            const double w = coeffs[i] * std::pow(one_minus_tt, static_cast<int>(n - i)) * std::pow(tt, static_cast<int>(i));
            x += w * ctrl_pts_ptr[i * 3];
            y += w * ctrl_pts_ptr[i * 3 + 1];
            z += w * ctrl_pts_ptr[i * 3 + 2];
        }
        out_curve_ptr[t_idx * 3] = x;
        out_curve_ptr[t_idx * 3 + 1] = y;
        out_curve_ptr[t_idx * 3 + 2] = z;
    }
    return 0;
}

int32_t nf_center_of_mass_ijk(
    const double* vol_ptr,
    size_t dim_x,
    size_t dim_y,
    size_t dim_z,
    double* out_com_ptr
) {
    if (!vol_ptr || !out_com_ptr) return -1;
    double sum_w = 0.0, sum_i = 0.0, sum_j = 0.0, sum_k = 0.0;
    for (size_t k = 0; k < dim_z; ++k) {
        const double kf = static_cast<double>(k);
        const size_t k_off = k * dim_y;
        for (size_t j = 0; j < dim_y; ++j) {
            const double jf = static_cast<double>(j);
            const size_t row_off = (k_off + j) * dim_x;
            for (size_t i = 0; i < dim_x; ++i) {
                const double w = vol_ptr[row_off + i];
                if (w != 0.0) {
                    sum_w += w;
                    sum_i += w * static_cast<double>(i);
                    sum_j += w * jf;
                    sum_k += w * kf;
                }
            }
        }
    }
    if (sum_w == 0.0) {
        out_com_ptr[0] = 0.0;
        out_com_ptr[1] = 0.0;
        out_com_ptr[2] = 0.0;
    } else {
        out_com_ptr[0] = sum_i / sum_w;
        out_com_ptr[1] = sum_j / sum_w;
        out_com_ptr[2] = sum_k / sum_w;
    }
    return 0;
}

int32_t nf_compute_3d_sym_response(
    const double* vol_ptr,
    size_t dim_x,
    size_t dim_y,
    size_t dim_z,
    double sp_x,
    double sp_y,
    size_t i0,
    size_t i1,
    size_t j0,
    size_t j1,
    size_t k0,
    size_t k1,
    double body_thresh,
    float* out_resp_ptr,
    float* out_csmooth_ptr,
    float* out_ringsym_ptr
) {
    if (!vol_ptr || !out_resp_ptr || !out_csmooth_ptr || !out_ringsym_ptr || dim_x < 25 || dim_y < 25 || dim_z == 0) {
        return -1;
    }
    const size_t k_start = std::min(k0, dim_z);
    const size_t k_end = std::min(k1, dim_z);
    if (k_end <= k_start) {
        return 0;
    }

    const double radii[3] = {1.7, 2.5, 3.4};
    ptrdiff_t di_tab[3][6];
    ptrdiff_t dj_tab[3][6];
    for (int r = 0; r < 3; ++r) {
        for (int a = 0; a < 6; ++a) {
            const double th = static_cast<double>(a) * (kPi / 6.0);
            di_tab[r][a] = std::clamp(static_cast<ptrdiff_t>(std::lround(radii[r] * std::cos(th) / sp_x)), ptrdiff_t(-11), ptrdiff_t(11));
            dj_tab[r][a] = std::clamp(static_cast<ptrdiff_t>(std::lround(radii[r] * std::sin(th) / sp_y)), ptrdiff_t(-11), ptrdiff_t(11));
        }
    }

    const size_t margin = 12;
    const size_t ja_s = std::max(size_t(1), j0 > margin ? j0 - margin : 1);
    const size_t jb_s = std::min(dim_y - 1, j1 + margin);
    const size_t ia_s = std::max(size_t(1), i0 > margin ? i0 - margin : 1);
    const size_t ib_s = std::min(dim_x - 1, i1 + margin);

    const size_t ja = std::max(margin, j0);
    const size_t jb = std::min(dim_y - margin, j1);
    const size_t ia = std::max(margin, i0);
    const size_t ib = std::min(dim_x - margin, i1);
    const float min_tissue = std::max(12.0f, static_cast<float>(body_thresh) * 0.45f);
    const size_t slice_elems = dim_x * dim_y;

    const size_t num_slices = k_end - k_start;
    const size_t hw = std::max(1u, std::thread::hardware_concurrency());
    const size_t n_threads = std::min(num_slices, hw);
    const size_t chunk = (num_slices + n_threads - 1) / n_threads;

    auto worker = [&](size_t ks, size_t ke) {
        for (size_t k = ks; k < ke; ++k) {
            const double* sl = vol_ptr + k * slice_elems;
            float* cs = out_csmooth_ptr + k * slice_elems;
            float* resp = out_resp_ptr + k * slice_elems;
            float* rsym = out_ringsym_ptr + k * slice_elems;

            for (size_t j = ja_s; j < jb_s; ++j) {
                const size_t row = j * dim_x;
                const size_t row_m = (j - 1) * dim_x;
                const size_t row_p = (j + 1) * dim_x;
                for (size_t i = ia_s; i < ib_s; ++i) {
                    const double sum = 4.0 * sl[row + i]
                        + 2.0 * (sl[row + i - 1] + sl[row + i + 1] + sl[row_m + i] + sl[row_p + i])
                        + (sl[row_m + i - 1] + sl[row_m + i + 1] + sl[row_p + i - 1] + sl[row_p + i + 1]);
                    cs[row + i] = static_cast<float>(sum * 0.0625);
                }
            }

            for (size_t j = ja; j < jb; ++j) {
                const size_t row = j * dim_x;
                for (size_t i = ia; i < ib; ++i) {
                    const float c = cs[row + i];
                    float best_sym = 0.0f;
                    float best_med = 0.0f;
                    for (int r = 0; r < 3; ++r) {
                        float pmins[6];
                        float all12[12];
                        for (int a = 0; a < 6; ++a) {
                            const ptrdiff_t di = di_tab[r][a];
                            const ptrdiff_t dj = dj_tab[r][a];
                            const size_t idx_pos = static_cast<size_t>(static_cast<ptrdiff_t>(j) + dj) * dim_x + static_cast<size_t>(static_cast<ptrdiff_t>(i) + di);
                            const size_t idx_neg = static_cast<size_t>(static_cast<ptrdiff_t>(j) - dj) * dim_x + static_cast<size_t>(static_cast<ptrdiff_t>(i) - di);
                            const float v_pos = cs[idx_pos];
                            const float v_neg = cs[idx_neg];
                            pmins[a] = std::min(v_pos, v_neg);
                            all12[2 * a] = v_pos;
                            all12[2 * a + 1] = v_neg;
                        }
                        std::sort(pmins, pmins + 6);
                        std::nth_element(all12, all12 + 6, all12 + 12);
                        if (pmins[1] > best_sym) best_sym = pmins[1];
                        if (all12[6] > best_med) best_med = all12[6];
                    }
                    rsym[row + i] = best_sym;
                    if (best_med > min_tissue && best_sym > c) {
                        const float drop = best_sym - c;
                        const float rel_drop = drop / (best_med + 12.0f);
                        const float dark_ratio = 1.0f - c / (best_sym + 1e-3f);
                        resp[row + i] = drop * rel_drop * std::sqrt(rel_drop) * (dark_ratio * dark_ratio);
                    } else {
                        resp[row + i] = 0.0f;
                    }
                }
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (size_t t = 0; t < n_threads; ++t) {
        const size_t ks = k_start + t * chunk;
        if (ks >= k_end) break;
        const size_t ke = std::min(k_end, ks + chunk);
        threads.emplace_back(worker, ks, ke);
    }
    for (auto& th : threads) {
        th.join();
    }
    return 0;
}

} // extern "C"
