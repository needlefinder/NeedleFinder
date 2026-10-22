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

    const double radii[4] = {1.35, 1.85, 2.5, 3.3};
    ptrdiff_t di_tab[4][6];
    ptrdiff_t dj_tab[4][6];
    for (int r = 0; r < 4; ++r) {
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
                    const double sum = 8.0 * sl[row + i]
                        + 2.0 * (sl[row + i - 1] + sl[row + i + 1] + sl[row_m + i] + sl[row_p + i])
                        + (sl[row_m + i - 1] + sl[row_m + i + 1] + sl[row_p + i - 1] + sl[row_p + i + 1]);
                    cs[row + i] = static_cast<float>(sum * 0.05);
                }
            }

            for (size_t j = ja; j < jb; ++j) {
                const size_t row = j * dim_x;
                for (size_t i = ia; i < ib; ++i) {
                    const float c = cs[row + i];
                    float best_sym = 0.0f;
                    float best_med = 0.0f;
                    for (int r = 0; r < 4; ++r) {
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
                        const float sym_r = 0.45f * pmins[1] + 0.55f * pmins[2];
                        if (sym_r > best_sym) best_sym = sym_r;
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

int32_t nf_find_2d_nms_peaks(
    const float* slice_ptr,
    size_t dim_x,
    size_t dim_y,
    size_t nms_px,
    float thresh,
    int* out_i,
    int* out_j,
    float* out_val,
    size_t max_out,
    size_t* out_count
) {
    if (!slice_ptr || !out_i || !out_j || !out_val || !out_count || dim_x == 0 || dim_y == 0) {
        return -1;
    }
    size_t count = 0;
    for (size_t j = 0; j < dim_y; ++j) {
        const size_t j_lo = (j > nms_px) ? (j - nms_px) : 0;
        const size_t j_hi = std::min(dim_y - 1, j + nms_px);
        const float* row_ptr = slice_ptr + j * dim_x;
        for (size_t i = 0; i < dim_x; ++i) {
            const float val = row_ptr[i];
            if (!(val >= thresh)) {
                continue;
            }
            const size_t i_lo = (i > nms_px) ? (i - nms_px) : 0;
            const size_t i_hi = std::min(dim_x - 1, i + nms_px);
            bool is_max = true;
            for (size_t jj = j_lo; jj <= j_hi && is_max; ++jj) {
                const float* w_row = slice_ptr + jj * dim_x;
                for (size_t ii = i_lo; ii <= i_hi; ++ii) {
                    if (w_row[ii] > val) {
                        is_max = false;
                        break;
                    }
                }
            }
            if (is_max) {
                if (count < max_out) {
                    out_i[count] = static_cast<int>(i);
                    out_j[count] = static_cast<int>(j);
                    out_val[count] = val;
                    ++count;
                }
            }
        }
    }
    *out_count = count;
    return 0;
}

int32_t nf_trace_seed_3d(
    const float* resp_sub,
    const float* c_smooth,
    const float* ring_sym,
    size_t dim_x,
    size_t dim_y,
    size_t dim_z,
    size_t k_start,
    double i_start,
    double j_start,
    double s_val,
    double cand_thresh,
    double max_step_px,
    size_t search_r_px,
    size_t max_gap_slices,
    size_t min_pts_slices,
    int pass_idx,
    int has_prior_slope,
    double prior_slope_i,
    double prior_slope_j,
    const double* gauss_base,
    int* out_ks,
    double* out_i,
    double* out_j,
    double* out_r,
    size_t* out_n,
    double* out_shaft_p75
) {
    if (!resp_sub || !c_smooth || !ring_sym || !gauss_base || !out_ks || !out_i || !out_j || !out_r || !out_n || !out_shaft_p75) {
        return -1;
    }
    if (k_start >= dim_z || dim_x <= 2 * search_r_px || dim_y <= 2 * search_r_px) {
        *out_n = 0;
        return 0;
    }

    const size_t patch_w = 2 * search_r_px + 1;
    const size_t slice_stride = dim_y * dim_x;
    double peak_r = std::min(s_val, cand_thresh * 40.0);
    const double base_min_r = cand_thresh * (pass_idx == 1 ? 0.28 : 0.20);

    std::vector<int> fwd_k;
    std::vector<double> fwd_i, fwd_j, fwd_r;
    std::vector<int> bwd_k;
    std::vector<double> bwd_i, bwd_j, bwd_r;
    fwd_k.reserve(dim_z);
    fwd_i.reserve(dim_z);
    fwd_j.reserve(dim_z);
    fwd_r.reserve(dim_z);
    bwd_k.reserve(dim_z);
    bwd_i.reserve(dim_z);
    bwd_j.reserve(dim_z);
    bwd_r.reserve(dim_z);

    const int dirs[2] = {1, -1};
    for (int d_idx = 0; d_idx < 2; ++d_idx) {
        const int direction = dirs[d_idx];
        double curr_i = i_start;
        double curr_j = j_start;
        double vel_i = has_prior_slope ? (prior_slope_i * direction) : 0.0;
        double vel_j = has_prior_slope ? (prior_slope_j * direction) : 0.0;
        size_t gap = 0;
        ptrdiff_t k_cur = static_cast<ptrdiff_t>(k_start) + direction;

        while (k_cur >= 0 && static_cast<size_t>(k_cur) < dim_z) {
            const double pred_i = curr_i + vel_i;
            const double pred_j = curr_j + vel_j;
            const ptrdiff_t ic = static_cast<ptrdiff_t>(std::rint(pred_i));
            const ptrdiff_t jc = static_cast<ptrdiff_t>(std::rint(pred_j));
            if (ic < static_cast<ptrdiff_t>(search_r_px) ||
                ic >= static_cast<ptrdiff_t>(dim_x - search_r_px) ||
                jc < static_cast<ptrdiff_t>(search_r_px) ||
                jc >= static_cast<ptrdiff_t>(dim_y - search_r_px)) {
                break;
            }

            const size_t k_idx = static_cast<size_t>(k_cur);
            const float* resp_sl = resp_sub + k_idx * slice_stride;
            const float* cs_sl = c_smooth + k_idx * slice_stride;
            const float* sym_sl = ring_sym + k_idx * slice_stride;

            double best_eff = -1.0e300;
            size_t best_dy = 0;
            size_t best_dx = 0;
            const size_t j0_p = static_cast<size_t>(jc) - search_r_px;
            const size_t i0_p = static_cast<size_t>(ic) - search_r_px;

            for (size_t dy = 0; dy < patch_w; ++dy) {
                const float* r_row = resp_sl + (j0_p + dy) * dim_x + i0_p;
                const double* g_row = gauss_base + dy * patch_w;
                for (size_t dx = 0; dx < patch_w; ++dx) {
                    const double eff = static_cast<double>(r_row[dx]) * g_row[dx];
                    if (eff > best_eff) {
                        best_eff = eff;
                        best_dy = dy;
                        best_dx = dx;
                    }
                }
            }

            const size_t best_j_idx = j0_p + best_dy;
            const size_t best_i_idx = i0_p + best_dx;
            const double best_r = static_cast<double>(resp_sl[best_j_idx * dim_x + best_i_idx]);
            const double best_c = static_cast<double>(cs_sl[best_j_idx * dim_x + best_i_idx]);
            const double best_sym = static_cast<double>(sym_sl[best_j_idx * dim_x + best_i_idx]);

            double ni = static_cast<double>(best_i_idx);
            double nj = static_cast<double>(best_j_idx);
            const size_t ii0 = best_i_idx;
            const size_t jj0 = best_j_idx;

            if (ii0 >= 1 && ii0 + 1 < dim_x && jj0 >= 1 && jj0 + 1 < dim_y && best_r > 0.5) {
                double w_sum = 0.0;
                double wx_sum = 0.0;
                double wy_sum = 0.0;
                for (int dj = -1; dj <= 1; ++dj) {
                    const float* w_row = resp_sl + static_cast<size_t>(static_cast<ptrdiff_t>(jj0) + dj) * dim_x;
                    for (int di = -1; di <= 1; ++di) {
                        const float rv = w_row[static_cast<size_t>(static_cast<ptrdiff_t>(ii0) + di)];
                        if (rv > 0.0f) {
                            const double w = static_cast<double>(rv);
                            w_sum += w;
                            wx_sum += static_cast<double>(di) * w;
                            wy_sum += static_cast<double>(dj) * w;
                        }
                    }
                }
                if (w_sum > 0.0) {
                    ni = static_cast<double>(ii0) + wx_sum / w_sum;
                    nj = static_cast<double>(jj0) + wy_sum / w_sum;
                }
            }

            const double min_step_r = std::max(base_min_r, 0.045 * peak_r);
            const bool is_dark = (best_r >= min_step_r) ||
                                 (best_sym > 14.0 && best_c < 0.56 * best_sym && best_r >= 0.25 * min_step_r);
            if (is_dark) {
                if (direction == 1) {
                    fwd_k.push_back(static_cast<int>(k_cur));
                    fwd_i.push_back(ni);
                    fwd_j.push_back(nj);
                    fwd_r.push_back(best_r);
                } else {
                    bwd_k.push_back(static_cast<int>(k_cur));
                    bwd_i.push_back(ni);
                    bwd_j.push_back(nj);
                    bwd_r.push_back(best_r);
                }
                peak_r = std::max(peak_r, std::min(best_r, cand_thresh * 42.0));
                const double step_i = std::max(-max_step_px, std::min(max_step_px, ni - curr_i));
                const double step_j = std::max(-max_step_px, std::min(max_step_px, nj - curr_j));
                vel_i = 0.65 * vel_i + 0.35 * step_i;
                vel_j = 0.65 * vel_j + 0.35 * step_j;
                curr_i = ni;
                curr_j = nj;
                gap = 0;
            } else {
                ++gap;
                if (gap > max_gap_slices) {
                    break;
                }
                curr_i = pred_i;
                curr_j = pred_j;
            }
            k_cur += direction;
        }
    }

    const size_t total_raw = bwd_k.size() + 1 + fwd_k.size();
    if (total_raw < min_pts_slices) {
        *out_n = 0;
        *out_shaft_p75 = 0.0;
        return 0;
    }

    size_t idx = 0;
    for (size_t b = bwd_k.size(); b > 0; --b) {
        out_ks[idx] = bwd_k[b - 1];
        out_i[idx] = bwd_i[b - 1];
        out_j[idx] = bwd_j[b - 1];
        out_r[idx] = bwd_r[b - 1];
        ++idx;
    }
    out_ks[idx] = static_cast<int>(k_start);
    out_i[idx] = i_start;
    out_j[idx] = j_start;
    out_r[idx] = s_val;
    ++idx;
    for (size_t f = 0; f < fwd_k.size(); ++f) {
        out_ks[idx] = fwd_k[f];
        out_i[idx] = fwd_i[f];
        out_j[idx] = fwd_j[f];
        out_r[idx] = fwd_r[f];
        ++idx;
    }

    std::vector<double> r_sorted(out_r, out_r + idx);
    std::sort(r_sorted.begin(), r_sorted.end());
    const double pos = 0.75 * static_cast<double>(idx - 1);
    const size_t lo = static_cast<size_t>(pos);
    const size_t hi = std::min(idx - 1, lo + 1);
    const double shaft_p75 = r_sorted[lo] + (pos - static_cast<double>(lo)) * (r_sorted[hi] - r_sorted[lo]);
    *out_shaft_p75 = shaft_p75;

    const double tip_cut = std::max(cand_thresh * 0.35, 0.08 * shaft_p75);
    const double base_cut = std::max(cand_thresh * 0.22, 0.05 * shaft_p75);
    size_t i_lo = 0;
    size_t i_hi = idx;
    while ((i_hi - i_lo) > 2 && out_r[i_hi - 1] < tip_cut) {
        --i_hi;
    }
    while ((i_hi - i_lo) > 2 && out_r[i_lo] < base_cut) {
        ++i_lo;
    }
    const size_t n_trim = i_hi - i_lo;
    if (n_trim < min_pts_slices) {
        *out_n = 0;
        return 0;
    }
    if (i_lo > 0) {
        for (size_t k = 0; k < n_trim; ++k) {
            out_ks[k] = out_ks[i_lo + k];
            out_i[k] = out_i[i_lo + k];
            out_j[k] = out_j[i_lo + k];
            out_r[k] = out_r[i_lo + k];
        }
    }
    *out_n = n_trim;
    return 0;
}

static inline void fast_line_fit_strided_c(
    const double* ras_pts,
    size_t n,
    double& sx,
    double& bx,
    double& sy,
    double& by
) {
    double zm = 0.0, xm = 0.0, ym = 0.0;
    for (size_t i = 0; i < n; ++i) {
        xm += ras_pts[3 * i + 0];
        ym += ras_pts[3 * i + 1];
        zm += ras_pts[3 * i + 2];
    }
    const double inv_n = 1.0 / static_cast<double>(n);
    zm *= inv_n;
    xm *= inv_n;
    ym *= inv_n;
    double denom = 0.0, num_x = 0.0, num_y = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double dz = ras_pts[3 * i + 2] - zm;
        denom += dz * dz;
        num_x += dz * (ras_pts[3 * i + 0] - xm);
        num_y += dz * (ras_pts[3 * i + 1] - ym);
    }
    if (denom < 1e-6) {
        sx = 0.0;
        bx = xm;
        sy = 0.0;
        by = ym;
    } else {
        sx = num_x / denom;
        bx = xm - sx * zm;
        sy = num_y / denom;
        by = ym - sy * zm;
    }
}

int32_t nf_trim_trajectory_inside_out(
    const double* ras_pts,
    const double* arr_pts,
    size_t n,
    double shaft_p75,
    int check_shallow,
    size_t* out_start_idx,
    size_t* out_end_idx
) {
    if (!ras_pts || !arr_pts || !out_start_idx || !out_end_idx) {
        return -1;
    }
    if (n < 6) {
        *out_start_idx = 0;
        *out_end_idx = n;
        return 0;
    }
    if (check_shallow != 0) {
        const double inv_n = 1.0 / static_cast<double>(n);
        double zm = 0.0, xm = 0.0, ym = 0.0;
        for (size_t i = 0; i < n; ++i) {
            xm += ras_pts[3 * i + 0];
            ym += ras_pts[3 * i + 1];
            zm += ras_pts[3 * i + 2];
        }
        zm *= inv_n;
        xm *= inv_n;
        ym *= inv_n;
        double s2 = 0.0, s3 = 0.0, s4 = 0.0;
        double rx1 = 0.0, rx2 = 0.0, ry1 = 0.0, ry2 = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double u = ras_pts[3 * i + 2] - zm;
            const double u2 = u * u;
            const double dx = ras_pts[3 * i + 0] - xm;
            const double dy = ras_pts[3 * i + 1] - ym;
            s2 += u2;
            s3 += u2 * u;
            s4 += u2 * u2;
            rx1 += u2 * dx;
            rx2 += u * dx;
            ry1 += u2 * dy;
            ry2 += u * dy;
        }
        double ax = 0.0, bx = 0.0, cx = 0.0;
        double ay = 0.0, by = 0.0, cy = 0.0;
        if (n >= 7 && s2 > 1e-12) {
            const double a11 = s4 - (s2 * s2) * inv_n;
            const double a12 = s3;
            const double a22 = s2;
            const double det = a11 * a22 - a12 * a12;
            if (std::abs(det) > 1e-12) {
                const double inv_det = 1.0 / det;
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
        } else if (s2 > 1e-6) {
            bx = rx2 / s2;
            by = ry2 / s2;
        }
        double sq_err = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double u = ras_pts[3 * i + 2] - zm;
            const double fx = xm + ax * u * u + bx * u + cx;
            const double fy = ym + ay * u * u + by * u + cy;
            const double dx = ras_pts[3 * i + 0] - fx;
            const double dy = ras_pts[3 * i + 1] - fy;
            sq_err += dx * dx + dy * dy;
        }
        if (std::sqrt(sq_err * inv_n) <= 1.30) {
            *out_start_idx = 0;
            *out_end_idx = n;
            return 0;
        }
    }
    const size_t win_n = std::min(n, std::max<size_t>(5, std::min<size_t>(16, static_cast<size_t>(std::rint(0.55 * static_cast<double>(n))))));
    const size_t max_start = std::max<size_t>(1, static_cast<size_t>(std::rint(0.50 * static_cast<double>(n - win_n))) + 1);

    double best_rmse = 1e9;
    size_t best_i0 = 0;
    double gsx = 0.0, gbx = 0.0, gsy = 0.0, gby = 0.0;
    bool found_fit = false;

    for (size_t s_i = 0; s_i < max_start; ++s_i) {
        const size_t e_i = s_i + win_n;
        if (std::abs(ras_pts[3 * (e_i - 1) + 2] - ras_pts[3 * s_i + 2]) < 3.5) {
            continue;
        }
        double sx, bx, sy, by;
        fast_line_fit_strided_c(ras_pts + 3 * s_i, win_n, sx, bx, sy, by);
        double sq_sum = 0.0;
        for (size_t k = s_i; k < e_i; ++k) {
            const double zk = ras_pts[3 * k + 2];
            const double dx = ras_pts[3 * k + 0] - (sx * zk + bx);
            const double dy = ras_pts[3 * k + 1] - (sy * zk + by);
            sq_sum += dx * dx + dy * dy;
        }
        const double err = std::sqrt(sq_sum / static_cast<double>(win_n));
        if (err < best_rmse) {
            best_rmse = err;
            best_i0 = s_i;
            gsx = sx;
            gbx = bx;
            gsy = sy;
            gby = by;
            found_fit = true;
        }
    }
    if (!found_fit) {
        *out_start_idx = 0;
        *out_end_idx = n;
        return 0;
    }

    size_t end_idx = n;
    size_t bad_streak = 0;
    for (size_t idx = best_i0 + win_n; idx < n; ++idx) {
        const size_t r0 = (idx > 14 && (idx - 14) > best_i0) ? (idx - 14) : best_i0;
        double lsx, lbx, lsy, lby;
        fast_line_fit_strided_c(ras_pts + 3 * r0, idx - r0, lsx, lbx, lsy, lby);
        const double xi = ras_pts[3 * idx + 0];
        const double yi = ras_pts[3 * idx + 1];
        const double zi = ras_pts[3 * idx + 2];
        const double dev_loc = std::hypot(xi - (lsx * zi + lbx), yi - (lsy * zi + lby));
        const double dev_glob = std::hypot(xi - (gsx * zi + gbx), yi - (gsy * zi + gby));
        const double max_glob = std::max(3.25, 0.048 * std::abs(zi - ras_pts[3 * best_i0 + 2]));
        if (dev_loc > 1.85 || dev_glob > max_glob || (dev_loc > 1.35 && arr_pts[4 * idx + 3] < 0.35 * shaft_p75)) {
            ++bad_streak;
            if (bad_streak >= 2 || dev_loc > 2.35 || idx == n - 1) {
                end_idx = idx - bad_streak + 1;
                break;
            }
        } else {
            bad_streak = 0;
        }
    }

    size_t start_idx = 0;
    bad_streak = 0;
    for (ptrdiff_t idx = static_cast<ptrdiff_t>(best_i0) - 1; idx >= 0; --idx) {
        const size_t u_idx = static_cast<size_t>(idx);
        const size_t r1 = std::min(end_idx, u_idx + 15);
        double lsx, lbx, lsy, lby;
        fast_line_fit_strided_c(ras_pts + 3 * (u_idx + 1), r1 - (u_idx + 1), lsx, lbx, lsy, lby);
        const double xi = ras_pts[3 * u_idx + 0];
        const double yi = ras_pts[3 * u_idx + 1];
        const double zi = ras_pts[3 * u_idx + 2];
        const double dev_loc = std::hypot(xi - (lsx * zi + lbx), yi - (lsy * zi + lby));
        const double dev_glob = std::hypot(xi - (gsx * zi + gbx), yi - (gsy * zi + gby));
        const double max_glob = std::max(3.25, 0.048 * std::abs(ras_pts[3 * (best_i0 + win_n - 1) + 2] - zi));
        if (dev_loc > 1.95 || dev_glob > max_glob || (dev_loc > 1.45 && arr_pts[4 * u_idx + 3] < 0.30 * shaft_p75)) {
            ++bad_streak;
            if (bad_streak >= 2 || dev_loc > 2.45 || u_idx == 0) {
                start_idx = u_idx + bad_streak;
                break;
            }
        } else {
            bad_streak = 0;
        }
    }

    *out_start_idx = start_idx;
    *out_end_idx = end_idx;
    return 0;
}

} // extern "C"
