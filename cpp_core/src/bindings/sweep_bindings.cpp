#include "binding_api.hpp"
#include "bindings_common.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

namespace nlds::bindings {

namespace {

enum class SystemKind {
    Lorenz,
    Rossler,
    HenonHeiles,
};

std::string to_lower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

bool parse_system_key(const std::string& key, SystemKind& out) {
    const std::string k = to_lower(key);
    if (k == "lorenz") {
        out = SystemKind::Lorenz;
        return true;
    }
    if (k == "rossler") {
        out = SystemKind::Rossler;
        return true;
    }
    if (k == "henon_heiles") {
        out = SystemKind::HenonHeiles;
        return true;
    }
    return false;
}

size_t system_dim(SystemKind kind) {
    switch (kind) {
        case SystemKind::Lorenz: return 3;
        case SystemKind::Rossler: return 3;
        case SystemKind::HenonHeiles: return 4;
    }
    return 0;
}

size_t system_param_count(SystemKind kind) {
    switch (kind) {
        case SystemKind::Lorenz: return 3;
        case SystemKind::Rossler: return 3;
        case SystemKind::HenonHeiles: return 1;
    }
    return 0;
}

void rhs_eval(SystemKind kind, const Vec& y, const Vec& params, Vec& out) {
    if (kind == SystemKind::Lorenz) {
        const double sigma = params[0];
        const double rho = params[1];
        const double beta = params[2];
        const double x = y[0];
        const double yv = y[1];
        const double z = y[2];
        out.resize(3);
        out[0] = sigma * (yv - x);
        out[1] = x * (rho - z) - yv;
        out[2] = x * yv - beta * z;
        return;
    }
    if (kind == SystemKind::Rossler) {
        const double a = params[0];
        const double b = params[1];
        const double c = params[2];
        const double x = y[0];
        const double yv = y[1];
        const double z = y[2];
        out.resize(3);
        out[0] = -yv - z;
        out[1] = x + a * yv;
        out[2] = b + z * (x - c);
        return;
    }
    if (kind == SystemKind::HenonHeiles) {
        const double lam = params[0];
        const double q1 = y[0];
        const double q2 = y[1];
        const double p1 = y[2];
        const double p2 = y[3];
        out.resize(4);
        out[0] = p1;
        out[1] = p2;
        out[2] = -(q1 + 2.0 * lam * q1 * q2);
        out[3] = -(q2 + lam * (q1 * q1 - q2 * q2));
        return;
    }
    throw std::runtime_error("Unknown system kind.");
}

void rk4_step(SystemKind kind, const Vec& params, double t, const Vec& y, double h, Vec& y_out) {
    const size_t n = y.size();
    Vec k1, k2, k3, k4, tmp;
    rhs_eval(kind, y, params, k1);

    tmp.resize(n);
    for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + 0.5 * h * k1[i];
    rhs_eval(kind, tmp, params, k2);

    for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + 0.5 * h * k2[i];
    rhs_eval(kind, tmp, params, k3);

    for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + h * k3[i];
    rhs_eval(kind, tmp, params, k4);

    y_out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        y_out[i] = y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

std::vector<double> frange_inclusive(double start, double stop, double step) {
    if (step <= 0.0) {
        throw std::runtime_error("step must be > 0.");
    }
    if (stop < start - 1e-12) {
        return {};
    }
    const int n = static_cast<int>(std::floor((stop - start) / step + 1e-12)) + 1;
    std::vector<double> vals;
    vals.reserve(static_cast<size_t>(std::max(0, n)));
    for (int i = 0; i < n; ++i) {
        const double v = start + step * static_cast<double>(i);
        if (v <= stop + 1e-12) {
            vals.push_back(v);
        }
    }
    return vals;
}

}  // namespace

py::dict sweep_poincare_rk4_cpp(
    const std::string& system_key,
    const py::array_t<double>& base_params_arr,
    int sweep_param_index,
    const py::array_t<double>& y0_arr,
    double t0,
    double tf,
    double dt,
    double sweep_start,
    double sweep_stop,
    double sweep_step,
    int section_index,
    double section_value,
    int direction,
    const std::string& method,
    double tol,
    int transient_steps,
    int output_index,
    bool warm_start,
    int max_hits
) {
    SystemKind kind;
    if (!parse_system_key(system_key, kind)) {
        throw std::runtime_error("Unsupported system_key for C++ sweep.");
    }
    const size_t n = system_dim(kind);
    const size_t n_params = system_param_count(kind);

    Vec base_params = vec_from_array(base_params_arr);
    if (base_params.size() != n_params) {
        throw std::runtime_error("base_params size mismatch.");
    }
    if (sweep_param_index < 0 || static_cast<size_t>(sweep_param_index) >= n_params) {
        throw std::runtime_error("sweep_param_index out of range.");
    }

    Vec y0 = vec_from_array(y0_arr);
    if (y0.size() != n) {
        throw std::runtime_error("y0 dimension mismatch for system.");
    }

    if (dt <= 0.0) {
        throw std::runtime_error("dt must be > 0.");
    }
    if (tf < t0) {
        throw std::runtime_error("tf must be >= t0.");
    }
    if (direction != -1 && direction != 0 && direction != 1) {
        throw std::runtime_error("direction must be one of: -1, 0, +1.");
    }
    if (section_index < 0 || static_cast<size_t>(section_index) >= n) {
        throw std::runtime_error("section_index out of range.");
    }
    if (output_index < 0 || static_cast<size_t>(output_index) >= n) {
        throw std::runtime_error("output_index out of range.");
    }

    const std::string method_lc = to_lower(method);
    const bool method_crossing = (method_lc == "crossing");
    const bool method_slab = (method_lc == "slab");
    if (!method_crossing && !method_slab) {
        throw std::runtime_error("method must be 'crossing' or 'slab'.");
    }

    const int keep_hits = std::max(1, max_hits);

    std::vector<double> out_param;
    std::vector<double> out_t;
    std::vector<double> out_y;

    std::vector<double> param_vals = frange_inclusive(sweep_start, sweep_stop, sweep_step);
    out_param.reserve(param_vals.size() * static_cast<size_t>(keep_hits));
    out_t.reserve(param_vals.size() * static_cast<size_t>(keep_hits));
    out_y.reserve(param_vals.size() * static_cast<size_t>(keep_hits));

    Vec y0_base = y0;
    Vec y0_curr = y0;

    for (double pv : param_vals) {
        Vec params = base_params;
        params[static_cast<size_t>(sweep_param_index)] = pv;

        Vec y = warm_start ? y0_curr : y0_base;
        double t = t0;

        bool have_prev = false;
        double prev_t = 0.0;
        double prev_ds = 0.0;
        Vec prev_y;

        std::deque<double> hits_t;
        std::deque<double> hits_y;

        const int n_steps = compute_n_steps(t0, tf, dt, 0);

        for (int step_idx = 0; step_idx < n_steps; ++step_idx) {
            if (step_idx >= transient_steps) {
                const double ds = y[static_cast<size_t>(section_index)] - section_value;
                if (!have_prev) {
                    if (method_slab && direction == 0) {
                        if (std::abs(ds) <= tol) {
                            hits_t.push_back(t);
                            hits_y.push_back(y[static_cast<size_t>(output_index)]);
                            if (static_cast<int>(hits_t.size()) > keep_hits) {
                                hits_t.pop_front();
                                hits_y.pop_front();
                            }
                        }
                    }
                    have_prev = true;
                    prev_t = t;
                    prev_ds = ds;
                    prev_y = y;
                } else {
                    if (method_crossing) {
                        bool cond = false;
                        if (direction == 1) {
                            cond = (prev_ds < 0.0) && (ds >= 0.0);
                        } else if (direction == -1) {
                            cond = (prev_ds > 0.0) && (ds <= 0.0);
                        } else {
                            cond = (prev_ds == 0.0) || (ds == 0.0) || (prev_ds * ds < 0.0);
                        }
                        if (cond) {
                            double alpha = 1.0;
                            const double denom = (ds - prev_ds);
                            if (denom != 0.0) {
                                alpha = (0.0 - prev_ds) / denom;
                                if (alpha < 0.0) alpha = 0.0;
                                if (alpha > 1.0) alpha = 1.0;
                            }
                            const double th = prev_t + alpha * (t - prev_t);
                            const double y_prev = prev_y[static_cast<size_t>(output_index)];
                            const double y_curr = y[static_cast<size_t>(output_index)];
                            const double yh = y_prev + alpha * (y_curr - y_prev);
                            hits_t.push_back(th);
                            hits_y.push_back(yh);
                            if (static_cast<int>(hits_t.size()) > keep_hits) {
                                hits_t.pop_front();
                                hits_y.pop_front();
                            }
                        }
                    } else {
                        if (std::abs(ds) <= tol) {
                            if (direction == 0) {
                                hits_t.push_back(t);
                                hits_y.push_back(y[static_cast<size_t>(output_index)]);
                            } else {
                                const double dt_loc = t - prev_t;
                                if (dt_loc != 0.0) {
                                    const double deriv = (ds - prev_ds) / dt_loc;
                                    if ((direction == 1 && deriv > 0.0) || (direction == -1 && deriv < 0.0)) {
                                        hits_t.push_back(t);
                                        hits_y.push_back(y[static_cast<size_t>(output_index)]);
                                    }
                                }
                            }
                            if (static_cast<int>(hits_t.size()) > keep_hits) {
                                hits_t.pop_front();
                                hits_y.pop_front();
                            }
                        }
                    }
                    prev_t = t;
                    prev_ds = ds;
                    prev_y = y;
                }
            }

            if (t >= tf) {
                break;
            }

            Vec y_next;
            rk4_step(kind, params, t, y, dt, y_next);
            y = y_next;
            t += dt;
        }

        if (warm_start) {
            y0_curr = y;
        } else {
            y0_curr = y0_base;
        }

        const size_t m = hits_t.size();
        for (size_t i = 0; i < m; ++i) {
            out_param.push_back(pv);
            out_t.push_back(hits_t[i]);
            out_y.push_back(hits_y[i]);
        }
    }

    py::dict out;
    out["param_vals"] = array_from_vec(out_param);
    out["t_hit"] = array_from_vec(out_t);
    out["y_hit"] = array_from_vec(out_y);
    return out;
}

}  // namespace nlds::bindings
