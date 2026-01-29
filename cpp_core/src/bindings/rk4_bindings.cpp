#include "binding_api.hpp"
#include "bindings_common.hpp"

#include <string>
#include <vector>

namespace nlds::bindings {

py::dict integrate_rk4_cpp(
    const py::function& rhs,
    const py::array_t<double>& y0,
    double t0,
    double tf,
    double dt,
    int max_steps
) {
    Vec y = vec_from_array(y0);
    const size_t n = y.size();
    int n_steps = compute_n_steps(t0, tf, dt, max_steps);
    std::vector<double> t_vec;
    t_vec.reserve(static_cast<size_t>(n_steps));
    std::vector<double> y_store;
    y_store.reserve(static_cast<size_t>(n_steps) * n);

    double t = t0;
    for (int i = 0; i < n_steps; ++i) {
        t_vec.push_back(t);
        for (size_t k = 0; k < n; ++k) {
            y_store.push_back(y[k]);
        }
        if (t >= tf) {
            break;
        }
        Vec y_next;
        auto rhs_simple = [&](double tt, const Vec& yy, Vec& out) {
            call_vec_fn(rhs, tt, yy, out);
        };

        Vec k1, k2, k3, k4, tmp;
        rhs_simple(t, y, k1);
        tmp.resize(n);
        for (size_t k = 0; k < n; ++k) tmp[k] = y[k] + 0.5 * dt * k1[k];
        rhs_simple(t + 0.5 * dt, tmp, k2);
        for (size_t k = 0; k < n; ++k) tmp[k] = y[k] + 0.5 * dt * k2[k];
        rhs_simple(t + 0.5 * dt, tmp, k3);
        for (size_t k = 0; k < n; ++k) tmp[k] = y[k] + dt * k3[k];
        rhs_simple(t + dt, tmp, k4);
        y_next.resize(n);
        for (size_t k = 0; k < n; ++k) {
            y_next[k] = y[k] + dt * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k]) / 6.0;
        }
        y = y_next;
        t += dt;
    }

    const size_t steps = t_vec.size();
    py::array_t<double> t_arr(static_cast<ssize_t>(steps));
    py::array_t<double> y_arr({static_cast<ssize_t>(n), static_cast<ssize_t>(steps)});

    auto t_buf = t_arr.mutable_unchecked<1>();
    for (size_t i = 0; i < steps; ++i) {
        t_buf(static_cast<ssize_t>(i)) = t_vec[i];
    }
    auto y_buf = y_arr.mutable_unchecked<2>();
    for (size_t j = 0; j < steps; ++j) {
        for (size_t i = 0; i < n; ++i) {
            y_buf(static_cast<ssize_t>(i), static_cast<ssize_t>(j)) = y_store[j * n + i];
        }
    }

    py::dict out;
    out["t"] = t_arr;
    out["y"] = y_arr;
    out["success"] = true;
    out["message"] = std::string("RK4 integration completed.");
    return out;
}

}  // namespace nlds::bindings
