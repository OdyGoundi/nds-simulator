#include "binding_api.hpp"
#include "bindings_common.hpp"

#include <string>
#include <vector>

namespace nlds::bindings {

py::dict integrate_symplectic_fr_cpp(
    const py::function& dq_dt,
    const py::function& dp_dt,
    const py::array_t<double>& y0,
    double t0,
    double tf,
    double dt,
    int max_steps
) {
    Vec y = vec_from_array(y0);
    const size_t n_states = y.size();
    if (n_states % 2 != 0) {
        throw std::runtime_error("State length must be even for symplectic FR.");
    }
    const size_t n_q = n_states / 2;

    int n_steps = compute_n_steps(t0, tf, dt, max_steps);
    std::vector<double> t_vec;
    t_vec.reserve(static_cast<size_t>(n_steps));
    std::vector<double> y_store;
    y_store.reserve(static_cast<size_t>(n_steps) * n_states);

    auto verlet_step = [&](double ti, double h, Vec& q, Vec& p) {
        Vec dp;
        Vec dq;
        call_vec_fn(dp_dt, ti, q, dp);
        for (size_t i = 0; i < n_q; ++i) {
            p[i] += 0.5 * h * dp[i];
        }
        call_vec_fn(dq_dt, ti, p, dq);
        for (size_t i = 0; i < n_q; ++i) {
            q[i] += h * dq[i];
        }
        call_vec_fn(dp_dt, ti + h, q, dp);
        for (size_t i = 0; i < n_q; ++i) {
            p[i] += 0.5 * h * dp[i];
        }
    };

    const double w = 1.0 / (2.0 - std::pow(2.0, 1.0 / 3.0));
    const double c1 = w;
    const double c2 = 1.0 - 2.0 * w;
    const double c3 = w;

    double t = t0;
    for (int i = 0; i < n_steps; ++i) {
        t_vec.push_back(t);
        for (size_t k = 0; k < n_states; ++k) {
            y_store.push_back(y[k]);
        }
        if (t >= tf) {
            break;
        }

        Vec q(n_q), p(n_q);
        for (size_t k = 0; k < n_q; ++k) {
            q[k] = y[k];
            p[k] = y[n_q + k];
        }

        verlet_step(t, c1 * dt, q, p);
        verlet_step(t + c1 * dt, c2 * dt, q, p);
        verlet_step(t + (c1 + c2) * dt, c3 * dt, q, p);

        for (size_t k = 0; k < n_q; ++k) {
            y[k] = q[k];
            y[n_q + k] = p[k];
        }
        t += dt;
    }

    const size_t steps = t_vec.size();
    py::array_t<double> t_arr(static_cast<ssize_t>(steps));
    py::array_t<double> y_arr({static_cast<ssize_t>(n_states), static_cast<ssize_t>(steps)});

    auto t_buf = t_arr.mutable_unchecked<1>();
    for (size_t i = 0; i < steps; ++i) {
        t_buf(static_cast<ssize_t>(i)) = t_vec[i];
    }
    auto y_buf = y_arr.mutable_unchecked<2>();
    for (size_t j = 0; j < steps; ++j) {
        for (size_t i = 0; i < n_states; ++i) {
            y_buf(static_cast<ssize_t>(i), static_cast<ssize_t>(j)) = y_store[j * n_states + i];
        }
    }

    py::dict out;
    out["t"] = t_arr;
    out["y"] = y_arr;
    out["success"] = true;
    out["message"] = std::string("Symplectic Forest-Ruth integration completed.");
    return out;
}

}  // namespace nlds::bindings
