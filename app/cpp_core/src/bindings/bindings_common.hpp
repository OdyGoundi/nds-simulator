#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace nlds::bindings {

namespace py = pybind11;

using Vec = std::vector<double>;

inline Vec vec_from_array(const py::array_t<double>& arr) {
    if (arr.ndim() != 1) {
        throw std::runtime_error("Expected 1D array.");
    }
    Vec v(static_cast<size_t>(arr.shape(0)));
    auto buf = arr.unchecked<1>();
    for (ssize_t i = 0; i < arr.shape(0); ++i) {
        v[static_cast<size_t>(i)] = buf(i);
    }
    return v;
}

inline py::array_t<double> array_from_vec(const Vec& v) {
    py::array_t<double> arr(static_cast<ssize_t>(v.size()));
    auto buf = arr.mutable_unchecked<1>();
    for (size_t i = 0; i < v.size(); ++i) {
        buf(static_cast<ssize_t>(i)) = v[i];
    }
    return arr;
}

inline void call_vec_fn(const py::function& fn, double t, const Vec& x, Vec& out) {
    py::array_t<double> x_arr(static_cast<ssize_t>(x.size()));
    auto x_buf = x_arr.mutable_unchecked<1>();
    for (size_t i = 0; i < x.size(); ++i) {
        x_buf(static_cast<ssize_t>(i)) = x[i];
    }

    py::object res = fn(t, x_arr);
    py::array arr = py::array::ensure(res);
    if (!arr) {
        throw std::runtime_error("Callable did not return an array.");
    }
    if (arr.ndim() != 1 || static_cast<size_t>(arr.shape(0)) != x.size()) {
        throw std::runtime_error("Callable returned wrong shape.");
    }

    auto arr_t = py::array_t<double>(arr);
    auto buf = arr_t.unchecked<1>();
    out.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = buf(static_cast<ssize_t>(i));
    }
}

inline void call_jac_fn(const py::function& fn, double t, const Vec& x, Vec& J, size_t n) {
    py::array_t<double> x_arr(static_cast<ssize_t>(x.size()));
    auto x_buf = x_arr.mutable_unchecked<1>();
    for (size_t i = 0; i < x.size(); ++i) {
        x_buf(static_cast<ssize_t>(i)) = x[i];
    }

    py::object res = fn(t, x_arr);
    py::array arr = py::array::ensure(res);
    if (!arr) {
        throw std::runtime_error("Jacobian callable did not return an array.");
    }
    if (arr.ndim() != 2 || static_cast<size_t>(arr.shape(0)) != n || static_cast<size_t>(arr.shape(1)) != n) {
        throw std::runtime_error("Jacobian returned wrong shape.");
    }

    auto arr_t = py::array_t<double>(arr);
    auto buf = arr_t.unchecked<2>();
    J.resize(n * n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            J[i * n + j] = buf(static_cast<ssize_t>(i), static_cast<ssize_t>(j));
        }
    }
}

inline void finite_difference_jacobian(
    const py::function& rhs,
    double t,
    const Vec& x,
    double eps,
    Vec& J,
    size_t n
) {
    Vec f0;
    call_vec_fn(rhs, t, x, f0);
    if (f0.size() != n) {
        throw std::runtime_error("rhs returned wrong shape.");
    }

    J.assign(n * n, 0.0);
    Vec x_pert = x;
    Vec fp;
    Vec fm;

    for (size_t j = 0; j < n; ++j) {
        x_pert[j] = x[j] + eps;
        call_vec_fn(rhs, t, x_pert, fp);
        x_pert[j] = x[j] - eps;
        call_vec_fn(rhs, t, x_pert, fm);
        x_pert[j] = x[j];

        for (size_t i = 0; i < n; ++i) {
            J[i * n + j] = (fp[i] - fm[i]) / (2.0 * eps);
        }
    }
}

inline int compute_n_steps(double t0, double tf, double dt, int max_steps) {
    if (dt <= 0.0) {
        throw std::runtime_error("t_step must be positive.");
    }
    if (max_steps > 0) {
        return max_steps;
    }
    return static_cast<int>(std::floor((tf - t0) / dt)) + 1;
}

}  // namespace nlds::bindings
