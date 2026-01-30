#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <string>

namespace nlds::bindings {

pybind11::dict compute_lyapunov_spectrum_cpp(
    const pybind11::function& rhs,
    const pybind11::array_t<double>& x0,
    double t0,
    double dt,
    double t_transient,
    double t_measure,
    int qr_every_steps,
    const pybind11::object& jac,
    double fd_eps
);

pybind11::dict integrate_rk4_cpp(
    const pybind11::function& rhs,
    const pybind11::array_t<double>& y0,
    double t0,
    double tf,
    double dt,
    int max_steps
);

pybind11::dict integrate_symplectic_fr_cpp(
    const pybind11::function& dq_dt,
    const pybind11::function& dp_dt,
    const pybind11::array_t<double>& y0,
    double t0,
    double tf,
    double dt,
    int max_steps
);

pybind11::dict sweep_poincare_rk4_cpp(
    const std::string& system_key,
    const pybind11::array_t<double>& base_params,
    int sweep_param_index,
    const pybind11::array_t<double>& y0,
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
);

}  // namespace nlds::bindings
