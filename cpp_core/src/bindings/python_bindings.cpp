#include <pybind11/pybind11.h>

#include "binding_api.hpp"

namespace py = pybind11;

PYBIND11_MODULE(nlds_cpp, m) {
    m.doc() = "C++ backend for NLDS simulator (RK4, Lyapunov, Forest-Ruth).";

    m.def(
        "compute_lyapunov_spectrum",
        &nlds::bindings::compute_lyapunov_spectrum_cpp,
        py::arg("rhs"),
        py::arg("x0"),
        py::arg("t0"),
        py::arg("dt"),
        py::arg("t_transient"),
        py::arg("t_measure"),
        py::arg("qr_every_steps") = 1,
        py::arg("jac") = py::none(),
        py::arg("fd_eps") = 1e-8
    );

    m.def(
        "integrate_rk4",
        &nlds::bindings::integrate_rk4_cpp,
        py::arg("rhs"),
        py::arg("y0"),
        py::arg("t0"),
        py::arg("tf"),
        py::arg("dt"),
        py::arg("max_steps") = 0
    );

    m.def(
        "integrate_symplectic_fr",
        &nlds::bindings::integrate_symplectic_fr_cpp,
        py::arg("dq_dt"),
        py::arg("dp_dt"),
        py::arg("y0"),
        py::arg("t0"),
        py::arg("tf"),
        py::arg("dt"),
        py::arg("max_steps") = 0
    );

    m.def(
        "sweep_poincare_rk4",
        &nlds::bindings::sweep_poincare_rk4_cpp,
        py::arg("system_key"),
        py::arg("base_params"),
        py::arg("sweep_param_index"),
        py::arg("y0"),
        py::arg("t0"),
        py::arg("tf"),
        py::arg("dt"),
        py::arg("sweep_start"),
        py::arg("sweep_stop"),
        py::arg("sweep_step"),
        py::arg("section_index"),
        py::arg("section_value"),
        py::arg("direction"),
        py::arg("method"),
        py::arg("tol"),
        py::arg("transient_steps"),
        py::arg("output_index"),
        py::arg("warm_start"),
        py::arg("max_hits")
    );
}
