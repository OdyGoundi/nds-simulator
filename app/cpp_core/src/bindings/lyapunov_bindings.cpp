#include "binding_api.hpp"
#include "bindings_common.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace nlds::bindings {

namespace {

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

void qr_accumulate(Vec& Q, Vec& sums_log, size_t n) {
    const auto n_idx = static_cast<Eigen::Index>(n);
    Eigen::Map<Matrix> Qmat(Q.data(), n_idx, n_idx);

    Eigen::HouseholderQR<Matrix> qr(Qmat);
    Matrix Qnew = qr.householderQ() * Matrix::Identity(n_idx, n_idx);
    Matrix R = qr.matrixQR().template triangularView<Eigen::Upper>();

    for (size_t i = 0; i < n; ++i) {
        const double diag = R(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i));
        const double safe_norm = std::max(std::abs(diag), 1e-300);
        sums_log[i] += std::log(safe_norm);
    }

    Qmat = Qnew;
}

void rhs_aug(
    const py::function& rhs,
    const py::object& jac,
    double fd_eps,
    double t,
    const Vec& y,
    Vec& dydt,
    size_t n
) {
    Vec x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = y[i];
    }

    Vec dx;
    call_vec_fn(rhs, t, x, dx);

    Vec J;
    if (!jac.is_none()) {
        call_jac_fn(jac.cast<py::function>(), t, x, J, n);
    } else {
        finite_difference_jacobian(rhs, t, x, fd_eps, J, n);
    }

    Vec Q(n * n);
    for (size_t i = 0; i < n * n; ++i) {
        Q[i] = y[n + i];
    }

    Vec dQ(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += J[i * n + k] * Q[k * n + j];
            }
            dQ[i * n + j] = sum;
        }
    }

    dydt.resize(n + n * n);
    for (size_t i = 0; i < n; ++i) {
        dydt[i] = dx[i];
    }
    for (size_t i = 0; i < n * n; ++i) {
        dydt[n + i] = dQ[i];
    }
}

void rk4_step(
    const py::function& rhs,
    const py::object& jac,
    double fd_eps,
    double t,
    const Vec& y,
    double h,
    Vec& y_out,
    size_t n
) {
    Vec k1, k2, k3, k4, tmp;
    rhs_aug(rhs, jac, fd_eps, t, y, k1, n);

    tmp.resize(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        tmp[i] = y[i] + 0.5 * h * k1[i];
    }
    rhs_aug(rhs, jac, fd_eps, t + 0.5 * h, tmp, k2, n);

    for (size_t i = 0; i < y.size(); ++i) {
        tmp[i] = y[i] + 0.5 * h * k2[i];
    }
    rhs_aug(rhs, jac, fd_eps, t + 0.5 * h, tmp, k3, n);

    for (size_t i = 0; i < y.size(); ++i) {
        tmp[i] = y[i] + h * k3[i];
    }
    rhs_aug(rhs, jac, fd_eps, t + h, tmp, k4, n);

    y_out.resize(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        y_out[i] = y[i] + h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
}

Vec integrate_chunk_rk4(
    const py::function& rhs,
    const py::object& jac,
    double fd_eps,
    double t0,
    const Vec& y0,
    double t1,
    double dt,
    size_t n
) {
    if (dt <= 0.0) {
        throw std::runtime_error("dt must be > 0.");
    }
    double duration = t1 - t0;
    if (duration < 0.0) {
        throw std::runtime_error("t1 must be >= t0.");
    }
    if (duration <= 1e-15) {
        return y0;
    }

    Vec y = y0;
    double t = t0;
    int n_full = static_cast<int>(std::floor(duration / dt));
    for (int i = 0; i < n_full; ++i) {
        Vec y_next;
        rk4_step(rhs, jac, fd_eps, t, y, dt, y_next, n);
        y = y_next;
        t += dt;
    }
    double rem = t1 - t;
    if (rem > 1e-15) {
        Vec y_next;
        rk4_step(rhs, jac, fd_eps, t, y, rem, y_next, n);
        y = y_next;
    }
    return y;
}

}  // namespace

py::dict compute_lyapunov_spectrum_cpp(
    const py::function& rhs,
    const py::array_t<double>& x0,
    double t0,
    double dt,
    double t_transient,
    double t_measure,
    int qr_every_steps,
    const py::object& jac,
    double fd_eps
) {
    Vec x = vec_from_array(x0);
    size_t n = x.size();
    if (n < 1) {
        throw std::runtime_error("x0 must have at least one component.");
    }
    if (dt <= 0.0) {
        throw std::runtime_error("dt must be > 0.");
    }
    if (t_transient < 0.0) {
        throw std::runtime_error("t_transient must be >= 0.");
    }
    if (t_measure <= 0.0) {
        throw std::runtime_error("t_measure must be > 0.");
    }
    if (qr_every_steps < 1) {
        throw std::runtime_error("qr_every_steps must be >= 1.");
    }

    Vec Q(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        Q[i * n + i] = 1.0;
    }

    Vec y_aug(n + n * n);
    for (size_t i = 0; i < n; ++i) {
        y_aug[i] = x[i];
    }
    for (size_t i = 0; i < n * n; ++i) {
        y_aug[n + i] = Q[i];
    }

    double chunk_dt = static_cast<double>(qr_every_steps) * dt;
    double t = t0;

    int n_full = static_cast<int>(std::floor(t_transient / chunk_dt));
    double rem = t_transient - n_full * chunk_dt;

    for (int i = 0; i < n_full; ++i) {
        y_aug = integrate_chunk_rk4(rhs, jac, fd_eps, t, y_aug, t + chunk_dt, dt, n);
        t += chunk_dt;
        for (size_t k = 0; k < n * n; ++k) {
            Q[k] = y_aug[n + k];
        }
        Vec sums_dummy(n, 0.0);
        qr_accumulate(Q, sums_dummy, n);
        for (size_t k = 0; k < n * n; ++k) {
            y_aug[n + k] = Q[k];
        }
    }

    if (rem > 1e-15) {
        y_aug = integrate_chunk_rk4(rhs, jac, fd_eps, t, y_aug, t + rem, dt, n);
        t += rem;
        for (size_t k = 0; k < n * n; ++k) {
            Q[k] = y_aug[n + k];
        }
        Vec sums_dummy(n, 0.0);
        qr_accumulate(Q, sums_dummy, n);
        for (size_t k = 0; k < n * n; ++k) {
            y_aug[n + k] = Q[k];
        }
    }

    Vec sums_log(n, 0.0);
    int n_qr = 0;

    n_full = static_cast<int>(std::floor(t_measure / chunk_dt));
    rem = t_measure - n_full * chunk_dt;

    for (int i = 0; i < n_full; ++i) {
        y_aug = integrate_chunk_rk4(rhs, jac, fd_eps, t, y_aug, t + chunk_dt, dt, n);
        t += chunk_dt;
        for (size_t k = 0; k < n * n; ++k) {
            Q[k] = y_aug[n + k];
        }
        qr_accumulate(Q, sums_log, n);
        n_qr += 1;
        for (size_t k = 0; k < n * n; ++k) {
            y_aug[n + k] = Q[k];
        }
    }

    if (rem > 1e-15) {
        y_aug = integrate_chunk_rk4(rhs, jac, fd_eps, t, y_aug, t + rem, dt, n);
        t += rem;
        for (size_t k = 0; k < n * n; ++k) {
            Q[k] = y_aug[n + k];
        }
        qr_accumulate(Q, sums_log, n);
        n_qr += 1;
        for (size_t k = 0; k < n * n; ++k) {
            y_aug[n + k] = Q[k];
        }
    }

    if (n_qr == 0) {
        throw std::runtime_error("No QR steps performed. Increase t_measure or reduce qr_every_steps.");
    }

    Vec lambdas(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        lambdas[i] = sums_log[i] / t_measure;
    }

    Vec x_final(n);
    for (size_t i = 0; i < n; ++i) {
        x_final[i] = y_aug[i];
    }

    py::dict out;
    out["lambdas"] = array_from_vec(lambdas);
    out["sums_log"] = array_from_vec(sums_log);
    out["t_meas"] = t_measure;
    out["n_qr"] = n_qr;
    out["x_final"] = array_from_vec(x_final);
    return out;
}

}  // namespace nlds::bindings
