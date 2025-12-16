from __future__ import annotations

from typing import Any, Callable, Optional, TypeAlias, cast
import numpy as np

from .solver import OdeSolution, _compute_n_steps


# ---------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------
Array: TypeAlias = np.ndarray
DQDT: TypeAlias = Callable[[float, Array], Array]   # dq_dt(t, p) -> dq/dt
DPDT: TypeAlias = Callable[[float, Array], Array]   # dp_dt(t, q) -> dp/dt


# ---------------------------------------------------------------------
# Symplectic Verlet (order 2)
# ---------------------------------------------------------------------
def integrate_system_symplectic_verlet(
    rhs: Any,
    t_span: tuple[float, float],
    y0: Array | list[float],
    t_step: float = 0.01,
    max_steps: Optional[int] = None,
    **options: Any,
) -> OdeSolution:
    """
    Symplectic Verlet (2nd order) for separable Hamiltonians H(q,p)=T(p)+V(q).

    State ordering:
        y = [q1, q2, ..., p1, p2, ...]

    Required kwargs
    ----------------
    dp_dt : callable
        dp_dt(t, q) -> dp/dt evaluated at (t, q)

    Optional kwargs
    ----------------
    dq_dt : callable
        dq_dt(t, p) -> dq/dt evaluated at (t, p)
        Default: dq_dt(t, p) = p
    n_q : int
        Number of q variables. Default: inferred as len(y0)//2
    """
    t0, tf = float(t_span[0]), float(t_span[1])
    y0_arr = np.asarray(y0, dtype=float)

    if y0_arr.ndim != 1:
        raise ValueError("y0 must be a 1D state vector.")

    n_states = int(y0_arr.size)
    n_q = int(options.get("n_q", n_states // 2))

    if n_q <= 0:
        raise ValueError("n_q must be a positive integer.")
    if n_states != 2 * n_q:
        raise ValueError(
            f"Expected len(y0)=2*n_q={2*n_q}, got {n_states}. "
            "State must be ordered as [q..., p...]."
        )

    dp_dt_any = options.get("dp_dt")
    if dp_dt_any is None:
        raise ValueError("integrate_system_symplectic_verlet requires dp_dt(t, q).")
    if not callable(dp_dt_any):
        raise TypeError("dp_dt must be callable: dp_dt(t, q) -> dp/dt.")

    dp_dt: DPDT = cast(DPDT, dp_dt_any)

    dq_dt_any = options.get("dq_dt")
    if dq_dt_any is None:
        def dq_dt(t: float, p: Array) -> Array:
            return p
        dq_dt_fn: DQDT = dq_dt
    else:
        if not callable(dq_dt_any):
            raise TypeError("dq_dt must be callable: dq_dt(t, p) -> dq/dt.")
        dq_dt_fn = cast(DQDT, dq_dt_any)

    n_steps = _compute_n_steps(t0, tf, t_step, max_steps)
    t = np.linspace(t0, tf, n_steps)

    y = np.zeros((n_states, n_steps), dtype=float)
    y[:, 0] = y0_arr

    def split_qp(z: Array) -> tuple[Array, Array]:
        return z[:n_q], z[n_q:]

    h = float(t_step)

    for i in range(1, n_steps):
        ti = float(t[i - 1])
        q, p = split_qp(y[:, i - 1])

        # Kick (h/2)
        p_half = p + 0.5 * h * np.asarray(dp_dt(ti, q), dtype=float)

        # Drift (h)
        q_new = q + h * np.asarray(dq_dt_fn(ti, p_half), dtype=float)

        # Kick (h/2)
        p_new = p_half + 0.5 * h * np.asarray(dp_dt(ti + h, q_new), dtype=float)

        y[:, i] = np.concatenate((q_new, p_new))

    return OdeSolution(
        t=t,
        y=y,
        success=True,
        message="Symplectic Verlet integration completed.",
    )


# ---------------------------------------------------------------------
# Symplectic Forest–Ruth (order 4)
# ---------------------------------------------------------------------
def integrate_system_symplectic_fr(
    rhs: Any,
    t_span: tuple[float, float],
    y0: Array | list[float],
    t_step: float = 0.01,
    max_steps: Optional[int] = None,
    **options: Any,
) -> OdeSolution:
    """
    Symplectic Forest–Ruth (4th order) for separable Hamiltonians H(q,p)=T(p)+V(q).

    Implemented as:
        S4(h) = S2(w*h) o S2((1-2w)*h) o S2(w*h),
    where w = 1 / (2 - 2^(1/3)).
    """
    t0, tf = float(t_span[0]), float(t_span[1])
    y0_arr = np.asarray(y0, dtype=float)

    if y0_arr.ndim != 1:
        raise ValueError("y0 must be a 1D state vector.")

    n_states = int(y0_arr.size)
    n_q = int(options.get("n_q", n_states // 2))

    if n_q <= 0:
        raise ValueError("n_q must be a positive integer.")
    if n_states != 2 * n_q:
        raise ValueError(
            f"Expected len(y0)=2*n_q={2*n_q}, got {n_states}. "
            "State must be ordered as [q..., p...]."
        )

    dp_dt_any = options.get("dp_dt")
    if dp_dt_any is None:
        raise ValueError("integrate_system_symplectic_fr requires dp_dt(t, q).")
    if not callable(dp_dt_any):
        raise TypeError("dp_dt must be callable: dp_dt(t, q) -> dp/dt.")

    dp_dt: DPDT = cast(DPDT, dp_dt_any)

    dq_dt_any = options.get("dq_dt")
    if dq_dt_any is None:
        def dq_dt(t: float, p: Array) -> Array:
            return p
        dq_dt_fn: DQDT = dq_dt
    else:
        if not callable(dq_dt_any):
            raise TypeError("dq_dt must be callable: dq_dt(t, p) -> dq/dt.")
        dq_dt_fn = cast(DQDT, dq_dt_any)

    n_steps = _compute_n_steps(t0, tf, t_step, max_steps)
    t = np.linspace(t0, tf, n_steps)

    y = np.zeros((n_states, n_steps), dtype=float)
    y[:, 0] = y0_arr

    def split_qp(z: Array) -> tuple[Array, Array]:
        return z[:n_q], z[n_q:]

    def verlet_step(ti: float, q: Array, p: Array, h: float) -> tuple[float, Array, Array]:
        p_half = p + 0.5 * h * np.asarray(dp_dt(ti, q), dtype=float)
        q_new = q + h * np.asarray(dq_dt_fn(ti, p_half), dtype=float)
        p_new = p_half + 0.5 * h * np.asarray(dp_dt(ti + h, q_new), dtype=float)
        return ti + h, q_new, p_new

    w = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    c1, c2, c3 = w, 1.0 - 2.0 * w, w
    h = float(t_step)

    for i in range(1, n_steps):
        ti = float(t[i - 1])
        q, p = split_qp(y[:, i - 1])

        ti, q, p = verlet_step(ti, q, p, c1 * h)
        ti, q, p = verlet_step(ti, q, p, c2 * h)
        ti, q, p = verlet_step(ti, q, p, c3 * h)

        y[:, i] = np.concatenate((q, p))

    return OdeSolution(
        t=t,
        y=y,
        success=True,
        message="Symplectic Forest–Ruth (FR) integration completed.",
    )
