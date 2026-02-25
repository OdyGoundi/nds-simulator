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


def _resolve_store_steps(n_steps: int, max_store_steps: Optional[int]) -> int:
    if max_store_steps is None:
        return int(n_steps)
    try:
        n_store = int(max_store_steps)
    except Exception:
        return int(n_steps)
    if n_store <= 0:
        return int(n_steps)
    if int(n_steps) <= 1:
        return 1
    n_store = min(int(n_steps), n_store)
    return max(2, int(n_store))


def _output_stride(n_steps: int, n_store_steps: int) -> int:
    if n_store_steps >= n_steps:
        return 1
    stride = int(np.ceil((int(n_steps) - 1) / float(max(1, int(n_store_steps) - 1))))
    return max(1, int(stride))


# ---------------------------------------------------------------------
# Symplectic Verlet (order 2)
# ---------------------------------------------------------------------
def integrate_system_symplectic_verlet(
    rhs: Any,
    t_span: tuple[float, float],
    y0: Array | list[float],
    t_step: float = 0.01,
    max_steps: Optional[int] = None,
    max_store_steps: Optional[int] = None,
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
    n_store_steps = _resolve_store_steps(int(n_steps), max_store_steps)
    stride = _output_stride(int(n_steps), int(n_store_steps))

    t_out = np.zeros(n_store_steps, dtype=float)
    y_out = np.zeros((n_states, n_store_steps), dtype=float)

    def split_qp(z: Array) -> tuple[Array, Array]:
        return z[:n_q], z[n_q:]

    h = float(t_step)
    q = y0_arr[:n_q].copy()
    p = y0_arr[n_q:].copy()
    ti = float(t0)
    out_idx = 0

    for i in range(int(n_steps)):
        should_store = (
            i == 0
            or i == int(n_steps) - 1
            or stride == 1
            or (i % stride == 0)
        )
        if should_store:
            if out_idx < n_store_steps:
                t_out[out_idx] = ti
                y_out[:n_q, out_idx] = q
                y_out[n_q:, out_idx] = p
                out_idx += 1
            else:
                t_out[n_store_steps - 1] = ti
                y_out[:n_q, n_store_steps - 1] = q
                y_out[n_q:, n_store_steps - 1] = p
        if i == int(n_steps) - 1 or ti >= tf:
            break

        # Kick (h/2)
        p_half = p + 0.5 * h * np.asarray(dp_dt(ti, q), dtype=float)

        # Drift (h)
        q_new = q + h * np.asarray(dq_dt_fn(ti, p_half), dtype=float)

        # Kick (h/2)
        p_new = p_half + 0.5 * h * np.asarray(dp_dt(ti + h, q_new), dtype=float)
        q = q_new
        p = p_new
        ti = ti + h

    if out_idx <= 0:
        t_out[0] = float(t0)
        y_out[:, 0] = y0_arr
        out_idx = 1

    return OdeSolution(
        t=t_out[:out_idx],
        y=y_out[:, :out_idx],
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
    max_store_steps: Optional[int] = None,
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
    n_store_steps = _resolve_store_steps(int(n_steps), max_store_steps)
    stride = _output_stride(int(n_steps), int(n_store_steps))

    t_out = np.zeros(n_store_steps, dtype=float)
    y_out = np.zeros((n_states, n_store_steps), dtype=float)

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
    q, p = split_qp(y0_arr)
    q = np.asarray(q, dtype=float).copy()
    p = np.asarray(p, dtype=float).copy()
    ti = float(t0)
    out_idx = 0

    for i in range(int(n_steps)):
        should_store = (
            i == 0
            or i == int(n_steps) - 1
            or stride == 1
            or (i % stride == 0)
        )
        if should_store:
            if out_idx < n_store_steps:
                t_out[out_idx] = ti
                y_out[:n_q, out_idx] = q
                y_out[n_q:, out_idx] = p
                out_idx += 1
            else:
                t_out[n_store_steps - 1] = ti
                y_out[:n_q, n_store_steps - 1] = q
                y_out[n_q:, n_store_steps - 1] = p
        if i == int(n_steps) - 1 or ti >= tf:
            break

        ti, q, p = verlet_step(ti, q, p, c1 * h)
        ti, q, p = verlet_step(ti, q, p, c2 * h)
        ti, q, p = verlet_step(ti, q, p, c3 * h)

    if out_idx <= 0:
        t_out[0] = float(t0)
        y_out[:, 0] = y0_arr
        out_idx = 1

    return OdeSolution(
        t=t_out[:out_idx],
        y=y_out[:, :out_idx],
        success=True,
        message="Symplectic Forest–Ruth (FR) integration completed.",
    )
