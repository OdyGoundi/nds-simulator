import numpy as np
from scipy.integrate import solve_ivp

class OdeSolution:
    """
    Simple container class for the solution of an ODE system.

    Attributes
    ----------
    t : np.ndarray, shape (n_points,)
        Time at each step.
    y : np.ndarray, shape (n_states, n_points)
        System state at each time.
    success : bool
        Whether the integration was successful (as defined by the solver).
    message : str
        Message from the solver (useful for debugging).
    """
    def __init__(self, t, y, success=True, message=""):
        self.t = t
        self.y = y
        self.success = bool(success)
        self.message = str(message)


def _compute_n_steps(t0, tf, t_step, max_steps):
    """
    helper function to compute number of steps.
    """
    if t_step <= 0:
        raise ValueError("t_step must be positive.")

    if max_steps is not None:
        return int(max_steps)

    return int(np.floor((tf - t0) / t_step)) + 1


def _resolve_store_steps(n_steps, max_store_steps):
    """
    Number of trajectory samples to keep in memory.
    """
    if max_store_steps is None:
        return int(n_steps)
    try:
        n_store = int(max_store_steps)
    except Exception:
        return int(n_steps)
    if n_store <= 0:
        return int(n_steps)
    n_steps_i = int(n_steps)
    if n_steps_i <= 1:
        return 1
    n_store = min(n_store, n_steps_i)
    return max(2, n_store)


def _output_stride(n_steps: int, n_store_steps: int) -> int:
    """Compute stride for downsampling output when storage is limited."""
    if n_store_steps >= n_steps:
        return 1
    stride = int(np.ceil((int(n_steps) - 1) / float(max(1, int(n_store_steps) - 1))))
    return max(1, int(stride))


class StoreBuffer:
    """Helper for managing output buffering in integrators."""

    def __init__(self, n_states: int, n_steps: int, n_store_steps: int):
        self.n_states = int(n_states)
        self.n_steps = int(n_steps)
        self.n_store_steps = int(n_store_steps)
        self.stride = _output_stride(n_steps, n_store_steps)

        self.t_out = np.zeros(n_store_steps, dtype=float)
        self.y_out = np.zeros((n_states, n_store_steps), dtype=float)
        self.out_idx = 0

    def should_store(self, i: int) -> bool:
        """Check if step i should be stored."""
        return (
            i == 0
            or i == self.n_steps - 1
            or self.stride == 1
            or (i % self.stride == 0)
        )

    def commit(self, i: int, t: float, y: np.ndarray) -> None:
        """Store (t, y) at the current position, with cap handling."""
        if self.out_idx < self.n_store_steps:
            self.t_out[self.out_idx] = t
            self.y_out[:, self.out_idx] = y
            self.out_idx += 1
        else:
            # Keep final endpoint even if we hit storage cap
            self.t_out[self.n_store_steps - 1] = t
            self.y_out[:, self.n_store_steps - 1] = y

    def finalize(self, t0: float, y0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return sliced (t, y) arrays; handle empty case."""
        if self.out_idx <= 0:
            self.t_out[0] = float(t0)
            self.y_out[:, 0] = y0
            self.out_idx = 1
        return self.t_out[:self.out_idx], self.y_out[:, :self.out_idx]


def integrate_system(rhs, t_span, y0, t_step=0.01, max_steps=None, max_store_steps=None, **solve_options):
    """
    Integrates an ODE system using scipy.solve_ivp (variable step but with t_eval).

    Parameters
    ----------
    rhs : function
        Right-hand side f(t, y) -> dy/dt.
    t_span : tuple
        (t_start, t_final).
    y0 : list or array-like
        Initial state.
    t_step : float
        Time step for evaluation points (t_eval).
    max_steps : int or None
        If given, overrides and sets the number of t_eval points.
    **solve_options :
        Additional options passed to solve_ivp.

    Returns
    -------
    OdeSolution
        Object with fields (t, y, success, message).
    """
    t0, tf = t_span

    # Always convert y0 to numpy array for consistency
    y0_arr = np.array(y0, dtype=float)

    n_steps = _compute_n_steps(t0, tf, t_step, max_steps)
    n_store_steps = _resolve_store_steps(n_steps, max_store_steps)
    t_eval = np.linspace(t0, tf, n_store_steps)

    sol = solve_ivp(
        rhs,
        (t0, tf),
        y0_arr,
        t_eval=t_eval,
        **solve_options
    )

    return OdeSolution(
        t=sol.t,
        y=sol.y,
        success=sol.success,
        message=sol.message,
    )


def integrate_system_rk4(rhs, t_span, y0, t_step=0.01, max_steps=None, max_store_steps=None):
    """
    Fixed-step RK4 integrator.

    Parameters
    ----------
    rhs : function
        right-hand side f(t, y) -> dy/dt.
    t_span : tuple
        (t_start, t_final).
    y0 : list or array-like
        initial state
    t_step : float
        step size
    max_steps : int or None
        if given, overrides t_step and sets the number of steps.

    Returns
    -------
    OdeSolution
        Object with fields (t, y, success, message).
    """
    t0, tf = t_span

    # Convert to numpy array (as in the solve_ivp solver)
    y0_arr = np.array(y0, dtype=float)

    n_steps = _compute_n_steps(t0, tf, t_step, max_steps)
    n_store_steps = _resolve_store_steps(n_steps, max_store_steps)
    n_states = y0_arr.size

    buf = StoreBuffer(n_states, n_steps, n_store_steps)
    t_curr = float(t0)
    y_curr = y0_arr.copy()

    for i in range(n_steps):
        if buf.should_store(i):
            buf.commit(i, t_curr, y_curr)

        if i == n_steps - 1 or t_curr >= tf:
            break

        k1 = rhs(t_curr, y_curr)
        k2 = rhs(t_curr + 0.5 * t_step, y_curr + 0.5 * t_step * k1)
        k3 = rhs(t_curr + 0.5 * t_step, y_curr + 0.5 * t_step * k2)
        k4 = rhs(t_curr + t_step, y_curr + t_step * k3)

        y_curr = y_curr + (t_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t_curr = t_curr + t_step

    t_store, y_store = buf.finalize(t0, y0_arr)
    return OdeSolution(
        t=t_store,
        y=y_store,
        success=True,
        message="RK4 integration completed.",
    )
