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

    # Allocate bounded output arrays.
    t_store = np.zeros(n_store_steps)
    y_store = np.zeros((n_states, n_store_steps))

    stride = 1
    if n_store_steps < n_steps:
        stride = int(np.ceil((n_steps - 1) / float(n_store_steps - 1)))
        stride = max(1, stride)

    store_idx = 0
    t_curr = float(t0)
    y_curr = y0_arr.copy()

    for i in range(n_steps):
        should_store = (
            i == 0
            or i == n_steps - 1
            or stride == 1
            or (i % stride == 0)
        )
        if should_store:
            if store_idx < n_store_steps:
                t_store[store_idx] = t_curr
                y_store[:, store_idx] = y_curr
                store_idx += 1
            else:
                # Keep final endpoint even if we hit the storage cap.
                t_store[n_store_steps - 1] = t_curr
                y_store[:, n_store_steps - 1] = y_curr

        if i == n_steps - 1 or t_curr >= tf:
            break

        k1 = rhs(t_curr, y_curr)
        k2 = rhs(t_curr + 0.5 * t_step, y_curr + 0.5 * t_step * k1)
        k3 = rhs(t_curr + 0.5 * t_step, y_curr + 0.5 * t_step * k2)
        k4 = rhs(t_curr + t_step, y_curr + t_step * k3)

        y_curr = y_curr + (t_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t_curr = t_curr + t_step

    if store_idx <= 0:
        t_store[0] = float(t0)
        y_store[:, 0] = y0_arr
        store_idx = 1

    return OdeSolution(
        t=t_store[:store_idx],
        y=y_store[:, :store_idx],
        success=True,
        message="RK4 integration completed.",
    )
