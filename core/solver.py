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


def integrate_system(rhs, t_span, y0, t_step=0.01, max_steps=None, **solve_options):
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
    t_eval = np.linspace(t0, tf, n_steps)

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


def integrate_system_rk4(rhs, t_span, y0, t_step=0.01, max_steps=None):
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

    n_states = y0_arr.size

    # Allocate arrays
    t = np.zeros(n_steps)
    y = np.zeros((n_states, n_steps))

    # Initial conditions
    t[0] = t0
    y[:, 0] = y0_arr

    # RK4 loop
    for i in range(1, n_steps):
        ti = t[i - 1]
        yi = y[:, i - 1]

        # if we have reached or exceeded tf, stop
        if ti >= tf:
            t = t[:i]
            y = y[:, :i]
            break

        k1 = rhs(ti, yi)
        k2 = rhs(ti + 0.5 * t_step, yi + 0.5 * t_step * k1)
        k3 = rhs(ti + 0.5 * t_step, yi + 0.5 * t_step * k2)
        k4 = rhs(ti + t_step,       yi + t_step * k3)

        y[:, i] = yi + (t_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t[i] = ti + t_step

    return OdeSolution(
        t=t,
        y=y,
        success=True,
        message="RK4 integration completed.",
    )
