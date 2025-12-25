import numpy as np

def rossler_rhs(_t, y, a=0.2, b=0.2, c=5.7):
    """
    Right-hand side of the 3D Rössler system.

    Parameters
    ----------
    _t : float
        Time (unused, system is autonomous).
    y : array-like, shape (3,)
        State vector [x, y, z].
    a, b, c : float
        Rössler system parameters.

    Returns
    -------
    dydt : ndarray, shape (3,)
        Time derivatives [dx/dt, dy/dt, dz/dt].
    """
    x, y_, z = y

    dx = -y_ - z
    dy = x + a * y_
    dz = b + z * (x - c)

    return np.array([dx, dy, dz])