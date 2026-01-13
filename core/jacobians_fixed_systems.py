# jacobians_3d.py
# Ready-to-drop analytic Jacobians for Lorenz / Rössler / Thomas (3D)
# Signature matches RHS style: jac(t, state, params...) -> (3,3) ndarray

import numpy as np


def lorenz_jac(_t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0) -> np.ndarray:
    """
    Jacobian of Lorenz system:
      dx = sigma*(y - x)
      dy = x*(rho - z) - y
      dz = x*y - beta*z
    """
    x, y, z = state
    return np.array(
        [
            [-sigma,  sigma,    0.0],
            [rho - z, -1.0,    -x  ],
            [y,        x,     -beta],
        ],
        dtype=float,
    )


def rossler_jac(_t, state, a=0.2, b=0.2, c=5.7) -> np.ndarray:
    """
    Jacobian of Rössler system:
      dx = -y - z
      dy = x + a*y
      dz = b + z*(x - c)
    """
    x, y, z = state
    return np.array(
        [
            [0.0, -1.0, -1.0],
            [1.0,  a,    0.0],
            [z,    0.0,  x - c],
        ],
        dtype=float,
    )


def thomas_jac(_t, state, beta=0.208186) -> np.ndarray:
    """
    Jacobian of Thomas' cyclically symmetric system (common form):
      dx = sin(y) - beta*x
      dy = sin(z) - beta*y
      dz = sin(x) - beta*z

    Note: parameter often denoted b or beta. Default here is a common chaotic value.
    """
    x, y, z = state
    return np.array(
        [
            [-beta,       np.cos(y), 0.0],
            [0.0,        -beta,      np.cos(z)],
            [np.cos(x),   0.0,      -beta],
        ],
        dtype=float,
    )
