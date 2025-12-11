import numpy as np

def lorenz_rhs(_t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Right-hand side of the 3D Lorenz system.

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

    state : [x, y, z]
    """
    x, y, z = state

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return np.array([dx, dy, dz])
