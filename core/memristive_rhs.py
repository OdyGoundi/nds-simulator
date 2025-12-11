import numpy as np

def memristive_rhs(_t, y, a=0.0, b=0.1, c=0.0):
    """
    Right side of the 3D memristive system equation.
    Returns dy/dt as a numpy array of size 3.
    """
    y1, y2, y3 = y

    dy1 = y2
    dy2 = 0.4 * y1 * y3 - a
    dy3 = 0.3 * y2 - 0.1 * y3 - 1.4 * y2**2 - b * y1 * y2 - c

    return np.array([dy1, dy2, dy3])
