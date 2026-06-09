import numpy as np


def henon_heiles_rhs(_t, state, lam=1.0, **kwargs):
    """
    Right-hand side of the Henon-Heiles Hamiltonian system.

    State ordering: [q1, q2, p1, p2]

    H = 0.5 * (p1^2 + p2^2 + q1^2 + q2^2) + lam * (q1^2 * q2 - q2^3 / 3)
    """
    if "lambda" in kwargs:
        lam = kwargs["lambda"]

    q1, q2, p1, p2 = state

    dq1 = p1
    dq2 = p2
    dp1 = -(q1 + 2.0 * lam * q1 * q2)
    dp2 = -(q2 + lam * (q1 * q1 - q2 * q2))

    return np.array([dq1, dq2, dp1, dp2], dtype=float)


def henon_heiles_dq_dt(_t, p, lam=1.0):
    """
    dq/dt for Henon-Heiles with state split y = [q..., p...].
    dq/dt depends only on p for separable Hamiltonians.
    """
    p1, p2 = p
    return np.array([p1, p2], dtype=float)


def henon_heiles_dp_dt(_t, q, lam=1.0):
    """
    dp/dt for Henon-Heiles with state split y = [q..., p...].
    dp/dt depends only on q for separable Hamiltonians.
    """
    q1, q2 = q
    dp1 = -(q1 + 2.0 * lam * q1 * q2)
    dp2 = -(q2 + lam * (q1 * q1 - q2 * q2))
    return np.array([dp1, dp2], dtype=float)
