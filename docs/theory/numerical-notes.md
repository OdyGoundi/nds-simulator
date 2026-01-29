# Numerical Techniques

This app uses several numerical methods for integration, Lyapunov estimation, and Poincare sections.
This note summarizes what is implemented and when it is used.

---

## ODE solvers

- **RK45 (adaptive)**: SciPy `solve_ivp` with RK45. Good default for general systems.
- **DOP853 (adaptive)**: High-order explicit solver for smooth, non-stiff problems.
- **RK4 (fixed step)**: Classic fixed-step Runge-Kutta; faster per step but needs smaller `dt`.
- **Symplectic Verlet (2nd order)**: For separable Hamiltonians; state = `[q..., p...]`.
- **Symplectic Forest-Ruth (4th order)**: Higher-order symplectic method, same assumptions.

Symplectic methods require an even number of variables and Hamiltonian structure.
If the system is not compatible, the app falls back to RK45.

---

## Lyapunov spectrum

- Uses **variational equations**: augment state with a tangent matrix `Q` and integrate `dQ/dt = J(x) Q`.
- Performs **periodic QR** re-orthonormalization every `qr_interval` and accumulates
  `log(|diag(R)|)` to estimate exponents.
- **Jacobian**:
  - Analytic Jacobian can be supplied.
  - For custom systems, a symbolic Jacobian can be generated (Sympy).
  - Otherwise, a central finite-difference Jacobian is used (epsilon `fd_eps`).

**Backend behavior**:
- Python path uses the selected solver: RK4 fixed step or adaptive `solve_ivp`.
- C++ backend (optional) uses fixed-step RK4 and Householder QR (Eigen), and is
  used when solver kind is RK4.

---

## Poincare sections and bifurcation sweeps

- **Section definition**:
  - Plane: `y[section_index] = section_value`
  - Expression: `f(t, y, params) = 0` (Sympy-based for custom systems)
- **Methods**:
  - `crossing`: detect sign changes and linearly interpolate
  - `slab`: keep points with `|f| <= tol`
- **Direction** filter: `+1` up, `-1` down, `0` both.
- For IVP sweeps, `crossing` can use `solve_ivp` events and early-stop once
  enough hits are collected.

---

## Accuracy tips

- Use smaller `dt` for RK4 and symplectic solvers if results look noisy.
- Increase `tf` (or reduce transient fraction) for more stable Lyapunov estimates.
- Provide analytic/symbolic Jacobians when possible for sharper Lyapunov results.
- For Poincare sections, prefer `crossing` when using adaptive solvers.
