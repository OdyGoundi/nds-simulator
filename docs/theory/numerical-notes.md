# Numerical Techniques

This note summarises the numerical methods used by the app: ODE solvers,
Poincaré sections, and Lyapunov spectrum estimation. It matches the
level of Chapter 1 of the thesis — readable, not exhaustive.

---

## Why numerical solutions

A continuous-time system is written as

```
ẋ = f(x, t; θ),    x(t₀) = x₀
```

For simple problems the solution `x(t)` can be found analytically. For
most nonlinear systems it cannot, so the app approximates the solution
at discrete times `t_k = t₀ + k·Δt`. The samples `x_k ≈ x(t_k)` are
then plotted as the trajectory.

---

## ODE solvers

Three categories, each with a clear trade-off:

- **Fixed-step (RK4)**: the user picks `Δt` and the same step is used
  throughout. Predictable cost and uniform sampling. Quality depends
  entirely on whether `Δt` is small enough.
- **Adaptive (RK45, DOP853)**: SciPy `solve_ivp`. The step shrinks
  where the solution changes quickly and grows where it is smooth, so
  you get good accuracy without tuning `Δt`. Default choice for most
  non-Hamiltonian systems.
- **Symplectic (Forest-Ruth, 4th order)**: designed for Hamiltonian
  systems. Preserves geometric structure so the total energy stays
  bounded over long times. Requires a separable Hamiltonian and an
  even number of variables (state splits into `q` and `p`).

A general-purpose method like RK45 can be locally accurate on a
Hamiltonian system but show a slow energy drift over long integrations.
A symplectic method does not. If the system is not compatible with a
symplectic step, the app falls back to RK45.

---

## Poincaré sections

The trajectory is reduced to its intersections with a chosen surface
(a coordinate plane, or a custom expression). Crossings are detected
either by sign change with linear interpolation, or by accepting every
sample inside a thin tolerance band. See
[`poincare-section.md`](./poincare-section.md) for the details.

---

## Lyapunov spectrum

### What it measures

Lyapunov exponents quantify how sensitive a system is to its initial
conditions. If we follow two trajectories that start very close to
each other, the Lyapunov exponent describes whether the distance
between them grows or shrinks, and how fast.

For a small initial perturbation `δx(0)`, the distance between the two
trajectories evolves approximately as

```
‖δx(t)‖ ≈ e^{λ t} · ‖δx(0)‖
```

so the exponent `λ` is the average exponential rate of stretching (or
shrinking) along the orbit:

- `λ > 0`: nearby trajectories diverge → chaos
- `λ < 0`: they converge → stable behaviour
- `λ ≈ 0`: neutral direction (expected once for any autonomous flow)

### The full spectrum

In `n` dimensions there are `n` exponents written in decreasing order
`λ_1 ≥ λ_2 ≥ … ≥ λ_n`. The largest exponent is the simplest
diagnostic; the full spectrum gives a more complete picture. A few
useful patterns:

- one positive exponent → chaotic
- two positive exponents → hyperchaotic
- one exponent ≈ 0, all others negative → periodic orbit
- all negative → stable equilibrium

### How the app computes them

Instead of comparing two random trajectories, the app linearises the
dynamics along the reference orbit — the **variational equations**:

```
δẋ = J(x(t), t) · δx
```

where `J` is the Jacobian. A whole orthonormal frame of perturbations
is integrated alongside the trajectory. Without intervention, the
columns of the frame would all collapse onto the most-expanding
direction. To prevent this, the frame is periodically re-orthonormalised
with a QR decomposition `Y = QR`. The diagonal entries of `R` record
the local stretching factors, and the exponents come from their time
average:

```
λ_i ≈ (1 / T_m) · Σ_k ln |R_ii^{(k)}|
```

**Jacobian source**: analytic Jacobian for built-in systems, symbolic
Jacobian (sympy) for custom systems, or central finite differences as
the fallback.

**Backend**: the Python path uses the selected solver (RK4 fixed-step
or `solve_ivp`). Numba accelerates the fixed-step RK4 + QR loop when
available.

---

## Accuracy tips

- Use a smaller `Δt` for RK4 and symplectic solvers if results look noisy.
- Increase `tf` (or reduce the transient fraction) until Lyapunov
  estimates stop changing.
- Prefer analytic / symbolic Jacobians when available.
- For Poincaré sections, prefer `crossing` with adaptive solvers.
- The QR interval is a knob: more frequent improves stability but costs
  more.
