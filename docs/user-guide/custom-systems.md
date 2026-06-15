# Custom systems

Besides the built-in Lorenz, Rössler and Hénon–Heiles systems, you can define
your own *n*-dimensional system directly in the sidebar (`Choose system →
Custom (nD)`). Equations are parsed with [SymPy](https://www.sympy.org), so you
write them as plain math.

This page is the syntax reference. For the theory behind the outputs, see
[../theory/](../theory/).

## The three input boxes

When `Custom (nD)` is selected, the sidebar shows three text areas, plus a
`Number of equations (n)` field.

### 1. Variable names — one per line

```
x
y
z
```

These become the state variables, **in order**. The first line is `y[0]`, the
second `y[1]`, and so on — that order is what the axis pickers, the Poincaré
section index, and the output variable refer to.

### 2. Equations `dy/dt` — one per line

One right-hand side per variable, in the same order as the names. The number of
equations **must equal** the number of variables, or parsing fails with
`Need exactly n equations`.

```
sigma*(y - x)
x*(rho - z) - y
x*y - beta*z
```

### 3. Parameters — `name = value`, one per line

```
sigma = 10
rho = 28
beta = 2.6667
```

Every symbol in the equations that is not a variable, a parameter, or `t` is
rejected with `Missing parameters in equations: ...`. So a stray symbol is
caught immediately rather than silently treated as zero.

## Writing the equations

- **Multiplication is explicit.** Write `a*x`, not `ax`. `xy` is read as one
  unknown symbol, not `x*y`.
- **Powers use `**`.** `x**2`, `x**4` — not `x^2`.
- **Time is `t`.** You may use `t` for non-autonomous (time-dependent) systems,
  e.g. a forcing term `f0*cos(w*t)`. Do not name a variable or parameter `t`.
- **Allowed functions:** `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `sinh`,
  `cosh`, `tanh`, `abs`. Nothing else is in scope.
- Constants like `pi`/`E` are not pre-defined — pass them as parameters if you
  need them.

## Jacobian (for Lyapunov exponents)

Lyapunov spectra need the Jacobian of the right-hand side. For custom systems
the sidebar offers:

- **Auto-compute Jacobian (symbolic)** — SymPy differentiates your equations and
  shows the symbolic matrix. This is exact and is the recommended default.
- If you turn the analytic Jacobian off, the app falls back to a
  central-difference (finite-difference) approximation. Slightly slower and less
  accurate, but it works for expressions that are awkward to differentiate.

## Symplectic (Forest–Ruth / Verlet) systems

If you pick a symplectic solver, the custom system must be a **separable
Hamiltonian** written in `[q..., p...]` order:

- The variable count must be **even**. The first half are the positions `q`, the
  second half the momenta `p`.
- The first half of the equations is `dq/dt` and may depend **only on `p`** (and
  `t`, parameters) — not on any `q`.
- The second half is `dp/dt` and may depend **only on `q`** — not on any `p`.

Violating either split raises an explicit error (`dq/dt equation … depends on q
vars`). For non-separable or non-Hamiltonian systems, use RK4/RK45/DOP853
instead.

## Save / load a configuration

The **Save configuration** button (Export tab) writes everything above —
variables, equations, parameters, Jacobian choice, ICs and integration settings
— to a JSON file. Loading it back restores the exact setup, so a system you tuned
once can be reproduced later or shared as a single file.

## A real gotcha: signs matter

A sign slip in one term can turn a bounded attractor into a trajectory that
diverges to infinity, after which a Poincaré section collects **zero** crossings
and the diagram comes out empty.

Concrete example — the 4-D modified Lorenz system (Volos et al. 2016):

```
y - x        ← correct  (ẋ = y − x)
-x*z + u
x*y - c
-d*y
```

Writing the first equation as `x - y` instead flips the linear part to `+1`,
every trajectory blows up, and the bifurcation diagram is blank. If a custom
sweep produces nothing, check the signs and the bounded-ness of a single
trajectory in Tab 1 first.
