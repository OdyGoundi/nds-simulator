# Developer README

This file documents the execution flow of the app with emphasis on the Numba and Python paths.

## Execution Map (Full)

Legend:
- "Numba" paths are JIT-compiled (first call compiles, next calls are fast).
- "Python" paths are the pure-Python fallbacks.

### 1) Single trajectory (Phase portrait / Time series)

```
UI (app/nlds_app.py)
  └─ solve_cached(...)  [app/cache.py]
       ├─ solver_kind = rk45 | dop853 | ivp
       │    └─ core/solver.integrate_system() -> scipy.solve_ivp  [Python]
       │
       ├─ solver_kind = rk4
       │    ├─ Numba available?
       │    │    ├─ built-in: core/numba_backend.build_builtin_system()
       │    │    │            -> rhs_nb -> build_rk4_integrator() -> rk4_nb(...)  [Numba]
       │    │    └─ custom: app/numba_custom.build_custom_numba_rhs()
       │    │               -> rhs_nb -> build_rk4_integrator() -> rk4_nb(...)  [Numba]
       │    └─ fallback: core/solver.integrate_system_rk4()                     [Python]
       │
       └─ solver_kind = symplectic_fr
            ├─ Numba available?
            │    ├─ built-in (Henon-Heiles): build_builtin_symplectic()
            │    │      -> dq_dt_nb/dp_dt_nb -> build_symplectic_fr_integrator()
            │    │      -> fr_nb(...)                                         [Numba]
            │    └─ custom: app/numba_custom.build_custom_numba_symplectic_functions()
            │           -> dq_dt_nb/dp_dt_nb -> build_symplectic_fr_integrator()
            │           -> fr_nb(...)                                         [Numba]
            └─ fallback: core/symplectic_solver.integrate_system_symplectic_fr()  [Python]
```

### 2) Lyapunov (single run)

```
UI -> compute_lyapunov_cached(...)  [app/logic/lyapunov_cached.py]
  └─ solver_kind = rk4
       ├─ Numba available?
       │    ├─ built-in: build_builtin_system() -> build_lyapunov_solver()  [Numba]
       │    └─ custom:
       │         - if analytic jac: build_custom_numba_rhs_and_jacobian()
       │         - else: build_custom_numba_rhs() + FD Jacobian
       │       -> build_lyapunov_solver()                                  [Numba]
       └─ fallback: core/lyapunov.compute_lyapunov_spectrum()
             └─ Python (RK4 or solve_ivp + QR)                         [Python]
```

### 3) Bifurcation / Poincare sweep (Tab 3)

```
UI -> run_sweep_chunk(...)  [app/sweep.py]
  ├─ system = custom
  │    ├─ solver_kind = rk4 and Numba available?
  │    │    └─ build_custom_numba_rhs() -> build_rk4_integrator()  [Numba]
  │    └─ fallback:
  │         ├─ core/solver.integrate_system_rk4()  [Python]
  │         └─ or core/solver.integrate_system()  [Python]
  │
  └─ system = built-in (lorenz/rossler/henon)
       ├─ solver_kind = rk4 + crossing/slab + no section_expr?
       │    └─ build_poincare_sweep_rk4() -> Numba sweep                  [Numba]
       └─ fallback:
            ├─ core/poincare_sweep.sweep_poincare_events_ivp() (ivp + crossing)
            └─ core/poincare_sweep.sweep_poincare()                       [Python]
```

### 4) Lyapunov sweep (Tab 3)

```
UI -> _run_lyapunov_sweep(...)  [app/logic/lyapunov_sweep.py]
  └─ per-chunk:
       ├─ solver_kind = rk4 and Numba available?
       │    ├─ built-in: build_builtin_system() -> build_lyapunov_solver()  [Numba]
       │    └─ custom: build_custom_numba_rhs_and_jacobian() OR RHS + FD
       │              -> build_lyapunov_solver()                           [Numba]
       └─ fallback: core/lyapunov.compute_lyapunov_spectrum()              [Python]
```

### 5) Benchmark

```
tests/benchmark_lyapunov_cpp_vs_python.py
  └─ compares: Python vs Numba
```

## Notes

- Numba compilation happens on first use; benchmark after the first run.
- The app uses `@st.cache_data` in `app/cache.py` and `app/logic/lyapunov_cached.py` to reuse results.
- Symplectic solvers in the app are Forest-Ruth (4th order) only.
