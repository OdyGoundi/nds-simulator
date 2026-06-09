# Developer README

This file documents the execution flow of the app with emphasis on the Numba and Python paths.

For the higher-level shape of the codebase, see
[architecture.md](./architecture.md) and
[code-reader-guide.md](./code-reader-guide.md). For Tab 3 specifically,
see [sweep-engine.md](./sweep-engine.md). For session state,
[session-state.md](./session-state.md).

## Execution Map (Full)

Legend:
- "Numba" paths are JIT-compiled (first call compiles, next calls are fast).
- "Python" paths are the pure-Python fallbacks.

System and solver dispatch are not inline `if/elif` anymore. Each entry
point goes through:

- [`app/services/system_registry.py`](../../app/services/system_registry.py) —
  `get_builtin(key)` returns a `SystemAdapter` with the system's RHS,
  Jacobian, parameter extractor, dimension, and Numba builder.
- [`app/services/solver_policy.py`](../../app/services/solver_policy.py) —
  `resolve_solver(raw)` normalises the solver kind (`rk4`, `rk45`,
  `dop853`, `symplectic_fr`, `ivp`) into a `SolverPolicy`.

The execution-map trees below describe what each leaf calls; the
registry and policy are simply how the path is selected.

### 1) Single trajectory (Phase portrait / Time series)

```
UI (app/nlds_app.py)
  └─ solve_cached(...)  [app/cache.py]
       ├─ solver_kind = rk45 | dop853 | ivp
       │    └─ core/solver.integrate_system() -> scipy.solve_ivp  [Python]
       │         (supports bounded output via integration.max_store_steps)
       │
       ├─ solver_kind = rk4
       │    ├─ Numba available?
       │    │    ├─ built-in: core/numba_backend.build_builtin_system()
       │    │    │            -> rhs_nb -> build_rk4_integrator() -> rk4_nb(...)  [Numba]
       │    │    └─ custom: app/numba_custom.build_custom_numba_rhs()
       │    │               -> rhs_nb -> build_rk4_integrator() -> rk4_nb(...)  [Numba]
       │    └─ fallback: core/solver.integrate_system_rk4()                     [Python]
       │         (supports bounded output via integration.max_store_steps)
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
               (supports bounded output via integration.max_store_steps)
```

### 2) Lyapunov (single run)

```
UI -> compute_lyapunov_cached(...)  [app/logic/lyapunov_cached.py]
  └─ app/services/lyapunov_service.compute_single_lyapunov(...)
       (resolves time window, builds RHS/Jacobian, picks Numba vs SciPy)
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

`lyapunov_service.py` also exposes `resolve_time_window`,
`build_lyapunov_rhs_jac`, `build_numba_lyap_solver`,
`run_lyapunov_numba`, and `run_lyapunov_scipy` — shared by the
single-run Tab 1 path and the Tab 3 sweep loop.

### 3) Bifurcation / Poincare sweep (Tab 3)

```
UI -> run_sweep_chunk(...)  [app/sweep.py]
  ├─ observable = poincare | extrema
  │    └─ results are bounded in-memory (row budget cap) and hit count is budget-aware
  │
  ├─ system = custom
  │    ├─ solver_kind = rk4 and Numba available?
  │    │    └─ build_custom_numba_rhs() -> build_rk4_integrator()  [Numba]
  │    ├─ solver_kind = symplectic_fr and Numba available?
  │    │    └─ build_custom_numba_symplectic_functions()
  │    │         -> build_symplectic_fr_integrator()             [Numba]
  │    └─ fallback:
  │         ├─ core/solver.integrate_system_rk4()  [Python]
  │         └─ or core/solver.integrate_system()  [Python]
  │
  └─ system = built-in (lorenz/rossler/henon)
       ├─ solver_kind = rk4 + crossing/slab + no section_expr?
       │    └─ build_poincare_sweep_rk4() -> Numba sweep                  [Numba]
       ├─ solver_kind = symplectic_fr (henon_heiles only)
       │    ├─ Numba available? build_builtin_symplectic()
       │    │      -> build_symplectic_fr_integrator()                   [Numba]
       │    └─ fallback: core/symplectic_solver.integrate_system_symplectic_fr()  [Python]
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
- When users select RK4 or Symplectic Forest-Ruth, the Numba path is attempted automatically if available.

### Numba backend is builder-based

There are no fixed JIT kernels. The flow is always:

1. Build RHS/Jacobian (built-in via
   `numba_backend.build_builtin_system`, or custom via
   `numba_custom.build_custom_numba_*`).
2. Pass the compiled RHS/Jacobian to a builder
   (`build_rk4_integrator`, `build_symplectic_fr_integrator`,
   `build_lyapunov_solver`, `build_poincare_sweep_rk4`).
3. Call the resulting JIT function.

Compilation happens on first call. Custom-system kernels are cached
per `(var_names, eq_lines, param_names)` in
[`app/numba_custom.py`](../../app/numba_custom.py).

### Large-run safeguards

- Tab 1 plotting uses decimation (`max_plot_points`) to keep rendering bounded.
- Trajectory storage can be capped (`integration.max_store_steps`).
- Tab 4 supports chunked trajectory CSV export.
- Tab 3 sweep accumulation is bounded by a row budget; older rows are
  folded into a reservoir sample. See
  [sweep-engine.md](./sweep-engine.md) for details.
