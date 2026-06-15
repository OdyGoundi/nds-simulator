# Architecture Overview

`dynaSim / nds-simulator` is a Streamlit application for exploring nonlinear
dynamical systems. The codebase is organized around a strict one-direction
dependency rule and a small set of typed contracts between layers.

This document describes the current shape of the code. Companion documents:
- [performance.md](./performance.md), [sweep-engine.md](./sweep-engine.md), [session-state.md](./session-state.md)

---

## 1. Layers and direction of imports

```
        ┌─────────────────────────────────────────────┐
        │                  app/nlds_app.py            │  composition root
        └──────────┬───────────┬──────────┬───────────┘
                   │           │          │
        ┌──────────▼───┐  ┌────▼─────┐  ┌─▼────────────┐
        │  app/ui/*    │  │ app/state│  │ app/logic/*  │  controllers
        │  (widgets,   │  │ (defaults│  │ (sweep,      │  near numerics
        │   tabs/)     │  │  + keys) │  │  poincare,
        └──────┬───────┘  └────┬─────┘  │  reservoir)
               │               │        └──────┬──────┘
        ┌──────▼───────────────▼───────────────▼──────┐
        │              app/services/*                 │  registries,
        │  (system_registry, solver_policy,           │  dispatch,
        │   lyapunov, sweep_run, sweep_state, export) │  business logic
        └────────────────────────┬────────────────────┘
                                 │
        ┌────────────────────────▼────────────────────┐
        │     app/cache.py, app/sweep.py              │  glue
        │     app/parsing/*, app/plotting/*           │  pure utilities
        │     app/export/*, app/numba_custom.py       │
        └────────────────────────┬────────────────────┘
                                 │
                       ┌─────────▼──────────┐
                       │     core/*         │  pure numerics
                       │  core/numba_backend│  JIT mirror
                       └────────────────────┘
```

**Rule:** UI → services → core. No back-edges. `core/` knows nothing about
Streamlit. Services know nothing about widgets. The UI never imports from
`core/` directly — always through a service or the cache layer.

---

## 2. Layer responsibilities

### `app/nlds_app.py` (~320 lines)

Streamlit entry point and composition root. Sets the page config, renders
the header and sidebar, then delegates each tab to a single function:

```python
phase_result = render_phase_tab(tabs[0], ...)
render_time_series_tab(tabs[1], phase_result=phase_result)
render_sweep_tab(tab=tabs[2], system=..., integration=..., ...)
render_export_tab(tabs[3], phase_result=phase_result, ...)
```

No widget logic, no system dispatch, no sweep orchestration here.

### `app/ui/`

The Streamlit layer. Every file is small (40–700 lines) and renders a single
piece of the UI.

- `app/ui/tabs/` — one file per tab, each is a thin orchestrator
- `app/ui/sidebar.py`, `branding.py`, `help_panels.py` — global chrome
- `app/ui/sweep_controls.py`, `sweep_plot_panel.py` — Tab 3 widgets / plots
- `app/ui/poincare_map_panel.py`, `widgets.py` — reusable pieces

UI files compute arrays and hand them to `app/plotting/` functions. They do
not call `core/` directly.

### `app/services/`

Business logic with a small public surface per file. No Streamlit decorators
here — services are callable from anywhere.

- `system_registry.py` — `BUILTIN_SYSTEMS` mapping + `SystemAdapter` per
  built-in (lorenz, rossler, henon_heiles). Holds RHS, Jacobian, parameter
  extraction, dimension.
- `solver_policy.py` — `resolve_solver(raw)` normalizes solver selection
  (`rk4`, `rk45`, `dop853`, `symplectic_fr`) and returns a `SolverPolicy`
  with the SciPy method to use.
- `lyapunov_service.py` — single-run Lyapunov + helpers reused by sweep.
- `sweep_run_service.py` — four entry points: new/cont × bif/lya.
- `sweep_state_service.py` — sweep accumulation, reservoir, clipping.
- `export_service.py` — bundle assembly + zip packaging.

### `app/state/`

Session-state defaults, typed key constants, and the apply-config layer.

- `defaults.py` — system labels, solver labels, UI defaults, the few
  reserved-key constants (e.g. `PENDING_STATIC_CFG_KEY`).
- `session_keys.py` — typed groups (`PhaseKeys`, `SweepControlsKeys`,
  `SweepDataKeys`, …) that replace bare `"phase_xlim_min_tab1"` strings.
- `apply_config.py` — `apply_state_values()`, `apply_static_config_to_state()`,
  `flush_pending_static_config_apply()`.

### `app/parsing/`

Symbolic parsing for custom user systems.

- `params_parser.py`, `vector_parser.py` — text → values.
- `custom_rhs_builder.py`, `custom_jacobian_builder.py`,
  `custom_symplectic_builder.py` — sympy → Python callables.
- `numba_rhs_builder.py` — sympy → Python source for Numba kernels (no JIT
  here; that lives in `numba_custom.py`).

### `app/plotting/`

Single home for figures and the helpers around them.

- `phase.py`, `time_series.py`, `sweep.py`, `lyapunov.py` — one module per
  plot kind, each pure (data + view bounds in, Matplotlib `Figure` out).
- `bounds.py` — `axis_bounds`, `square_xy_bounds` (deduped from UI files).
- `style.py` — `LINE_COLORS` palette.
- `downsampling.py` — `decimate_indices`, `downsample_trajectory`,
  `downsample_xy`, `apply_transient_cut`.

### `app/export/`

Streaming CSV builders. Bundle/zip assembly lives in
`app/services/export_service.py`.

### `app/logic/`

Controllers near numerics. Slightly higher level than `core/` but lower than
services — they own loops and orchestration.

- `bifurcation_sweep.py` — parallel sweep orchestration.
- `lyapunov_sweep.py` — parametric Lyapunov loop (uses
  `lyapunov_service.py`).
- `lyapunov_cached.py` — `@st.cache_data` shim over the service.
- `poincare_map.py` — typed wrapper over `core.poincare_sweep`.
- `reservoir_sampling.py` — bounded sweep visualization sampling.
- `sweep_utils.py` — worker count, fingerprint, chunk helpers.

### `app/cache.py`, `app/sweep.py`

Glue between UI/services and `core/`.

- `cache.py` — `solve_cached(...)` decorated with `@st.cache_data`. Picks
  Numba vs scalar, built-in vs custom via `system_registry`.
- `sweep.py` — `run_sweep_chunk(...)` plus six per-path helpers
  (`_run_custom_chunk`, `_run_symplectic_builtin_chunk`,
  `_run_builtin_extrema_chunk`, `_try_builtin_numba_poincare`,
  `_run_builtin_poincare_chunk`). Single dispatch point for all sweep
  observables.

### `core/`

Pure numerics, no Streamlit, no `app.*` imports.

- `solver.py` — SciPy `solve_ivp` wrapper, fixed-step RK4.
- `symplectic_solver.py` — Verlet, Forest-Ruth.
- `poincare_sweep.py` — Poincaré section + sweep entry points.
- `lyapunov.py` — QR-based Lyapunov spectrum.
- `<system>_rhs.py`, `jacobians_fixed_systems.py` — built-in systems.
- `numba_backend/` — JIT-accelerated mirror (RK4, Forest-Ruth, Lyapunov,
  Poincaré sweep, built-in compiled systems).

---

## 3. Key contracts

Boundaries between layers carry typed dataclasses, not loose dictionaries:

| Contract | Defined in | Producer | Consumer |
|---|---|---|---|
| `IntegrationConfig`, `InitialConditions`, `SolverTolerances`, `SystemConfig` | `app/params.py` | sidebar + tabs | services, cache, core |
| `PhaseTabResult` | `app/ui/tabs/phase_tab.py` | Tab 1 | Tabs 2, 3, 4 (read trajectory without resolving) |
| `SweepControlsResult` | `app/ui/sweep_controls.py` | sweep controls | sweep plot panel + sweep run service |
| `SystemAdapter` | `app/services/system_registry.py` | registry entries | cache, sweep, lyapunov |
| `SolverPolicy` | `app/services/solver_policy.py` | `resolve_solver()` | cache, sweep |
| `LyapunovTimeWindow` | `app/services/lyapunov_service.py` | `resolve_time_window()` | sweep + single Lyapunov |

Frozen dataclasses make boundaries explicit and make refactoring inside a
layer safe.

---

## 4. Caching and reuse

- `@st.cache_data` wraps `solve_cached` and the Lyapunov single-run shim.
  Cache keys are the dataclass arguments — re-runs with identical configs
  return instantly.
- `PhaseTabResult` carries the trajectory so Tabs 2 (time-series) and 4
  (export) read the same arrays Tab 1 already produced. No re-solve.
- The Numba backend is **builder-based**: build RHS/Jacobian → pass to a
  builder in `core/numba_backend/*` → get a JIT-compiled callable. Compiled
  kernels are cached per-config in `app/numba_custom.py` for custom systems.

---

## 5. Numba and fallback

Every `solver_kind` has a Numba path and a Python fallback:

| Solver kind | Numba path | Fallback |
|---|---|---|
| `rk45`, `dop853`, `ivp` | — (SciPy only) | `core/solver.integrate_system` |
| `rk4` | `numba_backend.build_rk4_integrator` | `core/solver.integrate_system_rk4` |
| `symplectic_fr` | `numba_backend.build_symplectic_fr_integrator` | `core/symplectic_solver.integrate_system_symplectic_fr` |

The cache layer attempts Numba first; on any failure it returns `None` and
the caller falls through to the scalar path. Behavior is identical between
the two — only speed differs after first JIT compile.

For the full Numba decision tree, see [README.md](./README.md).

---

## 6. Concurrency

- Bifurcation sweeps and Lyapunov sweeps support parallel execution via
  `concurrent.futures.ProcessPoolExecutor`. Each worker takes a chunk of
  parameter values and returns its own dataframe; the orchestrator joins.
- Streamlit itself is single-threaded per session. Long-running sweeps run
  in chunks so the UI stays responsive.
- The Numba JIT cache and the Streamlit `@st.cache_data` cache are both
  in-process; restart clears them.

---

## 7. Where to look for what

| Looking for | Start here |
|---|---|
| The entry point | `app/nlds_app.py` |
| How a tab is structured | `app/ui/tabs/<tab>_tab.py` |
| How to add a built-in system | `app/services/system_registry.py` |
| How a solver is selected | `app/services/solver_policy.py` |
| The Numba/scalar decision | `app/cache.py` (`_try_numba_*` helpers) |
| Sweep dispatch | `app/sweep.py` (`run_sweep_chunk`) |
| Lyapunov semantics | `app/services/lyapunov_service.py` |
| A plot's exact shape | `app/plotting/<kind>.py` |
| A session-state key | `app/state/session_keys.py` |
| Pure math | `core/<topic>.py` |

---

## 8. What's intentionally not abstracted

- **No plugin system.** Built-in systems are defined directly in
  `system_registry.py`. Adding one is an edit to that file plus a kernel.
- **No DI container.** Services import each other directly. The dependency
  graph is small enough that this is clearer than indirection.
- **No ORM / persistence layer.** All state is in-memory or in flat files
  (CSV, JSON, ZIP).
- **No abstract solver interface in `core/`.** SciPy's `solve_ivp` and the
  RK4/Forest-Ruth functions are called directly with their natural
  signatures; `solver_policy.py` lives one layer up.
- **No premature service splits.** A service exists when there is a second
  caller.
