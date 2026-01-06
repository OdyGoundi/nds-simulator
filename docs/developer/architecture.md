# Architecture Overview

Minimal map of the NLDS app, matching the structure used in `performance.md`.

---

## 1. UI layer (Streamlit)

- Owns layout (tabs, sidebar, columns) and user inputs.
- Delegates heavy work to controller functions; no computation inside widgets.

## 2. Controller

- Orchestrates solves and sweeps based on UI state.
- Manages session state (accumulated sweeps, last Poincaré hits, meta).
- Decides solver mode (event-based vs fallback) and chunking parameters.

## 3. Numerical core

- Deterministic RHS functions (Lorenz, Rossler, custom).
- Integration via `integrate_system` and Poincaré sweep utilities.
- Event-based sweep path for fast bifurcation diagrams; fallback keeps API stable.

## 4. Caching and reuse

- `solve_cached` and `sweep_cached` wrap expensive calls with `st.cache_data`.
- Warm-start paths reuse final states when continuation mode is selected.

## 5. Plotting/export

- Small plotting helpers (phase, time series, sweep scatter) produce Matplotlib figures.
- Export helpers emit CSV for trajectories and sweeps without re-running solvers.
