# Sweep Engine (Tab 3)

Tab 3 is no longer a single monolithic file. After refactor 4.5, it is
**three files joined by one contract**, plus a small set of services
that run the actual numerics and own the accumulated state.

Companion docs: [architecture.md](./architecture.md),
[session-state.md](./session-state.md).

---

## 1. Layout: three files, one contract

```
app/ui/tabs/sweep_tab.py        ← orchestrator (~127 lines)
  ├── app/ui/sweep_controls.py        ← widgets    (~689 lines)
  └── app/ui/sweep_plot_panel.py      ← plots      (~387 lines)

joined by:
  app/ui/sweep_controls.py::SweepControlsResult   (frozen dataclass)
```

`SweepControlsResult` is the contract. It carries every config object,
button state, and Streamlit column handle that the plot panel and the
sweep services need:

- column handles (`left_col`, `right_col`)
- pass-through context (`system`, `integration`, `initial`,
  `solve_tols`, `var_names`, `repo_root`, …)
- built config objects (`sweep_cfg`, `poincare_cfg`,
  `integration_sweep`, `run_cfg`, `lyapunov_cfg`,
  `solve_tols_sweep`, `sweep_meta`, `lya_meta`)
- raw widget values (`sweep_param`, `ycol`, `use_extrema`,
  `extrema_kind`, `transient_frac`, `warm_start`, `descending`,
  `max_hits`, `qr_interval_lya`, …)
- button states (`run_new`, `run_cont`, `run_lya`, `run_lya_cont`,
  `continue_stop`, `continue_stop_lya`)

The orchestrator [`app/ui/tabs/sweep_tab.py`](../../app/ui/tabs/sweep_tab.py)
reduces to:

```python
with tab:
    ctrl = render_sweep_controls(...)              # → SweepControlsResult
    df_plot = execute_*_sweep(...)                 # 4 service entry points
    render_sweep_plots(ctrl, df_plot)
```

---

## 2. Service entry points (4)

Tab 3 dispatches into four functions in
[`app/services/sweep_run_service.py`](../../app/services/sweep_run_service.py).
Each owns a complete execution path and writes its results back to
session state via the typed `SweepDataKeys` / `LyapunovDataKeys`
constants:

| Entry point | Purpose |
|---|---|
| `execute_new_bif_sweep(...)` | Fresh bifurcation sweep — clears accumulators, runs the full range, clips to row budget, seeds reservoir. |
| `execute_cont_bif_sweep(...)` | Continue from `LAST_PV + step`, validates `sweep_meta` against `prev_meta` (refuses to continue if settings changed), appends to `ACC_DF`. |
| `execute_new_lya_sweep(...)` | Fresh Lyapunov sweep — clears `LyapunovDataKeys.*` and runs the parametric loop. |
| `execute_cont_lya_sweep(...)` | Continue Lyapunov sweep with the same fingerprint guard. |

The two bifurcation entry points share a private helper
`_run_bif_chunk(...)` which delegates to `run_sweep_chunk` after
deciding observable (`OBSERVABLE_POINCARE` vs `OBSERVABLE_EXTREMA`)
and parallel-worker setup.

### Direction (ascending / descending)

The sweep mode radio has three options: `Bifurcation (reset ICs)`,
`Continuation (warm start, low->high)`, and
`Continuation (warm start, high->low)`. The third one sets
`SweepConfig.descending = True` — the step itself stays positive; the
parameter grid is built as usual and then reversed. The reversal
happens in `_param_values` (`app/sweep.py`), in `sweep_poincare` /
`sweep_poincare_events_ivp` (`core/poincare_sweep.py`), and inside the
Numba Poincaré kernel, which iterates the grid in reverse without
leaving JIT. Direction is part of the settings fingerprint, and
**Continue is disabled in descending mode** — a high->low run always
starts fresh.

---

## 3. Per-path runners (6)

[`app/sweep.py`](../../app/sweep.py) is now a thin dispatcher
(`run_sweep_chunk`) on top of six per-path helpers. Each
helper handles one (system × solver × observable) combination:

| Runner | Path |
|---|---|
| `_try_custom_numba_poincare` | Fused JIT Poincaré sweep for custom systems: feeds the compiled custom RHS into the same `build_poincare_sweep_rk4` kernel used by built-ins. Returns `None` if Numba isn't available, the observable is extrema, the section is expression-based, or the solver isn't RK4 + crossing — the caller falls through. |
| `_run_custom_chunk` | Custom systems (any solver): tries the fused Numba path above, then per-value integration. |
| `_run_symplectic_builtin_chunk` | Built-in Hénon-Heiles, symplectic Forest-Ruth. |
| `_run_builtin_extrema_chunk` | Built-in systems with the `extrema` observable. |
| `_try_builtin_numba_poincare` | Fast JIT Poincaré sweep for built-in systems. Returns `None` if Numba isn't available, the section is expression-based, or the solver isn't RK4 + crossing — the caller falls through. |
| `_run_builtin_poincare_chunk` | Built-in systems with Poincaré observable: tries Numba via the helper above, then `solve_ivp` events, then dense scanning. |
| `run_sweep_chunk` | Public dispatcher: normalises observable + solver, picks the runner, returns the chunk dataframe. |

A `dispatch` table for `run_sweep_chunk`:

```
custom system                  → _run_custom_chunk
                                   ├─ _try_custom_numba_poincare  (if RK4 + crossing + plane)
                                   └─ per-value integration       (fallback)
built-in + symplectic_fr       → _run_symplectic_builtin_chunk
built-in + extrema             → _run_builtin_extrema_chunk
built-in + poincare            → _run_builtin_poincare_chunk
                                   ├─ _try_builtin_numba_poincare (if RK4 + crossing + plane)
                                   ├─ sweep_poincare_events_ivp   (RK45/DOP853 + crossing)
                                   └─ sweep_poincare              (fallback)
```

---

## 4. State accumulation

[`app/services/sweep_state_service.py`](../../app/services/sweep_state_service.py)
owns the bookkeeping. Public concerns:

- **Initialise / reset** sweep buckets in `st.session_state`.
- **Continuation compatibility**: `_meta_mismatches(prev, curr)` —
  before continuing, the saved `sweep_meta` is compared with the new
  one; the four `execute_cont_*` paths refuse to extend if anything
  load-bearing changed (sweep param, section, solver, dt, …).
- **Row budget clipping** (`_clip_sweep_df`): when a sweep accumulates
  past `MAX_SWEEP_ROWS_IN_MEMORY`, oldest rows are dropped.
- **Reservoir fold-in** (`_append_dropped_rows_to_reservoir`): dropped
  rows are not lost — they are sampled into a bounded XY reservoir,
  rendered alongside recent rows by the plot panel.

All session-state access goes through typed constants from
[`app/state/session_keys.py`](../../app/state/session_keys.py)
(`SweepDataKeys.*`, `LyapunovDataKeys.*`), not bare strings. See
[session-state.md](./session-state.md).

---

## 5. Modes

| Mode | Trigger | Path |
|---|---|---|
| **Event-based IVP** | `solver_kind ∈ {rk45, dop853}` + `crossing` | `core/poincare_sweep.sweep_poincare_events_ivp` — keeps the last `max_hits` hits per parameter value. |
| **Numba Poincaré sweep** | RK4 + `crossing` + plane section + built-in or custom system | `core/numba_backend/sweep.build_poincare_sweep_rk4`. |
| **Fixed-step sweep** | RK4 / Symplectic FR | `core/solver.integrate_system_rk4` or `core/symplectic_solver`; Numba when available. |
| **Generic fallback** | Anything else | `core/poincare_sweep.sweep_poincare` with dense scanning. |
| **Lyapunov sweep** | `execute_*_lya_sweep` | `app/logic/lyapunov_sweep._run_lyapunov_sweep` → `app/services/lyapunov_service` → `core/lyapunov`. Optional `parallel_lya`. |

---

## 6. Performance knobs

- `max_hits` — caps the stored Poincaré hits per parameter value (the
  **last** `max_hits` crossings are kept); further clipped against the
  row budget via `_effective_max_hits` in `app/sweep.py`.
- `tf_sweep`, `dt_sweep`, `transient_frac` — integration effort per
  parameter value.
- `qr_interval` — Lyapunov re-orthonormalisation period.
- `warm_start` (continuation, low->high or high->low) vs reset ICs.
- `parallel_bif` / `parallel_lya` + `workers` — only honoured when
  `warm_start` is off.

See [performance.md](./performance.md) for the rationale.

---

## 7. Data contracts

| What | Shape |
|---|---|
| Bifurcation sweep result | DataFrame with columns `param`, `t_hit`, `y{idx}`, plus the chosen `out_var`. |
| Lyapunov sweep result | Dict with `param_vals`, `lambdas` (list of arrays), `errors`. |
| Bifurcation state | `SweepDataKeys.{ACC_DF, LAST_PV, META, BOUNDARIES, RESERVOIR, ROWS_CLIPPED, STOP_INTERNAL, LAST_DF, LAST_META}`. |
| Lyapunov state | `LyapunovDataKeys.{ACC_DATA, LAST_PV, META, BOUNDARIES}`. |
