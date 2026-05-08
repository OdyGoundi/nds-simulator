# Session State

Streamlit session state in this app is **typed**. After refactor 4.9,
bare string keys like `"phase_xlim_min_tab1"` or `"sweep_acc_df"` no
longer appear in app code — they live as constants on grouped key
classes in [`app/state/session_keys.py`](../../app/state/session_keys.py).

Companion docs: [architecture.md](./architecture.md),
[sweep-engine.md](./sweep-engine.md).

---

## 1. Why typed keys

The pre-refactor codebase had ~370 sites that referenced session-state
strings directly. Renaming or grepping for usage was painful, and
typos silently produced new keys. The migration was mechanical: each
constant is set to exactly the string Streamlit sees, so a global
`"phase_xlim_min_tab1"` → `PhaseKeys.XLIM_MIN` substitution preserved
behaviour.

The win is everyday: code completion shows the available keys per
feature, `grep PhaseKeys` finds every Tab 1 phase usage, and renames
are a one-line edit.

---

## 2. Usage pattern

Import the relevant class once, then access constants:

```python
from app.state import (
    PhaseKeys,
    IntegrationKeys,
    SweepDataKeys,
    LyapunovDataKeys,
)

# Reading and writing session state
st.session_state[PhaseKeys.X_IDX] = 0
df = st.session_state.get(SweepDataKeys.ACC_DF)

# Wiring widgets
st.number_input("t0", key=IntegrationKeys.T0)
st.checkbox("Square axes", key=PhaseKeys.SQUARE_AXES)

# Reset accumulators (e.g. after settings change)
for key in (SweepDataKeys.ACC_DF, SweepDataKeys.LAST_PV,
            SweepDataKeys.META, SweepDataKeys.BOUNDARIES):
    st.session_state.pop(key, None)
```

The constant value is the literal Streamlit key, so mixing typed
constants and stale string literals at the same site still works
during gradual migrations.

---

## 3. Key classes

Sixteen classes, grouped by feature/tab. One-line purpose each:

### Per-tab UI state

| Class | What it holds |
|---|---|
| `SidebarKeys` | System / solver / custom equations / `y0` text inputs in the global sidebar. |
| `SystemParamKeys` | Built-in parameter widgets (`sigma`, `rho`, `beta`, `ross_*`, `hh_lambda`). |
| `TolKeys` | Solver tolerances `rtol`, `atol`. |
| `IntegrationKeys` | Tab 1 integration window (`t0`, `tf`, `dt`, `max_store_steps`, `max_plot_points`, `transient_cut_time`). |
| `PhaseKeys` | Tab 1 phase plot — projection axes, axis limits, line width, square-axes toggle. |
| `LyapunovTab1Keys` | Tab 1 Lyapunov controls + cached result + result fingerprint. |
| `StaticConfigKeys` | Tab 1 static-config save/load slot. |
| `TimeSeriesKeys` | Tab 2 time window (`start`, `end`, `range`). |
| `SweepControlsKeys` | Tab 3 widgets — sweep param, section, observable, runner, transient, parallel toggles. |
| `BifPlotKeys` | Tab 3 bifurcation-plot view (axis limits + bounds signature). |
| `LyaPlotKeys` | Tab 3 Lyapunov-plot view. |
| `ExportKeys` | Tab 4 controls (`prepare_full_traj_export`, `traj_export_chunk_index`). |
| `HelpPanelKeys` | Help/manual popup toggles. |
| `PlotSettingsKeys` | Per-plot `PlotSettings` values edited via the popup dialog (one key per plot location). |

### Cross-tab data buckets

| Class | What it holds |
|---|---|
| `SweepDataKeys` | Bifurcation accumulation: `ACC_DF`, `LAST_PV`, `META`, `RESERVOIR`, `ROWS_CLIPPED`, `STOP_INTERNAL`, `BOUNDARIES`, `CONFIG`, `LAST_DF`, `LAST_META`. |
| `LyapunovDataKeys` | Lyapunov accumulation: `ACC_DATA`, `LAST_PV`, `META`, `BOUNDARIES`. |

These two are the buckets the four `execute_*_sweep` services write to
(see [sweep-engine.md](./sweep-engine.md)).

---

## 4. The `app/state/` layer

Three small files, each with one job:

| File | Purpose |
|---|---|
| [`defaults.py`](../../app/state/defaults.py) (32 lines) | System / solver labels, UI defaults, and the few reserved-key constants (e.g. `PENDING_STATIC_CFG_KEY`). |
| [`session_keys.py`](../../app/state/session_keys.py) (183 lines) | All 16 typed key classes documented above. |
| [`apply_config.py`](../../app/state/apply_config.py) (217 lines) | The bridge between exported JSON configs and session state: `apply_state_values(...)`, `apply_static_config_to_state(...)`, `flush_pending_static_config_apply()`, casting helpers. |

### How config upload works

1. User uploads `StaticConfig.json` or `SweepParamConfig.json`.
2. The handler stages the parsed dict under
   `defaults.PENDING_STATIC_CFG_KEY` and triggers a rerun.
3. Early in the next render, `flush_pending_static_config_apply()`
   spots the staged config, walks `apply_static_config_to_state(...)`,
   and writes each value into the right typed key with the right
   cast.
4. Widgets pick up the new values on this rerun.

The reason for the staging step is that Streamlit will not let you
write to a widget's session-state key after the widget has been
created in the same run.

---

## 5. What stays as a literal string

JSON config-dict keys are intentionally **not** migrated. They reflect
the exported schema, not session state:

```python
solve_opts.get("rtol")          # JSON dict key → stays a literal
sweep_meta["section_var"]       # JSON dict key → stays a literal

st.session_state[TolKeys.RTOL]  # Streamlit session state → typed
```

Renaming a typed constant is mechanical and local. Renaming a JSON
dict key would silently break configs saved by older app versions, so
those names are part of a public surface and we don't move them.
