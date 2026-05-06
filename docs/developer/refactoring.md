# Refactoring Notes

This document records the structural refactoring carried out on the
`refactor/codebase-cleanup` branch and the patterns that drove it. It
replaces the earlier `refactoring-proposals.md` (forward-looking plan) and
`refactoring-workflow.md` (process notes) — both folded in here.

Companion documents:
- [architecture.md](./architecture.md) — the resulting shape of the code
- [code-reader-guide.md](./code-reader-guide.md) — a reading map of the
  current files

---

## 1. Why this refactor

The codebase had grown two clusters of pain:

- **Oversized files.** `app/nlds_app.py` was 793 lines and held Tabs 2 and 4
  inline. `app/sweep.py` was 635 lines of mixed dispatch. `app/cache.py`
  mixed Streamlit caching with system/solver `if/elif` branches.
  `app/numba_custom.py` mixed sympy parsing with JIT compilation.
- **Scattered keys and helpers.** Session-state keys like
  `"phase_xlim_min_tab1"` appeared as bare strings across many files.
  Plotting was split between `app/plots.py`, `sweep_plot_panel.py`, and
  inline code in tabs.

The goal was steady, incremental cleanup with no behavior changes — small
modules, single responsibilities, simple call paths. No big rewrites.

---

## 2. Guiding principles

These rules were applied to every commit on the branch:

1. **One responsibility per file.** Not one function per file — one
   concern.
2. **Realistic file sizes.** UI: 80–250 lines. Services: 100–300 lines.
   Numerics: 150–500 if needed for cohesion.
3. **Direction of imports.** UI → services → core. No back-edges.
4. **Frozen dataclass contracts** between layers (`PhaseTabResult`,
   `SweepControlsResult`, `SystemAdapter`, `SolverPolicy`, …).
5. **Registry over `if/elif`** for dispatch.
6. **No behavior changes mixed with structural changes.** Each commit is a
   refactor *or* a bug-fix, not both.
7. **No new abstractions for hypothetical needs.** Extract a service when
   there is a second caller, not before.

---

## 3. What was done

The branch was applied in four phases (A–D). Each phase contains one or
more numbered items. Items are independent and could be skipped or
reordered.

### Phase A — UI cleanup

#### 4.1 Extract Tab 2 (time-series)

`nlds_app.py` no longer holds Tab 2 inline. Moved to
`app/ui/tabs/time_series_tab.py`. The new renderer reads the trajectory
from `PhaseTabResult`, so Tab 1's solve is reused — Tab 2 never re-runs the
solver.

Files: `app/ui/tabs/time_series_tab.py` (new, ~136 lines).

#### 4.2 Extract Tab 4 (export)

Same treatment for Tab 4. Bundle assembly was moved into
`app/services/export_service.py`; the tab itself is now widget rendering
plus calls into the service.

Files: `app/ui/tabs/export_tab.py` (new, ~354 lines).

After 4.1 and 4.2, `app/nlds_app.py` dropped from **793 → ~318 lines** and
became a true composition root.

### Phase B — Service consolidation

#### 4.3 Slim `app/sweep.py`

`run_sweep_chunk(...)` was a 635-line function mixing observable
normalization, row budgeting, solver selection, and per-system dispatch.
Split into six per-path runners plus a thin dispatcher:

- `_run_custom_chunk` — custom systems
- `_run_symplectic_builtin_chunk` — Henon-Heiles symplectic
- `_run_builtin_extrema_chunk` — built-in extrema
- `_try_builtin_numba_poincare` — JIT Poincaré (returns `None` on miss)
- `_run_builtin_poincare_chunk` — JIT/events/standard chain
- `run_sweep_chunk` — 88-line dispatcher

The file grew slightly (635 → 752 lines) because the dispatch is now
explicit rather than nested `if`s — but each runner is single-purpose and
trivially readable.

A 4-month-old bug was caught and fixed during this work: a stray `else:
raise` had silently broken non-custom bifurcation sweeps since
2026-01-13. Fixed in a separate commit before the structural change.

#### 4.4 Unify Lyapunov execution

`app/services/lyapunov_service.py` and `app/logic/lyapunov_sweep.py` had
duplicated setup logic (RHS/Jacobian build, time-window resolution, Numba
solver build, SciPy fallback). Pulled the shared helpers up into
`lyapunov_service.py`:

- `resolve_time_window(integration, lyapunov)`
- `build_lyapunov_rhs_jac(system, params)`
- `build_numba_lyap_solver(system_key, ...)`
- `run_lyapunov_numba(...)`, `run_lyapunov_scipy(...)`

`lyapunov_sweep.py` shrank from 314 → 206 lines and now calls into the
service for every step.

#### 4.5 Trim `app/cache.py`

Removed dead `sweep_cached` (zero callers). Replaced inline
`if system.key == ...` branches with `system_registry.get_builtin(...)` and
`adapter.dq_dp_builder` calls. Split the Numba/scalar decision into small
private helpers (`_try_numba_rk4`, `_try_numba_symplectic`,
`_build_scalar_symplectic_funcs`).

`cache.py` shrank from 330 → 229 lines.

### Phase C — Boundary cleanup

#### 4.6 Split `app/numba_custom.py`

The file mixed sympy parsing with `nb.njit` compilation. Split:

- `app/parsing/numba_rhs_builder.py` (new, 233 lines) — sympy → Python
  source for RHS / Jacobian / symplectic kernels.
- `app/numba_custom.py` (now 120 lines) — JIT compile + cache + builder
  factory.

Public API unchanged.

#### 4.7 Slim `app/export_utils.py`

Bundle/manifest concerns had already moved into `export_service.py`
during 4.2. The remaining cleanup here was to make the public API of
`export_utils.py` explicit: only `build_static_config` and
`build_sweep_config` are public; sub-block builders
(`_build_app_block`, `_build_system_block`, `_build_integration_block`)
and utilities (`_utc_timestamp`, `_get_git_commit`) are now private.

Result: clear split — **`export_utils.py` describes a run,
`export_service.py` packages it.**

### Phase D — Polish

#### 4.8 Centralize plotting

Plotting was scattered: `app/plots.py` (top-level), inline `plt.subplots`
calls in `sweep_plot_panel.py`, `phase_tab.py`, `time_series_tab.py`.
Built out `app/plotting/`:

- `phase.py` — `plot_phase_2d`, `plot_phase_3d`
- `time_series.py` — `plot_time_series` (multi-var), `plot_single_variable`
- `sweep.py` — `plot_bifurcation`
- `lyapunov.py` — `plot_lyapunov_sweep`
- `bounds.py` — `axis_bounds`, `square_xy_bounds` (deduped from two UI files)
- `style.py` — shared `LINE_COLORS` palette

Plot functions are now pure: data + view bounds in, `Figure` out. UI files
became declarative ("compute arrays, hand to `plot_x`, render"). UI files
shrank by ~84 lines net. The typo'd `plot_time_seiries_functional` alias
is gone. `app/plots.py` was renamed into `app/plotting/phase.py` so git
history follows the move.

#### 4.9 Typed session-state keys

Strings like `"phase_xlim_min_tab1"`, `"sweep_acc_df"`,
`"lya_result_tab1"` appeared in 100+ places across 10+ files. Created
`app/state/session_keys.py` with grouped constant classes:

- `PhaseKeys`, `IntegrationKeys`, `LyapunovTab1Keys`, `SystemParamKeys`,
  `TolKeys`, `SidebarKeys`, `StaticConfigKeys`
- `TimeSeriesKeys`
- `SweepControlsKeys`, `BifPlotKeys`, `LyaPlotKeys`
- `SweepDataKeys`, `LyapunovDataKeys` (cross-tab data buckets)
- `ExportKeys`, `HelpPanelKeys`

Each member equals the literal string Streamlit sees, so the migration
was mechanical and zero-risk. Migrated one tab at a time across three
commits:

- Tab 1 (phase_tab + apply_config + sidebar) — 161 string sites
- Tab 2 (time_series_tab) — 18 sites
- Tab 3 + cross-tab (sweep_controls, sweep_plot_panel, sweep_tab,
  export_tab, sweep_state_service, sweep_run_service) — 198 sites

JSON config dict keys (e.g. `solve_opts.get("rtol")`) were intentionally
left as literals — they reflect the exported JSON schema, not session
state.

#### 4.10 Move legacy code aside

Four notebook-era artifacts at the repo root were confusing readers
about which path is live. Moved into `legacy/`:

- `nds_app.py` — early single-file Streamlit app
- `app_lorenz.py` — minimal Lorenz-only demo
- `in_out/` — early I/O helpers
- `plotting/` — early plotting helpers (predecessor to `app/plotting/`)

Plus a one-paragraph `legacy/README.md` pointing readers to
`app/nlds_app.py` as the live entry point. None of the artifacts were on
the runtime path.

---

## 4. Patterns used

Four patterns recurred across the branch. Re-use them.

### 4.1 Extraction

Pull a cohesive group of functions out of an oversized file into its own
module.

1. Create the new module.
2. Move the smallest cohesive group of functions.
3. Define a return-type dataclass if the boundary is non-trivial.
4. Update the old call site.
5. Commit.

Used for: Tab 1 (`phase_tab.py` + `PhaseTabResult`), Tab 2, Tab 4, sidebar,
branding, help panels.

### 4.2 Split

Used when one file holds multiple concerns that share state through a
return value.

1. Identify the concerns (controls, plots, orchestration).
2. Create one file per concern.
3. Define a frozen dataclass as the contract between them.
4. The original file becomes a thin orchestrator.

Used for: Tab 3 split (`sweep_tab.py` orchestrator + `sweep_controls.py`
widgets + `sweep_plot_panel.py` plots, joined by `SweepControlsResult`).

### 4.3 Registry

Replaces repeated `if/elif` dispatch with a typed lookup.

1. Create a registry module.
2. Add a read-only mapping (e.g. `BUILTIN_SYSTEMS: dict[str,
   SystemAdapter]`).
3. Replace one call site to use the registry.
4. In a later commit, replace the remaining call sites.
5. Eventually remove the original `if/elif` blocks.

Used for: `system_registry.py` (built-in systems), `solver_policy.py`
(solver kind normalization).

### 4.4 Service

Used when UI or controllers know too many numerics details.

1. Create a service module with a small public surface.
2. Move setup/dispatch logic into it.
3. UI calls the service. The service hides the details.
4. Keep the contract simple (1–3 functions, dataclass inputs/outputs).

Used for: `lyapunov_service.py`, `sweep_run_service.py`,
`sweep_state_service.py`, `export_service.py`.

### 4.5 Mechanical migration (specific to 4.9)

For type-only refactors with hundreds of mostly-mechanical replacements:

1. Define the typed constants such that each member equals the literal it
   replaces. The substitution is then string → identifier.
2. Run a scripted bulk replace on quoted strings (`"foo"` →
   `Module.FOO`).
3. Watch for collisions where a literal serves double duty (session-state
   key vs JSON dict key vs UI label) — restore the literal in those
   spots.
4. Add the import.
5. Compile + import-check + run the affected tab.

---

## 5. Branching and commit style

The branch followed extraction → adoption → cleanup, spread across many
small commits.

- Commits are titled verb-first (`Extract`, `Split`, `Move`, `Slim`,
  `Migrate`, `Unify`).
- Each commit either restructures or fixes behavior — never both.
- File counts per commit: typically 2–6, occasionally up to 10 for the
  mechanical migrations in 4.9.

For follow-up refactors, prefer the same approach: keep `main` clean,
branch as `refactor/<topic>`, and make commits small enough to revert
individually. When in doubt, **make it smaller**.

---

## 6. Outcomes

After this branch:

- `app/nlds_app.py` is **318 lines** (was 793). It is a pure composition
  root.
- No file in `app/` exceeds 750 lines, and the only one over 500 is
  `app/sweep.py` (752 lines, single dispatch surface for sweep
  observables).
- Plotting goes through `app/plotting/`. The Matplotlib backend is now
  swappable from one place.
- Session-state keys are typed constants. Renames are mechanical; usage
  is greppable through the constant.
- `legacy/` exists; the repo root reflects only the live application.

A new contributor should be able to:

1. Read [code-reader-guide.md](./code-reader-guide.md) (~15 min).
2. Open one tab file and trace its full call path within five minutes.
3. Add a new built-in system or solver in under an hour using the
   registry pattern.

---

## 7. What was deliberately not done

- **No rewrite of `core/`.** The numerics are in good shape. Targeted
  improvements only.
- **No new top-level directories** (`app_v2/`, `new/`, etc.). Refactor in
  place.
- **No micro-files.** A 30-line file with one tiny function is usually
  wrong; cohesive groups stay together.
- **No mixing structure and behavior.** Each commit is a refactor or a
  bug-fix, not both.
- **No new abstractions for hypothetical needs.** Extract a service when
  there is a second caller, not before.
