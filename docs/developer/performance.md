# ⚡ Performance Optimization

This document explains how the NLDS sweep engine achieves practical
performance for large bifurcation and continuation diagrams.

The focus is on **Tab 3 (Bifurcation / Poincaré sweep)**.

---

## 1. Event-based IVP integration

### What
Instead of sampling the full trajectory at fixed time steps and
post-processing crossings, the solver uses **event detection** to
identify Poincaré crossings *during* integration.

### Why
- Avoids storing dense trajectories
- Avoids scanning arrays of size O(N)
- Detects only physically relevant points

### How
- Uses `solve_ivp` with event functions
- Each event corresponds to a Poincaré crossing
- Integration covers the full `t0 → tf` window; internally it runs in
  short fixed chunks (an implementation detail, not a user setting),
  carrying the state from one chunk to the next
- Only the **last** `max_hits` crossings per parameter value are kept
  (bounded deque)

**Result:**  
Memory scales with the number of kept crossings, not the number of
time steps — no dense trajectory is ever stored.

---

## 2. Capped Poincaré hits (`max_hits`)

### What

Limit the number of Poincaré points stored per parameter value.

### Why

- Prevents memory explosion

- Matches bibliography (usually last 50–200 crossings)

- Improves plot readability

### How

"if len(hits) > max_hits:
    hits = hits[-max_hits:]"

This keeps only the last crossings, i.e. closest to the attractor.
The same tail-keeping semantics hold on every path: bounded deque in
`sweep_poincare_events_ivp`, ring buffer in the Numba kernel. The
effective cap is also clipped against the in-memory row budget
(`_effective_max_hits` in `app/sweep.py`).

## 3. Warm start (continuation mode)

### What

Reuse the final state of parameter value pᵢ as the initial condition
for pᵢ₊₁.

### Why

- Dramatically reduces transient length

- Ideal for smooth continuation diagrams

- Essential for large parameter ranges

### How 

- Store last state after each sweep step

- Use it as y0 for the next run

- Enabled via Sweep mode → Continuation (low->high or high->low; the
  descending variant reverses the parameter grid and always starts
  fresh — Continue is disabled there)

## 4. Parallel workers (independent mode)

### What

Distribute parameter values across multiple workers when the sweep
is **independent** (reset ICs, no warm start).

### Why

- Speeds up large parameter ranges on multi-core machines
- Keeps UI responsive for local runs
- Preserves correctness because runs are independent

### How

- Uses a process pool to evaluate chunks in parallel
- Disabled automatically for continuation (warm start)
- Typically limited to local runs (Streamlit Cloud disables it)

## 5. Transient removal (sweep-only)

### What

Ignore an initial fraction of each sweep integration.

### Why

- Ensures crossings belong to the attractor

- Matches standard practice in nonlinear dynamics

### How

- Compute estimated number of steps

- Convert transient fraction → transient steps

- Applied before event collection

## 6. Bounded plotting memory in Tab 3

### What

Keep only a bounded recent sweep buffer in full resolution and move older
rows into a bounded reservoir sample.

### Why

- Prevents unbounded Streamlit session-state growth
- Preserves recent rows exactly for continuation and inspection
- Retains older visual history without storing every point

### How

- Recent sweep rows are clipped to a fixed in-memory row budget
- Dropped rows are inserted into a fixed-capacity XY reservoir sample
- Plotting combines recent rows plus a downsampled view of the reservoir
- Recent and reservoir points now use the same black marker styling; the
  distinction is internal storage, not color coding

## 7. Computational complexity (qualitative)

Let:

- *P* = number of sweep parameter values
- *N* = integration steps per parameter value (`≈ (tf − t0) / dt`)
- *H* = `max_hits`

Then, approximately:

> **Compute cost ≈** `O(P × N)`
> **Memory ≈** `O(P × H)`, with `H ≪ N`

---

**Key insight**

Integration time still scales with the full time window — what the
design bounds is *storage*. No dense trajectory is kept; each parameter
value contributes only its last `H` crossings, and Tab 3 plotting state
is further bounded by the row budget plus reservoir (section 6).

The speedups come from three places: event detection (no dense output
to scan), warm start (shorter transients per parameter value), and the
Numba JIT kernels (cheaper per-step cost). Combined, sweeps that
originally took **tens of minutes** complete in **a few minutes**.

## 8. Numba acceleration (fixed-step)

### What
When the user selects **RK4** or **Symplectic Forest-Ruth**, the app attempts to
use the Numba backend for the fixed-step integrators.

### Why
- JIT-compiled loops are significantly faster for long fixed-step runs.
- Particularly helpful for dense sweeps and Hamiltonian systems.

### Notes
- The first run pays a compilation cost; later runs are fast.
- Event-based IVP acceleration still applies only to RK45/DOP853 + crossing.
- The Numba-vs-Python decision is centralised in `app/cache.py` (single
  trajectories) and `app/sweep.py` (sweeps); system and solver dispatch
  go through `app/services/system_registry.py` and
  `app/services/solver_policy.py`.


## 9. Practical tuning guidelines

Rules of thumb for **Lorenz bifurcation sweeps**. Express integration length as total steps:

`N_steps ~= (tf_sweep - t0) / dt_sweep`

Smaller `dt_sweep` means larger `N_steps` (slower) but more accurate crossings.

| Goal                | Recommendation                                                                 |
| ------------------- | ------------------------------------------------------------------------------ |
| Fast preview        | `N_steps` in the low thousands, `max_hits ~ 50`                                |
| Publication-quality | Increase `N_steps` and `max_hits` vs preview until the diagram stabilizes      |
| Very large sweeps   | Use warm start (continuation); for independent mode, enable parallel workers   |
| Chaotic regimes     | Increase `N_steps` if crossings are sparse; raise `max_hits` for a longer tail |

## 10. Why this design fits Streamlit

- No long blocking calls

- Deterministic memory usage

- Incremental generation supported

- Safe for cloud deployment limits

## 11. Summary


### Core performance principle

> **Store only what you will actually plot.**

Event detection, capped Poincaré hits, and Numba-compiled kernels
transform bifurcation diagrams from a **brute-force numerical task**
into a **scalable and efficient workflow**.

As a result, **NLDS remains interactive**, even for large parameter
sweeps and chaotic regimes.
