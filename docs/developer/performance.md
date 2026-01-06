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
- Integration stops or continues based on event count

**Result:**  
Cost scales with number of crossings, not number of time steps.

---

## 2. Chunked integration (`chunk_time`)

### What
Instead of integrating from `t0 → tf` in a single solver call, the
integration is split into **time chunks** of fixed duration.

Example:

t ∈ [0, 2] → [2, 4] → [4, 6] → …

### Why

- Allows early stopping once enough hits are collected

- Prevents runaway integrations in chaotic regimes

- Improves responsiveness in Streamlit

### How

- Each chunk is an independent IVP call

- Final state of one chunk is initial state of the next

- Controlled by chunk_time parameter

### Rule of thumb:

- Small chunk_time → more solver calls, finer control

- Large chunk_time → fewer solver calls, faster but less reactive

## 3. Early Stopping 

### What 

Stop integrating as soon as enough Poincaré hits are collected.

### Why

- Bifurcation diagrams only need the asymptotic regime

- Integrating past that gives no new information

- Prevents wasting time in long chaotic transients

### How

- Track number of detected events

- Stop when hits >= max_hits

- Enabled via early_stop=True

## 4. Capped Poincaré hits (`max_hits`)

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

## 5. Warm start (continuation mode)

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

- Enabled via Sweep mode → Continuation

## 6. Transient removal (sweep-only)

### What

Ignore an initial fraction of each sweep integration.

### Why

- Ensures crossings belong to the attractor

- Matches standard practice in nonlinear dynamics

### How

- Compute estimated number of steps

- Convert transient fraction → transient steps

- Applied before event collection

## 7. Computational complexity (qualitative)

Let:

- *P* = number of sweep parameter values  
- *H* = `max_hits`  
- *C* = number of time chunks per run  

Then, approximately:

> **Cost ≈** `O(P × C × cost(IVP_chunk))`

Instead of the classical approach:

> **Cost ≈** `O(P × N_steps)`

where:

- `N_steps ≫ H`

---

**Key insight**

Because event-based integration detects only *relevant crossings* and
terminates early, the computational cost depends on the number of
*events* rather than the number of *time samples*.

-> This is the key reason why sweeps that originally took **tens of minutes**
now complete in **a few minutes**.


## 8. Practical tuning guidelines

| Goal                | Recommendation                           |
| ------------------- | ---------------------------------------- |
| Fast preview        | `tf_sweep ≈ 10–20`, `max_hits ≈ 50`      |
| Publication-quality | `tf_sweep ≈ 50–80`, `max_hits ≈ 100–200` |
| Very large sweeps   | Enable warm start + early stop           |
| Chaotic regimes     | Smaller `chunk_time`                     |

## 9. Why this design fits Streamlit

- No long blocking calls

- Deterministic memory usage

- Incremental generation supported

- Safe for cloud deployment limits

## 10. Summary


### Core performance principle

> **Compute only what you will actually plot.**

Event detection, chunked integration, and capped Poincaré hits transform
bifurcation diagrams from a **brute-force numerical task** into a
**scalable and efficient workflow**.

As a result, **NLDS remains interactive**, even for large parameter
sweeps and chaotic regimes.



