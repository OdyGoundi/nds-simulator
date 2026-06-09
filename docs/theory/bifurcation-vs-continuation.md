# Bifurcation vs Continuation

Two ways to sweep a parameter and visualise how a system's long-term
behaviour changes.

---

## What a bifurcation diagram shows

For each value of the control parameter `μ`, the system is integrated,
the transient is discarded, and a few characteristic points are kept —
usually Poincaré crossings (see
[`poincare-section.md`](./poincare-section.md)) or the local extrema of
one variable. Plotting those points against `μ` reveals the
**asymptotic** behaviour of the system:

- One point per `μ` → simple periodic motion.
- A few discrete points → period-doubled / multi-period motion.
- A dense cloud or band → likely chaotic.

Transitions like period-doubling cascades and the route to chaos
become directly visible.

---

## Bifurcation mode (reset ICs)

- Each `μ` starts from the same initial conditions.
- Runs are independent of each other.
- Mimics the standard textbook diagram.
- Captures coexisting attractors and abrupt jumps between branches.

**Trade-off:** every step pays the full transient cost.

---

## Continuation mode (warm start)

- Each `μ` starts from the previous run's final state, so the numerical
  procedure "follows" a single solution branch as the parameter changes.
- Transients are short, so it runs much faster.
- Better suited to gradual, smooth parameter changes.

**Trade-off:** can stick to one branch and miss coexisting attractors
or sudden jumps.

---

## When to use which

- **Exploring / preview**: continuation, for speed.
- **Mapping multiple branches or chaotic jumps**: bifurcation (reset).
- **Publication-quality**: often combine both — continuation through
  smooth regions, selective resets near suspected jumps.
