# Poincaré Sections

A Poincaré section turns a continuous trajectory into a discrete
sequence of intersection points with a chosen surface. The reduced
picture makes it much easier to tell whether the motion is periodic or
chaotic, and forms the basis for bifurcation diagrams (see
[`bifurcation-vs-continuation.md`](./bifurcation-vs-continuation.md)).

---

## Section definitions

- **Plane**: `y[section_index] = section_value`
- **Expression**: `f(t, y, params) = 0` (sympy-backed; for custom systems)

If an expression is provided, the plane settings are ignored.

---

## Hit detection

The trajectory is only known at the integration steps `t_k`, so a
crossing rarely lands exactly on a sample. The app supports two
strategies.

### `crossing` (sign change + interpolation)

Define `h(t) = x_s(t) - v`. The crossing happens where `h(t) = 0`. If

```
h(t_k) · h(t_{k+1}) < 0
```

then `h` changed sign between `t_k` and `t_{k+1}`, so the crossing
point is estimated by linear interpolation between the two samples.
The direction of the crossing follows the sign of `ḣ = ẋ_s`:

- `+1` (upward): trajectory passed from negative to positive `h`.
- `-1` (downward): from positive to negative.
- `0`: both directions accepted.

With adaptive IVP solvers this is implemented through SciPy event
detection, with optional early stop once enough hits are collected.
With fixed-step solvers, detection uses sampled points with the same
interpolation rule.

### `slab` (tolerance band)

Accept every sample with `|f| ≤ tol`. Simpler but less precise — the
"hits" are now a thin band rather than the surface itself.

---

## Time sampling vs crossing detection

A common alternative is to keep trajectory values at fixed time
intervals (stroboscopic sampling). That is **not** the same as a
Poincaré section: it does not check whether a crossing actually
occurred at that moment. The `crossing` and `slab` methods both
target the section explicitly.

---

## Notes

- Apply a long enough transient cut so early points are not still
  startup behaviour.
- For chaotic regimes, increase the integration horizon so the section
  has enough crossings to be representative.
