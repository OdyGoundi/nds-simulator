# Poincare Sections

A Poincare section reduces a continuous trajectory to a discrete set of
intersections with a surface. It is used for bifurcation diagrams and
qualitative analysis of attractors.

---

## Section definitions

- **Plane section**: `y[section_index] = section_value`
- **Expression**: `f(t, y, params) = 0` (custom equation, Sympy-backed)

The app lets you switch between the plane and the expression. If an expression
is provided, the plane settings are ignored.

---

## Hit detection methods

- **crossing**: Detects sign changes and linearly interpolates to the section.
- **slab**: Accepts points within a tolerance band `|f| <= tol`.

Direction filtering is available: `+1` (upward), `-1` (downward), `0` (both).

---

## Notes

- With adaptive IVP solvers, `crossing` can use event detection and early-stop
  once enough hits are collected.
- With fixed-step solvers, detection uses sampled points (with interpolation).
- Use a longer transient cut if early points are dominated by transients.
