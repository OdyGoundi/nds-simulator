from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import math

try:
    from scipy.integrate import solve_ivp
except Exception:  # pragma: no cover
    solve_ivp = None

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


RhsFn = Callable[..., np.ndarray]


@dataclass(frozen=True)
class PoincareConfig:
    # section: y[section_index] = section_value
    section_index: int
    section_value: float = 0.0

    # direction: +1 upward crossing, -1 downward crossing, 0 both
    direction: int = +1

    # method:
    # - "crossing": detect sign change and interpolate
    # - "slab": accept points within |y - value| <= tol
    method: str = "crossing"
    tol: float = 1e-3

    # discard transient:
    transient_steps: int = 0
    # optional general section: expression f(t, y, params) = 0
    section_expr: str = ""
    # variable names for expression mapping (one per state dimension)
    section_vars: Tuple[str, ...] = ()


def _normalize_section_expr(expr_text: str) -> str:
    s = (expr_text or "").strip()
    if "=" in s:
        lhs, rhs = s.split("=", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if not lhs or not rhs:
            raise ValueError("Section equation must include both sides of '='.")
        return f"({lhs}) - ({rhs})"
    return s


def _build_section_expr_func(
    expr_text: str,
    var_names: Sequence[str],
    params: Dict[str, float],
):
    try:
        import sympy as sp
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("sympy is required for section expressions.") from exc

    safe_funcs = {
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
    }

    t_sym = sp.Symbol("t")
    var_syms = sp.symbols(list(var_names))
    param_syms = {k: sp.Symbol(k) for k in params.keys()}

    locals_dict = {
        **safe_funcs,
        "t": t_sym,
        **{name: sym for name, sym in zip(var_names, var_syms)},
        **param_syms,
    }

    expr_str = _normalize_section_expr(expr_text)
    expr = sp.sympify(expr_str, locals=locals_dict)

    known = {t_sym} | set(var_syms) | set(param_syms.values())
    unknown = expr.free_symbols - known
    if unknown:
        names = ", ".join(sorted(str(s) for s in unknown))
        raise ValueError(f"Unknown symbols in section expression: {names}")

    args = [t_sym] + list(var_syms) + [param_syms[k] for k in params.keys()]
    f = sp.lambdify(args, expr, modules=["numpy"])
    param_values = [float(params[k]) for k in params.keys()]

    def section_eval(t, y):
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim == 1:
            vals = [float(t)] + list(y_arr) + param_values
            return float(f(*vals))
        if y_arr.ndim == 2:
            t_arr = np.asarray(t, dtype=float)
            vals = [t_arr] + [y_arr[i, :] for i in range(y_arr.shape[0])] + param_values
            return np.asarray(f(*vals), dtype=float)
        raise ValueError("y must be 1D or 2D")

    return section_eval


@dataclass(frozen=True)
class SweepConfig:
    param_name: str
    start: float
    stop: float
    step: float


def _validate_direction(direction: int) -> None:
    if direction not in (-1, 0, +1):
        raise ValueError("direction must be one of: -1, 0, +1")


def _frange_inclusive(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("step must be > 0")
    # include stop (numerically safe)
    n = int(np.floor((stop - start) / step + 1e-12)) + 1
    vals = start + step * np.arange(n, dtype=float)
    vals = vals[vals <= stop + 1e-12]
    return vals


def poincare_section(
    t: np.ndarray,
    y: np.ndarray,
    cfg: PoincareConfig,
    params: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract Poincaré section hits from a trajectory.

    Parameters
    ----------
    t : (n,) array
    y : (d, n) array
    cfg : PoincareConfig
    params : optional dict
        Parameter values used when cfg.section_expr is set.

    Returns
    -------
    t_hits : (m,) array
    y_hits : (d, m) array
    """
    _validate_direction(cfg.direction)

    if y.ndim != 2:
        raise ValueError("y must be shape (d, n)")
    d, n = y.shape
    if n != t.shape[0]:
        raise ValueError("t and y length mismatch")
    if not (0 <= cfg.section_index < d):
        raise ValueError("section_index out of range")

    i0 = int(cfg.transient_steps)
    if i0 < 0:
        i0 = 0
    if i0 >= n - 1:
        return np.array([], dtype=float), np.zeros((d, 0), dtype=float)

    t2 = t[i0:]
    y2 = y[:, i0:]

    s = cfg.section_index
    v = cfg.section_value
    params = params or {}

    section_expr = str(cfg.section_expr or "").strip()
    if section_expr:
        if not cfg.section_vars:
            raise ValueError("section_vars must be provided when section_expr is set.")
        if len(cfg.section_vars) != d:
            raise ValueError("section_vars length must match y dimension.")
        section_fn = _build_section_expr_func(section_expr, cfg.section_vars, params)
        ds = np.asarray(section_fn(t2, y2), dtype=float).ravel()
        if ds.size != t2.size:
            raise ValueError("section_expr must evaluate to a 1D array matching t.")
    else:
        ds = y2[s, :] - v

    if cfg.method.lower() == "slab":
        # slab filter: |y_s - v| <= tol, optional direction via dy_s/dt sign
        mask = np.abs(ds) <= float(cfg.tol)

        if cfg.direction != 0:
            # finite diff for dg/dt sign
            dt = np.diff(t2)
            dy = np.diff(ds)
            # align with points 1..end
            deriv = np.zeros_like(t2)
            deriv[1:] = np.divide(dy, dt, out=np.zeros_like(dy), where=dt != 0.0)
            if cfg.direction == +1:
                mask = mask & (deriv > 0)
            else:
                mask = mask & (deriv < 0)

        idx = np.where(mask)[0]
        if idx.size == 0:
            return np.array([], dtype=float), np.zeros((d, 0), dtype=float)

        return t2[idx].copy(), y2[:, idx].copy()

    # default: "crossing"
    t_hits: List[float] = []
    y_hits: List[np.ndarray] = []

    for i in range(1, ds.shape[0]):
        prev = ds[i - 1]
        curr = ds[i]

        # direction filtering
        if cfg.direction == +1:
            cond = (prev < 0.0) and (curr >= 0.0)
        elif cfg.direction == -1:
            cond = (prev > 0.0) and (curr <= 0.0)
        else:
            cond = (prev == 0.0) or (curr == 0.0) or (prev * curr < 0.0)

        if not cond:
            continue

        denom = (curr - prev)
        if denom == 0.0:
            # rare: both exactly equal; fall back to taking the second point
            alpha = 1.0
        else:
            alpha = (0.0 - prev) / denom  # crossing at ds = 0

        # clamp alpha in [0,1] for numerical safety
        if alpha < 0.0:
            alpha = 0.0
        elif alpha > 1.0:
            alpha = 1.0

        th = t2[i - 1] + alpha * (t2[i] - t2[i - 1])
        yh = y2[:, i - 1] + alpha * (y2[:, i] - y2[:, i - 1])

        t_hits.append(float(th))
        y_hits.append(yh)

    if not t_hits:
        return np.array([], dtype=float), np.zeros((d, 0), dtype=float)

    t_hits_arr = np.array(t_hits, dtype=float)
    y_hits_arr = np.stack(y_hits, axis=1)  # (d, m)
    return t_hits_arr, y_hits_arr

def _make_poincare_event(
    section_index: int,
    section_value: float,
    direction: int,
    section_fn=None,
):
    def event(t, y):
        if section_fn is None:
            return float(y[section_index] - section_value)
        return float(section_fn(t, y))
    # assign event attributes using setattr to satisfy static type checkers
    setattr(event, "terminal", False)
    setattr(event, "direction", int(direction))
    return event

def sweep_poincare_events_ivp(
    rhs: RhsFn,
    y0: Sequence[float],
    t_span: Tuple[float, float],
    base_params: Dict[str, float],
    sweep: SweepConfig,
    poincare: PoincareConfig,
    t_step: float = 0.01,
    solve_options: Optional[Dict] = None,
    output_indices: Optional[Sequence[int]] = None,
    include_all_state: bool = False,
    warm_start: bool = False,
    max_hits: int = 100,
    chunk_time: float = 2.0,
    early_stop: bool = True,
):
    """
    Fast IVP sweep using solve_ivp EVENTS for Poincaré crossings.
    Collects up to `max_hits` hits per parameter value and can early-stop.

    Notes
    -----
    - Only supports poincare.method == "crossing" (events are crossings).
    - Transient removal is approximated as transient_time = transient_steps * t_step.
    """
    if solve_ivp is None:
        raise RuntimeError("scipy is required for event-based IVP sweep")

    if solve_options is None:
        solve_options = {}
    if output_indices is None:
        output_indices = [0]

    if poincare.method.lower() != "crossing":
        raise ValueError("Event-based sweep supports only method='crossing'")

    _validate_direction(poincare.direction)

    t0, tf = float(t_span[0]), float(t_span[1])
    if tf <= t0:
        raise ValueError("t_span must satisfy tf > t0")

    # convert transient_steps (samples) -> transient time (approx)
    transient_time = max(0.0, float(poincare.transient_steps) * float(t_step))
    t_trans_end = min(tf, t0 + transient_time)

    param_vals = _frange_inclusive(sweep.start, sweep.stop, sweep.step)

    y0_base = np.array(y0, dtype=float).copy()
    y0_curr = y0_base.copy()

    rows: List[Dict[str, float]] = []

    for pv in param_vals:
        params = dict(base_params)
        params[sweep.param_name] = float(pv)

        def rhs_wrapped(t, y):
            return rhs(t, y, **params)

        # define event: g(t, y) = 0 (plane or general expression)
        section_fn = None
        section_expr = str(poincare.section_expr or "").strip()
        if section_expr:
            if not poincare.section_vars:
                raise ValueError("section_vars must be provided when section_expr is set.")
            section_fn = _build_section_expr_func(section_expr, poincare.section_vars, params)

        event_fn = _make_poincare_event(
            poincare.section_index,
            poincare.section_value,
            poincare.direction,
            section_fn=section_fn,
        )

        # --- 1) burn transient (no event collection needed) ---
        y_start = y0_curr.copy() if warm_start else y0_base.copy()

        if t_trans_end > t0:
            sol_tr = solve_ivp(
                rhs_wrapped,
                (t0, t_trans_end),
                y_start,
                events=None,
                dense_output=False,
                t_eval=None,              # important: do NOT force output grid
                **solve_options,
            )
            if not sol_tr.success:
                # reset if bifurcation mode
                if not warm_start:
                    y0_curr = y0_base.copy()
                continue
            y_start = np.array(sol_tr.y[:, -1], dtype=float).copy()

        # --- 2) collect events in chunks until max_hits or tf ---
        hits_t: List[float] = []
        hits_y: List[np.ndarray] = []

        t_cur = t_trans_end
        y_cur = y_start

        # guard: chunk_time should be positive
        ct = float(chunk_time)
        if ct <= 0.0:
            ct = max(1.0, 20.0 * float(t_step))
        
        # If early_stop is disabled, use a single chunk to tf and ignore max_hits stopping
        if not bool(early_stop):
            ct = max(0.0, tf - t_cur)

        while t_cur < tf and len(hits_t) < int(max_hits):
            t_next = min(tf, t_cur + ct)

            sol = solve_ivp(
                rhs_wrapped,
                (t_cur, t_next),
                y_cur,
                events=event_fn,
                dense_output=False,
                t_eval=None,              # important
                **solve_options,
            )

            if not sol.success:
                break

            # events: sol.t_events[0], sol.y_events[0]
            if sol.t_events and len(sol.t_events[0]) > 0:
                te = sol.t_events[0]
                ye = sol.y_events[0]  # shape (m, d)
                for k in range(te.shape[0]):
                    hits_t.append(float(te[k]))
                    hits_y.append(np.array(ye[k], dtype=float))
                    if len(hits_t) >= int(max_hits):
                        break

            # advance
            t_cur = float(sol.t[-1])
            y_cur = np.array(sol.y[:, -1], dtype=float).copy()

        # update warm-start state for next pv
        if warm_start:
            y0_curr = y_cur.copy()
        else:
            y0_curr = y0_base.copy()

        if len(hits_t) == 0:
            continue

        # keep last max_hits (safety)
        if len(hits_t) > int(max_hits):
            hits_t = hits_t[-int(max_hits):]
            hits_y = hits_y[-int(max_hits):]

        # stack to (d, m)
        y_hits = np.stack(hits_y, axis=1)
        t_hits = np.array(hits_t, dtype=float)

        # build rows
        for j in range(t_hits.size):
            row = {sweep.param_name: float(pv), "t_hit": float(t_hits[j])}

            if include_all_state:
                for kk in range(y_hits.shape[0]):
                    row[f"y{kk}"] = float(y_hits[kk, j])
            else:
                for kk in output_indices:
                    row[f"y{int(kk)}"] = float(y_hits[int(kk), j])

            rows.append(row)

    if pd is None:
        return rows
    return pd.DataFrame(rows)

def sweep_poincare(
    rhs: RhsFn,
    y0: Sequence[float],
    t_span: Tuple[float, float],
    base_params: Dict[str, float],
    sweep: SweepConfig,
    poincare: PoincareConfig,
    solver_kind: str = "ivp",
    t_step: float = 0.01,
    solve_options: Optional[Dict] = None,
    output_indices: Optional[Sequence[int]] = None,
    include_all_state: bool = False,
    warm_start: bool = False, 
):
    """
    Parametric sweep producing Poincaré hits suitable for plotting & CSV export.

    Returns a pandas DataFrame if pandas is available; otherwise returns a list of dict rows.
    """
    if solve_options is None:
        solve_options = {}
    if output_indices is None:
        output_indices = [0]

    # local import to avoid circular deps (core/solver.py)
    from core.solver import integrate_system, integrate_system_rk4  # type: ignore

    param_vals = _frange_inclusive(sweep.start, sweep.stop, sweep.step)

    rows: List[Dict[str, float]] = []
    
    y0_curr = np.array(y0, dtype=float).copy()

    for pv in param_vals:
        params = dict(base_params)
        params[sweep.param_name] = float(pv)

        def rhs_wrapped(t, y):
            return rhs(t, y, **params)

        if solver_kind.lower() == "ivp":
            sol = integrate_system(rhs_wrapped, t_span, y0_curr, t_step=t_step, **solve_options)
        else:
            sol = integrate_system_rk4(rhs_wrapped, t_span, y0_curr, t_step=t_step)

        if not sol.success:
            # keep a marker row (optional)
            continue
        
        if warm_start:
            # next parameter value starts from last state of current run
            y0_curr = np.array(sol.y[:, -1], dtype=float).copy()
        else:
            # reset for bifurcation-style sweep
            y0_curr = np.array(y0, dtype=float).copy()


        t_hits, y_hits = poincare_section(sol.t, sol.y, poincare, params=params)

        MAX_HITS = 100
        if t_hits.size > MAX_HITS:
            t_hits = t_hits[-MAX_HITS:]
            y_hits = y_hits[:, -MAX_HITS:]

        if t_hits.size == 0:
            continue

        # Build output rows
        for j in range(t_hits.size):
            row = {
                sweep.param_name: float(pv),
                "t_hit": float(t_hits[j]),
            }

            if include_all_state:
                for k in range(y_hits.shape[0]):
                    row[f"y{k}"] = float(y_hits[k, j])
            else:
                for k in output_indices:
                    row[f"y{k}"] = float(y_hits[int(k), j])

            rows.append(row)

    if pd is None:
        return rows

    return pd.DataFrame(rows)


def export_csv(df_or_rows, filename: str) -> None:
    """
    Export sweep output to CSV.
    Accepts either a pandas DataFrame or list-of-dicts rows.
    """
    if pd is not None and hasattr(df_or_rows, "to_csv"):
        df_or_rows.to_csv(filename, index=False)
        return

    # fallback without pandas
    rows: List[Dict[str, float]] = list(df_or_rows)
    if not rows:
        # create empty file with no content (or raise)
        with open(filename, "w", encoding="utf-8") as f:
            f.write("")
        return

    cols = list(rows[0].keys())
    with open(filename, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
