from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple, cast

import numpy as np
import sympy as sp
from sympy.printing.pycode import pycode

try:
    import numba as nb  # type: ignore
except Exception:  # pragma: no cover
    nb = None


SAFE_FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "abs": sp.Abs,
}

_CUSTOM_CACHE: Dict[Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]], Tuple[object, object]] = {}
_CUSTOM_SYMPLECTIC_CACHE: Dict[
    Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]],
    Tuple[object, object],
] = {}


def _require_numba():
    if nb is None:  # pragma: no cover
        raise RuntimeError("Numba is required for custom Numba compilation.")
    return nb


def _expr_to_code(expr: sp.Expr) -> str:
    code = cast(str, pycode(expr, standard="numpy"))
    return str(code).replace("numpy.", "np.")


def _parse_expressions(
    var_names: Sequence[str],
    eq_lines: Sequence[str],
    param_names: Sequence[str],
) -> Tuple[List[sp.Expr], Sequence[sp.Symbol], Sequence[sp.Symbol]]:
    n = len(var_names)
    if len(eq_lines) != n:
        raise ValueError(f"Need exactly {n} equations (one per variable). Got {len(eq_lines)}.")

    t_sym = sp.Symbol("t")
    orig_var_syms = sp.symbols(list(var_names))
    if isinstance(orig_var_syms, sp.Symbol):
        orig_var_syms = (orig_var_syms,)
    orig_param_syms = {k: sp.Symbol(k) for k in param_names}

    locals_dict = {
        **SAFE_FUNCS,
        "t": t_sym,
        **{name: sym for name, sym in zip(var_names, orig_var_syms)},
        **orig_param_syms,
    }

    exprs: List[sp.Expr] = []
    for i, line in enumerate(eq_lines):
        s = (line or "").strip()
        if not s:
            raise ValueError(f"Equation {i+1} is empty.")
        exprs.append(sp.sympify(s, locals=locals_dict))

    missing = (
        set().union(*(e.free_symbols for e in exprs))
        - {t_sym}
        - set(orig_var_syms)
        - set(orig_param_syms.values())
    )
    if missing:
        missing_names = ", ".join(sorted(str(sym) for sym in missing))
        raise ValueError(f"Missing parameters in equations: {missing_names}")

    safe_var_syms = sp.symbols([f"_x{i}" for i in range(n)])
    if isinstance(safe_var_syms, sp.Symbol):
        safe_var_syms = (safe_var_syms,)
    safe_param_syms = sp.symbols([f"_p{i}" for i in range(len(param_names))])
    if isinstance(safe_param_syms, sp.Symbol):
        safe_param_syms = (safe_param_syms,)

    subs_map = {
        **{orig_var_syms[i]: safe_var_syms[i] for i in range(n)},
        **{orig_param_syms[name]: safe_param_syms[i] for i, name in enumerate(param_names)},
    }
    exprs_safe = [expr.xreplace(subs_map) for expr in exprs]
    return exprs_safe, safe_var_syms, safe_param_syms


def build_custom_numba_rhs(
    var_names: Sequence[str],
    eq_lines: Sequence[str],
    param_names: Sequence[str],
):
    nb = _require_numba()
    key = (tuple(var_names), tuple(eq_lines), tuple(param_names))
    cached = _CUSTOM_CACHE.get(key)
    if cached is not None and cached[0] is not None:
        return cached[0]

    exprs, safe_vars, safe_params = _parse_expressions(var_names, eq_lines, param_names)

    n = len(exprs)
    lines = ["def rhs(t, y, p):"]
    for i in range(n):
        lines.append(f"    {safe_vars[i]} = y[{i}]")
    for i in range(len(param_names)):
        lines.append(f"    {safe_params[i]} = p[{i}]")
    lines.append(f"    out = np.empty({n}, dtype=np.float64)")
    for i, expr in enumerate(exprs):
        lines.append(f"    out[{i}] = {_expr_to_code(expr)}")
    lines.append("    return out")
    src = "\n".join(lines)
    ns = {
        "np": np,
        "math": math,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "abs": np.abs,
    }
    exec(src, ns)
    rhs_py = ns["rhs"]
    rhs_nb = nb.njit(cache=True, fastmath=True)(rhs_py)

    _CUSTOM_CACHE[key] = (rhs_nb, cached[1] if cached is not None else None)
    return rhs_nb


def build_custom_numba_rhs_and_jacobian(
    var_names: Sequence[str],
    eq_lines: Sequence[str],
    param_names: Sequence[str],
):
    nb = _require_numba()
    key = (tuple(var_names), tuple(eq_lines), tuple(param_names))
    cached = _CUSTOM_CACHE.get(key)
    if cached is not None and cached[0] is not None and cached[1] is not None:
        return cached[0], cached[1]

    exprs, safe_vars, safe_params = _parse_expressions(var_names, eq_lines, param_names)
    J = sp.Matrix(exprs).jacobian(list(safe_vars))
    n = len(exprs)

    rhs_lines = ["def rhs(t, y, p):"]
    for i in range(n):
        rhs_lines.append(f"    {safe_vars[i]} = y[{i}]")
    for i in range(len(param_names)):
        rhs_lines.append(f"    {safe_params[i]} = p[{i}]")
    rhs_lines.append(f"    out = np.empty({n}, dtype=np.float64)")
    for i, expr in enumerate(exprs):
        rhs_lines.append(f"    out[{i}] = {_expr_to_code(expr)}")
    rhs_lines.append("    return out")

    jac_lines = ["def jac(t, y, p):"]
    for i in range(n):
        jac_lines.append(f"    {safe_vars[i]} = y[{i}]")
    for i in range(len(param_names)):
        jac_lines.append(f"    {safe_params[i]} = p[{i}]")
    jac_lines.append(f"    out = np.empty(({n}, {n}), dtype=np.float64)")
    for i in range(n):
        for j in range(n):
            jac_lines.append(f"    out[{i}, {j}] = {_expr_to_code(J[i, j])}")
    jac_lines.append("    return out")

    src = "\n".join(rhs_lines + [""] + jac_lines)
    ns = {
        "np": np,
        "math": math,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "abs": np.abs,
    }
    exec(src, ns)
    rhs_py = ns["rhs"]
    jac_py = ns["jac"]

    rhs_nb = nb.njit(cache=True, fastmath=True)(rhs_py)
    jac_nb = nb.njit(cache=True, fastmath=True)(jac_py)

    _CUSTOM_CACHE[key] = (rhs_nb, jac_nb)
    return rhs_nb, jac_nb


def build_custom_numba_symplectic_functions(
    var_names: Sequence[str],
    eq_lines: Sequence[str],
    param_names: Sequence[str],
):
    nb = _require_numba()
    key = (tuple(var_names), tuple(eq_lines), tuple(param_names))
    cached = _CUSTOM_SYMPLECTIC_CACHE.get(key)
    if cached is not None:
        return cached

    n = len(var_names)
    if n % 2 != 0:
        raise ValueError("Symplectic solvers require an even number of variables [q..., p...].")
    if len(eq_lines) != n:
        raise ValueError(f"Need exactly {n} equations (one per variable). Got {len(eq_lines)}.")

    n_q = n // 2
    t_sym = sp.Symbol("t")
    orig_var_syms = sp.symbols(list(var_names))
    if isinstance(orig_var_syms, sp.Symbol):
        orig_var_syms = (orig_var_syms,)
    q_syms = orig_var_syms[:n_q]
    p_syms = orig_var_syms[n_q:]
    orig_param_syms = {k: sp.Symbol(k) for k in param_names}

    locals_dict = {
        **SAFE_FUNCS,
        "t": t_sym,
        **{name: sym for name, sym in zip(var_names, orig_var_syms)},
        **orig_param_syms,
    }

    exprs: List[sp.Expr] = []
    for i, line in enumerate(eq_lines):
        s = (line or "").strip()
        if not s:
            raise ValueError(f"Equation {i+1} is empty.")
        exprs.append(sp.sympify(s, locals=locals_dict))

    missing = (
        set().union(*(e.free_symbols for e in exprs))
        - {t_sym}
        - set(orig_var_syms)
        - set(orig_param_syms.values())
    )
    if missing:
        missing_names = ", ".join(sorted(str(sym) for sym in missing))
        raise ValueError(f"Missing parameters in equations: {missing_names}")

    dq_exprs = exprs[:n_q]
    dp_exprs = exprs[n_q:]
    q_set = set(q_syms)
    p_set = set(p_syms)

    for i, expr in enumerate(dq_exprs):
        bad = expr.free_symbols & q_set
        if bad:
            bad_names = ", ".join(sorted(str(sym) for sym in bad))
            raise ValueError(
                f"Symplectic dq/dt equation {i+1} depends on q vars: {bad_names}."
            )

    for i, expr in enumerate(dp_exprs):
        bad = expr.free_symbols & p_set
        if bad:
            bad_names = ", ".join(sorted(str(sym) for sym in bad))
            eq_idx = n_q + i + 1
            raise ValueError(
                f"Symplectic dp/dt equation {eq_idx} depends on p vars: {bad_names}."
            )

    safe_var_syms = sp.symbols([f"_x{i}" for i in range(n)])
    if isinstance(safe_var_syms, sp.Symbol):
        safe_var_syms = (safe_var_syms,)
    safe_param_syms = sp.symbols([f"_p{i}" for i in range(len(param_names))])
    if isinstance(safe_param_syms, sp.Symbol):
        safe_param_syms = (safe_param_syms,)

    subs_map = {
        **{orig_var_syms[i]: safe_var_syms[i] for i in range(n)},
        **{orig_param_syms[name]: safe_param_syms[i] for i, name in enumerate(param_names)},
    }
    dq_exprs_safe = [expr.xreplace(subs_map) for expr in dq_exprs]
    dp_exprs_safe = [expr.xreplace(subs_map) for expr in dp_exprs]

    dq_lines = ["def dq_dt(t, p, params):"]
    for i in range(n_q):
        dq_lines.append(f"    {safe_var_syms[n_q + i]} = p[{i}]")
    for i in range(len(param_names)):
        dq_lines.append(f"    {safe_param_syms[i]} = params[{i}]")
    dq_lines.append(f"    out = np.empty({n_q}, dtype=np.float64)")
    for i, expr in enumerate(dq_exprs_safe):
        dq_lines.append(f"    out[{i}] = {_expr_to_code(expr)}")
    dq_lines.append("    return out")

    dp_lines = ["def dp_dt(t, q, params):"]
    for i in range(n_q):
        dp_lines.append(f"    {safe_var_syms[i]} = q[{i}]")
    for i in range(len(param_names)):
        dp_lines.append(f"    {safe_param_syms[i]} = params[{i}]")
    dp_lines.append(f"    out = np.empty({n_q}, dtype=np.float64)")
    for i, expr in enumerate(dp_exprs_safe):
        dp_lines.append(f"    out[{i}] = {_expr_to_code(expr)}")
    dp_lines.append("    return out")

    src = "\n".join(dq_lines + [""] + dp_lines)
    ns = {
        "np": np,
        "math": math,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "abs": np.abs,
    }
    exec(src, ns)
    dq_dt_py = ns["dq_dt"]
    dp_dt_py = ns["dp_dt"]

    dq_dt_nb = nb.njit(cache=True, fastmath=True)(dq_dt_py)
    dp_dt_nb = nb.njit(cache=True, fastmath=True)(dp_dt_py)

    _CUSTOM_SYMPLECTIC_CACHE[key] = (dq_dt_nb, dp_dt_nb)
    return dq_dt_nb, dp_dt_nb
