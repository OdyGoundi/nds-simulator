"""Parsing/codegen for Numba-compatible custom system functions.

This module turns user equation strings into Python *source code* for
RHS/Jacobian/symplectic kernels. It does no JIT compilation — that lives
in `app/numba_custom.py`. The split keeps sympy concerns away from the
Numba decorator and the cache layer.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import sympy as sp
from sympy.printing.pycode import pycode

from app.parsing._safe_funcs import SAFE_FUNCS


def _expr_to_code(expr: sp.Expr) -> str:
    """Render a sympy expression as numpy-flavored Python source."""
    try:
        code = pycode(expr, standard="numpy")
    except ValueError:
        # Sympy >= 1.14 accepts only "python3" as standard.
        code = pycode(expr, standard="python3")
    return str(code).replace("numpy.", "np.")


def _parse_orig_exprs(
    var_names: Sequence[str],
    eq_lines: Sequence[str],
    param_names: Sequence[str],
) -> Tuple[List[sp.Expr], Sequence[sp.Symbol], dict]:
    """Parse equations through sympy. Validates equation count and missing params.

    Returns (exprs, orig_var_syms, orig_param_syms_dict). The expressions still
    use the user's original variable/parameter names — no safe substitution yet.
    """
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

    return exprs, orig_var_syms, orig_param_syms


def _substitute_safe_symbols(
    exprs: List[sp.Expr],
    orig_var_syms: Sequence[sp.Symbol],
    orig_param_syms: dict,
    param_names: Sequence[str],
) -> Tuple[List[sp.Expr], Sequence[sp.Symbol], Sequence[sp.Symbol]]:
    """Replace user variable/param symbols with safe placeholders (_x0, _p0, ...)."""
    n = len(orig_var_syms)
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


def parse_safe_exprs(
    var_names: Sequence[str],
    eq_lines: Sequence[str],
    param_names: Sequence[str],
) -> Tuple[List[sp.Expr], Sequence[sp.Symbol], Sequence[sp.Symbol]]:
    """Parse equations and return them with safe variable/param placeholders.

    Returns (safe_exprs, safe_var_syms, safe_param_syms).
    """
    exprs, orig_vars, orig_params = _parse_orig_exprs(var_names, eq_lines, param_names)
    return _substitute_safe_symbols(exprs, orig_vars, orig_params, param_names)


def parse_symplectic_safe_exprs(
    var_names: Sequence[str],
    eq_lines: Sequence[str],
    param_names: Sequence[str],
) -> Tuple[List[sp.Expr], List[sp.Expr], Sequence[sp.Symbol], Sequence[sp.Symbol], int]:
    """Parse symplectic equations and validate q/p separability.

    Raises if dq/dt depends on q vars or dp/dt depends on p vars.
    Returns (dq_safe_exprs, dp_safe_exprs, safe_var_syms, safe_param_syms, n_q).
    """
    n = len(var_names)
    if n % 2 != 0:
        raise ValueError("Symplectic solvers require an even number of variables [q..., p...].")

    n_q = n // 2
    exprs, orig_var_syms, orig_param_syms = _parse_orig_exprs(var_names, eq_lines, param_names)

    q_set = set(orig_var_syms[:n_q])
    p_set = set(orig_var_syms[n_q:])

    for i, expr in enumerate(exprs[:n_q]):
        bad = expr.free_symbols & q_set
        if bad:
            bad_names = ", ".join(sorted(str(sym) for sym in bad))
            raise ValueError(
                f"Symplectic dq/dt equation {i+1} depends on q vars: {bad_names}."
            )

    for i, expr in enumerate(exprs[n_q:]):
        bad = expr.free_symbols & p_set
        if bad:
            bad_names = ", ".join(sorted(str(sym) for sym in bad))
            eq_idx = n_q + i + 1
            raise ValueError(
                f"Symplectic dp/dt equation {eq_idx} depends on p vars: {bad_names}."
            )

    safe_exprs, safe_var_syms, safe_param_syms = _substitute_safe_symbols(
        exprs, orig_var_syms, orig_param_syms, param_names
    )
    return safe_exprs[:n_q], safe_exprs[n_q:], safe_var_syms, safe_param_syms, n_q


def render_rhs_source(
    safe_exprs: Sequence[sp.Expr],
    safe_vars: Sequence[sp.Symbol],
    safe_params: Sequence[sp.Symbol],
    n_params: int,
) -> str:
    """Render `def rhs(t, y, p): ...` as a Python source string."""
    n = len(safe_exprs)
    lines = ["def rhs(t, y, p):"]
    for i in range(n):
        lines.append(f"    {safe_vars[i]} = y[{i}]")
    for i in range(n_params):
        lines.append(f"    {safe_params[i]} = p[{i}]")
    lines.append(f"    out = np.empty({n}, dtype=np.float64)")
    for i, expr in enumerate(safe_exprs):
        lines.append(f"    out[{i}] = {_expr_to_code(expr)}")
    lines.append("    return out")
    return "\n".join(lines)


def render_jacobian_source(
    safe_exprs: Sequence[sp.Expr],
    safe_vars: Sequence[sp.Symbol],
    safe_params: Sequence[sp.Symbol],
    n_params: int,
) -> str:
    """Render `def jac(t, y, p): ...` as a Python source string."""
    n = len(safe_exprs)
    J = sp.Matrix(list(safe_exprs)).jacobian(list(safe_vars))
    lines = ["def jac(t, y, p):"]
    for i in range(n):
        lines.append(f"    {safe_vars[i]} = y[{i}]")
    for i in range(n_params):
        lines.append(f"    {safe_params[i]} = p[{i}]")
    lines.append(f"    out = np.empty(({n}, {n}), dtype=np.float64)")
    for i in range(n):
        for j in range(n):
            lines.append(f"    out[{i}, {j}] = {_expr_to_code(J[i, j])}")
    lines.append("    return out")
    return "\n".join(lines)


def render_symplectic_dqdt_source(
    dq_safe_exprs: Sequence[sp.Expr],
    safe_vars: Sequence[sp.Symbol],
    safe_params: Sequence[sp.Symbol],
    n_params: int,
    n_q: int,
) -> str:
    """Render `def dq_dt(t, p, params): ...` — depends only on p variables."""
    lines = ["def dq_dt(t, p, params):"]
    for i in range(n_q):
        lines.append(f"    {safe_vars[n_q + i]} = p[{i}]")
    for i in range(n_params):
        lines.append(f"    {safe_params[i]} = params[{i}]")
    lines.append(f"    out = np.empty({n_q}, dtype=np.float64)")
    for i, expr in enumerate(dq_safe_exprs):
        lines.append(f"    out[{i}] = {_expr_to_code(expr)}")
    lines.append("    return out")
    return "\n".join(lines)


def render_symplectic_dpdt_source(
    dp_safe_exprs: Sequence[sp.Expr],
    safe_vars: Sequence[sp.Symbol],
    safe_params: Sequence[sp.Symbol],
    n_params: int,
    n_q: int,
) -> str:
    """Render `def dp_dt(t, q, params): ...` — depends only on q variables."""
    lines = ["def dp_dt(t, q, params):"]
    for i in range(n_q):
        lines.append(f"    {safe_vars[i]} = q[{i}]")
    for i in range(n_params):
        lines.append(f"    {safe_params[i]} = params[{i}]")
    lines.append(f"    out = np.empty({n_q}, dtype=np.float64)")
    for i, expr in enumerate(dp_safe_exprs):
        lines.append(f"    out[{i}] = {_expr_to_code(expr)}")
    lines.append("    return out")
    return "\n".join(lines)
