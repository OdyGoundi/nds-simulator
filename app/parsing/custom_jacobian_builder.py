from typing import Callable, Dict, List, Tuple

import numpy as np
import sympy as sp

from ._safe_funcs import SAFE_FUNCS


def build_custom_rhs_and_jacobian(
    var_names: List[str],
    eq_lines: List[str],
    params: Dict[str, float],
) -> Tuple[
    Callable[[float, np.ndarray], np.ndarray],
    Callable[[float, np.ndarray], np.ndarray],
]:
    n = len(var_names)
    if len(eq_lines) != n:
        raise ValueError(f"Need exactly {n} equations. Got {len(eq_lines)}.")

    t_sym = sp.Symbol("t")
    var_syms = sp.symbols(var_names)
    param_syms = {k: sp.Symbol(k) for k in params.keys()}

    locals_dict = {
        **SAFE_FUNCS,
        "t": t_sym,
        **{name: sym for name, sym in zip(var_names, var_syms)},
        **param_syms,
    }

    exprs = []
    for i, line in enumerate(eq_lines):
        s = (line or "").strip()
        if not s:
            raise ValueError(f"Equation {i+1} is empty.")
        exprs.append(sp.sympify(s, locals=locals_dict))

    missing = (
        set().union(*(e.free_symbols for e in exprs))
        - {t_sym}
        - set(var_syms)
        - set(param_syms.values())
    )
    if missing:
        missing_names = ", ".join(sorted(sym.name for sym in missing))
        raise ValueError(f"Missing parameters in equations: {missing_names}")

    J = sp.Matrix(exprs).jacobian(var_syms)

    args = [t_sym] + list(var_syms) + [param_syms[k] for k in params.keys()]
    f_rhs = sp.lambdify(args, exprs, modules=["numpy"])
    f_jac = sp.lambdify(args, J, modules=["numpy"])

    param_values = [float(params[k]) for k in params.keys()]

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        vals = [float(t)] + list(y) + param_values
        out = f_rhs(*vals)
        return np.array(out, dtype=float)

    def jac(t: float, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        vals = [float(t)] + list(y) + param_values
        Jn = f_jac(*vals)
        return np.array(Jn, dtype=float)

    return rhs, jac


def build_custom_symbolic_jacobian_str(
    var_names: List[str],
    eq_lines: List[str],
    params: Dict[str, float],
) -> str:
    n = len(var_names)
    if len(eq_lines) != n:
        raise ValueError(f"Need exactly {n} equations. Got {len(eq_lines)}.")

    t_sym = sp.Symbol("t")
    var_syms = sp.symbols(var_names)
    param_syms = {k: sp.Symbol(k) for k in params.keys()}

    locals_dict = {
        **SAFE_FUNCS,
        "t": t_sym,
        **{name: sym for name, sym in zip(var_names, var_syms)},
        **param_syms,
    }

    exprs = []
    for i, line in enumerate(eq_lines):
        s = (line or "").strip()
        if not s:
            raise ValueError(f"Equation {i+1} is empty.")
        exprs.append(sp.sympify(s, locals=locals_dict))

    J = sp.Matrix(exprs).jacobian(var_syms)
    return sp.pretty(J, use_unicode=False).strip()
