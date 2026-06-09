from typing import Dict, List

import numpy as np
import sympy as sp

from ._safe_funcs import SAFE_FUNCS


def build_custom_rhs(var_names: List[str], eq_lines: List[str], params: Dict[str, float]):
    n = len(var_names)
    if len(eq_lines) != n:
        raise ValueError(f"Need exactly {n} equations (one per variable). Got {len(eq_lines)}.")

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
        expr = sp.sympify(s, locals=locals_dict)
        exprs.append(expr)

    missing = (
        set().union(*(e.free_symbols for e in exprs))
        - {t_sym}
        - set(var_syms)
        - set(param_syms.values())
    )
    if missing:
        missing_names = ", ".join(sorted(sym.name for sym in missing))
        raise ValueError(f"Missing parameters in equations: {missing_names}")

    args = [t_sym] + list(var_syms) + [param_syms[k] for k in params.keys()]
    f = sp.lambdify(args, exprs, modules=["numpy"])
    param_values = [float(params[k]) for k in params.keys()]

    def rhs(t, y):
        vals = [t] + list(y) + param_values
        out = f(*vals)
        return np.array(out, dtype=float)

    return rhs
