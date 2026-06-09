from typing import Dict, List, Protocol, Tuple

import numpy as np
import sympy as sp

from ._safe_funcs import SAFE_FUNCS


class DQDT(Protocol):
    def __call__(self, t: float, p: np.ndarray) -> np.ndarray:
        ...


class DPDT(Protocol):
    def __call__(self, t: float, q: np.ndarray) -> np.ndarray:
        ...


def build_custom_symplectic_functions(
    var_names: List[str],
    eq_lines: List[str],
    params: Dict[str, float],
) -> Tuple[DQDT, DPDT]:
    n = len(var_names)
    if n % 2 != 0:
        raise ValueError("Symplectic solvers require an even number of variables [q..., p...].")
    if len(eq_lines) != n:
        raise ValueError(f"Need exactly {n} equations (one per variable). Got {len(eq_lines)}.")

    n_q = n // 2
    t_sym = sp.Symbol("t")
    var_syms = sp.symbols(var_names)
    q_syms = var_syms[:n_q]
    p_syms = var_syms[n_q:]
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

    dq_exprs = exprs[:n_q]
    dp_exprs = exprs[n_q:]
    q_set = set(q_syms)
    p_set = set(p_syms)

    for i, expr in enumerate(dq_exprs):
        bad = expr.free_symbols & q_set
        if bad:
            bad_names = ", ".join(sorted(sym.name for sym in bad))
            raise ValueError(
                f"Symplectic dq/dt equation {i+1} depends on q vars: {bad_names}."
            )

    for i, expr in enumerate(dp_exprs):
        bad = expr.free_symbols & p_set
        if bad:
            bad_names = ", ".join(sorted(sym.name for sym in bad))
            eq_idx = n_q + i + 1
            raise ValueError(
                f"Symplectic dp/dt equation {eq_idx} depends on p vars: {bad_names}."
            )

    dq_args = [t_sym] + list(p_syms) + [param_syms[k] for k in params.keys()]
    dp_args = [t_sym] + list(q_syms) + [param_syms[k] for k in params.keys()]
    f_dq = sp.lambdify(dq_args, dq_exprs, modules=["numpy"])
    f_dp = sp.lambdify(dp_args, dp_exprs, modules=["numpy"])
    param_values = [float(params[k]) for k in params.keys()]

    def dq_dt(t: float, p: np.ndarray) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        if p_arr.size != n_q:
            raise ValueError(f"dq_dt expects {n_q} p variables, got {p_arr.size}.")
        vals = [float(t)] + list(p_arr) + param_values
        out = f_dq(*vals)
        return np.array(out, dtype=float)

    def dp_dt(t: float, q: np.ndarray) -> np.ndarray:
        q_arr = np.asarray(q, dtype=float)
        if q_arr.size != n_q:
            raise ValueError(f"dp_dt expects {n_q} q variables, got {q_arr.size}.")
        vals = [float(t)] + list(q_arr) + param_values
        out = f_dp(*vals)
        return np.array(out, dtype=float)

    return dq_dt, dp_dt
