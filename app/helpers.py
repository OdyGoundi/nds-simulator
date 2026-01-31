import io
from typing import Callable, Dict, List, Protocol, Tuple

import numpy as np
import streamlit as st
import sympy as sp


SAFE_FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "abs": sp.Abs,
}


class DQDT(Protocol):
    def __call__(self, t: float, p: np.ndarray) -> np.ndarray:
        ...


class DPDT(Protocol):
    def __call__(self, t: float, q: np.ndarray) -> np.ndarray:
        ...


def parse_params(text: str) -> Dict[str, float]:
    """
    Parameters format:
      a=1.2
      b=3
    Empty lines ignored.
    """
    params: Dict[str, float] = {}
    for line in (text or "").splitlines():
        line = line.replace("\u00a0", " ").strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"Parameter line must be name=value. Got: '{line}'")
        name, val = line.split("=", 1)
        name = name.replace("\u00a0", " ").strip()
        val = val.replace("\u00a0", " ").strip()
        if name.lower() == "t":
            raise ValueError("Parameter name 't' is reserved for the independent variable; use other symbols for constants.")
        params[name] = float(val)
    return params


def parse_list_of_floats(text: str, n: int, label: str) -> np.ndarray:
    """
    Accept either:
      - one number per line
      - or comma/space separated
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError(f"{label} is empty.")
    tokens = raw.replace(",", " ").split()
    if len(tokens) != n:
        raise ValueError(f"{label} must have exactly {n} values. Got {len(tokens)}.")
    return np.array([float(t) for t in tokens], dtype=float)


def build_custom_rhs(var_names: List[str], eq_lines: List[str], params: Dict[str, float]):
    """
    Build rhs(t,y) from user equations using sympy. Supports time-dependent
    terms via the symbol `t`.

    Equations are expressions in var_names and parameters, e.g.:
      sigma*(y - x)
      x*(rho - z) - y
      x*y - beta*z
    """
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


def build_custom_symplectic_functions(
    var_names: List[str],
    eq_lines: List[str],
    params: Dict[str, float],
) -> Tuple[DQDT, DPDT]:
    """
    Build dq_dt(t, p) and dp_dt(t, q) for separable Hamiltonian systems.

    Assumes state ordering y = [q1..qN, p1..pN] and equations:
      dq/dt = f(p, t)
      dp/dt = g(q, t)
    """
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


def build_custom_rhs_and_jacobian(
    var_names: List[str],
    eq_lines: List[str],
    params: Dict[str, float],
) -> Tuple[
    Callable[[float, np.ndarray], np.ndarray],
    Callable[[float, np.ndarray], np.ndarray],
]:
    """
    Returns:
      rhs(t, y) -> (n,)
      jac(t, y) -> (n, n)

    Notes:
    - Supports time-dependent terms via symbol 't'.
    - Parameter ordering follows params.keys() insertion order.
    """
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
    """
    Returns a pretty-printed symbolic Jacobian matrix for display.
    """
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


def slider_with_input(label: str, min_value: float, max_value: float,
                      value: float, step: float, key: str, fmt: str = "%.6f") -> float:
    """
    Slider + number_input where the input can exceed the slider range.
    Slider updates the input, but typing does not clamp to the slider bounds.
    """
    if key not in st.session_state:
        st.session_state[key] = float(value)

    slider_key = f"{key}_slider"
    input_key = f"{key}_input"
    if slider_key not in st.session_state:
        slider_val = max(min_value, min(max_value, float(value)))
        st.session_state[slider_key] = slider_val
    if input_key not in st.session_state:
        st.session_state[input_key] = float(value)

    def sync_from_slider():
        val = float(st.session_state[slider_key])
        st.session_state[key] = val
        st.session_state[input_key] = val

    def sync_from_input():
        st.session_state[key] = float(st.session_state[input_key])

    c1, c2 = st.columns([2, 1], gap="small")

    with c1:
        st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            value=float(st.session_state[slider_key]),
            step=step,
            key=slider_key,
            on_change=sync_from_slider,
        )

    with c2:
        st.number_input(
            " ",
            min_value=min_value,
            value=float(st.session_state[input_key]),
            step=step,
            format=fmt,
            key=input_key,
            on_change=sync_from_input,
        )

    return float(st.session_state[key])


def build_csv_bytes(t: np.ndarray, y: np.ndarray, var_names: List[str]) -> bytes:
    buf = io.StringIO()
    header = "t," + ",".join(var_names)
    data = np.column_stack([t] + [y[i, :] for i in range(y.shape[0])])
    np.savetxt(buf, data, delimiter=",", header=header, comments="")
    return buf.getvalue().encode("utf-8")
