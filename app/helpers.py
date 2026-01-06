import io
from typing import Dict, List

import numpy as np
import streamlit as st
import sympy as sp


SAFE_FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
    "abs": sp.Abs,
}


def parse_params(text: str) -> Dict[str, float]:
    """
    Parameters format:
      a=1.2
      b=3
    Empty lines ignored.
    """
    params: Dict[str, float] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"Parameter line must be name=value. Got: '{line}'")
        name, val = line.split("=", 1)
        name = name.strip()
        val = val.strip()
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
    Build rhs(t,y) from user equations using sympy.

    Equations are expressions in var_names and parameters, e.g.:
      sigma*(y - x)
      x*(rho - z) - y
      x*y - beta*z
    """
    n = len(var_names)
    if len(eq_lines) != n:
        raise ValueError(f"Need exactly {n} equations (one per variable). Got {len(eq_lines)}.")

    var_syms = sp.symbols(var_names)
    param_syms = {k: sp.Symbol(k) for k in params.keys()}

    locals_dict = {
        **SAFE_FUNCS,
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

    args = list(var_syms) + [param_syms[k] for k in params.keys()]
    f = sp.lambdify(args, exprs, modules=["numpy"])
    param_values = [float(params[k]) for k in params.keys()]

    def rhs(t, y):
        vals = list(y) + param_values
        out = f(*vals)
        return np.array(out, dtype=float)

    return rhs


def slider_with_input(label: str, min_value: float, max_value: float,
                      value: float, step: float, key: str, fmt: str = "%.6f") -> float:
    """
    Slider + number_input with true two-way synchronization.
    Both widgets stay synchronized at all times.
    """
    if key not in st.session_state:
        st.session_state[key] = float(value)

    def sync_to_main(widget_key: str):
        """Callback: sync widget value to main session_state key"""
        val = st.session_state[widget_key]
        val = max(min_value, min(max_value, float(val)))
        st.session_state[key] = val

    c1, c2 = st.columns([2, 1], gap="small")

    with c1:
        st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            value=float(st.session_state[key]),
            step=step,
            key=f"{key}_slider",
            on_change=sync_to_main,
            args=(f"{key}_slider",),
        )

    with c2:
        st.number_input(
            " ",
            min_value=min_value,
            max_value=max_value,
            value=float(st.session_state[key]),
            step=step,
            format=fmt,
            key=f"{key}_input",
            on_change=sync_to_main,
            args=(f"{key}_input",),
        )

    return float(st.session_state[key])


def build_csv_bytes(t: np.ndarray, y: np.ndarray, var_names: List[str]) -> bytes:
    buf = io.StringIO()
    header = "t," + ",".join(var_names)
    data = np.column_stack([t] + [y[i, :] for i in range(y.shape[0])])
    np.savetxt(buf, data, delimiter=",", header=header, comments="")
    return buf.getvalue().encode("utf-8")
