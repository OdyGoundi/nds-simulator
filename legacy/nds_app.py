import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import sympy as sp

# Ensure project root import works
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.lorenz_system_rhs import lorenz_rhs
from core.solver import integrate_system


# ----------------------------
# Helpers: parsing & plotting
# ----------------------------

SAFE_FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
    "abs": sp.Abs,
}

def parse_params(text: str) -> Dict[str, float]:
    """
    Parse parameters in form:
      a=1.2
      b=3
    Returns dict {name: float}.
    Empty lines are ignored.
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
      - or comma/space separated in one line
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError(f"{label} is empty.")
    # allow commas or spaces or newlines
    tokens = raw.replace(",", " ").split()
    if len(tokens) != n:
        raise ValueError(f"{label} must have exactly {n} values. Got {len(tokens)}.")
    return np.array([float(t) for t in tokens], dtype=float)

def build_custom_rhs(var_names: List[str], eq_lines: List[str], params: Dict[str, float]):
    """
    Build rhs(t,y) from user equations using sympy.

    Equations are expressions in var_names and parameters, e.g.:
      x*(rho - z) - y
      x*y - beta*z
      -sigma*x + sigma*y

    Returns rhs(t, y) -> np.ndarray shape (n,)
    """
    n = len(var_names)
    if len(eq_lines) != n:
        raise ValueError(f"Need exactly {n} equations (one per variable). Got {len(eq_lines)}.")

    # Symbols for variables + parameters
    var_syms = sp.symbols(var_names)
    param_syms = {k: sp.Symbol(k) for k in params.keys()}

    # Locals allowed in sympify
    locals_dict = {**SAFE_FUNCS, **{name: sym for name, sym in zip(var_names, var_syms)}, **param_syms}

    exprs = []
    for i, line in enumerate(eq_lines):
        s = (line or "").strip()
        if not s:
            raise ValueError(f"Equation {i+1} is empty.")
        expr = sp.sympify(s, locals=locals_dict)
        exprs.append(expr)

    # Lambdify into a fast numeric function f(vars..., params...) -> list
    args = list(var_syms) + [param_syms[k] for k in params.keys()]
    f = sp.lambdify(args, exprs, modules=["numpy"])

    param_values = [float(params[k]) for k in params.keys()]

    def rhs(t, y):
        # y is shape (n,)
        vals = list(y) + param_values
        out = f(*vals)
        return np.array(out, dtype=float)

    return rhs

def plot_phase(y: np.ndarray, x_idx: int, y_idx: int, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(y[x_idx, :], y[y_idx, :], linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.3)
    ax.set_aspect("equal", adjustable="box")
    return fig


# ----------------------------
# Caching: store solution only
# ----------------------------

@st.cache_data(show_spinner=False)
def solve_cached(system_key: str,
                 t0: float, tf: float, dt: float,
                 y0_tuple: Tuple[float, ...],
                 # system-specific:
                 sigma: float, rho: float, beta: float,
                 mem_a: float, mem_b: float, mem_c: float,
                 # custom:
                 var_names_tuple: Tuple[str, ...],
                 eq_lines_tuple: Tuple[str, ...],
                 params_text: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (t, y) where:
      t: shape (n_steps,)
      y: shape (n_vars, n_steps)
    """
    y0 = np.array(y0_tuple, dtype=float)

    if system_key == "lorenz":
        def rhs(t, y):
            return lorenz_rhs(t, y, sigma=sigma, rho=rho, beta=beta)

    elif system_key == "custom":
        var_names = list(var_names_tuple)
        eq_lines = list(eq_lines_tuple)
        params = parse_params(params_text)
        rhs = build_custom_rhs(var_names, eq_lines, params)

    else:
        raise ValueError(f"Unknown system_key: {system_key}")

    sol = integrate_system(rhs, t_span=(t0, tf), y0=y0, t_step=dt)
    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.t, sol.y


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="NDS Simulator", layout="wide")
st.title("NDS Simulator (Streamlit)")

with st.sidebar:
    st.header("System")

    system_label = st.selectbox(
        "Choose system",
        ["Lorenz (3D)", "Memristive (3D)", "Custom (nD)"],
        index=0
    )

    if system_label.startswith("Lorenz"):
        system_key = "lorenz"
        n_vars = 3
    elif system_label.startswith("Memristive"):
        system_key = "memristive"
        n_vars = 3
    else:
        system_key = "custom"
        n_vars = st.number_input("Number of equations (n)", min_value=1, max_value=12, value=3, step=1)

    st.divider()
    st.header("Integration")

    t0 = st.number_input("t0", value=0.0, step=1.0)
    tf = st.number_input("tf", value=50.0, step=1.0)
    dt = st.number_input("dt", value=0.01, step=0.01, format="%.5f")

    st.divider()
    st.header("Initial conditions")

    # For simplicity and scalability (nD), we use text input for y0
    y0_text = st.text_area(
        "y0 values (comma/space/newline separated)",
        value="1, 1, 1" if n_vars == 3 else "\n".join(["0"] * int(n_vars)),
        height=90,
    )

    st.divider()
    st.header("Phase plane")
    # Axis dropdowns use indices, displayed with variable names when available
    # We'll set names below for custom; for fixed systems we use x,y,z.
    run_btn = st.button("Run / Refresh")

# --- Defaults so variables are always defined ---
eqs_text: str = ""
params_text: str = ""
var_names_text: str = ""

# Build variable names
if system_key in ("lorenz", "memristive"):
    var_names = ["x", "y", "z"]
else:
    # Custom names: allow user to input names; fallback to y1..yn
    with st.sidebar:
        st.header("Custom definitions")
        default_names = "\n".join([f"y{i+1}" for i in range(int(n_vars))])
        var_names_text = st.text_area("Variable names (one per line)", value=default_names, height=120)
        var_names = [ln.strip() for ln in var_names_text.splitlines() if ln.strip()]
        if len(var_names) != int(n_vars):
            st.warning(f"Need exactly {n_vars} variable names. Using y1..y{n_vars} temporarily.")
            var_names = [f"y{i+1}" for i in range(int(n_vars))]

        default_eq = "\n".join(["0"] * int(n_vars))
        eqs_text = st.text_area("Equations dy/dt (one per line)", value=default_eq, height=180)

        params_text = st.text_area("Parameters (name=value per line)", value="", height=120)

# Phase-plane selectors (main area so user sees it)
colA, colB = st.columns([1, 2], gap="large")

with colA:
    st.subheader("Controls")

    axis_options = [(f"{name} (index {i})", i) for i, name in enumerate(var_names)]
    x_idx = st.selectbox("x-axis", options=[o[1] for o in axis_options],
                         format_func=lambda i: axis_options[i][0], index=0)
    y_idx_default = 1 if len(var_names) > 1 else 0
    y_idx = st.selectbox("y-axis", options=[o[1] for o in axis_options],
                         format_func=lambda i: axis_options[i][0], index=y_idx_default)

    # System-specific params
    sigma = rho = beta = 0.0
    mem_a = mem_b = mem_c = 0.0

    if system_key == "lorenz":
        st.markdown("**Lorenz parameters**")
        sigma = st.slider("sigma", 0.1, 50.0, 10.0, 0.1)
        rho   = st.slider("rho",   0.0, 80.0, 28.0, 0.5)
        beta  = st.slider("beta",  0.1, 10.0, float(8.0/3.0), 0.05)

    elif system_key == "memristive":
        st.markdown("**Memristive parameters**")
        mem_a = st.slider("a", -5.0, 5.0, 0.0, 0.1)
        mem_b = st.slider("b", -5.0, 5.0, 0.1, 0.1)
        mem_c = st.slider("c", -5.0, 5.0, 0.0, 0.1)

    if system_key == "custom":
        # Prepare equations list
        eq_lines = [ln.strip() for ln in (eqs_text or "").splitlines()]
        # pad/truncate defensively
        eq_lines = (eq_lines + ["0"] * int(n_vars))[:int(n_vars)]
    else:
        eq_lines = [""] * int(n_vars)
        params_text = ""


with colB:
    st.subheader("Phase portrait")

    try:
        y0 = parse_list_of_floats(y0_text, int(n_vars), label="y0")
        t, y = solve_cached(
            system_key=system_key,
            t0=float(t0), tf=float(tf), dt=float(dt),
            y0_tuple=tuple(float(v) for v in y0),
            sigma=float(sigma), rho=float(rho), beta=float(beta),
            mem_a=float(mem_a), mem_b=float(mem_b), mem_c=float(mem_c),
            var_names_tuple=tuple(var_names),
            eq_lines_tuple=tuple(eq_lines),
            params_text=params_text,
        )

        title = f"{system_label} – {var_names[y_idx]} vs {var_names[x_idx]}"
        fig = plot_phase(
            y=y,
            x_idx=int(x_idx),
            y_idx=int(y_idx),
            title=title,
            xlabel=var_names[int(x_idx)],
            ylabel=var_names[int(y_idx)],
        )
        st.pyplot(fig, clear_figure=True)

        st.caption(f"Steps: {len(t)} | n_vars: {y.shape[0]} | t in [{t[0]:.2f}, {t[-1]:.2f}]")

    except Exception as e:
        st.error(str(e))
        st.info("Check: variable names count, equations count, parameter format, and y0 length.")

