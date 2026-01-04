import sys
import io
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
from core.rossler_system_rhs import rossler_rhs
from core.solver import integrate_system
# from core.symplectic_solver import integrate_system_symplectic_fr
#Poincaré sweep
from core.poincare_sweep import (
    poincare_section,
    sweep_poincare,
    sweep_poincare_events_ivp,
    PoincareConfig,
    SweepConfig,
)


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


def plot_phase_2d(y: np.ndarray, i: int, j: int, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot(y[i, :], y[j, :], linewidth=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, linewidth=0.3)
    ax.set_aspect("equal", adjustable="box")
    return fig

def plot_phase_3d(y: np.ndarray, i: int, j: int, k: int, title: str, labels: Tuple[str, str, str]):
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(y[i, :], y[j, :], y[k, :], linewidth=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(labels[0], fontsize=9)
    ax.set_ylabel(labels[1], fontsize=9)
    ax.set_zlabel(labels[2], fontsize=9)
    ax.tick_params(labelsize=8)
    return fig

def plot_time_seiries_functional(t: np.ndarray, y: np.ndarray, indices: List[int], var_names: List[str], title: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    for i in indices:
        ax.plot(t, y[i, :], linewidth=0.9, label=var_names[i])
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.grid(True, linewidth=0.3)
    ax.legend(loc="best")
    return fig

def plot_time_series(t: np.ndarray, y: np.ndarray, indices: List[int], var_names: List[str], title: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    for i in indices:
        ax.plot(t, y[i, :], linewidth=0.9, label=var_names[i])
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.grid(True, linewidth=0.3)
    ax.legend(loc="best")
    return fig


def build_csv_bytes(t: np.ndarray, y: np.ndarray, var_names: List[str]) -> bytes:
    buf = io.StringIO()
    header = "t," + ",".join(var_names)
    data = np.column_stack([t] + [y[i, :] for i in range(y.shape[0])])
    np.savetxt(buf, data, delimiter=",", header=header, comments="")
    return buf.getvalue().encode("utf-8")


# ----------------------------
# Caching: store solution only
# ----------------------------

@st.cache_data(show_spinner=False)
def solve_cached(system_key: str,
                 t0: float, tf: float, dt: float,
                 y0_tuple: Tuple[float, ...],
                 # Lorenz:
                 sigma: float, rho: float, beta: float,
                 # Rossler:
                 ross_a: float, ross_b: float, ross_c: float,
                 # Custom:
                 var_names_tuple: Tuple[str, ...],
                 eq_lines_tuple: Tuple[str, ...],
                 params_text: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (t, y):
      t: shape (n_steps,)
      y: shape (n_vars, n_steps)
    """
    y0 = np.array(y0_tuple, dtype=float)

    if system_key == "lorenz":
        def rhs(t, y):
            return lorenz_rhs(t, y, sigma=sigma, rho=rho, beta=beta)

    elif system_key == "rossler":
        def rhs(t, y):
            return rossler_rhs(t, y, a=ross_a, b=ross_b, c=ross_c)

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

@st.cache_data(show_spinner=False)
def sweep_cached(
    system_key: str,
    t0: float, tf: float, dt: float,
    y0_tuple: Tuple[float, ...],
    # built-in params
    sigma: float, rho: float, beta: float,
    ross_a: float, ross_b: float, ross_c: float,
    # custom definitions
    var_names_tuple: Tuple[str, ...],
    eq_lines_tuple: Tuple[str, ...],
    params_text: str,
    # sweep + poincare settings
    sweep_param: str, sweep_start: float, sweep_stop: float, sweep_step: float,
    section_index: int, section_value: float, direction: int,
    method: str, tol: float, transient_steps: int,
    # output selection
    output_index: int,
    solver_kind: str = "ivp",
    warm_start: bool = False,
    max_hits: int = 100,
    early_stop: bool = True,
    chunk_time: float = 2.0,
):
    import pandas as pd

    y0 = np.array(y0_tuple, dtype=float)

    # Build base rhs + base_params (everything except swept param)
    if system_key == "lorenz":
        rhs_fn = lorenz_rhs
        base_params = {"sigma": float(sigma), "rho": float(rho), "beta": float(beta)}
    elif system_key == "rossler":
        rhs_fn = rossler_rhs
        base_params = {"a": float(ross_a), "b": float(ross_b), "c": float(ross_c)}
    elif system_key == "custom":
        var_names = list(var_names_tuple)
        eq_lines = list(eq_lines_tuple)
        params = parse_params(params_text)
        rhs_user = build_custom_rhs(var_names, eq_lines, params)

        # Για custom rhs που είναι ήδη rhs(t,y) (χωρίς kwargs),
        # we make wrapper that ignores params .
        # So base_params will be the params dict.
        rhs_fn = None  # handled below
        base_params = dict(params)
    else:
        raise ValueError(f"Unknown system_key: {system_key}")

    sweep = SweepConfig(param_name=str(sweep_param), start=float(sweep_start),
                        stop=float(sweep_stop), step=float(sweep_step))

    poincare = PoincareConfig(
        section_index=int(section_index),
        section_value=float(section_value),
        direction=int(direction),
        method=str(method),
        tol=float(tol),
        transient_steps=int(transient_steps_sweep),
    )

    # -----------------------
    # EARLY RETURN: custom
    # -----------------------
    if system_key == "custom":
        var_names = list(var_names_tuple)
        eq_lines = list(eq_lines_tuple)
        base_params = parse_params(params_text)

        # Generate inclusive sweep values (safe for floats)
        if sweep.step <= 0:
            raise ValueError("Sweep step must be > 0.")
        n = int(np.floor((sweep.stop - sweep.start) / sweep.step + 1e-12)) + 1
        param_vals = sweep.start + sweep.step * np.arange(n, dtype=float)
        param_vals = param_vals[param_vals <= sweep.stop + 1e-12]

        rows = []
        ycol = f"y{int(output_index)}"

        for pv in param_vals:
            params2 = dict(base_params)
            params2[sweep.param_name] = float(pv)

            rhs2 = build_custom_rhs(var_names, eq_lines, params2)

            sol = integrate_system(rhs2, t_span=(t0, tf), y0=y0, t_step=dt)
            
            if not sol.success:
                continue

            t_hits, y_hits = poincare_section(sol.t, sol.y, poincare)
            
            # Keep only last K Poincaré hits (bibliography-style)
            MAX_HITS = 100   

            if t_hits.size > MAX_HITS:
                t_hits = t_hits[-MAX_HITS:]
                y_hits = y_hits[:, -MAX_HITS:]
                
            if t_hits.size == 0:
                continue

            for j in range(t_hits.size):
                rows.append({
                    sweep.param_name: float(pv),
                    "t_hit": float(t_hits[j]),
                    ycol: float(y_hits[int(output_index), j]),
                })

        return pd.DataFrame(rows)

    # -----------------------
    # Non-custom: use sweep_poincare
    # -----------------------
    if system_key == "lorenz":
        rhs_fn = lorenz_rhs
        base_params = {"sigma": float(sigma), "rho": float(rho), "beta": float(beta)}
    elif system_key == "rossler":
        rhs_fn = rossler_rhs
        base_params = {"a": float(ross_a), "b": float(ross_b), "c": float(ross_c)}
    else:
        raise ValueError(f"Unknown system_key: {system_key}")
    
    solve_options = {
    "rtol": 3e-4,
    "atol": 1e-6,
    }

    # Event-based fast path (only for ivp + crossing)
    if str(solver_kind).lower() == "ivp" and str(method).lower() == "crossing":
        df = sweep_poincare_events_ivp(
            rhs=rhs_fn,
            y0=tuple(y0),
            t_span=(float(t0), float(tf)),
            base_params=base_params,
            sweep=sweep,
            poincare=poincare,
            t_step=float(dt),
            solve_options=solve_options,
            output_indices=[int(output_index)],
            include_all_state=False,
            warm_start=bool(warm_start),
            max_hits=int(max_hits),
            early_stop=bool(early_stop),     
            chunk_time=float(chunk_time),    
        )

    else:
        # Fallback: existing implementation
        df = sweep_poincare(
            rhs=rhs_fn,
            y0=tuple(y0),
            t_span=(float(t0), float(tf)),
            base_params=base_params,
            sweep=sweep,
            poincare=poincare,
            solver_kind=str(solver_kind),
            t_step=float(dt),
            solve_options=solve_options,
            output_indices=[int(output_index)],
            include_all_state=False,
            warm_start=bool(warm_start),
        )

    return df



# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Non Linear Dynamics Simulator", layout="wide")
st.title("Non Linear Dynamics Simulator (NLDS)")

# -------- Sidebar: system + integration + initial conditions --------
with st.sidebar:
    st.header("System")

    system_label = st.selectbox(
        "Choose system",
        ["Lorenz (3D)", "Rossler (3D)", "Custom (nD)"],
        index=0
    )

    if system_label.startswith("Lorenz"):
        system_key = "lorenz"
        n_vars = 3
    elif system_label.startswith("Rossler"):
        system_key = "rossler"
        n_vars = 3
    else:
        system_key = "custom"
        n_vars = st.number_input("Number of equations (n)", min_value=1, max_value=12, value=3, step=1)

    st.divider()
    st.header("Integration")

    t0 = st.number_input("initial time", value=0.0, step=1.0)
    tf = st.number_input("final time", value=50.0, step=1.0)
    dt = st.number_input("time step", value=0.01, step=0.01, format="%.5f")

    st.divider()
    st.header("Initial conditions")

    y0_default = "1, 1, 1" if int(n_vars) == 3 else "\n".join(["0"] * int(n_vars))
    y0_text = st.text_area(
        "y0 values (comma/space/newline separated)",
        value=y0_default,
        height=90,
    )

    st.divider()
    st.header("Plot settings")
    plot_mode = st.selectbox("Plot mode", ["2D phase plane", "3D phase plot"], index=0)

    transient_steps = st.number_input(
        "Transient cut (steps to skip)",
        min_value=0,
        value=0,
        step=100,
        help="Ignores the first N integration samples before plotting/export."
    )

    ## Optional "run" button (Streamlit reruns anyway)
    #run_btn = st.button("Run / Refresh")


# -------- Define variables for custom (avoid unbound issues) --------
eq_lines: List[str] = [""] * int(n_vars)
params_text: str = ""
var_names_text: str = ""

# Variable names
if system_key in ("lorenz", "rossler"):
    var_names = ["x", "y", "z"]
else:
    with st.sidebar:
        st.header("Custom definitions")

        default_names = "\n".join([f"y{i+1}" for i in range(int(n_vars))])
        var_names_text = st.text_area(
            "Variable names (one per line)",
            value=default_names,
            height=120,
        )
        tmp_names = [ln.strip() for ln in (var_names_text or "").splitlines() if ln.strip()]
        if len(tmp_names) != int(n_vars):
            st.warning(f"Need exactly {n_vars} variable names. Using y1..y{n_vars} temporarily.")
            var_names = [f"y{i+1}" for i in range(int(n_vars))]
        else:
            var_names = tmp_names

        default_eq = "\n".join(["0"] * int(n_vars))
        eqs_text = st.text_area(
            "Equations dy/dt (one per line)",
            value=default_eq,
            height=180,
        )

        params_text = st.text_area(
            "Parameters (name=value per line)",
            value="",
            height=120,
        )

    # Build equation list for custom
    eq_lines = [ln.strip() for ln in (eqs_text or "").splitlines()]
    eq_lines = (eq_lines + ["0"] * int(n_vars))[:int(n_vars)]


# -------- Main layout: left controls + right outputs --------
colA, colB = st.columns([1, 2], gap="large")

with colA:
    st.subheader("Controls")

    # Axes selection (based on var_names)
    axis_options = [(f"{name} (index {i})", i) for i, name in enumerate(var_names)]
    idx_list = [o[1] for o in axis_options]

    x_idx = st.selectbox(
        "x-axis",
        options=idx_list,
        format_func=lambda i: axis_options[i][0],
        index=0 if len(idx_list) > 0 else 0
    )

    y_default = 1 if len(idx_list) > 1 else 0
    y_idx = st.selectbox(
        "y-axis",
        options=idx_list,
        format_func=lambda i: axis_options[i][0],
        index=y_default
    )

    z_idx = 2 if len(idx_list) > 2 else 0
    if plot_mode == "3D phase plot":
        z_idx = st.selectbox(
            "z-axis",
            options=idx_list,
            format_func=lambda i: axis_options[i][0],
            index=2 if len(idx_list) > 2 else 0
        )

    st.divider()
    st.subheader("System parameters")

    # Default values
    sigma = rho = beta = 0.0
    ross_a = ross_b = ross_c = 0.0

    if system_key == "lorenz":
        sigma = slider_with_input("sigma", 0.1, 50.0, 10.0, 0.1, key="sigma", fmt="%.3f")
        rho   = slider_with_input("rho",   0.0, 80.0, 28.0, 0.5, key="rho",   fmt="%.3f")
        beta  = slider_with_input("beta",  0.1, 10.0, float(8.0/3.0), 0.05, key="beta", fmt="%.4f")

    elif system_key == "rossler":
        ross_a = slider_with_input("a", 0.0, 1.0, 0.2, 0.01, key="ross_a", fmt="%.4f")
        ross_b = slider_with_input("b", 0.0, 1.0, 0.2, 0.01, key="ross_b", fmt="%.4f")
        ross_c = slider_with_input("c", 0.0, 10.0, 5.7, 0.1, key="ross_c", fmt="%.3f")

    else:
        st.caption("Custom: parameters are defined in the sidebar.")


with colB:
    st.subheader("Outputs")

    tabs = st.tabs(["Phase portrait", "Time series", "Bifurcation Diagram", "Lyapunov Exponents", "Export"])

    # Solve once, then all outputs derive from (t, y)
    try:
        y0 = parse_list_of_floats(y0_text, int(n_vars), label="y0")

        t, y = solve_cached(
            system_key=system_key,
            t0=float(t0), tf=float(tf), dt=float(dt),
            y0_tuple=tuple(float(v) for v in y0),
            sigma=float(sigma), rho=float(rho), beta=float(beta),
            ross_a=float(ross_a), ross_b=float(ross_b), ross_c=float(ross_c),
            var_names_tuple=tuple(var_names),
            eq_lines_tuple=tuple(eq_lines),
            params_text=params_text,
        )

        # Apply transient cut safely (keep >= 2 samples)
        N = int(transient_steps)
        N = max(0, min(N, y.shape[1] - 2))
        t_plot = t[N:]
        y_plot = y[:, N:]

        # --- Tab 1: Phase portrait (functional) ---
        with tabs[0]:
            if plot_mode == "2D phase plane":
                title = f"{system_label} – {var_names[int(y_idx)]} vs {var_names[int(x_idx)]}"
                fig = plot_phase_2d(
                    y=y_plot,
                    i=int(x_idx),
                    j=int(y_idx),
                    title=title,
                    xlabel=var_names[int(x_idx)],
                    ylabel=var_names[int(y_idx)],
                )
                st.pyplot(fig, clear_figure=True)

            else:
                title = f"{system_label} – 3D phase ({var_names[int(x_idx)]}, {var_names[int(y_idx)]}, {var_names[int(z_idx)]})"
                fig = plot_phase_3d(
                    y=y_plot,
                    i=int(x_idx),
                    j=int(y_idx),
                    k=int(z_idx),
                    title=title,
                    labels=(var_names[int(x_idx)], var_names[int(y_idx)], var_names[int(z_idx)]),
                )
                st.pyplot(fig, clear_figure=True)

            st.caption(
                f"Total steps: {len(t)} | plotted: {len(t_plot)} | transient cut: {N} | "
                f"n_vars: {y.shape[0]} | t in [{t[0]:.2f}, {t[-1]:.2f}]"
            )

        # --- Tab 2: Time series (one plot per variable)
        with tabs[1]:
            st.markdown("**Time series (post-transient)**")

            # Variable selection
            default_sel = [0] if len(var_names) > 0 else []
            selected_names = st.multiselect(
            "Select variable(s)",
            options=var_names,
            default=[var_names[i] for i in default_sel] if default_sel else [],
            )

            if not selected_names:
                st.info("Select at least one variable to plot.")
            else:
                selected_indices = [var_names.index(name) for name in selected_names]

                fig_ts = plot_time_seiries_functional(
                    t=t_plot,
                    y=y_plot,
                    indices=selected_indices,
                    var_names=var_names,
                    title=f"{system_label} – time series",
                )   
                st.pyplot(fig_ts, clear_figure=True)

            # Allow user to select which variables to display
            selected_names = st.multiselect(
                "Select variable(s) to display (one plot per variable)",
                options=var_names,
                default=[],
            )

            if selected_names:
                plot_indices = [var_names.index(name) for name in selected_names]
            else:
                # Default: show all variables
                plot_indices = list(range(len(var_names)))
            # Render one plot per chosen variable
            COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
            for plot_pos, var_idx in enumerate(plot_indices):
                fig, ax = plt.subplots(figsize=(9, 3))
                color = COLORS[plot_pos % len(COLORS)]

                ax.plot(
                    t_plot,
                    y_plot[var_idx, :],
                    linewidth=0.9,
                    label=var_names[var_idx],
                    color=color,
                )
                ax.set_title(
                    f"{system_label} – {var_names[var_idx]} vs time",
                    fontsize=10,
                )
                ax.set_xlabel("t", fontsize=9)
                ax.set_ylabel(var_names[var_idx], fontsize=9)
                ax.tick_params(labelsize=8)
                ax.grid(True, linewidth=0.3)
                ax.legend(loc="best")

                st.pyplot(fig, clear_figure=True)

        # --- Tab 3: Bifurcation diagram  ---
        with tabs[2]:
            st.markdown("**Bifurcation / Poincaré sweep**")

            # --- Sweep controls ---
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="small")

            # Choose sweep parameter
            if system_key == "lorenz":
                sweep_choices = ["sigma", "rho", "beta"]
            elif system_key == "rossler":
                sweep_choices = ["a", "b", "c"]
            else:
                try:
                    sweep_choices = list(parse_params(params_text).keys())
                except Exception:
                    sweep_choices = []

            if not sweep_choices:
                st.warning("No sweep parameters available (check parameters).")
                st.stop()

            with c1:
                sweep_param = st.selectbox("Sweep param", sweep_choices, index=0)

            with c2:
                sweep_start = st.number_input("start", value=0.0, step=0.1, format="%.6f", key="sw_start")
            with c3:
                sweep_stop = st.number_input("stop", value=50.0, step=0.1, format="%.6f", key="sw_stop")
            with c4:
                sweep_step = st.number_input("step", value=0.1, step=0.01, format="%.6f", key="sw_step")

            st.divider()

            # --- Poincaré section controls ---
            d1, d2, d3, d4, d5 = st.columns([1, 1, 1, 1, 1], gap="small")

            with d1:
                section_var = st.selectbox("Section var", var_names, index=0)
                section_index = var_names.index(section_var)

            with d2:
                section_value = st.number_input("Section value", value=0.0, step=0.1, format="%.6f", key="sec_val")

            with d3:
                direction_label = st.selectbox("Direction", ["+1 (up)", "-1 (down)", "0 (both)"], index=0)
                direction = +1 if direction_label.startswith("+1") else (-1 if direction_label.startswith("-1") else 0)

            with d4:
                method = st.selectbox("Method", ["crossing", "slab"], index=0)

            with d5:
                tol = st.number_input("tol (slab only)", value=1e-3, step=1e-3, format="%.6f", key="sec_tol")
            # output variable (what to plot on y-axis)
            out_var = st.selectbox("Output var (plotted)", var_names, index=min(2, len(var_names)-1))
            output_index = var_names.index(out_var)

            # use same transient as global
            st.caption("Uses the same transient cut (steps) as in the Integration panel.")

            SWEEP_SOLVER = "ivp"

            st.divider()
            st.markdown("**Sweep performance settings**")

            # -------- Row 1: time & mode --------
            r1c1, r1c2, r1c3 = st.columns([1, 1, 1], gap="small")

            with r1c1:
                dt_sweep = st.number_input(
                    "dt (sweep)",
                    min_value=1e-6,
                    value=max(float(dt), 0.1),
                    step=0.01,
                    format="%.6f",
                    key="dt_sweep_tab3",
                    help="Time step used ONLY for sweep."
                )

            with r1c2:
                tf_sweep = st.number_input(
                    "final time (sweep)",
                    min_value=float(t0) + 1e-6,
                    value=min(float(tf), 80.0),
                    step=5.0,
                    format="%.3f",
                    key="tf_sweep_tab3",
                    help="Final integration time for sweep."
                )

            with r1c3:
                sweep_mode = st.selectbox(
                    "Sweep mode",
                    ["Bifurcation (reset ICs)", "Continuation (warm start)"],
                    index=0,
                    key="sweep_mode_tab3",
                    help="Reset ICs = bibliography-style. Warm start = faster continuation."
                )

            warm_start = sweep_mode.startswith("Continuation")
            
            # -------- Row 2: events & early-stop --------
            r2c1, r2c2, r2c3 = st.columns([1, 1, 1], gap="small")

            with r2c1:
                early_stop = st.checkbox(
                    "Early stop (events)",
                    value=True,
                    key="early_stop_tab3",
                    help="Stop each run after collecting enough Poincaré hits."
                )

            with r2c2:
                max_hits = st.number_input(
                    "Max hits kept",
                    min_value=10,
                    max_value=2000,
                    value=200,
                    step=10,
                    key="max_hits_tab3",
                    disabled=not early_stop,
                    help="Maximum number of Poincaré crossings kept per parameter value."
                )

            with r2c3:
                chunk_time = st.number_input(
                    "Chunk time",
                    min_value=0.1,
                    value=2.0,
                    step=0.5,
                    format="%.2f",
                    key="chunk_time_tab3",
                    disabled=not early_stop,
                    help="Integration time window for event detection. Larger = fewer solver calls."
                )


            st.markdown("**Transient removal (sweep only)**")
            tc1, tc2 = st.columns([1, 1], gap="small")

            with tc1:
                transient_frac = st.slider(
                    "Transient fraction",
                    min_value=0.0,
                    max_value=0.95,
                    value=0.80,
                    step=0.05,
                    key="sw_transient_frac_tab3",
                    help="Fraction of sweep integration steps to discard before computing crossings."
                )

            with tc2:
                n_steps_est = int(max(1.0, (float(tf_sweep) - float(t0)) / float(dt_sweep)))
                transient_steps_sweep = int(transient_frac * n_steps_est)
                st.metric("Transient steps (estimated)", transient_steps_sweep)

            run_sweep = st.button("Generate Bifurcation Diagram", type="primary", key="run_sweep_tab3")

            
            df = None
            if run_sweep:
                with st.spinner("Running sweep..."):
                    df = sweep_cached(
                        system_key=system_key,
                        t0=float(t0), tf=float(tf_sweep), dt=float(dt_sweep),
                        y0_tuple=tuple(float(v) for v in y0),
                        sigma=float(sigma), rho=float(rho), beta=float(beta),
                        ross_a=float(ross_a), ross_b=float(ross_b), ross_c=float(ross_c),
                        var_names_tuple=tuple(var_names),
                        eq_lines_tuple=tuple(eq_lines),
                        params_text=params_text,
                        sweep_param=str(sweep_param),
                        sweep_start=float(sweep_start), sweep_stop=float(sweep_stop), sweep_step=float(sweep_step),
                        section_index=int(section_index), section_value=float(section_value),
                        direction=int(direction),
                        method=str(method), tol=float(tol),
                        transient_steps=int(transient_steps_sweep),
                        output_index=int(output_index),
                        solver_kind=SWEEP_SOLVER,
                        warm_start=bool(warm_start),
                        max_hits=int(max_hits),
                        chunk_time=float(chunk_time),
                        early_stop=bool(early_stop),
                    )

                if df is None or len(df) == 0:
                    st.warning("No Poincaré hits found for these settings.")
                else:
                    import pandas as pd
                    if not isinstance(df, pd.DataFrame):
                        df = pd.DataFrame(df)

                    ycol = f"y{int(output_index)}"
                    fig, ax = plt.subplots(figsize=(7.5, 4.0))
                    ax.scatter(
                        df[sweep_param].to_numpy(),
                        df[ycol].to_numpy(),
                        s=2,                 
                        c="black",           
                        marker=".",
                        linewidths=0,
                        alpha=0.8,
                    )


                    ax.set_xlabel(sweep_param)
                    ax.set_ylabel(f"{out_var} on section ({section_var}={section_value})")

   
                    ax.set_xlim(float(sweep_start), float(sweep_stop))

                    ax.grid(True, linewidth=0.3)
                    st.pyplot(fig, clear_figure=True)


            # CSV export
            if df is not None:
                import pandas as pd
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download sweep CSV",
                    data=csv_bytes,
                    file_name=f"{system_key}_poincare_sweep.csv",
                    mime="text/csv",
                )

                st.caption(f"Rows: {len(df)} | Columns: {', '.join(df.columns)}")

            

        # --- Tab 4: Lyapunov Exponents (placeholder) ---
        with tabs[3]:
            st.info("Lyapunov exponents will be added here.")
            st.empty()

        # --- Tab 5: Export (CSV functional) ---
        with tabs[4]:
            st.markdown("**Export results**")

            csv_bytes = build_csv_bytes(t_plot, y_plot, var_names)

            st.download_button(
                label="Download CSV (post-transient)",
                data=csv_bytes,
                file_name=f"{system_key}_trajectory.csv",
                mime="text/csv",
            )

            st.caption("CSV columns: t, " + ", ".join(var_names))

    except Exception as e:
        st.error(str(e))
        st.info("Check: variable names count, equations count, parameter format, y0 length, and dt/tf values.")
