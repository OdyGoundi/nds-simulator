import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app.cache import solve_cached
from app.helpers import build_csv_bytes, parse_list_of_floats, slider_with_input
from app.plots import (
    plot_phase_2d,
    plot_phase_3d,
    plot_time_seiries_functional,
    plot_time_series,
)
from app.tab_bifurcation import render_bifurcation_tab


# Ensure project root import works
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Non Linear Dynamics Simulator", layout="wide")
st.title("Non Linear Dynamics Simulator (NLDS)")

# -------- Sidebar: system + initial conditions --------
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
    st.header("Initial conditions")

    y0_default = "1, 1, 1" if int(n_vars) == 3 else "\n".join(["0"] * int(n_vars))
    y0_text = st.text_area(
        "y0 values (comma/space/newline separated)",
        value=y0_default,
        height=90,
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


# -------- System parameters defaults --------
# Default values
sigma = rho = beta = 0.0
ross_a = ross_b = ross_c = 0.0


# -------- Main layout: outputs only --------
st.subheader("Outputs")

tabs = st.tabs(["Phase portrait", "Time series", "Bifurcation Diagram", "Lyapunov Exponents", "Export"])

# Solve once, then all outputs derive from (t, y)
try:
    # --- Tab 1: Phase portrait (controls) ---
    with tabs[0]:
        phase_col_controls, phase_col_plot = st.columns([1, 2], gap="large")

        with phase_col_controls:
            st.header("Integration")
            t0 = st.number_input("initial time", value=0.0, step=1.0)
            tf = st.number_input("final time", value=50.0, step=1.0)
            dt = st.number_input("time step", value=0.01, step=0.01, format="%.5f")

            st.divider()
            st.header("System parameters")

            if system_key == "lorenz":
                sigma = slider_with_input("sigma", 0.1, 50.0, 10.0, 0.1, key="sigma", fmt="%.3f")
                rho   = slider_with_input("rho",   0.0, 80.0, 28.0, 0.5, key="rho",   fmt="%.3f")
                beta  = slider_with_input("beta",  0.1, 10.0, float(8.0/3.0), 0.05, key="beta", fmt="%.4f")

            elif system_key == "rossler":
                ross_a = slider_with_input("a", 0.0, 1.0, 0.2, 0.01, key="ross_a", fmt="%.4f")
                ross_b = slider_with_input("b", 0.0, 1.0, 0.2, 0.01, key="ross_b", fmt="%.4f")
                ross_c = slider_with_input("c", 0.0, 10.0, 5.7, 0.1, key="ross_c", fmt="%.3f")

            else:
                st.caption("Custom: parameters are defined above.")

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

            st.divider()
            st.markdown("**Axis selection**")
            axis_options = [(f"{name} (index {i})", i) for i, name in enumerate(var_names)]
            idx_list = [o[1] for o in axis_options]

            x_idx = st.selectbox(
                "x-axis",
                options=idx_list,
                format_func=lambda i: axis_options[i][0],
                index=0 if len(idx_list) > 0 else 0,
            )

            y_default = 1 if len(idx_list) > 1 else 0
            y_idx = st.selectbox(
                "y-axis",
                options=idx_list,
                format_func=lambda i: axis_options[i][0],
                index=y_default,
            )

            z_idx = 2 if len(idx_list) > 2 else 0
            if plot_mode == "3D phase plot":
                z_idx = st.selectbox(
                    "z-axis",
                    options=idx_list,
                    format_func=lambda i: axis_options[i][0],
                    index=2 if len(idx_list) > 2 else 0,
                )

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

    # --- Tab 1: Phase portrait (plot) ---
    with phase_col_plot:
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
            fig, ax = plt.subplots(figsize=(7.0, 2.5))
            fig.set_dpi(140)
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

        # -----------------------------
        # Tab 3: Bifurcation diagram
        # -----------------------------
        render_bifurcation_tab(
            tab=tabs[2],
            system_key=system_key,
            t0=float(t0),
            tf=float(tf),
            dt=float(dt),
            y0=np.array(y0, dtype=float),
            sigma=float(sigma),
            rho=float(rho),
            beta=float(beta),
            ross_a=float(ross_a),
            ross_b=float(ross_b),
            ross_c=float(ross_c),
            var_names=list(var_names),
            eq_lines=list(eq_lines),
            params_text=params_text,
        )

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
            
            st.divider()
            st.markdown("**Export: Sweep (bifurcation / Poincaré)**")

            df_sweep = st.session_state.get("last_sweep_df", None)
            meta = st.session_state.get("last_sweep_meta", {})

            if df_sweep is None or len(df_sweep) == 0:
                st.info("No sweep results available yet. Run a sweep in Tab 3 first.")
            else:
                import pandas as pd
                if not isinstance(df_sweep, pd.DataFrame):
                    df_sweep = pd.DataFrame(df_sweep)

                csv_bytes = df_sweep.to_csv(index=False).encode("utf-8")

                # informative filename
                sys_key = meta.get("system_key", "system")
                sp = meta.get("sweep_param", "param")
                a = meta.get("sweep_start", 0.0)
                b = meta.get("sweep_stop", 0.0)
                stp = meta.get("sweep_step", 0.0)
                fname = f"{sys_key}_sweep_{sp}_{a:g}_{b:g}_step{stp:g}.csv"

                st.download_button(
                    label="Download sweep CSV",
                    data=csv_bytes,
                    file_name=fname,
                    mime="text/csv",
                    key="dl_sweep_csv",
                )

                st.caption(f"Rows: {len(df_sweep)} | Columns: {', '.join(df_sweep.columns)}")


except Exception as e:
    st.error(str(e))
    st.info("Check: variable names count, equations count, parameter format, y0 length, and dt/tf values.")
