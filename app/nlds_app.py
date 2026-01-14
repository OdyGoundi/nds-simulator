import sys
from pathlib import Path
from typing import List

# Ensure project root import works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app.cache import solve_cached
from app.helpers import build_csv_bytes, parse_list_of_floats
from app.logic.lyapunov_cached import compute_lyapunov_cached
from app.plots import (
    plot_phase_2d,
    plot_phase_3d,
    plot_time_seiries_functional,
    plot_time_series,
)
from app.params import (
    CustomSystemDefinition,
    InitialConditions,
    IntegrationConfig,
    LorenzParams,
    LyapunovConfig,
    RosslerParams,
    SolverTolerances,
    SystemConfig,
)
from app.ui.bifurcation_tab import render_bifurcation_tab

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

tabs = st.tabs(["Phase portrait", "Time series", "Parameter Sweep Analysis", "Export"])

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
                sigma = st.number_input("sigma", value=10.0, step=0.1, format="%.3f", key="sigma")
                rho = st.number_input("rho", value=28.0, step=0.5, format="%.3f", key="rho")
                beta = st.number_input(
                    "beta",
                    value=float(8.0 / 3.0),
                    step=0.05,
                    format="%.4f",
                    key="beta",
                )

            elif system_key == "rossler":
                ross_a = st.number_input("a", value=0.2, step=0.01, format="%.4f", key="ross_a")
                ross_b = st.number_input("b", value=0.2, step=0.01, format="%.4f", key="ross_b")
                ross_c = st.number_input("c", value=5.7, step=0.1, format="%.3f", key="ross_c")

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

            st.markdown("**Lyapunov settings**")
            qr_interval = st.number_input(
                "QR interval (time)",
                min_value=1e-6,
                value=0.1,
                step=0.01,
                format="%.4f",
                help="Time between orthonormalizations during Lyapunov computation.",
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
            
            st.divider()
            st.header("Solver tolerances")
            rtol = st.number_input(
                "relative tolerance (rtol)",
                min_value=0.0,
                value=1e-6,
                step=1e-6,
                format="%.1e",
                key="rtol",
            )
            atol = st.number_input(
                "absolute tolerance (atol)",
                min_value=0.0,
                value=1e-8,
                step=1e-8,
                format="%.1e",
                key="atol",
            )
            solve_tols = SolverTolerances(rtol=float(rtol), atol=float(atol))

    y0 = parse_list_of_floats(y0_text, int(n_vars), label="y0")
    initial = InitialConditions(tuple(float(v) for v in y0))
    integration = IntegrationConfig(t0=float(t0), tf=float(tf), dt=float(dt))
    system = SystemConfig(
        key=system_key,
        lorenz=LorenzParams(sigma=float(sigma), rho=float(rho), beta=float(beta)),
        rossler=RosslerParams(a=float(ross_a), b=float(ross_b), c=float(ross_c)),
        custom=CustomSystemDefinition(
            var_names=tuple(var_names),
            eq_lines=tuple(eq_lines),
            params_text=params_text,
        ),
    )
    lyapunov_cfg = LyapunovConfig(
        transient_steps=int(transient_steps),
        qr_interval=float(qr_interval),
    )

    t, y = solve_cached(
        system=system,
        integration=integration,
        initial=initial,
        solve_tols=solve_tols,
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

        st.divider()
        st.markdown("**Lyapunov exponents**")
        t_transient_lya = float(transient_steps) * float(dt)
        total_time_lya = float(tf) - float(t0)
        t_measure_lya = total_time_lya - t_transient_lya
        if t_measure_lya <= 0.0:
            st.warning("Not enough time for Lyapunov measurement. Increase tf or reduce transient cut.")
        else:
            try:
                with st.spinner("Computing Lyapunov spectrum..."):
                    lambdas = compute_lyapunov_cached(
                        system=system,
                        integration=integration,
                        initial=initial,
                        lyapunov=lyapunov_cfg,
                        solve_tols=solve_tols,
                    )
                formatted = ", ".join(f"{val:.5f}" for val in lambdas)
                st.write(f"lambda = [{formatted}]")
                st.caption(
                    f"n={len(lambdas)} | t_transient={t_transient_lya:.3f} | "
                    f"t_measure={t_measure_lya:.3f}"
                )
            except Exception as exc:
                st.warning(f"Lyapunov computation failed: {exc}")

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
        # Tab 3: Bifurcation and Lyapunov diagram of parameter sweep
        # -----------------------------
        render_bifurcation_tab(
            tab=tabs[2],
            system=system,
            integration=integration,
            initial=initial,
        )

        # --- Tab 4: Export (CSV functional) ---
        with tabs[3]:
            st.markdown("**Export results**")

            csv_bytes = build_csv_bytes(t_plot, y_plot, var_names)
            rtol_tag = f"{float(rtol):.0e}"
            atol_tag = f"{float(atol):.0e}"

            st.download_button(
                label="Download CSV (post-transient)",
                data=csv_bytes,
                file_name=f"{system_key}_trajectory_rtol{rtol_tag}_atol{atol_tag}.csv",
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
                rtol_meta = meta.get("rtol", rtol)
                atol_meta = meta.get("atol", atol)
                rtol_tag = f"{float(rtol_meta):.0e}"
                atol_tag = f"{float(atol_meta):.0e}"
                fname = (
                    f"{sys_key}_sweep_{sp}_{a:g}_{b:g}_step{stp:g}"
                    f"_rtol{rtol_tag}_atol{atol_tag}.csv"
                )

                st.download_button(
                    label="Download sweep CSV",
                    data=csv_bytes,
                    file_name=fname,
                    mime="text/csv",
                    key="dl_sweep_csv",
                )

                st.caption(f"Rows: {len(df_sweep)} | Columns: {', '.join(df_sweep.columns)}")

            st.divider()
            st.markdown("**Export: Lyapunov sweep**")

            lya_data = st.session_state.get("lya_acc_data", None)
            if lya_data is None:
                st.info("No Lyapunov sweep results available yet. Run Lyapunov in Tab 3 first.")
            else:
                import pandas as pd

                param_vals = np.array(lya_data.get("param_vals", []), dtype=float)
                lambdas_arr = np.array(lya_data.get("lambdas", []), dtype=float)
                if param_vals.size == 0 or lambdas_arr.size == 0:
                    st.info("No Lyapunov sweep results available yet. Run Lyapunov in Tab 3 first.")
                else:
                    meta = lya_data.get("meta", {})
                    sweep_param = meta.get("sweep_param", "param")

                    data = {str(sweep_param): param_vals}
                    if lambdas_arr.ndim == 1:
                        data["lambda0"] = lambdas_arr
                    else:
                        for k in range(lambdas_arr.shape[1]):
                            data[f"lambda{k}"] = lambdas_arr[:, k]

                    df_lya = pd.DataFrame(data)
                    csv_bytes = df_lya.to_csv(index=False).encode("utf-8")

                    sys_key = meta.get("system_key", "system")
                    a = meta.get("sweep_start", float(param_vals[0]) if param_vals.size else 0.0)
                    b = meta.get("sweep_stop", float(param_vals[-1]) if param_vals.size else 0.0)
                    stp = meta.get("sweep_step", 0.0)
                    rtol_meta = meta.get("rtol", rtol)
                    atol_meta = meta.get("atol", atol)
                    rtol_tag = f"{float(rtol_meta):.0e}"
                    atol_tag = f"{float(atol_meta):.0e}"
                    fname = (
                        f"{sys_key}_lyapunov_{sweep_param}_{a:g}_{b:g}_step{stp:g}"
                        f"_rtol{rtol_tag}_atol{atol_tag}.csv"
                    )

                    st.download_button(
                        label="Download Lyapunov CSV",
                        data=csv_bytes,
                        file_name=fname,
                        mime="text/csv",
                        key="dl_lya_csv",
                    )

                    st.caption(f"Rows: {len(df_lya)} | Columns: {', '.join(df_lya.columns)}")


except Exception as e:
    st.error(str(e))
    st.info("Check: variable names count, equations count, parameter format, y0 length, and dt/tf values.")
