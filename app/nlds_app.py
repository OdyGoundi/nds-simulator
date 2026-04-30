import json
import sys
from pathlib import Path
from typing import Callable, List, Optional

# Ensure project root import works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app.cache import solve_cached
from app.services.export_service import build_run_bundle
from app.helpers import (
    apply_transient_cut,
    build_csv_bytes,
    build_custom_symplectic_functions,
    build_custom_symbolic_jacobian_str,
    downsample_trajectory,
    parse_params,
)
from app.state import (
    DIRECT_CSV_MAX_ROWS,
    EXPORT_CHUNK_ROWS_DEFAULT,
    TRAJ_EXPORT_SOURCE_STORED,
    TRAJ_EXPORT_SOURCE_FULL,
    TRAJ_EXPORT_READY_SIG_KEY,
)
from app.state.apply_config import (
    apply_state_values,
    flush_pending_static_config_apply,
)
from app.export_utils import build_static_config
from app.plots import plot_time_seiries_functional
from app.params import IntegrationConfig
from app.ui.tabs.sweep_tab import render_sweep_tab
from app.ui.branding import render_header_logo
from app.ui.help_panels import (
    get_dialog_decorator,
    render_info,
    render_quick_manual_el,
    render_quick_manual_eng,
)
from app.ui.sidebar import render_system_sidebar
from app.ui.tabs import render_phase_tab

APP_NAME = "DynaSim"
APP_SUBTITLE = "Non-linear Dynamical Systems Simulator"
HENON_HEILES_VAR_NAMES = ["q1", "q2", "p1", "p2"]
HENON_HEILES_EQ_LINES = [
    "p1",
    "p2",
    "-q1 - 2*lambda*q1*q2",
    "-q2 - lambda*(q1**2 - q2**2)",
]
HENON_HEILES_PARAMS_TEXT = "lambda=1.0"



st.set_page_config(page_title="dynaSim", layout="wide")

if "show_quick_manual_eng" not in st.session_state:
    st.session_state["show_quick_manual_eng"] = False
if "show_quick_manual_el" not in st.session_state:
    st.session_state["show_quick_manual_el"] = False
if "show_info_popup" not in st.session_state:
    st.session_state["show_info_popup"] = False

open_manual_eng = False
open_manual_el = False
open_info = False
header_logo_col, header_actions_col = st.columns([3, 1], gap="large")
with header_logo_col:
    if not render_header_logo(PROJECT_ROOT, width_px=282, align="left"):
        st.title("dynaSim")
        st.caption(APP_SUBTITLE)
with header_actions_col:
    open_manual_eng = st.button("Help (English)", key="open_quick_manual_btn")
    open_manual_el = st.button("Help(Ελληνικά)", key="open_quick_manual_el_btn")
    open_info = st.button("Info", key="open_info_btn")

if open_manual_eng:
    st.session_state["show_quick_manual_eng"] = True
    st.session_state["show_quick_manual_el"] = False
    st.session_state["show_info_popup"] = False
if open_manual_el:
    st.session_state["show_quick_manual_el"] = True
    st.session_state["show_quick_manual_eng"] = False
    st.session_state["show_info_popup"] = False
if open_info:
    st.session_state["show_info_popup"] = True
    st.session_state["show_quick_manual_eng"] = False
    st.session_state["show_quick_manual_el"] = False

dialog_decorator = get_dialog_decorator()
_quick_manual_eng_dialog: Optional[Callable[[], None]] = None
_quick_manual_el_dialog: Optional[Callable[[], None]] = None
_info_dialog: Optional[Callable[[], None]] = None
if dialog_decorator is not None:

    @dialog_decorator("Quick Start Manual")
    def _quick_manual_eng_dialog_impl() -> None:
        render_quick_manual_eng(PROJECT_ROOT)
        if st.button("Close manual", key="close_quick_manual_btn"):
            st.session_state["show_quick_manual_eng"] = False
            st.rerun()

    _quick_manual_eng_dialog = _quick_manual_eng_dialog_impl

    @dialog_decorator("Σύντομο Εγχειρίδιο")
    def _quick_manual_el_dialog_impl() -> None:
        render_quick_manual_el(PROJECT_ROOT)
        if st.button("Κλείσιμο εγχειριδίου", key="close_quick_manual_el_btn"):
            st.session_state["show_quick_manual_el"] = False
            st.rerun()

    _quick_manual_el_dialog = _quick_manual_el_dialog_impl

    @dialog_decorator("Info")
    def _info_dialog_impl() -> None:
        render_info(PROJECT_ROOT)
        if st.button("Close info", key="close_info_btn"):
            st.session_state["show_info_popup"] = False
            st.rerun()

    _info_dialog = _info_dialog_impl

if st.session_state.get("show_quick_manual_eng", False):
    if _quick_manual_eng_dialog is not None:
        _quick_manual_eng_dialog()
        st.session_state["show_quick_manual_eng"] = False
    else:
        with st.expander("Quick Start Manual", expanded=True):
            render_quick_manual_eng(PROJECT_ROOT)
            if st.button("Hide manual", key="hide_quick_manual_btn"):
                st.session_state["show_quick_manual_eng"] = False

if st.session_state.get("show_quick_manual_el", False):
    if _quick_manual_el_dialog is not None:
        _quick_manual_el_dialog()
        st.session_state["show_quick_manual_el"] = False
    else:
        with st.expander("Σύντομο Εγχειρίδιο", expanded=True):
            render_quick_manual_el(PROJECT_ROOT)
            if st.button("Απόκρυψη εγχειριδίου", key="hide_quick_manual_el_btn"):
                st.session_state["show_quick_manual_el"] = False

if st.session_state.get("show_info_popup", False):
    if _info_dialog is not None:
        _info_dialog()
        st.session_state["show_info_popup"] = False
    else:
        with st.expander("Info", expanded=True):
            render_info(PROJECT_ROOT)
            if st.button("Hide info", key="hide_info_btn"):
                st.session_state["show_info_popup"] = False

# Apply uploaded static config before sidebar widgets are instantiated.
flush_pending_static_config_apply()

sidebar_result = render_system_sidebar()
system_label = sidebar_result.system_label
system_key = sidebar_result.system_key
n_vars = sidebar_result.n_vars
solver_kind = sidebar_result.solver_kind
solver_kind_effective = sidebar_result.solver_kind_effective
y0_text = sidebar_result.y0_text


# -------- Define variables for custom (avoid unbound issues) --------
eq_lines: List[str] = [""] * int(n_vars)
params_text: str = ""
var_names_text: str = ""
custom_auto_jac = False
custom_use_jac = False

# Variable names
if system_key in ("lorenz", "rossler"):
    var_names = ["x", "y", "z"]
elif system_key == "henon_heiles":
    var_names = list(HENON_HEILES_VAR_NAMES)
    eq_lines = list(HENON_HEILES_EQ_LINES)
    params_text = HENON_HEILES_PARAMS_TEXT
    with st.sidebar:
        st.header("Henon-Heiles definition")
        st.caption("State order: q1, q2, p1, p2")
        st.code("\n".join(eq_lines), language="text")
        st.caption(f"Parameters: {HENON_HEILES_PARAMS_TEXT}")
else:
    with st.sidebar:
        st.header("Custom definitions")

        default_names = "\n".join([f"y{i+1}" for i in range(int(n_vars))])
        apply_state_values({"var_names_text_sidebar": default_names}, only_missing=True)
        var_names_text = st.text_area(
            "Variable names (one per line)",
            height=120,
            key="var_names_text_sidebar",
        )
        tmp_names = [ln.strip() for ln in (var_names_text or "").splitlines() if ln.strip()]
        if len(tmp_names) != int(n_vars):
            st.warning(f"Need exactly {n_vars} variable names. Using y1..y{n_vars} temporarily.")
            var_names = [f"y{i+1}" for i in range(int(n_vars))]
        else:
            var_names = tmp_names

        default_eq = "\n".join(["0"] * int(n_vars))
        apply_state_values({"eqs_text_sidebar": default_eq}, only_missing=True)
        eqs_text = st.text_area(
            "Equations dy/dt (one per line)",
            height=180,
            key="eqs_text_sidebar",
        )

        apply_state_values({"params_text_sidebar": ""}, only_missing=True)
        params_text = st.text_area(
            "Parameters (name=value per line)",
            height=120,
            key="params_text_sidebar",
        )

        eq_lines = [ln.strip() for ln in (eqs_text or "").splitlines()]
        eq_lines = (eq_lines + ["0"] * int(n_vars))[:int(n_vars)]

        st.markdown("**Jacobian (custom)**")
        apply_state_values(
            {
                "custom_auto_jac_sidebar": False,
                "custom_use_jac_sidebar": False,
            },
            only_missing=True,
        )
        custom_auto_jac = st.checkbox(
            "Auto-compute Jacobian (symbolic)",
            key="custom_auto_jac_sidebar",
            help="Builds a symbolic Jacobian for Lyapunov on custom systems.",
        )
        custom_use_jac = st.checkbox(
            "Use analytic Jacobian",
            key="custom_use_jac_sidebar",
            disabled=not custom_auto_jac,
            help="When off, Lyapunov uses finite differences as before.",
        )
        if not custom_auto_jac:
            custom_use_jac = False
        else:
            try:
                params = parse_params(params_text)
                jac_preview = build_custom_symbolic_jacobian_str(var_names, eq_lines, params)
                st.markdown("**Symbolic Jacobian**")
                st.code(jac_preview, language="text")
            except Exception as exc:
                st.warning(f"Jacobian preview failed: {exc}")

with st.sidebar:
    st.header("Symplectic preview")
    if system_key not in ("custom", "henon_heiles"):
        st.caption("Symplectic preview is available for custom or Henon-Heiles systems.")
    elif int(n_vars) % 2 != 0:
        st.caption("Symplectic solvers require an even number of variables [q..., p...].")
    else:
        n_q = int(n_vars) // 2
        q_vars = list(var_names[:n_q])
        p_vars = list(var_names[n_q:])
        st.markdown(f"**q vars:** {', '.join(q_vars)}")
        st.markdown(f"**p vars:** {', '.join(p_vars)}")

        dq_lines = eq_lines[:n_q]
        dp_lines = eq_lines[n_q:]
        st.markdown("**dq/dt (q):**")
        st.code("\n".join(dq_lines), language="text")
        st.markdown("**dp/dt (p):**")
        st.code("\n".join(dp_lines), language="text")

        if not str(solver_kind_effective).startswith("symplectic"):
            st.caption("Select a symplectic solver to validate the structure.")
        elif system_key == "custom":
            try:
                params = parse_params(params_text)
                build_custom_symplectic_functions(var_names, eq_lines, params)
                st.success("Symplectic check passed.")
            except Exception as exc:
                st.error(f"Symplectic check failed: {exc}")
        elif system_key == "henon_heiles":
            st.success("Symplectic check passed (Henon-Heiles).")


# -------- Main layout: outputs only --------
st.subheader("Outputs")

tabs = st.tabs(["Phase portrait & Lyapunov exponents", "Time series", "Parameter Sweep Analysis", "Export"])

try:
    phase_result = render_phase_tab(
        tabs[0],
        system_key=system_key,
        n_vars=n_vars,
        solver_kind_effective=solver_kind_effective,
        y0_text=y0_text,
        var_names=var_names,
        eq_lines=eq_lines,
        params_text=params_text,
        custom_auto_jac=custom_auto_jac,
        custom_use_jac=custom_use_jac,
        system_label=system_label,
        app_name=APP_NAME,
        repo_root=PROJECT_ROOT,
    )
    system = phase_result.system
    integration = phase_result.integration
    initial = phase_result.initial
    solve_tols = phase_result.solve_tols
    t_plot = phase_result.t_plot
    y_plot = phase_result.y_plot
    var_names = phase_result.var_names
    max_plot_points_i = phase_result.max_plot_points
    transient_steps = phase_result.transient_steps
    transient_cut_time = phase_result.transient_cut_time

    # --- Tab 2: Time series (one plot per variable)
    with tabs[1]:
        st.markdown("**Time series (post-transient)**")

        t_min = float(t_plot[0])
        t_max = float(t_plot[-1])
        time_step_ui = max(float(integration.dt), 1e-6)
        current_ts_range = (float(t_min), float(t_max))

        if "ts_window_start_tab2" not in st.session_state:
            st.session_state["ts_window_start_tab2"] = t_min
        if "ts_window_end_tab2" not in st.session_state:
            st.session_state["ts_window_end_tab2"] = t_max
        if "ts_window_range_tab2" not in st.session_state:
            st.session_state["ts_window_range_tab2"] = current_ts_range

        # Reset to full range whenever the available integration window changes.
        if tuple(st.session_state.get("ts_window_range_tab2", ())) != current_ts_range:
            st.session_state["ts_window_start_tab2"] = t_min
            st.session_state["ts_window_end_tab2"] = t_max
            st.session_state["ts_window_range_tab2"] = current_ts_range

        # Keep persisted values inside current bounds when t0/tf/transient changes.
        st.session_state["ts_window_start_tab2"] = min(
            max(float(st.session_state["ts_window_start_tab2"]), t_min),
            t_max,
        )
        st.session_state["ts_window_end_tab2"] = min(
            max(float(st.session_state["ts_window_end_tab2"]), t_min),
            t_max,
        )

        tw_header_col, twc1, twc2 = st.columns([1.2, 1, 1], gap="small")
        with tw_header_col:
            st.markdown("**Time window**")
            st.caption(
                f"Available range from Tab 1 integration: [{t_min:.3f}, {t_max:.3f}]"
            )
        with twc1:
            t_view_start = st.number_input(
                "start time",
                min_value=t_min,
                max_value=t_max,
                value=float(st.session_state["ts_window_start_tab2"]),
                step=time_step_ui,
                format="%.6f",
                key="ts_window_start_tab2",
            )
        with twc2:
            t_view_end = st.number_input(
                "end time",
                min_value=t_min,
                max_value=t_max,
                value=float(st.session_state["ts_window_end_tab2"]),
                step=time_step_ui,
                format="%.6f",
                key="ts_window_end_tab2",
            )

        if float(t_view_start) >= float(t_view_end):
            st.warning("Invalid time window: 'start time' must be smaller than 'end time'. Showing full range.")
            t_view_start = t_min
            t_view_end = t_max

        ts_mask = (t_plot >= float(t_view_start)) & (t_plot <= float(t_view_end))
        if int(np.count_nonzero(ts_mask)) < 2:
            st.warning("Time window contains fewer than 2 samples. Showing full range.")
            ts_mask = np.ones_like(t_plot, dtype=bool)

        t_ts = t_plot[ts_mask]
        y_ts = y_plot[:, ts_mask]
        t_ts_plot, y_ts_plot = downsample_trajectory(t_ts, y_ts, max_plot_points_i)
        st.caption(
            f"Showing t in [{float(t_view_start):.3f}, {float(t_view_end):.3f}] | "
            f"samples: {len(t_ts_plot)}/{len(t_ts)} (stored window) | stored total: {len(t_plot)}"
        )

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
                t=t_ts_plot,
                y=y_ts_plot,
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
                t_ts_plot,
                y_ts_plot[var_idx, :],
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
        render_sweep_tab(
            tab=tabs[2],
            system=system,
            integration=integration,
            initial=initial,
            solve_tols=solve_tols,
            app_name=APP_NAME,
            repo_root=PROJECT_ROOT,
        )

        # --- Tab 4: Export (CSV functional) ---
        with tabs[3]:
            st.markdown("**Export results**")

            st.markdown("**Export: Configurations**")
            static_cfg = st.session_state.get("static_config", None)
            if static_cfg is None:
                st.info("No StaticParamsConfig saved yet. Use Save configuration in Tab 1.")
            else:
                static_json = json.dumps(static_cfg, indent=2).encode("utf-8")
                st.download_button(
                    label="Download StaticParamsConfig.json",
                    data=static_json,
                    file_name="StaticParamsConfig.json",
                    mime="application/json",
                    key="dl_static_cfg",
                )

            sweep_cfg = st.session_state.get("sweep_config", None)
            if sweep_cfg is None:
                st.info("No SweepParamConfig saved yet. Use Save configuration in Tab 3.")
            else:
                sweep_json = json.dumps(sweep_cfg, indent=2).encode("utf-8")
                st.download_button(
                    label="Download SweepParamConfig.json",
                    data=sweep_json,
                    file_name="SweepParamConfig.json",
                    mime="application/json",
                    key="dl_sweep_cfg",
                )

            st.divider()
            st.markdown("**Export: Trajectory (post-transient)**")
            export_integration = IntegrationConfig(
                t0=float(integration.t0),
                tf=float(integration.tf),
                dt=float(integration.dt),
                solver_kind=str(getattr(integration, "solver_kind", "ivp")),
                max_store_steps=None,
            )
            export_sig = (
                repr(system),
                repr(export_integration),
                repr(initial),
                repr(solve_tols),
                int(transient_steps),
            )
            export_source = st.radio(
                "Trajectory export source",
                options=[TRAJ_EXPORT_SOURCE_STORED, TRAJ_EXPORT_SOURCE_FULL],
                index=1,
                key="traj_export_source_tab4",
                help=(
                    "Use the current in-memory trajectory for fast export, or prepare a separate "
                    "full-resolution trajectory for publication-oriented CSV/bundle export."
                ),
            )

            t_export = t_plot
            y_export = y_plot
            export_source_tag = "stored"
            export_ready = True
            if export_source == TRAJ_EXPORT_SOURCE_FULL:
                prep_c1, prep_c2 = st.columns([1.2, 2.2], gap="small")
                with prep_c1:
                    prepare_full_export = st.button(
                        "Prepare full-resolution trajectory",
                        key="prepare_full_traj_export_tab4",
                        use_container_width=True,
                    )
                if prepare_full_export:
                    st.session_state[TRAJ_EXPORT_READY_SIG_KEY] = export_sig
                export_ready = st.session_state.get(TRAJ_EXPORT_READY_SIG_KEY) == export_sig
                with prep_c2:
                    if export_ready:
                        st.caption(
                            "Full-resolution export is prepared for the current system/integration settings."
                        )
                    else:
                        st.caption(
                            "Prepare once to recompute the trajectory with full storage for export only."
                        )

                if export_ready:
                    with st.spinner("Preparing full-resolution trajectory for export..."):
                        t_export_full, y_export_full = solve_cached(
                            system=system,
                            integration=export_integration,
                            initial=initial,
                            solve_tols=solve_tols,
                        )
                    t_export, y_export = apply_transient_cut(
                        t_export_full,
                        y_export_full,
                        int(transient_steps),
                    )
                    export_source_tag = "fullres"
                    st.caption(
                        f"Export source: full-resolution recompute | rows after transient cut: {len(t_export):,}"
                    )
                else:
                    st.info(
                        "Full-resolution trajectory is not prepared yet for the current settings. "
                        "Use the button above to enable export."
                    )
            else:
                st.caption(
                    f"Export source: current stored trajectory | rows after transient cut: {len(t_export):,}"
                )

            rtol_tag = f"{float(solve_tols.rtol):.0e}"
            atol_tag = f"{float(solve_tols.atol):.0e}"
            traj_base = f"{system.key}_trajectory_{export_source_tag}_rtol{rtol_tag}_atol{atol_tag}"
            traj_rows = int(t_plot.size)
            if export_ready:
                traj_rows = int(t_export.size)
            else:
                traj_rows = 0

            if traj_rows <= 0:
                st.info("No trajectory samples available for export.")
            else:
                if traj_rows <= int(DIRECT_CSV_MAX_ROWS):
                    csv_bytes = build_csv_bytes(t_export, y_export, var_names)
                    st.download_button(
                        label="Download CSV (single file)",
                        data=csv_bytes,
                        file_name=f"{traj_base}.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning(
                        f"Trajectory has {traj_rows:,} rows; single-file export is disabled to avoid memory spikes."
                    )

                chunk_rows = int(
                    st.number_input(
                        "Trajectory chunk size (rows)",
                        min_value=10_000,
                        max_value=1_000_000,
                        value=EXPORT_CHUNK_ROWS_DEFAULT,
                        step=10_000,
                        key="traj_export_chunk_rows_tab4",
                    )
                )
                n_chunks = int(np.ceil(traj_rows / float(chunk_rows)))
                if "traj_export_chunk_index_tab4" in st.session_state:
                    st.session_state["traj_export_chunk_index_tab4"] = max(
                        1,
                        min(int(st.session_state["traj_export_chunk_index_tab4"]), max(1, n_chunks)),
                    )
                chunk_idx = int(
                    st.number_input(
                        "Chunk number",
                        min_value=1,
                        max_value=max(1, n_chunks),
                        value=1,
                        step=1,
                        key="traj_export_chunk_index_tab4",
                    )
                )
                start_row = (chunk_idx - 1) * chunk_rows
                end_row = min(traj_rows, start_row + chunk_rows)
                chunk_bytes = build_csv_bytes(
                    t_export,
                    y_export,
                    var_names,
                    start=start_row,
                    end=end_row,
                )
                st.download_button(
                    label=f"Download trajectory chunk {chunk_idx}/{n_chunks}",
                    data=chunk_bytes,
                    file_name=f"{traj_base}_part{chunk_idx:03d}-of-{n_chunks:03d}.csv",
                    mime="text/csv",
                    key="dl_traj_chunk_csv",
                )
                st.caption(
                    f"Chunk rows: {start_row + 1}-{end_row} of {traj_rows} | columns: t, {', '.join(var_names)}"
                )
            
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
                rtol_meta = meta.get("rtol", solve_tols.rtol)
                atol_meta = meta.get("atol", solve_tols.atol)
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
                    rtol_meta = meta.get("rtol", solve_tols.rtol)
                    atol_meta = meta.get("atol", solve_tols.atol)
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

            st.divider()
            st.markdown("**Export: Run bundle (zip)**")

            if static_cfg is None:
                _pm = phase_result.plot_mode
                z_idx_val = phase_result.z_idx if _pm == "3D phase plot" else None
                bundle_cfg = build_static_config(
                    app_name=APP_NAME,
                    repo_root=PROJECT_ROOT,
                    system=system,
                    integration=integration,
                    initial=initial,
                    solve_tols=solve_tols,
                    plot_mode=_pm,
                    x_idx=phase_result.x_idx,
                    y_idx=phase_result.y_idx,
                    z_idx=z_idx_val,
                    phase_linewidth=phase_result.phase_linewidth,
                    transient_steps=int(transient_steps),
                    lyapunov_transient_steps=phase_result.lyapunov_transient_steps,
                    lyapunov_transient_frac=phase_result.lyapunov_transient_frac,
                    qr_interval=phase_result.qr_interval,
                )
            else:
                bundle_cfg = static_cfg

            bundle_bytes = build_run_bundle(
                bundle_cfg=bundle_cfg,
                static_cfg=static_cfg,
                sweep_cfg=sweep_cfg,
                t_traj=t_export if export_ready else None,
                y_traj=y_export if export_ready else None,
                var_names=var_names,
                traj_ready=export_ready,
                traj_source=export_source,
                df_sweep=df_sweep,
                lya_data=lya_data,
            )
            st.download_button(
                label="Download Run Bundle (zip)",
                data=bundle_bytes,
                file_name="run_bundle.zip",
                mime="application/zip",
                key="dl_run_bundle_zip",
            )


except Exception as e:
    st.error(str(e))
    st.info("Check: variable names count, equations count, parameter format, y0 length, and dt/tf values.")
