"""
dynaSim
    A Streamlit app for simulating and analyzing non-linear dynamical systems.
    Interactive Nonlinear Dynamics Laboratory for Simulation, Trajectory Plotting,
    Strovoscopic Visualization, Lyapunov Exponent Calculation, Lyapunov Spectrum, 
    and Bifurcation or Continuation Analysis.

=====================================================================================

@author: Odysseas Gkountinakos
@for:             Postgraduate thesis project at Aristotle University of Thessaloniki, Greece
                  Msc in Computational Physics, supervised by Prof. Christos Volos
@created:         2026-06

=====================================================================================
overview:         https://nlds-simulator.streamlit.app/
full source code: https://github.com/OdyGoundi/nds-simulator

=====================================================================================
Before exploring the code, fully reccommended to read and follow the documentation.

=====================================================================================
version: 1.0
Gnu General Public License v3.0, Free Software
@info:            ody.gkount@gmail.com
"""


import sys
from pathlib import Path
from typing import Callable, List, Optional

# Ensure project root import works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.helpers import (
    build_custom_symplectic_functions,
    build_custom_symbolic_jacobian_str,
    parse_params,
)
from app.services.system_registry import BUILTIN_SYSTEMS, get_builtin
from app.state.apply_config import (
    apply_state_values,
    flush_pending_static_config_apply,
)
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
from app.ui.tabs.export_tab import render_export_tab
from app.ui.tabs.time_series_tab import render_time_series_tab

APP_NAME = "DynaSim"
APP_SUBTITLE = "Non-linear Dynamical Systems Simulator"


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
if system_key in BUILTIN_SYSTEMS:
    _adapter = get_builtin(system_key)
    var_names = list(_adapter.var_names)
    eq_lines = list(_adapter.eq_lines) if _adapter.eq_lines else [""] * int(n_vars)
    params_text = _adapter.params_text
    if _adapter.eq_lines:
        with st.sidebar:
            st.header(f"{_adapter.display_name} definition")
            st.caption(f"State order: {', '.join(_adapter.var_names)}")
            st.code("\n".join(_adapter.eq_lines), language="text")
            if _adapter.params_text:
                st.caption(f"Parameters: {_adapter.params_text}")
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
    _symplectic_eligible = system_key == "custom" or (
        system_key in BUILTIN_SYSTEMS and get_builtin(system_key).supports_symplectic
    )
    if not _symplectic_eligible:
        st.caption("Symplectic preview is available for custom or symplectic built-in systems.")
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
        elif system_key in BUILTIN_SYSTEMS:
            st.success(f"Symplectic check passed ({get_builtin(system_key).display_name}).")


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

    render_time_series_tab(tabs[1], phase_result=phase_result)

    render_sweep_tab(
        tab=tabs[2],
        system=system,
        integration=integration,
        initial=initial,
        solve_tols=solve_tols,
        app_name=APP_NAME,
        repo_root=PROJECT_ROOT,
    )

    render_export_tab(tabs[3], phase_result=phase_result, app_name=APP_NAME, repo_root=PROJECT_ROOT)


except Exception as e:
    st.error(str(e))
    st.info("Check: variable names count, equations count, parameter format, y0 length, and dt/tf values.")
