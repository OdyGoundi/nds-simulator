import base64
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

# Ensure project root import works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from app.cache import solve_cached
from app.helpers import (
    build_csv_bytes,
    build_custom_symplectic_functions,
    build_custom_symbolic_jacobian_str,
    downsample_trajectory,
    parse_list_of_floats,
    parse_params,
)
from app.export_utils import build_static_config
from app.logic.lyapunov_cached import compute_lyapunov_cached
from app.plots import (
    plot_phase_2d,
    plot_phase_3d,
    plot_time_seiries_functional,
    plot_time_series,
)
from app.params import (
    CustomSystemDefinition,
    HenonHeilesParams,
    InitialConditions,
    IntegrationConfig,
    LorenzParams,
    LyapunovConfig,
    RosslerParams,
    SolverTolerances,
    SystemConfig,
)
from app.ui.bifurcation_tab import render_bifurcation_tab
from core import numba_backend

APP_NAME = "DynaSim"
APP_SUBTITLE = "Non-linear Dynamical Systems Simulator"
APP_LOGO_CANDIDATES = [
    PROJECT_ROOT / "docs" / "thesis" / "figures" / "new_logo.png",
    PROJECT_ROOT / "docs" / "assets" / "new_logo.png",
]
APP_LOGO_INVERT_CANDIDATES = [
    PROJECT_ROOT / "docs" / "thesis" / "figures" / "new_logo.png",
    PROJECT_ROOT / "docs" / "assets" / "new_logo.png",
]
HENON_HEILES_VAR_NAMES = ["q1", "q2", "p1", "p2"]
HENON_HEILES_EQ_LINES = [
    "p1",
    "p2",
    "-q1 - 2*lambda*q1*q2",
    "-q2 - lambda*(q1**2 - q2**2)",
]
HENON_HEILES_PARAMS_TEXT = "lambda=1.0"
SYSTEM_LABEL_BY_KEY = {
    "lorenz": "Lorenz (3D)",
    "rossler": "Rossler (3D)",
    "henon_heiles": "Henon-Heiles (4D Hamiltonian)",
    "custom": "Custom (nD)",
}
SOLVER_LABEL_BY_KIND = {
    "ivp": "RK45 (adaptive)",
    "rk45": "RK45 (adaptive)",
    "dop853": "DOP853 (non-stiff, high order)",
    "rk4": "RK4 (fixed step)",
    "symplectic_fr": "Symplectic Forest-Ruth (4th order)",
}
PENDING_STATIC_CFG_KEY = "_pending_static_cfg_apply"
STATIC_CFG_APPLY_SUCCESS_KEY = "_static_cfg_apply_success_msg"
STATIC_CFG_APPLY_ERROR_KEY = "_static_cfg_apply_error_msg"
MAX_PLOT_POINTS_DEFAULT = 120_000
MAX_STORE_STEPS_DEFAULT = 200_000
DIRECT_CSV_MAX_ROWS = 600_000
EXPORT_CHUNK_ROWS_DEFAULT = 250_000


def _axis_bounds(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        delta = max(1e-6, 0.05 * max(1.0, abs(vmin)))
        return vmin - delta, vmax + delta
    return vmin, vmax


def _square_xy_bounds(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    x0, x1 = float(x_bounds[0]), float(x_bounds[1])
    y0, y1 = float(y_bounds[0]), float(y_bounds[1])
    dx = max(1e-12, x1 - x0)
    dy = max(1e-12, y1 - y0)
    half_span = 0.5 * max(dx, dy)
    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)
    return (x_mid - half_span, x_mid + half_span), (y_mid - half_span, y_mid + half_span)


def _to_float(value: Any, default: Any) -> float:
    try:
        return float(value)
    except Exception:
        try:
            return float(default)
        except Exception:
            return 0.0


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(int(low), min(int(value), int(high)))


def _image_path_to_data_uri(image_path: Path) -> Optional[str]:
    if not image_path.exists():
        return None
    suffix = image_path.suffix.lower()
    mime = "image/png"
    if suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    data_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data_b64}"


def _pick_latest_existing_path(paths: List[Path]) -> Optional[Path]:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def _get_runtime_theme_base() -> str:
    context_obj = getattr(st, "context", None)
    if context_obj is not None:
        theme_obj = getattr(context_obj, "theme", None)
        if isinstance(theme_obj, dict):
            base_val = theme_obj.get("base")
            if base_val is not None:
                return str(base_val).strip().lower()
        elif theme_obj is not None:
            base_attr = getattr(theme_obj, "base", None)
            if base_attr is not None:
                return str(base_attr).strip().lower()
    return ""


def _render_header_logo(width_px: int = 196, align: str = "center") -> bool:
    light_logo_path = _pick_latest_existing_path(APP_LOGO_CANDIDATES)
    dark_logo_path = _pick_latest_existing_path(APP_LOGO_INVERT_CANDIDATES)
    light_logo_uri = _image_path_to_data_uri(light_logo_path) if light_logo_path is not None else None
    dark_logo_uri = _image_path_to_data_uri(dark_logo_path) if dark_logo_path is not None else None
    if light_logo_uri is None and dark_logo_uri is None:
        return False
    if light_logo_uri is None:
        light_logo_uri = dark_logo_uri
    if dark_logo_uri is None:
        dark_logo_uri = light_logo_uri
    if light_logo_uri is None or dark_logo_uri is None:
        return False

    css_align = "left" if str(align).strip().lower() == "left" else "center"
    runtime_theme = _get_runtime_theme_base()
    light_default_display = "none" if runtime_theme == "dark" else "inline-block"
    dark_default_display = "inline-block" if runtime_theme == "dark" else "none"

    st.markdown(
        f"""
<style>
.dynasim-header-logo-wrap {{
  width: 100%;
  text-align: {css_align};
}}
.dynasim-header-logo-wrap img {{
  width: {int(width_px)}px;
  height: auto;
}}
.dynasim-header-logo-dark {{
  display: {dark_default_display};
}}
.dynasim-header-logo-light {{
  display: {light_default_display};
}}
html[data-theme="dark"] .dynasim-header-logo-light,
html[theme="dark"] .dynasim-header-logo-light,
body[data-theme="dark"] .dynasim-header-logo-light,
body[theme="dark"] .dynasim-header-logo-light,
body.dark .dynasim-header-logo-light {{
  display: none !important;
}}
html[data-theme="dark"] .dynasim-header-logo-dark,
html[theme="dark"] .dynasim-header-logo-dark,
body[data-theme="dark"] .dynasim-header-logo-dark,
body[theme="dark"] .dynasim-header-logo-dark,
body.dark .dynasim-header-logo-dark {{
  display: inline-block !important;
}}
</style>
<div class="dynasim-header-logo-wrap">
  <img class="dynasim-header-logo-light" src="{light_logo_uri}" alt="dynaSim logo">
  <img class="dynasim-header-logo-dark" src="{dark_logo_uri}" alt="dynaSim logo dark">
</div>
        """,
        unsafe_allow_html=True,
    )
    return True


def _params_dict_to_text(params_obj: object) -> str:
    if not isinstance(params_obj, dict):
        return ""
    lines: List[str] = []
    for key, val in params_obj.items():
        try:
            lines.append(f"{str(key)}={float(val):g}")
        except Exception:
            continue
    return "\n".join(lines)


def _apply_static_config_to_state(cfg: Dict[str, object]) -> None:
    system_obj = cfg.get("system")
    integration_obj = cfg.get("integration")
    postprocess_obj = cfg.get("postprocess")
    plots_obj = cfg.get("plots")
    lyapunov_obj = cfg.get("lyapunov")
    if not isinstance(system_obj, dict) or not isinstance(integration_obj, dict):
        raise ValueError("Invalid config: missing 'system' or 'integration' blocks.")

    system_key = str(system_obj.get("system_key", "")).strip().lower()
    system_label = SYSTEM_LABEL_BY_KEY.get(system_key)
    if system_label is not None:
        st.session_state["system_label_sidebar"] = system_label

    if system_key == "lorenz":
        params = system_obj.get("params") if isinstance(system_obj.get("params"), dict) else {}
        st.session_state["sigma"] = _to_float((params or {}).get("sigma", 10.0), 10.0)
        st.session_state["rho"] = _to_float((params or {}).get("rho", 28.0), 28.0)
        st.session_state["beta"] = _to_float((params or {}).get("beta", 8.0 / 3.0), 8.0 / 3.0)
    elif system_key == "rossler":
        params = system_obj.get("params") if isinstance(system_obj.get("params"), dict) else {}
        st.session_state["ross_a"] = _to_float((params or {}).get("a", 0.2), 0.2)
        st.session_state["ross_b"] = _to_float((params or {}).get("b", 0.2), 0.2)
        st.session_state["ross_c"] = _to_float((params or {}).get("c", 5.7), 5.7)
    elif system_key == "henon_heiles":
        params = system_obj.get("params") if isinstance(system_obj.get("params"), dict) else {}
        st.session_state["hh_lambda"] = _to_float((params or {}).get("lambda", 1.0), 1.0)
    elif system_key == "custom":
        var_names = system_obj.get("var_names") if isinstance(system_obj.get("var_names"), list) else []
        eq_lines = system_obj.get("eq_lines") if isinstance(system_obj.get("eq_lines"), list) else []
        params_text = str(system_obj.get("params_text", "") or "").strip()
        if not params_text:
            params_text = _params_dict_to_text(system_obj.get("params"))
        var_names_list = var_names if isinstance(var_names, list) else []
        eq_lines_list = eq_lines if isinstance(eq_lines, list) else []
        n_vars_custom = len(var_names_list) if len(var_names_list) > 0 else len(eq_lines_list)
        if n_vars_custom > 0:
            st.session_state["n_vars_sidebar"] = int(n_vars_custom)
        if var_names_list and len(var_names_list) > 0:
            st.session_state["var_names_text_sidebar"] = "\n".join(str(v) for v in var_names_list)
        if eq_lines_list and len(eq_lines_list) > 0:
            st.session_state["eqs_text_sidebar"] = "\n".join(str(v) for v in eq_lines_list)
        st.session_state["params_text_sidebar"] = params_text
        auto_jac = bool(system_obj.get("auto_jacobian", False))
        use_jac = bool(system_obj.get("use_jacobian", auto_jac))
        st.session_state["custom_auto_jac_sidebar"] = auto_jac
        st.session_state["custom_use_jac_sidebar"] = bool(use_jac and auto_jac)

    t0 = _to_float(integration_obj.get("t0", 0.0), 0.0)
    tf = _to_float(integration_obj.get("tf", 50.0), 50.0)
    dt = max(1e-12, _to_float(integration_obj.get("dt", 0.01), 0.01))
    max_store_cfg_raw = integration_obj.get("max_store_steps", MAX_STORE_STEPS_DEFAULT)
    try:
        max_store_cfg = int(max_store_cfg_raw) if max_store_cfg_raw is not None else 0
    except Exception:
        max_store_cfg = MAX_STORE_STEPS_DEFAULT
    if max_store_cfg < 0:
        max_store_cfg = 0
    st.session_state["t0_tab1"] = float(t0)
    st.session_state["tf_tab1"] = float(tf)
    st.session_state["dt_tab1"] = float(dt)
    st.session_state["max_store_steps_tab1"] = int(max_store_cfg)

    y0 = integration_obj.get("y0")
    if isinstance(y0, list) and len(y0) > 0:
        try:
            y0_text = ", ".join(f"{float(v):g}" for v in y0)
            st.session_state["y0_text_sidebar"] = y0_text
        except Exception:
            pass

    solver_kind = str(integration_obj.get("solver_kind", "")).strip().lower()
    solver_label = SOLVER_LABEL_BY_KIND.get(solver_kind)
    if solver_label is not None:
        st.session_state["solver_kind_label_sidebar"] = solver_label

    solve_opts = integration_obj.get("solve_options")
    if isinstance(solve_opts, dict):
        if "rtol" in solve_opts:
            st.session_state["rtol"] = _to_float(solve_opts.get("rtol"), 1e-6)
        if "atol" in solve_opts:
            st.session_state["atol"] = _to_float(solve_opts.get("atol"), 1e-8)

    if isinstance(postprocess_obj, dict):
        transient_steps_cfg = max(0, _to_int(postprocess_obj.get("transient_steps", 0), 0))
        st.session_state["transient_cut_time_tab1"] = float(transient_steps_cfg) * float(dt)

    n_vars_axes = 3
    if system_key == "henon_heiles":
        n_vars_axes = 4
    elif system_key == "custom":
        var_names_obj = system_obj.get("var_names")
        var_names_count = len(var_names_obj) if isinstance(var_names_obj, (list, tuple)) else 0
        n_vars_axes = max(
            1,
            _to_int(st.session_state.get("n_vars_sidebar", 0), 0),
            var_names_count,
        )

    if isinstance(plots_obj, dict):
        plot_mode = str(plots_obj.get("plot_mode", "")).strip()
        if plot_mode in ("2D phase plane", "3D phase plot"):
            st.session_state["plot_mode_tab1"] = plot_mode
        phase_axes_obj = plots_obj.get("phase_axes")
        phase_axes = phase_axes_obj if isinstance(phase_axes_obj, dict) else {}
        x_idx_cfg = _to_int(phase_axes.get("x_idx", 0), 0)
        y_default = 1 if n_vars_axes > 1 else 0
        y_idx_cfg = _to_int(phase_axes.get("y_idx", y_default), y_default)
        z_default = 2 if n_vars_axes > 2 else 0
        z_idx_cfg = _to_int(phase_axes.get("z_idx", z_default), z_default)
        st.session_state["phase_x_idx_tab1"] = _clamp_int(x_idx_cfg, 0, n_vars_axes - 1)
        st.session_state["phase_y_idx_tab1"] = _clamp_int(y_idx_cfg, 0, n_vars_axes - 1)
        st.session_state["phase_z_idx_tab1"] = _clamp_int(z_idx_cfg, 0, n_vars_axes - 1)

    if isinstance(lyapunov_obj, dict):
        lya_settings = lyapunov_obj.get("settings")
        if isinstance(lya_settings, dict):
            if "qr_interval" in lya_settings:
                st.session_state["qr_interval_tab1"] = max(
                    1e-6, _to_float(lya_settings.get("qr_interval", 0.1), 0.1)
                )
            frac = None
            if "transient_fraction" in lya_settings:
                frac = _to_float(lya_settings.get("transient_fraction", 0.3), 0.3)
            elif "transient_steps" in lya_settings:
                n_steps_est = max(1.0, (float(tf) - float(t0)) / float(dt))
                frac = _to_float(lya_settings.get("transient_steps", 0), 0.0) / float(n_steps_est)
            if frac is not None:
                st.session_state["lya_transient_frac_tab1"] = float(max(0.0, min(0.99, frac)))


def _zip_bytes(file_map: Dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in file_map.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _flush_pending_static_config_apply() -> None:
    pending_cfg = st.session_state.pop(PENDING_STATIC_CFG_KEY, None)
    if pending_cfg is None:
        return
    try:
        if not isinstance(pending_cfg, dict):
            raise ValueError("JSON root must be an object.")
        _apply_static_config_to_state(pending_cfg)
        st.session_state["static_config"] = pending_cfg
        st.session_state[STATIC_CFG_APPLY_SUCCESS_KEY] = (
            "Static configuration loaded. Settings were applied."
        )
        st.session_state.pop(STATIC_CFG_APPLY_ERROR_KEY, None)
    except Exception as exc:
        st.session_state[STATIC_CFG_APPLY_ERROR_KEY] = str(exc)
        st.session_state.pop(STATIC_CFG_APPLY_SUCCESS_KEY, None)

st.set_page_config(page_title="dynaSim", layout="wide")

def _render_manual(manual_html_path: Path, manual_pdf_path: Path, fallback_markdown: str) -> None:
    if manual_html_path.exists():
        html = manual_html_path.read_text(encoding="utf-8")
        components.html(html, height=640, scrolling=True)
        return
    if manual_pdf_path.exists():
        pdf_bytes = manual_pdf_path.read_bytes()
        b64 = base64.b64encode(pdf_bytes).decode("ascii")
        pdf_html = (
            "<iframe "
            f"src=\"data:application/pdf;base64,{b64}\" "
            "width=\"100%\" height=\"640\" style=\"border:0;\" "
            "></iframe>"
        )
        components.html(pdf_html, height=640, scrolling=True)
        return
    st.markdown(fallback_markdown)


def _render_quick_manual_eng() -> None:
    manual_html_path = PROJECT_ROOT / "docs" / "user-guide" / "manual.html"
    manual_pdf_path = PROJECT_ROOT / "docs" / "user-guide" / "manual.pdf"
    _render_manual(
        manual_html_path,
        manual_pdf_path,
        """
**Manual not available**

Please check that `docs/user-guide/manual.html` (or `manual.pdf`) exists.
        """,
    )


def _render_quick_manual_el() -> None:
    manual_html_path = PROJECT_ROOT / "docs" / "user-guide" / "manual-el.html"
    manual_pdf_path = PROJECT_ROOT / "docs" / "user-guide" / "manual-el.pdf"
    _render_manual(
        manual_html_path,
        manual_pdf_path,
        """
**Το εγχειρίδιο δεν είναι διαθέσιμο**

Ελέγξτε ότι υπάρχει το `docs/user-guide/manual-el.html` (ή `manual-el.pdf`).
        """,
    )

def _render_info() -> None:
    info_html_path = PROJECT_ROOT / "docs" / "user-guide" / "info.html"
    if info_html_path.exists():
        html = info_html_path.read_text(encoding="utf-8")
        components.html(html, height=520, scrolling=True)
        return
    st.markdown(
        """
**Info not available**

Please check that `docs/user-guide/info.html` exists.
        """
    )


DialogDecorator = Callable[[str], Callable[[Callable[[], None]], Callable[[], None]]]


def _get_dialog_decorator() -> Optional[DialogDecorator]:
    dialog = getattr(st, "dialog", None)
    if callable(dialog):
        return cast(DialogDecorator, dialog)
    dialog = getattr(st, "experimental_dialog", None)
    if callable(dialog):
        return cast(DialogDecorator, dialog)
    return None


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
    if not _render_header_logo(width_px=282, align="left"):
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

dialog_decorator = _get_dialog_decorator()
_quick_manual_eng_dialog: Optional[Callable[[], None]] = None
_quick_manual_el_dialog: Optional[Callable[[], None]] = None
_info_dialog: Optional[Callable[[], None]] = None
if dialog_decorator is not None:

    @dialog_decorator("Quick Start Manual")
    def _quick_manual_eng_dialog_impl() -> None:
        _render_quick_manual_eng()
        if st.button("Close manual", key="close_quick_manual_btn"):
            st.session_state["show_quick_manual_eng"] = False
            st.rerun()

    _quick_manual_eng_dialog = _quick_manual_eng_dialog_impl

    @dialog_decorator("Σύντομο Εγχειρίδιο")
    def _quick_manual_el_dialog_impl() -> None:
        _render_quick_manual_el()
        if st.button("Κλείσιμο εγχειριδίου", key="close_quick_manual_el_btn"):
            st.session_state["show_quick_manual_el"] = False
            st.rerun()

    _quick_manual_el_dialog = _quick_manual_el_dialog_impl

    @dialog_decorator("Info")
    def _info_dialog_impl() -> None:
        _render_info()
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
            _render_quick_manual_eng()
            if st.button("Hide manual", key="hide_quick_manual_btn"):
                st.session_state["show_quick_manual_eng"] = False

if st.session_state.get("show_quick_manual_el", False):
    if _quick_manual_el_dialog is not None:
        _quick_manual_el_dialog()
        st.session_state["show_quick_manual_el"] = False
    else:
        with st.expander("Σύντομο Εγχειρίδιο", expanded=True):
            _render_quick_manual_el()
            if st.button("Απόκρυψη εγχειριδίου", key="hide_quick_manual_el_btn"):
                st.session_state["show_quick_manual_el"] = False

if st.session_state.get("show_info_popup", False):
    if _info_dialog is not None:
        _info_dialog()
        st.session_state["show_info_popup"] = False
    else:
        with st.expander("Info", expanded=True):
            _render_info()
            if st.button("Hide info", key="hide_info_btn"):
                st.session_state["show_info_popup"] = False

# Apply uploaded static config before sidebar widgets are instantiated.
_flush_pending_static_config_apply()

# -------- Sidebar: system + initial conditions --------
with st.sidebar:
    st.header("System")

    system_label = st.selectbox(
        "Choose system",
        ["Lorenz (3D)", "Rossler (3D)", "Henon-Heiles (4D Hamiltonian)", "Custom (nD)"],
        index=0,
        key="system_label_sidebar",
    )

    if system_label.startswith("Lorenz"):
        system_key = "lorenz"
        n_vars = 3
    elif system_label.startswith("Rossler"):
        system_key = "rossler"
        n_vars = 3
    elif system_label.startswith("Henon-Heiles"):
        system_key = "henon_heiles"
        n_vars = 4
    else:
        system_key = "custom"
        n_vars = st.number_input(
            "Number of equations (n)",
            min_value=1,
            max_value=12,
            value=3,
            step=1,
            key="n_vars_sidebar",
        )

    st.markdown("**Solver kind**")
    solver_kind_labels = [
        "RK45 (adaptive)",
        "DOP853 (non-stiff, high order)",
        "RK4 (fixed step)",
        "Symplectic Forest-Ruth (4th order)",
    ]
    solver_kind_map = {
        "RK45 (adaptive)": "rk45",
        "DOP853 (non-stiff, high order)": "dop853",
        "RK4 (fixed step)": "rk4",
        "Symplectic Forest-Ruth (4th order)": "symplectic_fr",
    }
    solver_default = "RK4 (fixed step)"
    solver_kind_label = st.selectbox(
        "Solver kind",
        solver_kind_labels,
        index=solver_kind_labels.index(solver_default),
        key="solver_kind_label_sidebar",
    )
    solver_kind = solver_kind_map[solver_kind_label]
    st.markdown(
        "- RK45 adaptive: default choice, uses rtol/atol.\n"
        "- DOP853: high-order solver for non-stiff problems, uses rtol/atol.\n"
        "- RK4 fixed: fixed dt, faster but needs smaller dt for accuracy.\n"
        "- Symplectic Forest-Ruth: separable Hamiltonians, state = [q..., p...], dq/dt uses p only, dp/dt uses q only."
    )
    solver_kind_effective = solver_kind
    if solver_kind.startswith("symplectic"):
        if system_key not in ("custom", "henon_heiles"):
            st.warning("Symplectic solvers require Hamiltonian systems. Using RK45 instead.")
            solver_kind_effective = "rk45"
        elif int(n_vars) % 2 != 0:
            st.warning("Symplectic solvers require an even number of variables [q..., p...]. Using RK45 instead.")
            solver_kind_effective = "rk45"
    elif system_key == "henon_heiles":
        st.caption("Henon-Heiles is Hamiltonian: symplectic solvers are recommended.")

    st.divider()
    st.header("Initial conditions")

    if system_key == "henon_heiles":
        y0_default = "0.1, 0.0, 0.0, 0.1"
    elif int(n_vars) == 3:
        y0_default = "1, 1, 1"
    else:
        y0_default = "\n".join(["0"] * int(n_vars))
    y0_text = st.text_area(
        "y0 values (comma/space/newline separated)",
        value=y0_default,
        height=90,
        key="y0_text_sidebar",
    )

    ## Optional "run" button (Streamlit reruns anyway)
    #run_btn = st.button("Run / Refresh")


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
        var_names_text = st.text_area(
            "Variable names (one per line)",
            value=default_names,
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
        eqs_text = st.text_area(
            "Equations dy/dt (one per line)",
            value=default_eq,
            height=180,
            key="eqs_text_sidebar",
        )

        params_text = st.text_area(
            "Parameters (name=value per line)",
            value="",
            height=120,
            key="params_text_sidebar",
        )

        eq_lines = [ln.strip() for ln in (eqs_text or "").splitlines()]
        eq_lines = (eq_lines + ["0"] * int(n_vars))[:int(n_vars)]

        st.markdown("**Jacobian (custom)**")
        custom_auto_jac = st.checkbox(
            "Auto-compute Jacobian (symbolic)",
            value=False,
            key="custom_auto_jac_sidebar",
            help="Builds a symbolic Jacobian for Lyapunov on custom systems.",
        )
        custom_use_jac = st.checkbox(
            "Use analytic Jacobian",
            value=custom_auto_jac,
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


# -------- System parameters defaults --------
# Default values
sigma = rho = beta = 0.0
ross_a = ross_b = ross_c = 0.0
hh_lambda = 1.0


# -------- Main layout: outputs only --------
st.subheader("Outputs")

tabs = st.tabs(["Phase portrait & Lyapunov exponents", "Time series", "Parameter Sweep Analysis", "Export"])

# Solve once, then all outputs derive from (t, y)
try:
    # --- Tab 1: Phase portrait (controls) ---
    with tabs[0]:
        phase_col_controls, phase_col_plot = st.columns([1, 2], gap="large")

        with phase_col_controls:
            st.header("Integration")
            t0 = st.number_input("initial time", value=0.0, step=1.0, key="t0_tab1")
            tf = st.number_input("final time", value=50.0, step=1.0, key="tf_tab1")
            dt = st.number_input("time step", value=0.01, step=0.01, format="%.5f", key="dt_tab1")

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

            elif system_key == "henon_heiles":
                hh_lambda = st.number_input(
                    "lambda",
                    value=1.0,
                    step=0.05,
                    format="%.4f",
                    key="hh_lambda",
                )

            else:
                st.caption("Custom: parameters are defined above.")

            st.divider()
            st.header("Plot settings")
            plot_mode = st.selectbox(
                "Plot mode",
                ["2D phase plane", "3D phase plot"],
                index=0,
                key="plot_mode_tab1",
            )

            tc_c1, tc_c2 = st.columns([1.2, 1], gap="small")
            with tc_c1:
                transient_cut_time = st.number_input(
                    "Transient cut (time to skip)",
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                    format="%.6f",
                    key="transient_cut_time_tab1",
                    help=(
                        "Skips trajectory samples from t0 up to t0 + this duration before plotting/export. "
                        "The equivalent number of steps is shown on the right. Does not affect Lyapunov."
                    ),
                )
            with tc_c2:
                dt_abs = max(abs(float(dt)), 1e-12)
                transient_steps = int(max(0.0, float(transient_cut_time)) / dt_abs)
                st.metric("Equivalent transient steps", transient_steps)

            perf_c1, perf_c2 = st.columns([1, 1], gap="small")
            with perf_c1:
                max_plot_points = st.number_input(
                    "Max points per plot",
                    min_value=10_000,
                    max_value=500_000,
                    value=MAX_PLOT_POINTS_DEFAULT,
                    step=10_000,
                    key="max_plot_points_tab1",
                    help="Uniformly downsamples trajectories for plotting only.",
                )
            with perf_c2:
                max_store_steps_ui = st.number_input(
                    "Max stored trajectory samples (0=all)",
                    min_value=0,
                    max_value=5_000_000,
                    value=MAX_STORE_STEPS_DEFAULT,
                    step=50_000,
                    key="max_store_steps_tab1",
                    help=(
                        "Limits in-memory trajectory samples to avoid Streamlit Cloud crashes. "
                        "Integration still runs full duration."
                    ),
                )

            st.divider()
            st.header("Lyapunov exponents calculation settings")
            qr_interval = st.number_input(
                "QR interval (time)",
                min_value=1e-6,
                value=0.1,
                step=0.01,
                format="%.4f",
                key="qr_interval_tab1",
                help="Time between orthonormalizations during Lyapunov computation.",
            )
            lya_c1, lya_c2 = st.columns([1, 1], gap="small")
            with lya_c1:
                lyapunov_transient_frac = st.slider(
                    "Lyapunov transient fraction",
                    min_value=0.0,
                    max_value=0.99,
                    value=0.30,
                    step=0.05,
                    key="lya_transient_frac_tab1",
                    help="Fraction of integration steps discarded before Lyapunov accumulation.",
                )
            with lya_c2:
                n_steps_est_lya = int(max(1.0, (float(tf) - float(t0)) / float(dt)))
                transient_steps_lya = int(lyapunov_transient_frac * n_steps_est_lya)
                st.metric("Lyapunov transient steps (estimated)", transient_steps_lya)
            compute_lya_btn = st.button(
                "Compute Lyapunov exponents",
                key="compute_lya_tab1",
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
                key="phase_x_idx_tab1",
            )

            y_default = 1 if len(idx_list) > 1 else 0
            y_idx = st.selectbox(
                "y-axis",
                options=idx_list,
                format_func=lambda i: axis_options[i][0],
                index=y_default,
                key="phase_y_idx_tab1",
            )

            z_idx = 2 if len(idx_list) > 2 else 0
            if plot_mode == "3D phase plot":
                z_idx = st.selectbox(
                    "z-axis",
                    options=idx_list,
                    format_func=lambda i: axis_options[i][0],
                    index=2 if len(idx_list) > 2 else 0,
                    key="phase_z_idx_tab1",
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
            if str(solver_kind_effective) not in ("rk45", "dop853", "ivp"):
                st.caption("Note: rtol/atol are used only by RK45/DOP853.")
            solve_tols = SolverTolerances(rtol=float(rtol), atol=float(atol))

        st.divider()
        st.markdown("**Configuration**")
        static_cfg_apply_success = st.session_state.pop(STATIC_CFG_APPLY_SUCCESS_KEY, None)
        if static_cfg_apply_success:
            st.success(str(static_cfg_apply_success))
        static_cfg_apply_error = st.session_state.pop(STATIC_CFG_APPLY_ERROR_KEY, None)
        if static_cfg_apply_error:
            st.error(f"Failed to load StaticParamsConfig: {static_cfg_apply_error}")
        _, cfg_center_tab1, _ = st.columns([1, 1.8, 1], gap="large")
        with cfg_center_tab1:
            save_static_cfg = st.button(
                "Save configuration",
                key="save_static_cfg_tab1",
                use_container_width=True,
            )
            static_cfg_upload = st.file_uploader(
                "Upload StaticParamsConfig.json",
                type=["json"],
                key="upload_static_cfg_tab1",
                help="Load settings from a previously exported static configuration file.",
            )
            apply_static_cfg = st.button(
                "Apply uploaded configuration",
                key="apply_static_cfg_tab1",
                disabled=static_cfg_upload is None,
                use_container_width=True,
            )

        if apply_static_cfg and static_cfg_upload is not None:
            try:
                loaded_static_cfg = json.loads(static_cfg_upload.getvalue().decode("utf-8"))
                if not isinstance(loaded_static_cfg, dict):
                    raise ValueError("JSON root must be an object.")
                st.session_state.pop(STATIC_CFG_APPLY_SUCCESS_KEY, None)
                st.session_state.pop(STATIC_CFG_APPLY_ERROR_KEY, None)
                st.session_state[PENDING_STATIC_CFG_KEY] = loaded_static_cfg
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load StaticParamsConfig: {exc}")

    y0 = parse_list_of_floats(y0_text, int(n_vars), label="y0")
    if system_key == "henon_heiles":
        params_text = f"lambda={float(hh_lambda)}"
    initial = InitialConditions(tuple(float(v) for v in y0))
    max_store_steps = int(max_store_steps_ui)
    if max_store_steps <= 0:
        max_store_steps = None
    integration = IntegrationConfig(
        t0=float(t0),
        tf=float(tf),
        dt=float(dt),
        solver_kind=str(solver_kind_effective),
        max_store_steps=max_store_steps,
    )
    system = SystemConfig(
        key=system_key,
        lorenz=LorenzParams(sigma=float(sigma), rho=float(rho), beta=float(beta)),
        rossler=RosslerParams(a=float(ross_a), b=float(ross_b), c=float(ross_c)),
        henon_heiles=HenonHeilesParams(lam=float(hh_lambda)),
        custom=CustomSystemDefinition(
            var_names=tuple(var_names),
            eq_lines=tuple(eq_lines),
            params_text=params_text,
            auto_jacobian=bool(custom_auto_jac),
            use_jacobian=bool(custom_use_jac),
        ),
    )
    lyapunov_cfg = LyapunovConfig(
        transient_steps=int(transient_steps_lya),
        qr_interval=float(qr_interval),
    )

    if save_static_cfg:
        z_idx_val = int(z_idx) if plot_mode == "3D phase plot" else None
        static_config = build_static_config(
            app_name=APP_NAME,
            repo_root=PROJECT_ROOT,
            system=system,
            integration=integration,
            initial=initial,
            solve_tols=solve_tols,
            plot_mode=plot_mode,
            x_idx=int(x_idx),
            y_idx=int(y_idx),
            z_idx=z_idx_val,
            transient_steps=int(transient_steps),
            lyapunov_transient_steps=int(transient_steps_lya),
            lyapunov_transient_frac=float(lyapunov_transient_frac),
            qr_interval=float(qr_interval),
        )
        st.session_state["static_config"] = static_config
        with phase_col_controls:
            st.success("Static configuration saved. Download from the Export tab.")

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
    max_plot_points_i = max(2, int(max_plot_points))
    t_plot_ds, y_plot_ds = downsample_trajectory(t_plot, y_plot, max_plot_points_i)

    # --- Tab 1: Phase portrait (plot) ---
    with phase_col_plot:
        x_auto = _axis_bounds(y_plot_ds[int(x_idx), :])
        y_auto = _axis_bounds(y_plot_ds[int(y_idx), :])
        z_auto = None
        if plot_mode == "3D phase plot":
            z_auto = _axis_bounds(y_plot_ds[int(z_idx), :])

        phase_bounds_sig = (
            str(plot_mode),
            int(x_idx),
            int(y_idx),
            int(z_idx) if plot_mode == "3D phase plot" else None,
            float(x_auto[0]),
            float(x_auto[1]),
            float(y_auto[0]),
            float(y_auto[1]),
            float(z_auto[0]) if z_auto is not None else None,
            float(z_auto[1]) if z_auto is not None else None,
        )
        if st.session_state.get("phase_bounds_sig_tab1") != phase_bounds_sig:
            st.session_state["phase_xlim_min_tab1"] = float(x_auto[0])
            st.session_state["phase_xlim_max_tab1"] = float(x_auto[1])
            st.session_state["phase_ylim_min_tab1"] = float(y_auto[0])
            st.session_state["phase_ylim_max_tab1"] = float(y_auto[1])
            if z_auto is not None:
                st.session_state["phase_zlim_min_tab1"] = float(z_auto[0])
                st.session_state["phase_zlim_max_tab1"] = float(z_auto[1])
            st.session_state["phase_bounds_sig_tab1"] = phase_bounds_sig

        x_view = (
            float(st.session_state.get("phase_xlim_min_tab1", x_auto[0])),
            float(st.session_state.get("phase_xlim_max_tab1", x_auto[1])),
        )
        y_view = (
            float(st.session_state.get("phase_ylim_min_tab1", y_auto[0])),
            float(st.session_state.get("phase_ylim_max_tab1", y_auto[1])),
        )
        z_view = None
        if z_auto is not None:
            z_view = (
                float(st.session_state.get("phase_zlim_min_tab1", z_auto[0])),
                float(st.session_state.get("phase_zlim_max_tab1", z_auto[1])),
            )

        valid_xy = x_view[0] < x_view[1] and y_view[0] < y_view[1]
        valid_z = (z_view is None) or (z_view[0] < z_view[1])
        if not valid_xy or not valid_z:
            st.warning("Invalid axis limits detected. Reverting to data bounds.")
            x_view = x_auto
            y_view = y_auto
            if z_auto is not None:
                z_view = z_auto
            st.session_state["phase_xlim_min_tab1"] = float(x_view[0])
            st.session_state["phase_xlim_max_tab1"] = float(x_view[1])
            st.session_state["phase_ylim_min_tab1"] = float(y_view[0])
            st.session_state["phase_ylim_max_tab1"] = float(y_view[1])
            if z_view is not None:
                st.session_state["phase_zlim_min_tab1"] = float(z_view[0])
                st.session_state["phase_zlim_max_tab1"] = float(z_view[1])

        phase_square_axes = False
        if plot_mode == "2D phase plane":
            phase_square_axes = st.checkbox(
                "Square axes (equal x/y scale)",
                value=bool(st.session_state.get("phase_square_axes_tab1", False)),
                key="phase_square_axes_tab1",
                help="Use the same scale on both axes and keep the phase plot square.",
            )
            if phase_square_axes:
                x_view, y_view = _square_xy_bounds(x_view, y_view)
                st.session_state["phase_xlim_min_tab1"] = float(x_view[0])
                st.session_state["phase_xlim_max_tab1"] = float(x_view[1])
                st.session_state["phase_ylim_min_tab1"] = float(y_view[0])
                st.session_state["phase_ylim_max_tab1"] = float(y_view[1])

        if plot_mode == "2D phase plane":
            title = f"{system_label} – {var_names[int(y_idx)]} vs {var_names[int(x_idx)]}"
            fig = plot_phase_2d(
                y=y_plot_ds,
                i=int(x_idx),
                j=int(y_idx),
                title=title,
                xlabel=var_names[int(x_idx)],
                ylabel=var_names[int(y_idx)],
            )
            ax = fig.axes[0]
            ax.set_xlim(float(x_view[0]), float(x_view[1]))
            ax.set_ylim(float(y_view[0]), float(y_view[1]))
            if phase_square_axes:
                ax.set_aspect("equal", adjustable="box")
            st.pyplot(fig, clear_figure=True)

        else:
            title = f"{system_label} – 3D phase ({var_names[int(x_idx)]}, {var_names[int(y_idx)]}, {var_names[int(z_idx)]})"
            fig = plot_phase_3d(
                y=y_plot_ds,
                i=int(x_idx),
                j=int(y_idx),
                k=int(z_idx),
                title=title,
                labels=(var_names[int(x_idx)], var_names[int(y_idx)], var_names[int(z_idx)]),
            )
            ax3d = fig.axes[0]
            ax3d.set_xlim(float(x_view[0]), float(x_view[1]))
            ax3d.set_ylim(float(y_view[0]), float(y_view[1]))
            if z_view is not None and hasattr(ax3d, "set_zlim"):
                cast(Any, ax3d).set_zlim(float(z_view[0]), float(z_view[1]))
            st.pyplot(fig, clear_figure=True)

        st.markdown("**Axis limits (view window)**")
        if plot_mode == "2D phase plane":
            lim_c1, lim_c2, lim_c3, lim_c4 = st.columns([1, 1, 1, 1], gap="small")
            with lim_c1:
                st.number_input(
                    f"{var_names[int(x_idx)]} min",
                    format="%.6f",
                    key="phase_xlim_min_tab1",
                )
            with lim_c2:
                st.number_input(
                    f"{var_names[int(x_idx)]} max",
                    format="%.6f",
                    key="phase_xlim_max_tab1",
                )
            with lim_c3:
                st.number_input(
                    f"{var_names[int(y_idx)]} min",
                    format="%.6f",
                    key="phase_ylim_min_tab1",
                )
            with lim_c4:
                st.number_input(
                    f"{var_names[int(y_idx)]} max",
                    format="%.6f",
                    key="phase_ylim_max_tab1",
                )
        else:
            lim_c1, lim_c2, lim_c3 = st.columns([1, 1, 1], gap="small")
            with lim_c1:
                st.number_input(
                    f"{var_names[int(x_idx)]} min",
                    format="%.6f",
                    key="phase_xlim_min_tab1",
                )
                st.number_input(
                    f"{var_names[int(x_idx)]} max",
                    format="%.6f",
                    key="phase_xlim_max_tab1",
                )
            with lim_c2:
                st.number_input(
                    f"{var_names[int(y_idx)]} min",
                    format="%.6f",
                    key="phase_ylim_min_tab1",
                )
                st.number_input(
                    f"{var_names[int(y_idx)]} max",
                    format="%.6f",
                    key="phase_ylim_max_tab1",
                )
            with lim_c3:
                st.number_input(
                    f"{var_names[int(z_idx)]} min",
                    format="%.6f",
                    key="phase_zlim_min_tab1",
                )
                st.number_input(
                    f"{var_names[int(z_idx)]} max",
                    format="%.6f",
                    key="phase_zlim_max_tab1",
                )

        st.caption(
            f"Default bounds: {var_names[int(x_idx)]} [{x_auto[0]:.4g}, {x_auto[1]:.4g}], "
            f"{var_names[int(y_idx)]} [{y_auto[0]:.4g}, {y_auto[1]:.4g}]"
            + (
                f", {var_names[int(z_idx)]} [{z_auto[0]:.4g}, {z_auto[1]:.4g}]"
                if z_auto is not None
                else ""
            )
        )

        st.caption(
            f"Total steps: {len(t)} | stored: {len(t_plot)} | plotted: {len(t_plot_ds)} | "
            f"transient cut: {float(transient_cut_time):.3f} time ({N} steps) | "
            f"n_vars: {y.shape[0]} | t in [{t[0]:.2f}, {t[-1]:.2f}]"
        )
        est_steps = int(np.floor((float(tf) - float(t0)) / max(float(dt), 1e-12))) + 1
        if max_store_steps is not None and est_steps > len(t):
            st.caption(
                f"Storage cap active: kept {len(t)} of ~{est_steps} trajectory samples in memory."
            )

        st.divider()
        st.markdown("**Lyapunov exponents**")
        t_transient_lya = float(transient_steps_lya) * float(dt)
        total_time_lya = float(tf) - float(t0)
        t_measure_lya = total_time_lya - t_transient_lya
        lya_sig = (
            repr(system),
            repr(integration),
            repr(initial),
            repr(lyapunov_cfg),
            repr(solve_tols),
        )
        if t_measure_lya <= 0.0:
            st.warning("Not enough time for Lyapunov measurement. Increase tf or reduce Lyapunov transient fraction.")
        else:
            if compute_lya_btn:
                try:
                    with st.spinner("Computing Lyapunov spectrum..."):
                        lambdas = compute_lyapunov_cached(
                            system=system,
                            integration=integration,
                            initial=initial,
                            lyapunov=lyapunov_cfg,
                            solve_tols=solve_tols,
                        )
                    st.session_state["lya_result_tab1"] = np.array(lambdas, dtype=float)
                    st.session_state["lya_result_sig"] = lya_sig
                except Exception as exc:
                    st.warning(f"Lyapunov computation failed: {exc}")

            lambdas = st.session_state.get("lya_result_tab1", None)
            sig_ok = st.session_state.get("lya_result_sig", None) == lya_sig
            if lambdas is not None and sig_ok:
                formatted = ", ".join(f"{val:.5f}" for val in lambdas)
                st.write(f"lambda = [{formatted}]")
                st.caption(
                    f"n={len(lambdas)} | t_transient={t_transient_lya:.3f} | "
                    f"t_measure={t_measure_lya:.3f} | lyapunov_transient_frac={lyapunov_transient_frac:.2f}"
                )
            else:
                st.info("Click 'Compute Lyapunov exponents' to run the calculation.")

    # --- Tab 2: Time series (one plot per variable)
    with tabs[1]:
        st.markdown("**Time series (post-transient)**")

        t_min = float(t_plot[0])
        t_max = float(t_plot[-1])
        time_step_ui = max(float(dt), 1e-6)
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
        render_bifurcation_tab(
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
            rtol_tag = f"{float(rtol):.0e}"
            atol_tag = f"{float(atol):.0e}"
            traj_base = f"{system_key}_trajectory_rtol{rtol_tag}_atol{atol_tag}"
            traj_rows = int(t_plot.size)

            if traj_rows <= 0:
                st.info("No trajectory samples available for export.")
            else:
                if traj_rows <= int(DIRECT_CSV_MAX_ROWS):
                    csv_bytes = build_csv_bytes(t_plot, y_plot, var_names)
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
                    t_plot,
                    y_plot,
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

            st.divider()
            st.markdown("**Export: Run bundle (zip)**")
            bundle_files: Dict[str, bytes] = {}

            if static_cfg is None:
                z_idx_val = int(z_idx) if plot_mode == "3D phase plot" else None
                bundle_cfg = build_static_config(
                    app_name=APP_NAME,
                    repo_root=PROJECT_ROOT,
                    system=system,
                    integration=integration,
                    initial=initial,
                    solve_tols=solve_tols,
                    plot_mode=plot_mode,
                    x_idx=int(x_idx),
                    y_idx=int(y_idx),
                    z_idx=z_idx_val,
                    transient_steps=int(transient_steps),
                    lyapunov_transient_steps=int(transient_steps_lya),
                    lyapunov_transient_frac=float(lyapunov_transient_frac),
                    qr_interval=float(qr_interval),
                )
            else:
                bundle_cfg = static_cfg

            bundle_files["config.json"] = json.dumps(bundle_cfg, indent=2).encode("utf-8")
            if static_cfg is not None:
                bundle_files["StaticParamsConfig.json"] = json.dumps(static_cfg, indent=2).encode("utf-8")
            if sweep_cfg is not None:
                bundle_files["SweepParamConfig.json"] = json.dumps(sweep_cfg, indent=2).encode("utf-8")

            if int(t_plot.size) <= int(DIRECT_CSV_MAX_ROWS):
                bundle_files["trajectory.csv"] = build_csv_bytes(t_plot, y_plot, var_names)
            else:
                end_first = min(int(t_plot.size), int(EXPORT_CHUNK_ROWS_DEFAULT))
                bundle_files["trajectory_part001.csv"] = build_csv_bytes(
                    t_plot,
                    y_plot,
                    var_names,
                    start=0,
                    end=end_first,
                )
                bundle_files["trajectory_manifest.txt"] = (
                    f"Trajectory rows: {int(t_plot.size)}\n"
                    "Only the first chunk is included in this zip to keep memory bounded.\n"
                    "Use Tab 4 chunk export to download the remaining chunks.\n"
                ).encode("utf-8")

            if df_sweep is not None and len(df_sweep) > 0:
                if not isinstance(df_sweep, pd.DataFrame):
                    df_sweep = pd.DataFrame(df_sweep)
                bundle_files["sweep.csv"] = df_sweep.to_csv(index=False).encode("utf-8")

            if lya_data is not None:
                param_vals = np.array(lya_data.get("param_vals", []), dtype=float)
                lambdas_arr = np.array(lya_data.get("lambdas", []), dtype=float)
                if param_vals.size and lambdas_arr.size:
                    meta = lya_data.get("meta", {})
                    sweep_param = meta.get("sweep_param", "param")
                    data = {str(sweep_param): param_vals}
                    if lambdas_arr.ndim == 1:
                        data["lambda0"] = lambdas_arr
                    else:
                        for k in range(lambdas_arr.shape[1]):
                            data[f"lambda{k}"] = lambdas_arr[:, k]
                    df_lya = pd.DataFrame(data)
                    bundle_files["lyapunov_sweep.csv"] = df_lya.to_csv(index=False).encode("utf-8")

            bundle_bytes = _zip_bytes(bundle_files)
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
