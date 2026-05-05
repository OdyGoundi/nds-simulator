from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from app.state import (
    MAX_STORE_STEPS_DEFAULT,
    PENDING_STATIC_CFG_KEY,
    PHASE_LINEWIDTH_DEFAULT,
    SOLVER_LABEL_BY_KIND,
    STATIC_CFG_APPLY_ERROR_KEY,
    STATIC_CFG_APPLY_SUCCESS_KEY,
    SYSTEM_LABEL_BY_KEY,
    IntegrationKeys,
    LyapunovTab1Keys,
    PhaseKeys,
    SidebarKeys,
    StaticConfigKeys,
    SystemParamKeys,
    TolKeys,
)


def to_float(value: Any, default: Any) -> float:
    try:
        return float(value)
    except Exception:
        try:
            return float(default)
        except Exception:
            return 0.0


def to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def clamp_int(value: int, low: int, high: int) -> int:
    return max(int(low), min(int(value), int(high)))


def apply_state_values(values: Dict[str, Any], *, only_missing: bool = False) -> None:
    for key, value in values.items():
        if only_missing and key in st.session_state:
            continue
        st.session_state[key] = value


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


def apply_static_config_to_state(cfg: Dict[str, object]) -> None:
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
        st.session_state[SidebarKeys.SYSTEM_LABEL] = system_label

    if system_key == "lorenz":
        params = system_obj.get("params") if isinstance(system_obj.get("params"), dict) else {}
        st.session_state[SystemParamKeys.SIGMA] = to_float((params or {}).get("sigma", 10.0), 10.0)
        st.session_state[SystemParamKeys.RHO] = to_float((params or {}).get("rho", 28.0), 28.0)
        st.session_state[SystemParamKeys.BETA] = to_float((params or {}).get("beta", 8.0 / 3.0), 8.0 / 3.0)
    elif system_key == "rossler":
        params = system_obj.get("params") if isinstance(system_obj.get("params"), dict) else {}
        st.session_state[SystemParamKeys.ROSS_A] = to_float((params or {}).get("a", 0.2), 0.2)
        st.session_state[SystemParamKeys.ROSS_B] = to_float((params or {}).get("b", 0.2), 0.2)
        st.session_state[SystemParamKeys.ROSS_C] = to_float((params or {}).get("c", 5.7), 5.7)
    elif system_key == "henon_heiles":
        params = system_obj.get("params") if isinstance(system_obj.get("params"), dict) else {}
        st.session_state[SystemParamKeys.HH_LAMBDA] = to_float((params or {}).get("lambda", 1.0), 1.0)
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
            st.session_state[SidebarKeys.N_VARS] = int(n_vars_custom)
        if var_names_list and len(var_names_list) > 0:
            st.session_state[SidebarKeys.VAR_NAMES_TEXT] = "\n".join(str(v) for v in var_names_list)
        if eq_lines_list and len(eq_lines_list) > 0:
            st.session_state[SidebarKeys.EQS_TEXT] = "\n".join(str(v) for v in eq_lines_list)
        st.session_state[SidebarKeys.PARAMS_TEXT] = params_text
        auto_jac = bool(system_obj.get("auto_jacobian", False))
        use_jac = bool(system_obj.get("use_jacobian", auto_jac))
        st.session_state[SidebarKeys.CUSTOM_AUTO_JAC] = auto_jac
        st.session_state[SidebarKeys.CUSTOM_USE_JAC] = bool(use_jac and auto_jac)

    t0 = to_float(integration_obj.get("t0", 0.0), 0.0)
    tf = to_float(integration_obj.get("tf", 50.0), 50.0)
    dt = max(1e-12, to_float(integration_obj.get("dt", 0.01), 0.01))
    max_store_cfg_raw = integration_obj.get("max_store_steps", MAX_STORE_STEPS_DEFAULT)
    try:
        max_store_cfg = int(max_store_cfg_raw) if max_store_cfg_raw is not None else 0
    except Exception:
        max_store_cfg = MAX_STORE_STEPS_DEFAULT
    if max_store_cfg < 0:
        max_store_cfg = 0
    st.session_state[IntegrationKeys.T0] = float(t0)
    st.session_state[IntegrationKeys.TF] = float(tf)
    st.session_state[IntegrationKeys.DT] = float(dt)
    st.session_state[IntegrationKeys.MAX_STORE_STEPS] = int(max_store_cfg)

    y0 = integration_obj.get("y0")
    if isinstance(y0, list) and len(y0) > 0:
        try:
            y0_text = ", ".join(f"{float(v):g}" for v in y0)
            st.session_state[SidebarKeys.Y0_TEXT] = y0_text
        except Exception:
            pass

    solver_kind = str(integration_obj.get("solver_kind", "")).strip().lower()
    solver_label = SOLVER_LABEL_BY_KIND.get(solver_kind)
    if solver_label is not None:
        st.session_state[SidebarKeys.SOLVER_KIND_LABEL] = solver_label

    solve_opts = integration_obj.get("solve_options")
    if isinstance(solve_opts, dict):
        if "rtol" in solve_opts:
            st.session_state[TolKeys.RTOL] = to_float(solve_opts.get("rtol"), 1e-6)
        if "atol" in solve_opts:
            st.session_state[TolKeys.ATOL] = to_float(solve_opts.get("atol"), 1e-8)

    if isinstance(postprocess_obj, dict):
        transient_steps_cfg = max(0, to_int(postprocess_obj.get("transient_steps", 0), 0))
        st.session_state[IntegrationKeys.TRANSIENT_CUT_TIME] = float(transient_steps_cfg) * float(dt)

    n_vars_axes = 3
    if system_key == "henon_heiles":
        n_vars_axes = 4
    elif system_key == "custom":
        var_names_obj = system_obj.get("var_names")
        var_names_count = len(var_names_obj) if isinstance(var_names_obj, (list, tuple)) else 0
        n_vars_axes = max(
            1,
            to_int(st.session_state.get(SidebarKeys.N_VARS, 0), 0),
            var_names_count,
        )

    if isinstance(plots_obj, dict):
        plot_mode = str(plots_obj.get("plot_mode", "")).strip()
        if plot_mode in ("2D phase plane", "3D phase plot"):
            st.session_state[PhaseKeys.PLOT_MODE] = plot_mode
        if "phase_linewidth" in plots_obj:
            phase_linewidth_cfg = max(
                0.001,
                to_float(plots_obj.get("phase_linewidth", PHASE_LINEWIDTH_DEFAULT), PHASE_LINEWIDTH_DEFAULT),
            )
            st.session_state[PhaseKeys.LINEWIDTH] = float(phase_linewidth_cfg)
        phase_axes_obj = plots_obj.get("phase_axes")
        phase_axes = phase_axes_obj if isinstance(phase_axes_obj, dict) else {}
        x_idx_cfg = to_int(phase_axes.get("x_idx", 0), 0)
        y_default = 1 if n_vars_axes > 1 else 0
        y_idx_cfg = to_int(phase_axes.get("y_idx", y_default), y_default)
        z_default = 2 if n_vars_axes > 2 else 0
        z_idx_cfg = to_int(phase_axes.get("z_idx", z_default), z_default)
        st.session_state[PhaseKeys.X_IDX] = clamp_int(x_idx_cfg, 0, n_vars_axes - 1)
        st.session_state[PhaseKeys.Y_IDX] = clamp_int(y_idx_cfg, 0, n_vars_axes - 1)
        st.session_state[PhaseKeys.Z_IDX] = clamp_int(z_idx_cfg, 0, n_vars_axes - 1)

    if isinstance(lyapunov_obj, dict):
        lya_settings = lyapunov_obj.get("settings")
        if isinstance(lya_settings, dict):
            if "qr_interval" in lya_settings:
                st.session_state[LyapunovTab1Keys.QR_INTERVAL] = max(
                    1e-6, to_float(lya_settings.get("qr_interval", 0.1), 0.1)
                )
            frac = None
            if "transient_fraction" in lya_settings:
                frac = to_float(lya_settings.get("transient_fraction", 0.3), 0.3)
            elif "transient_steps" in lya_settings:
                n_steps_est = max(1.0, (float(tf) - float(t0)) / float(dt))
                frac = to_float(lya_settings.get("transient_steps", 0), 0.0) / float(n_steps_est)
            if frac is not None:
                st.session_state[LyapunovTab1Keys.TRANSIENT_FRAC] = float(max(0.0, min(0.99, frac)))


def flush_pending_static_config_apply() -> None:
    pending_cfg = st.session_state.pop(PENDING_STATIC_CFG_KEY, None)
    if pending_cfg is None:
        return
    try:
        if not isinstance(pending_cfg, dict):
            raise ValueError("JSON root must be an object.")
        apply_static_config_to_state(pending_cfg)
        st.session_state[StaticConfigKeys.CURRENT] = pending_cfg
        st.session_state[STATIC_CFG_APPLY_SUCCESS_KEY] = (
            "Static configuration loaded. Settings were applied."
        )
        st.session_state.pop(STATIC_CFG_APPLY_ERROR_KEY, None)
    except Exception as exc:
        st.session_state[STATIC_CFG_APPLY_ERROR_KEY] = str(exc)
        st.session_state.pop(STATIC_CFG_APPLY_SUCCESS_KEY, None)
