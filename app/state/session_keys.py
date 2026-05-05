"""Typed session-state key constants, grouped by feature/tab.

Each class holds the literal string keys used with ``st.session_state``.
Constant names are uppercase; the value is exactly the string Streamlit sees.

Use these instead of bare string literals so renames are mechanical and
``grep`` reports every call site.
"""
from __future__ import annotations


class SidebarKeys:
    """Inputs from the global sidebar (system/solver/initial conditions)."""
    SYSTEM_LABEL = "system_label_sidebar"
    SOLVER_KIND_LABEL = "solver_kind_label_sidebar"
    N_VARS = "n_vars_sidebar"
    VAR_NAMES_TEXT = "var_names_text_sidebar"
    EQS_TEXT = "eqs_text_sidebar"
    PARAMS_TEXT = "params_text_sidebar"
    CUSTOM_AUTO_JAC = "custom_auto_jac_sidebar"
    CUSTOM_USE_JAC = "custom_use_jac_sidebar"
    Y0_TEXT = "y0_text_sidebar"


class SystemParamKeys:
    """Built-in system parameter widgets (Tab 1)."""
    SIGMA = "sigma"
    RHO = "rho"
    BETA = "beta"
    ROSS_A = "ross_a"
    ROSS_B = "ross_b"
    ROSS_C = "ross_c"
    HH_LAMBDA = "hh_lambda"


class TolKeys:
    """Solver tolerance widgets (Tab 1 phase)."""
    RTOL = "rtol"
    ATOL = "atol"


class IntegrationKeys:
    """Tab 1 integration window widgets."""
    T0 = "t0_tab1"
    TF = "tf_tab1"
    DT = "dt_tab1"
    MAX_STORE_STEPS = "max_store_steps_tab1"
    MAX_PLOT_POINTS = "max_plot_points_tab1"
    TRANSIENT_CUT_TIME = "transient_cut_time_tab1"


class PhaseKeys:
    """Tab 1 phase-plot widgets and view state."""
    PLOT_MODE = "plot_mode_tab1"
    X_IDX = "phase_x_idx_tab1"
    Y_IDX = "phase_y_idx_tab1"
    Z_IDX = "phase_z_idx_tab1"
    XLIM_MIN = "phase_xlim_min_tab1"
    XLIM_MAX = "phase_xlim_max_tab1"
    YLIM_MIN = "phase_ylim_min_tab1"
    YLIM_MAX = "phase_ylim_max_tab1"
    ZLIM_MIN = "phase_zlim_min_tab1"
    ZLIM_MAX = "phase_zlim_max_tab1"
    BOUNDS_SIG = "phase_bounds_sig_tab1"
    LINEWIDTH = "phase_linewidth_tab1"
    SQUARE_AXES = "phase_square_axes_tab1"


class LyapunovTab1Keys:
    """Tab 1 Lyapunov-spectrum controls and result cache."""
    QR_INTERVAL = "qr_interval_tab1"
    TRANSIENT_FRAC = "lya_transient_frac_tab1"
    RESULT = "lya_result_tab1"
    RESULT_SIG = "lya_result_sig"


class StaticConfigKeys:
    """Tab 1 static-config save/load slot."""
    CURRENT = "static_config"


class TimeSeriesKeys:
    """Tab 2 time-window state."""
    WINDOW_START = "ts_window_start_tab2"
    WINDOW_END = "ts_window_end_tab2"
    WINDOW_RANGE = "ts_window_range_tab2"


class SweepControlsKeys:
    """Tab 3 input widgets (parameter range, section, observable, runner)."""
    PARAM = "sw_param_tab3"
    START = "sw_start_tab3"
    STOP = "sw_stop_tab3"
    STEP = "sw_step_tab3"
    DT_SWEEP = "dt_sweep_tab3"
    TF_SWEEP = "tf_sweep_tab3"
    SECTION_VAR = "sec_var_tab3"
    SECTION_VAL = "sec_val_tab3"
    SECTION_EXPR = "sec_expr_tab3"
    SECTION_DIR = "sec_dir_tab3"
    SECTION_METHOD = "sec_method_tab3"
    SECTION_TOL = "sec_tol_tab3"
    OUTPUT_VAR = "out_var_tab3"
    OBS_KIND = "obs_kind_tab3"
    EXTREMA_KIND = "ext_kind_tab3"
    TRANSIENT_FRAC = "sw_transient_frac_tab3"
    MODE = "sweep_mode_tab3"
    EARLY_STOP = "early_stop_tab3"
    MAX_HITS = "max_hits_tab3"
    CHUNK_TIME = "chunk_time_tab3"
    BIF_PARALLEL = "bif_parallel_tab3"
    BIF_WORKERS = "bif_workers_tab3"
    RTOL_SWEEP = "rtol_sweep_tab3"
    ATOL_SWEEP = "atol_sweep_tab3"
    QR_INTERVAL_LYA = "qr_interval_lya_tab3"
    LYA_TRANSIENT_FRAC = "lya_transient_frac_tab3"
    LYA_PARALLEL = "lya_parallel_tab3"
    LYA_WORKERS = "lya_workers_tab3"
    CLIP_LYAPUNOV = "clip_lyapunov_tab3"
    CLIP_MIN_LYAPUNOV = "clip_min_lyapunov_tab3"


class BifPlotKeys:
    """Tab 3 bifurcation-plot view state."""
    XLIM_MIN = "bif_xlim_min_tab3"
    XLIM_MAX = "bif_xlim_max_tab3"
    YLIM_MIN = "bif_ylim_min_tab3"
    YLIM_MAX = "bif_ylim_max_tab3"
    BOUNDS_SIG = "bif_bounds_sig_tab3"


class LyaPlotKeys:
    """Tab 3 Lyapunov-plot view state."""
    XLIM_MIN = "lya_xlim_min_tab3"
    XLIM_MAX = "lya_xlim_max_tab3"
    YLIM_MIN = "lya_ylim_min_tab3"
    YLIM_MAX = "lya_ylim_max_tab3"
    BOUNDS_SIG = "lya_bounds_sig_tab3"


class SweepDataKeys:
    """Cross-tab bifurcation-sweep accumulation."""
    ACC_DF = "sweep_acc_df"
    LAST_PV = "sweep_last_pv"
    META = "sweep_meta"
    RESERVOIR = "sweep_reservoir"
    ROWS_CLIPPED = "sweep_rows_clipped"
    STOP_INTERNAL = "sweep_stop_internal"
    BOUNDARIES = "sweep_boundaries"
    CONFIG = "sweep_config"
    LAST_DF = "last_sweep_df"
    LAST_META = "last_sweep_meta"


class LyapunovDataKeys:
    """Cross-tab Lyapunov-sweep accumulation."""
    ACC_DATA = "lya_acc_data"
    LAST_PV = "lya_last_pv"
    META = "lya_meta"
    BOUNDARIES = "lya_boundaries"


class ExportKeys:
    """Tab 4 export controls."""
    PREPARE_FULL_TRAJ = "prepare_full_traj_export_tab4"
    TRAJ_CHUNK_INDEX = "traj_export_chunk_index_tab4"


class HelpPanelKeys:
    """Help/manual popup toggles."""
    SHOW_INFO_POPUP = "show_info_popup"
    SHOW_QUICK_MANUAL_EL = "show_quick_manual_el"
    SHOW_QUICK_MANUAL_ENG = "show_quick_manual_eng"
