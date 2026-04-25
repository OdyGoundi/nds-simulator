SYSTEM_LABEL_BY_KEY = {
    "lorenz": "Lorenz (3D)",
    "rossler": "Rossler (3D)",
    "henon_heiles": "Henon-Heiles (4D Hamiltonian)",
    "custom": "Custom (nD)",
}

SYSTEM_KEY_BY_LABEL = {label: key for key, label in SYSTEM_LABEL_BY_KEY.items()}

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

MAX_PLOT_POINTS_DEFAULT = 1_000_000
MAX_PLOT_POINTS_UI_MAX = 2_000_000
MAX_STORE_STEPS_DEFAULT = 0
PHASE_LINEWIDTH_DEFAULT = 0.07

DIRECT_CSV_MAX_ROWS = 600_000
EXPORT_CHUNK_ROWS_DEFAULT = 250_000

TRAJ_EXPORT_SOURCE_STORED = "Current stored trajectory (fast)"
TRAJ_EXPORT_SOURCE_FULL = "Prepared full-resolution trajectory (recommended)"
TRAJ_EXPORT_READY_SIG_KEY = "traj_export_full_ready_sig_tab4"
