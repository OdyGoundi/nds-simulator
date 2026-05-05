from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from app.logic.reservoir_sampling import ensure_xy_reservoir, update_xy_reservoir
from app.state import LyapunovDataKeys, SweepControlsKeys, SweepDataKeys

MAX_BIF_RESERVOIR_POINTS = 120_000
MAX_SWEEP_ROWS_IN_MEMORY = 300_000
DIRECTION_LABEL_BY_VALUE = {1: "+1 (up)", -1: "-1 (down)", 0: "0 (both)"}


def _to_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _meta_mismatches(prev: Dict[str, Any], curr: Dict[str, Any]) -> List[str]:
    mismatches: List[str] = []
    for k, v_prev in prev.items():
        if k not in curr:
            continue
        v_now = curr[k]
        if isinstance(v_prev, float) and isinstance(v_now, float):
            if not math.isclose(v_prev, v_now, rel_tol=0.0, abs_tol=1e-12):
                mismatches.append(k)
        elif v_prev != v_now:
            mismatches.append(k)
    return mismatches


def _clip_sweep_df(df: pd.DataFrame, max_rows: int) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    max_rows_i = max(1, int(max_rows))
    if len(df) <= max_rows_i:
        return df, pd.DataFrame(columns=df.columns), False
    drop_n = int(len(df) - max_rows_i)
    dropped = df.head(drop_n).reset_index(drop=True)
    kept = df.tail(max_rows_i).reset_index(drop=True)
    return kept, dropped, True


def _append_dropped_rows_to_reservoir(df_dropped: pd.DataFrame, x_col: str, y_col: str) -> None:
    if df_dropped is None or len(df_dropped) == 0:
        return
    if x_col not in df_dropped.columns or y_col not in df_dropped.columns:
        return
    x_old = np.asarray(df_dropped[x_col].to_numpy(), dtype=float)
    y_old = np.asarray(df_dropped[y_col].to_numpy(), dtype=float)
    reservoir_state = ensure_xy_reservoir(
        st.session_state.get(SweepDataKeys.RESERVOIR),
        capacity=MAX_BIF_RESERVOIR_POINTS,
    )
    reservoir_state = update_xy_reservoir(reservoir_state, x_old, y_old)
    st.session_state[SweepDataKeys.RESERVOIR] = reservoir_state


def _apply_sweep_config_to_state(
    cfg: dict,
    *,
    sweep_choices: list[str],
    var_names: list[str],
    t0_default: float,
    tf_default: float,
    dt_default: float,
) -> None:
    sweep_obj = cfg.get("sweep")
    lyapunov_obj = cfg.get("lyapunov")
    if not isinstance(sweep_obj, dict):
        raise ValueError("Invalid config: missing 'sweep' block.")
    sweep_settings = sweep_obj.get("settings")
    if not isinstance(sweep_settings, dict):
        raise ValueError("Invalid config: missing 'sweep.settings' block.")

    sweep_param = str(sweep_settings.get("sweep_param", "")).strip()
    if sweep_param in sweep_choices:
        st.session_state[SweepControlsKeys.PARAM] = sweep_param

    sweep_start = _to_float(sweep_settings.get("sweep_start", 0.0), 0.0)
    sweep_stop = _to_float(sweep_settings.get("sweep_stop", 50.0), 50.0)
    sweep_step = _to_float(sweep_settings.get("sweep_step", 0.1), 0.1)
    st.session_state[SweepControlsKeys.START] = float(sweep_start)
    st.session_state[SweepControlsKeys.STOP] = float(sweep_stop)
    st.session_state[SweepControlsKeys.STEP] = float(sweep_step)
    st.session_state[SweepDataKeys.STOP_INTERNAL] = float(sweep_stop)

    sweep_integration = sweep_settings.get("integration")
    if isinstance(sweep_integration, dict):
        dt_sweep = max(1e-6, _to_float(sweep_integration.get("dt", dt_default), dt_default))
        tf_sweep = max(float(t0_default) + 1e-6, _to_float(sweep_integration.get("tf", tf_default), tf_default))
        st.session_state[SweepControlsKeys.DT_SWEEP] = float(dt_sweep)
        st.session_state[SweepControlsKeys.TF_SWEEP] = float(tf_sweep)

    mode = sweep_settings.get("mode")
    if isinstance(mode, dict):
        warm_start = bool(mode.get("warm_start", False))
        st.session_state[SweepControlsKeys.MODE] = (
            "Continuation (warm start)" if warm_start else "Bifurcation (reset ICs)"
        )
        st.session_state[SweepControlsKeys.BIF_PARALLEL] = bool(mode.get("parallel", False))
        workers_bif = mode.get("parallel_workers")
        if workers_bif is not None:
            st.session_state[SweepControlsKeys.BIF_WORKERS] = max(1, _to_int(workers_bif, 1))

    sweep_solver = sweep_settings.get("solver")
    if isinstance(sweep_solver, dict):
        if "rtol" in sweep_solver:
            st.session_state[SweepControlsKeys.RTOL_SWEEP] = _to_float(sweep_solver.get("rtol"), 3e-4)
        if "atol" in sweep_solver:
            st.session_state[SweepControlsKeys.ATOL_SWEEP] = _to_float(sweep_solver.get("atol"), 1e-6)

    poincare = sweep_settings.get("poincare")
    if isinstance(poincare, dict):
        section_var = str(poincare.get("section_var", "")).strip()
        if section_var in var_names:
            st.session_state[SweepControlsKeys.SECTION_VAR] = section_var
        st.session_state[SweepControlsKeys.SECTION_VAL] = _to_float(poincare.get("section_value", 0.0), 0.0)
        st.session_state[SweepControlsKeys.SECTION_EXPR] = str(poincare.get("section_expr", "") or "")
        direction = _to_int(poincare.get("direction", 1), 1)
        st.session_state[SweepControlsKeys.SECTION_DIR] = DIRECTION_LABEL_BY_VALUE.get(direction, "+1 (up)")
        method = str(poincare.get("method", "crossing")).strip().lower()
        st.session_state[SweepControlsKeys.SECTION_METHOD] = "slab" if method == "slab" else "crossing"
        st.session_state[SweepControlsKeys.SECTION_TOL] = _to_float(poincare.get("tol", 1e-3), 1e-3)

    output = sweep_settings.get("output")
    if isinstance(output, dict):
        output_var = str(output.get("var", "")).strip()
        if output_var in var_names:
            st.session_state[SweepControlsKeys.OUTPUT_VAR] = output_var

    observable = str(sweep_settings.get("observable", "poincare") or "poincare").strip().lower()
    st.session_state[SweepControlsKeys.OBS_KIND] = (
        "Local extrema (max/min)" if observable == "extrema" else "Poincaré crossings"
    )
    extrema_kind = str(sweep_settings.get("extrema_kind", "max") or "max").strip().lower()
    if extrema_kind not in ("max", "min", "both"):
        extrema_kind = "max"
    st.session_state[SweepControlsKeys.EXTREMA_KIND] = extrema_kind

    transient = sweep_settings.get("transient")
    if isinstance(transient, dict) and "fraction" in transient:
        st.session_state[SweepControlsKeys.TRANSIENT_FRAC] = float(
            max(0.0, min(0.95, _to_float(transient.get("fraction", 0.75), 0.75)))
        )

    run = sweep_settings.get("run")
    if isinstance(run, dict):
        st.session_state[SweepControlsKeys.EARLY_STOP] = bool(run.get("early_stop", True))
        st.session_state[SweepControlsKeys.MAX_HITS] = max(10, _to_int(run.get("max_hits", 200), 200))
        st.session_state[SweepControlsKeys.CHUNK_TIME] = max(0.1, _to_float(run.get("chunk_time", 2.0), 2.0))

    if isinstance(lyapunov_obj, dict):
        lyapunov_settings = lyapunov_obj.get("settings")
        if isinstance(lyapunov_settings, dict):
            st.session_state[SweepControlsKeys.QR_INTERVAL_LYA] = max(
                1e-6, _to_float(lyapunov_settings.get("qr_interval", 0.1), 0.1)
            )
            if "transient_fraction" in lyapunov_settings:
                st.session_state[SweepControlsKeys.LYA_TRANSIENT_FRAC] = float(
                    max(0.0, min(0.95, _to_float(lyapunov_settings.get("transient_fraction", 0.30), 0.30)))
                )
            st.session_state[SweepControlsKeys.LYA_PARALLEL] = bool(lyapunov_settings.get("parallel", False))
            lya_workers = lyapunov_settings.get("parallel_workers")
            if lya_workers is not None:
                st.session_state[SweepControlsKeys.LYA_WORKERS] = max(1, _to_int(lya_workers, 1))
            clip = lyapunov_settings.get("clip")
            if isinstance(clip, dict):
                clip_enabled = bool(clip.get("enabled", False))
                st.session_state[SweepControlsKeys.CLIP_LYAPUNOV] = clip_enabled
                clip_min = clip.get("min", -50.0)
                if clip_min is not None:
                    st.session_state[SweepControlsKeys.CLIP_MIN_LYAPUNOV] = _to_float(clip_min, -50.0)


def _init_sweep_state() -> None:
    if SweepDataKeys.STOP_INTERNAL not in st.session_state:
        st.session_state[SweepDataKeys.STOP_INTERNAL] = float(
            st.session_state.get(SweepDataKeys.STOP_INTERNAL, 50.0)
        )
    if SweepDataKeys.ACC_DF not in st.session_state:
        st.session_state[SweepDataKeys.ACC_DF] = None
    if SweepDataKeys.LAST_PV not in st.session_state:
        st.session_state[SweepDataKeys.LAST_PV] = None
    if SweepDataKeys.BOUNDARIES not in st.session_state:
        st.session_state[SweepDataKeys.BOUNDARIES] = []
    if SweepDataKeys.META not in st.session_state:
        st.session_state[SweepDataKeys.META] = {}
    if SweepDataKeys.ROWS_CLIPPED not in st.session_state:
        st.session_state[SweepDataKeys.ROWS_CLIPPED] = False
    st.session_state[SweepDataKeys.RESERVOIR] = ensure_xy_reservoir(
        st.session_state.get(SweepDataKeys.RESERVOIR),
        capacity=MAX_BIF_RESERVOIR_POINTS,
    )
    if LyapunovDataKeys.ACC_DATA not in st.session_state:
        st.session_state[LyapunovDataKeys.ACC_DATA] = None
    if LyapunovDataKeys.LAST_PV not in st.session_state:
        st.session_state[LyapunovDataKeys.LAST_PV] = None
    if LyapunovDataKeys.META not in st.session_state:
        st.session_state[LyapunovDataKeys.META] = {}
    if LyapunovDataKeys.BOUNDARIES not in st.session_state:
        st.session_state[LyapunovDataKeys.BOUNDARIES] = []
