from typing import Dict, List, Optional

import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from app.helpers import parse_params
from app.sweep import run_sweep_chunk


def _init_sweep_state():
    if "sweep_stop_internal" not in st.session_state:
        st.session_state["sweep_stop_internal"] = float(
            st.session_state.get("sweep_stop_internal", 50.0)
        )
    if "sweep_acc_df" not in st.session_state:
        st.session_state["sweep_acc_df"] = None
    if "sweep_last_pv" not in st.session_state:
        st.session_state["sweep_last_pv"] = None
    if "sweep_boundaries" not in st.session_state:
        st.session_state["sweep_boundaries"] = []
    if "sweep_meta" not in st.session_state:
        st.session_state["sweep_meta"] = {}


def _sweep_settings_fingerprint(
    system_key: str,
    sweep_param: str,
    sweep_step: float,
    section_index: int,
    section_value: float,
    direction: int,
    method: str,
    tol: float,
    output_index: int,
    tf_sweep: float,
    dt_sweep: float,
    transient_frac: float,
    max_hits: int,
    early_stop: bool,
    chunk_time: float,
    warm_start: bool,
    rtol: Optional[float],
    atol: Optional[float],
) -> Dict[str, object]:
    return {
        "system_key": system_key,
        "sweep_param": str(sweep_param),
        "sweep_step": float(sweep_step),
        "section_index": int(section_index),
        "section_value": float(section_value),
        "direction": int(direction),
        "method": str(method),
        "tol": float(tol),
        "output_index": int(output_index),
        "tf_sweep": float(tf_sweep),
        "dt_sweep": float(dt_sweep),
        "transient_frac": float(transient_frac),
        "max_hits": int(max_hits),
        "early_stop": bool(early_stop),
        "chunk_time": float(chunk_time),
        "warm_start": bool(warm_start),
        "rtol": rtol,
        "atol": atol,
    }


def render_bifurcation_tab(
    tab,
    *,
    system_key: str,
    t0: float,
    tf: float,
    dt: float,
    y0: np.ndarray,
    sigma: float,
    rho: float,
    beta: float,
    ross_a: float,
    ross_b: float,
    ross_c: float,
    var_names: List[str],
    eq_lines: List[str],
    params_text: str,
):
    with tab:
        st.markdown("**Bifurcation / Poincaré sweep**")
        _init_sweep_state()

        left_col, right_col = st.columns([1, 1], gap="large")

        # Left column: sweep + section controls
        with left_col:
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="small")

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
                sweep_param = st.selectbox("Sweep param", sweep_choices, index=0, key="sw_param_tab3")
            with c2:
                sweep_start = st.number_input("start", value=0.0, step=0.1, format="%.6f", key="sw_start_tab3")
            with c3:
                sweep_stop = st.number_input(
                    "stop",
                    value=float(st.session_state["sweep_stop_internal"]),
                    step=0.1,
                    format="%.6f",
                    key="sw_stop_tab3",
                )

            with c4:
                sweep_step = st.number_input("step", value=0.1, step=0.01, format="%.6f", key="sw_step_tab3")

            st.divider()

            d1, d2, d3, d4, d5 = st.columns([1, 1, 1, 1, 1], gap="small")

            with d1:
                section_var = st.selectbox("Section var", var_names, index=0, key="sec_var_tab3")
                section_index = var_names.index(section_var)
            with d2:
                section_value = st.number_input("Section value", value=0.0, step=0.1, format="%.6f", key="sec_val_tab3")
            with d3:
                direction_label = st.selectbox("Direction", ["+1 (up)", "-1 (down)", "0 (both)"], index=0, key="sec_dir_tab3")
                direction = +1 if direction_label.startswith("+1") else (-1 if direction_label.startswith("-1") else 0)
            with d4:
                method = st.selectbox("Method", ["crossing", "slab"], index=0, key="sec_method_tab3")
            with d5:
                tol = st.number_input("tolerance (slab only)", value=1e-3, step=1e-3, format="%.6f", key="sec_tol_tab3")

            out_var = st.selectbox("Output var (plotted)", var_names, index=min(2, len(var_names) - 1), key="out_var_tab3")
            output_index = var_names.index(out_var)

            st.caption("Uses sweep-specific transient fraction (see right column).")

        # Right column: performance + transient
        with right_col:
            st.markdown("**Sweep performance settings**")
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
                    help="Maximum number of crossings kept per parameter value."
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
                    help="Integration time window for event detection."
                )

            st.markdown("**Sweep solver tolerances**")
            t1c1, t1c2 = st.columns([1, 1], gap="small")
            with t1c1:
                rtol_sweep = st.number_input(
                    "relative tolerance (sweep)",
                    min_value=0.0,
                    value=3e-4,
                    step=1e-4,
                    format="%.1e",
                    key="rtol_sweep_tab3",
                )
            with t1c2:
                atol_sweep = st.number_input(
                    "absolute tolerance (sweep)",
                    min_value=0.0,
                    value=1e-6,
                    step=1e-6,
                    format="%.1e",
                    key="atol_sweep_tab3",
                )

            solve_options_sweep = {"rtol": float(rtol_sweep), "atol": float(atol_sweep)}

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
                    help="Fraction of sweep integration steps to discard before crossings."
                )
            with tc2:
                n_steps_est = int(max(1.0, (float(tf_sweep) - float(t0)) / float(dt_sweep)))
                transient_steps_sweep = int(transient_frac * n_steps_est)
                st.metric("Transient steps (estimated)", transient_steps_sweep)

        # Buttons (full width)
        b1, b2, b3 = st.columns([1, 1, 1], gap="small")
        run_new = b1.button("Generate Bifurcation Diagram", type="primary", key="run_new_sweep")
        run_cont = b2.button("Continue Generation", type="secondary", key="run_cont_sweep")
        reset_acc = b3.button("Reset accumulated", type="secondary", key="reset_acc_sweep")

        if reset_acc:
            st.session_state["sweep_acc_df"] = None
            st.session_state["sweep_last_pv"] = None
            st.session_state["sweep_boundaries"] = []
            st.session_state["sweep_meta"] = {}
            st.success("Accumulated sweep cleared.")

        df_plot = None

        have_prev = (
            st.session_state.get("sweep_acc_df", None) is not None and
            st.session_state.get("sweep_last_pv", None) is not None
        )

        continue_stop = None
        if have_prev:
            last_pv_ui = float(st.session_state["sweep_last_pv"])
            continue_stop = st.number_input(
                f"Continue to (stop) [{sweep_param}]",
                min_value=last_pv_ui + float(sweep_step),
                value=max(float(sweep_stop), last_pv_ui + float(sweep_step)),
                step=float(sweep_step),
                format="%.6f",
                key="continue_stop_tab3",
                help="Sets the new stop for Continue Generation. Start is automatically last_pv + step."
            )

        sweep_meta = _sweep_settings_fingerprint(
            system_key=system_key,
            sweep_param=sweep_param,
            sweep_step=sweep_step,
            section_index=section_index,
            section_value=section_value,
            direction=direction,
            method=method,
            tol=tol,
            output_index=output_index,
            tf_sweep=tf_sweep,
            dt_sweep=dt_sweep,
            transient_frac=transient_frac,
            max_hits=max_hits,
            early_stop=early_stop,
            chunk_time=chunk_time,
            warm_start=warm_start,
            rtol=float(rtol_sweep),
            atol=float(atol_sweep),
        )

        if run_new:
            st.session_state["sweep_acc_df"] = None
            st.session_state["sweep_last_pv"] = None
            st.session_state["sweep_boundaries"] = []
            st.session_state["sweep_meta"] = sweep_meta

            start_here = float(sweep_start)
            stop_here = float(sweep_stop)

            with st.spinner("Running sweep..."):
                df_chunk = run_sweep_chunk(
                    system_key=system_key,
                    t0=float(t0), tf=float(tf_sweep), dt=float(dt_sweep),
                    y0=np.array(y0, dtype=float),
                    sigma=float(sigma), rho=float(rho), beta=float(beta),
                    ross_a=float(ross_a), ross_b=float(ross_b), ross_c=float(ross_c),
                    var_names=list(var_names), eq_lines=list(eq_lines),
                    params_text=params_text,
                    sweep_param=str(sweep_param),
                    sweep_start=float(start_here),
                    sweep_stop=float(stop_here),
                    sweep_step=float(sweep_step),
                    section_index=int(section_index),
                    section_value=float(section_value),
                    direction=int(direction),
                    method=str(method),
                    tol=float(tol),
                    transient_steps=int(transient_steps_sweep),
                    output_index=int(output_index),
                    warm_start=bool(warm_start),
                    max_hits=int(max_hits),
                    early_stop=bool(early_stop),
                    chunk_time=float(chunk_time),
                    solve_options=solve_options_sweep,
                )

            df_chunk = pd.DataFrame(df_chunk) if not isinstance(df_chunk, pd.DataFrame) else df_chunk
            st.session_state["sweep_acc_df"] = df_chunk
            st.session_state["sweep_last_pv"] = float(stop_here)

            st.session_state["last_sweep_df"] = df_chunk
            st.session_state["last_sweep_meta"] = st.session_state["sweep_meta"]

            df_plot = df_chunk

        elif run_cont:
            acc_df = st.session_state.get("sweep_acc_df", None)
            last_pv = st.session_state.get("sweep_last_pv", None)

            if acc_df is None or last_pv is None:
                st.warning("No previous sweep found. Run 'Generate' first.")
                st.stop()

            else:
                prev_meta = st.session_state.get("sweep_meta", {})
                now_meta = sweep_meta

                mismatches = []
                for k, v_prev in prev_meta.items():
                    if k not in now_meta:
                        continue
                    v_now = now_meta[k]
                    if isinstance(v_prev, float) and isinstance(v_now, float):
                        if not math.isclose(v_prev, v_now, rel_tol=0.0, abs_tol=1e-12):
                            mismatches.append(k)
                    else:
                        if v_prev != v_now:
                            mismatches.append(k)

                if mismatches:
                    st.error(
                        "Cannot continue: settings changed since last run. "
                        f"Changed: {', '.join(mismatches)}. Run 'Generate' to restart."
                    )
                else:
                    last_pv = float(st.session_state["sweep_last_pv"])
                    start_here = last_pv + float(sweep_step)

                    stop_here = float(continue_stop) if continue_stop is not None else float(sweep_stop)

                    st.session_state["sweep_stop_internal"] = stop_here

                    if start_here > stop_here + 1e-12:
                        st.warning("Nothing to continue: start is already beyond stop.")
                    else:
                        st.session_state["sweep_boundaries"].append(last_pv)

                        with st.spinner("Continuing sweep..."):
                            df_chunk = run_sweep_chunk(
                                system_key=system_key,
                                t0=float(t0), tf=float(tf_sweep), dt=float(dt_sweep),
                                y0=np.array(y0, dtype=float),
                                sigma=float(sigma), rho=float(rho), beta=float(beta),
                                ross_a=float(ross_a), ross_b=float(ross_b), ross_c=float(ross_c),
                                var_names=list(var_names), eq_lines=list(eq_lines),
                                params_text=params_text,
                                sweep_param=str(sweep_param),
                                sweep_start=float(start_here),
                                sweep_stop=float(stop_here),
                                sweep_step=float(sweep_step),
                                section_index=int(section_index),
                                section_value=float(section_value),
                                direction=int(direction),
                                method=str(method),
                                tol=float(tol),
                                transient_steps=int(transient_steps_sweep),
                                output_index=int(output_index),
                                warm_start=bool(warm_start),
                                max_hits=int(max_hits),
                                early_stop=bool(early_stop),
                                chunk_time=float(chunk_time),
                                solve_options=solve_options_sweep,
                            )

                        df_chunk = pd.DataFrame(df_chunk) if not isinstance(df_chunk, pd.DataFrame) else df_chunk
                        df_acc = st.session_state["sweep_acc_df"]
                        df_acc = pd.concat([df_acc, df_chunk], ignore_index=True)

                        st.session_state["sweep_acc_df"] = df_acc
                        st.session_state["sweep_last_pv"] = float(stop_here)

                        st.session_state["last_sweep_df"] = df_acc
                        st.session_state["last_sweep_meta"] = prev_meta

                        df_plot = df_acc

        if df_plot is None:
            df_plot = st.session_state.get("sweep_acc_df", None)

        if df_plot is None or len(df_plot) == 0:
            st.info("No sweep data yet. Click 'Generate' to start.")
            return

        if not isinstance(df_plot, pd.DataFrame):
            df_plot = pd.DataFrame(df_plot)

        ycol = f"y{int(output_index)}"
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        fig.set_dpi(140)
        ax.scatter(
            df_plot[sweep_param].to_numpy(),
            df_plot[ycol].to_numpy(),
            s=2,
            c="black",
            marker=".",
            linewidths=0,
            alpha=0.8,
        )

        for x_sep in st.session_state.get("sweep_boundaries", []):
            ax.axvline(float(x_sep), color="magenta", linewidth=0.3)

        ax.set_xlabel(sweep_param)
        ax.set_ylabel(f"{out_var} on section ({section_var}={section_value})")
        x_min = float(sweep_start)
        x_max = float(np.nanmax(df_plot[sweep_param].to_numpy()))
        ax.set_xlim(x_min, x_max)
        ax.grid(True, linewidth=0.3)
        st.pyplot(fig, clear_figure=True)

        last_pv = st.session_state.get("sweep_last_pv", None)
        if last_pv is not None:
            st.caption(f"Accumulated sweep up to {sweep_param} = {float(last_pv):g} | Rows: {len(df_plot)}")
        else:
            try:
                st.caption(f"Accumulated sweep | Rows: {len(df_plot)}")
            except Exception:
                pass
