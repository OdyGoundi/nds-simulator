import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from app.helpers import parse_params
from app.export_utils import build_sweep_config
from app.logic.bifurcation_sweep import _run_bifurcation_parallel
from app.logic.lyapunov_sweep import _run_lyapunov_sweep
from app.logic.sweep_utils import (
    _default_worker_count,
    _is_streamlit_cloud,
    _sweep_settings_fingerprint,
)
from app.params import (
    InitialConditions,
    IntegrationConfig,
    LyapunovConfig,
    SolverTolerances,
    SweepRunConfig,
    SystemConfig,
)
from app.sweep import run_sweep_chunk
from core.poincare_sweep import PoincareConfig, SweepConfig


COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


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
    if "lya_acc_data" not in st.session_state:
        st.session_state["lya_acc_data"] = None
    if "lya_last_pv" not in st.session_state:
        st.session_state["lya_last_pv"] = None
    if "lya_meta" not in st.session_state:
        st.session_state["lya_meta"] = {}
    if "lya_boundaries" not in st.session_state:
        st.session_state["lya_boundaries"] = []


def render_bifurcation_tab(
    tab,
    *,
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    solve_tols: SolverTolerances,
    app_name: str,
    repo_root: Path,
):
    with tab:
        system_key = system.key
        t0 = float(integration.t0)
        tf = float(integration.tf)
        dt = float(integration.dt)
        var_names = list(system.custom.var_names)
        params_text = system.custom.params_text

        _init_sweep_state()

        if system_key == "lorenz":
            sweep_choices = ["sigma", "rho", "beta"]
        elif system_key == "rossler":
            sweep_choices = ["a", "b", "c"]
        elif system_key == "henon_heiles":
            sweep_choices = ["lambda"]
        else:
            try:
                sweep_choices = list(parse_params(params_text).keys())
            except Exception:
                sweep_choices = []

        if not sweep_choices:
            st.warning("No sweep parameters available (check parameters).")
            st.stop()

        top_c1, top_c2, top_c3 = st.columns([2, 2, 1], gap="large")
        with top_c1:
            st.markdown("**Parameter sweep setup**")
            p1c1, p1c2, p1c3, p1c4 = st.columns([1, 1, 1, 1], gap="small")
            with p1c1:
                sweep_param = st.selectbox("Sweep param", sweep_choices, index=0, key="sw_param_tab3")
            with p1c2:
                sweep_start = st.number_input("start", value=0.0, step=0.1, format="%.6f", key="sw_start_tab3")
            with p1c3:
                sweep_stop = st.number_input(
                    "stop",
                    value=float(st.session_state["sweep_stop_internal"]),
                    step=0.1,
                    format="%.6f",
                    key="sw_stop_tab3",
                )
            with p1c4:
                sweep_step = st.number_input("step", value=0.1, step=0.01, format="%.6f", key="sw_step_tab3")

        with top_c2:
            st.markdown("**Sweep performance settings**")
            p2c1, p2c2, p2c3 = st.columns([1, 1, 1], gap="small")
            with p2c1:
                dt_sweep = st.number_input(
                    "dt",
                    min_value=1e-6,
                    value=max(float(dt), 0.1),
                    step=0.01,
                    format="%.6f",
                    key="dt_sweep_tab3",
                    help="Time step used for sweep."
                )
            with p2c2:
                tf_sweep = st.number_input(
                    "final time",
                    min_value=float(t0) + 1e-6,
                    value=min(float(tf), 80.0),
                    step=5.0,
                    format="%.3f",
                    key="tf_sweep_tab3",
                    help="Final integration time for sweep."
                )
            with p2c3:
                sweep_mode = st.selectbox(
                    "Sweep mode",
                    ["Bifurcation (reset ICs)", "Continuation (warm start)"],
                    index=0,
                    key="sweep_mode_tab3",
                    help="Reset ICs = bibliography-style. Warm start = faster continuation."
                )

        with top_c3:
            st.markdown("**Sweep solver tolerances**")
            p3c1, p3c2 = st.columns([1, 1], gap="small")
            with p3c1:
                rtol_sweep = st.number_input(
                    "relative tolerance",
                    min_value=0.0,
                    value=3e-4,
                    step=1e-4,
                    format="%.1e",
                    key="rtol_sweep_tab3",
                )
            with p3c2:
                atol_sweep = st.number_input(
                    "absolute tolerance",
                    min_value=0.0,
                    value=1e-6,
                    step=1e-6,
                    format="%.1e",
                    key="atol_sweep_tab3",
                )

        warm_start = sweep_mode.startswith("Continuation")
        solve_tols_sweep = SolverTolerances(rtol=float(rtol_sweep), atol=float(atol_sweep))

        continue_stop = None
        continue_stop_lya = None

        st.divider()

        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            st.markdown("**Bifurcation sweep settings**")
            on_cloud_bif = _is_streamlit_cloud()
            parallel_bif_disabled = bool(warm_start or on_cloud_bif)
            parallel_bif = st.checkbox(
                "Parallel sweep (local only)",
                value=False,
                disabled=parallel_bif_disabled,
                key="bif_parallel_tab3",
                help="On Streamlit Cloud this may not speed up.",
            )
            if on_cloud_bif:
                st.warning("Parallel sweep is disabled on Streamlit Cloud.")
            elif warm_start:
                st.caption("Continuation mode: sequential (smooth).")
            elif parallel_bif:
                st.caption("Parallel independent mode: faster (no warm start).")
            else:
                st.caption("Independent mode: sequential.")

            cpu_count_bif = os.cpu_count() or 1
            max_workers_ui_bif = max(1, min(int(cpu_count_bif), 32))
            workers_default_bif = min(_default_worker_count(), max_workers_ui_bif)
            workers_bif = st.slider(
                "Workers",
                min_value=1,
                max_value=max_workers_ui_bif,
                value=workers_default_bif,
                step=1,
                key="bif_workers_tab3",
                disabled=parallel_bif_disabled or not parallel_bif,
            )

            r1c1, r1c2, r1c3, r1c4 = st.columns([1, 1, 1, 1], gap="small")
            with r1c1:
                section_var = st.selectbox("Section var", var_names, index=0, key="sec_var_tab3")
                section_index = var_names.index(section_var)
            with r1c2:
                section_value = st.number_input(
                    "Section value",
                    value=0.0,
                    step=0.1,
                    format="%.6f",
                    key="sec_val_tab3",
                )
            with r1c3:
                direction_label = st.selectbox(
                    "Direction",
                    ["+1 (up)", "-1 (down)", "0 (both)"],
                    index=0,
                    key="sec_dir_tab3",
                )
                direction = +1 if direction_label.startswith("+1") else (-1 if direction_label.startswith("-1") else 0)
            with r1c4:
                out_var = st.selectbox(
                    "Output var (plotted)",
                    var_names,
                    index=min(2, len(var_names) - 1),
                    key="out_var_tab3",
                )
                output_index = var_names.index(out_var)

            var_hint = ", ".join(var_names)
            param_hint = ", ".join(sweep_choices)
            section_expr = st.text_input(
                "Section equation (optional, overrides plane)",
                value="",
                key="sec_expr_tab3",
                help=(
                    f"Vars: {var_hint}. Params: {param_hint}."
                ),
            )
            if str(section_expr).strip():
                st.caption("Using section equation; Section var/value is ignored.")

            st.divider()

            r2c1, r2c2, r2c3 = st.columns([1, 1, 1], gap="small")
            with r2c1:
                method = st.selectbox("Method", ["crossing", "slab"], index=0, key="sec_method_tab3")
            with r2c2:
                tol = st.number_input(
                    "Tolerance (slab only)",
                    value=1e-3,
                    step=1e-3,
                    format="%.1e",
                    key="sec_tol_tab3",
                )
            with r2c3:
                st.empty()

            r3c1, r3c2, r3c3 = st.columns([1, 1, 1], gap="small")
            with r3c1:
                early_stop = st.checkbox(
                    "Early stop (events)",
                    value=True,
                    key="early_stop_tab3",
                    help="Stop each run after collecting enough Poincaré hits."
                )
            with r3c2:
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
            with r3c3:
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

            st.markdown("**Bifurcation discard (not plotted)**")
            r4c1, r4c2 = st.columns([1, 1], gap="small")
            with r4c1:
                transient_frac = st.slider(
                    "Transient fraction",
                    min_value=0.0,
                    max_value=0.95,
                    value=0.75,
                    step=0.05,
                    key="sw_transient_frac_tab3",
                    help="Fraction of sweep integration steps to discard before crossings."
                )
            with r4c2:
                n_steps_est = int(max(1.0, (float(tf_sweep) - float(t0)) / float(dt_sweep)))
                transient_steps_sweep = int(transient_frac * n_steps_est)
                st.metric("Transient steps (estimated)", transient_steps_sweep)

            bbtn1, bbtn2, bbtn3 = st.columns([1, 1, 1], gap="small")
            with bbtn1:
                run_new = st.button("Generate Bifurcation Diagram", type="primary", key="run_new_sweep")
            with bbtn2:
                reset_bif = st.button("Reset bifurcation data", type="secondary", key="reset_acc_bif")
            with bbtn3:
                run_cont = st.button("Continue Bifurcation", type="secondary", key="run_cont_sweep")
            if reset_bif:
                st.session_state["sweep_acc_df"] = None
                st.session_state["sweep_last_pv"] = None
                st.session_state["sweep_boundaries"] = []
                st.session_state["sweep_meta"] = {}
                st.success("Bifurcation sweep cleared.")

            have_prev_bif = (
                st.session_state.get("sweep_acc_df", None) is not None and
                st.session_state.get("sweep_last_pv", None) is not None
            )
            if have_prev_bif:
                last_pv_ui = float(st.session_state["sweep_last_pv"])
                continue_stop = st.number_input(
                    f"Continue bifurcation to (stop) [{sweep_param}]",
                    min_value=last_pv_ui + float(sweep_step),
                    value=max(float(sweep_stop), last_pv_ui + float(sweep_step)),
                    step=float(sweep_step),
                    format="%.6f",
                    key="continue_stop_tab3",
                    help="Sets the new stop for Continue Bifurcation. Start is last_pv + step."
                )

        with right_col:
            st.markdown("**Lyapunov sweep settings**")
            on_cloud = _is_streamlit_cloud()
            parallel_disabled = bool(warm_start or on_cloud)
            parallel_lya = st.checkbox(
                "Parallel sweep (local only)",
                value=False,
                disabled=parallel_disabled,
                key="lya_parallel_tab3",
                help="On Streamlit Cloud this may not speed up.",
            )
            if on_cloud:
                st.warning("Parallel sweep is disabled on Streamlit Cloud.")
            elif warm_start:
                st.caption("Continuation mode: sequential (smooth).")
            elif parallel_lya:
                st.caption("Parallel independent mode: faster (no warm start).")
            else:
                st.caption("Independent mode: sequential.")

            cpu_count = os.cpu_count() or 1
            max_workers_ui = max(1, min(int(cpu_count), 32))
            workers_default = min(_default_worker_count(), max_workers_ui)
            workers = st.slider(
                "Workers",
                min_value=1,
                max_value=max_workers_ui,
                value=workers_default,
                step=1,
                key="lya_workers_tab3",
                disabled=parallel_disabled or not parallel_lya,
            )

            qr_interval_lya = st.number_input(
                "QR interval (time)",
                min_value=1e-6,
                value=0.1,
                step=0.01,
                format="%.4f",
                key="qr_interval_lya_tab3",
                help="Time between orthonormalizations during Lyapunov sweep.",
            )
            clip_lyapunov = st.checkbox(
                "Clip lower exponents",
                value=False,
                key="clip_lyapunov_tab3",
                help="Clamp very negative exponents for cleaner plots.",
            )
            clip_min = st.number_input(
                "Clip minimum",
                value=-50.0,
                step=1.0,
                format="%.3f",
                key="clip_min_lyapunov_tab3",
                disabled=not clip_lyapunov,
            )
            
            st.markdown("**Lyapunov transient cut**")
            ltc1, ltc2 = st.columns([1, 1], gap="small")
            with ltc1:
                transient_frac_lya = st.slider(
                    "Transient fraction",
                    min_value=0.0,
                    max_value=0.95,
                    value=0.30,
                    step=0.05,
                    key="lya_transient_frac_tab3",
                    help="Fraction of sweep integration steps discarded before Lyapunov computation.",
                )
            with ltc2:
                n_steps_est_lya = int(max(1.0, (float(tf_sweep) - float(t0)) / float(dt_sweep)))
                transient_steps_lya = int(transient_frac_lya * n_steps_est_lya)
                st.metric("Transient steps (estimated)", transient_steps_lya)

            lbtn1, lbtn2, lbtn3 = st.columns([1, 1, 1], gap="small")
            with lbtn1:
                run_lya = st.button("Generate Lyapunov Diagram", type="primary", key="run_lya_sweep")
            with lbtn2:
                reset_lya = st.button("Reset Lyapunov data", type="secondary", key="reset_acc_lya")
            with lbtn3:
                run_lya_cont = st.button("Continue Lyapunov", type="secondary", key="run_cont_lya")
            save_sweep_cfg = st.button("Save configuration", key="save_cfg_lya_tab3")
            if reset_lya:
                st.session_state["lya_acc_data"] = None
                st.session_state["lya_last_pv"] = None
                st.session_state["lya_meta"] = {}
                st.session_state["lya_boundaries"] = []
                st.success("Lyapunov sweep cleared.")

            have_prev_lya = (
                st.session_state.get("lya_acc_data", None) is not None and
                st.session_state.get("lya_last_pv", None) is not None
            )
            if have_prev_lya:
                last_pv_ui = float(st.session_state["lya_last_pv"])
                continue_stop_lya = st.number_input(
                    f"Continue Lyapunov to (stop) [{sweep_param}]",
                    min_value=last_pv_ui + float(sweep_step),
                    value=max(float(sweep_stop), last_pv_ui + float(sweep_step)),
                    step=float(sweep_step),
                    format="%.6f",
                    key="continue_stop_lya_tab3",
                    help="Sets the new stop for Continue Lyapunov. Start is last_pv + step."
                )

        sweep_cfg = SweepConfig(
            param_name=str(sweep_param),
            start=float(sweep_start),
            stop=float(sweep_stop),
            step=float(sweep_step),
        )
        poincare_cfg = PoincareConfig(
            section_index=int(section_index),
            section_value=float(section_value),
            section_expr=str(section_expr).strip(),
            section_vars=tuple(var_names),
            direction=int(direction),
            method=str(method),
            tol=float(tol),
            transient_steps=int(transient_steps_sweep),
        )
        integration_sweep = IntegrationConfig(
            t0=float(t0),
            tf=float(tf_sweep),
            dt=float(dt_sweep),
            solver_kind=str(getattr(integration, "solver_kind", "ivp")),
        )
        run_cfg = SweepRunConfig(
            output_index=int(output_index),
            warm_start=bool(warm_start),
            max_hits=int(max_hits),
            early_stop=bool(early_stop),
            chunk_time=float(chunk_time),
        )
        lyapunov_cfg = LyapunovConfig(
            transient_steps=int(transient_steps_lya),
            qr_interval=float(qr_interval_lya),
        )

        parallel_bif_enabled = bool(parallel_bif and not parallel_bif_disabled)
        parallel_enabled = bool(parallel_lya and not parallel_disabled)

        sweep_meta = _sweep_settings_fingerprint(
            system=system,
            sweep=sweep_cfg,
            poincare=poincare_cfg,
            run_cfg=run_cfg,
            integration=integration_sweep,
            transient_frac=transient_frac,
            solve_tols=solve_tols_sweep,
        )
        lya_meta = dict(sweep_meta)
        lya_meta.pop("transient_frac", None)
        lya_meta["lyapunov_transient_frac"] = float(transient_frac_lya)
        lya_meta["lyapunov_qr_interval"] = float(lyapunov_cfg.qr_interval)
        lya_meta["parallel"] = parallel_enabled
        lya_meta["parallel_workers"] = int(workers) if parallel_enabled else None

        if save_sweep_cfg:
            sweep_config = build_sweep_config(
                app_name=app_name,
                repo_root=repo_root,
                system=system,
                integration=integration,
                initial=initial,
                solve_tols=solve_tols,
                sweep_param=str(sweep_param),
                sweep_start=float(sweep_start),
                sweep_stop=float(sweep_stop),
                sweep_step=float(sweep_step),
                dt_sweep=float(dt_sweep),
                tf_sweep=float(tf_sweep),
                section_var=str(section_var),
                section_index=int(section_index),
                section_value=float(section_value),
                section_expr=str(section_expr or ""),
                direction=int(direction),
                method=str(method),
                tol=float(tol),
                output_var=str(out_var),
                output_index=int(output_index),
                transient_frac=float(transient_frac),
                transient_steps_est=int(transient_steps_sweep),
                warm_start=bool(warm_start),
                early_stop=bool(early_stop),
                max_hits=int(max_hits),
                chunk_time=float(chunk_time),
                parallel_bif=bool(parallel_bif),
                workers_bif=int(workers_bif),
                rtol_sweep=float(rtol_sweep),
                atol_sweep=float(atol_sweep),
                qr_interval_lya=float(qr_interval_lya),
                lyapunov_transient_frac=float(transient_frac_lya),
                lyapunov_transient_steps=int(transient_steps_lya),
                parallel_lya=bool(parallel_enabled),
                workers_lya=int(workers),
                clip_lyapunov=bool(clip_lyapunov),
                clip_min=float(clip_min),
                sweep_fingerprint=sweep_meta,
                lya_fingerprint=lya_meta,
            )
            st.session_state["sweep_config"] = sweep_config
            with right_col:
                st.success("Sweep configuration saved. Download from the Export tab.")

        df_plot = None

        if run_new:
            st.session_state["sweep_acc_df"] = None
            st.session_state["sweep_last_pv"] = None
            st.session_state["sweep_boundaries"] = []
            st.session_state["sweep_meta"] = sweep_meta

            start_here = float(sweep_start)
            stop_here = float(sweep_stop)
            sweep_run = SweepConfig(
                param_name=str(sweep_param),
                start=float(start_here),
                stop=float(stop_here),
                step=float(sweep_step),
            )

            with st.spinner("Running sweep..."):
                if parallel_bif_enabled:
                    df_chunk = _run_bifurcation_parallel(
                        system=system,
                        integration=integration_sweep,
                        initial=initial,
                        sweep=sweep_run,
                        poincare=poincare_cfg,
                        run_cfg=run_cfg,
                        solve_tols=solve_tols_sweep,
                        max_workers=int(workers_bif),
                    )
                else:
                    df_chunk = run_sweep_chunk(
                        system=system,
                        integration=integration_sweep,
                        initial=initial,
                        sweep=sweep_run,
                        poincare=poincare_cfg,
                        run_cfg=run_cfg,
                        solve_tols=solve_tols_sweep,
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
                st.warning("No previous sweep found. Run 'Generate Bifurcation Diagram' first.")
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
                        f"Changed: {', '.join(mismatches)}. Run 'Generate Bifurcation Diagram' to restart."
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
                        sweep_run = SweepConfig(
                            param_name=str(sweep_param),
                            start=float(start_here),
                            stop=float(stop_here),
                            step=float(sweep_step),
                        )

                        with st.spinner("Continuing sweep..."):
                            if parallel_bif_enabled:
                                df_chunk = _run_bifurcation_parallel(
                                    system=system,
                                    integration=integration_sweep,
                                    initial=initial,
                                    sweep=sweep_run,
                                    poincare=poincare_cfg,
                                    run_cfg=run_cfg,
                                    solve_tols=solve_tols_sweep,
                                    max_workers=int(workers_bif),
                                )
                            else:
                                df_chunk = run_sweep_chunk(
                                    system=system,
                                    integration=integration_sweep,
                                    initial=initial,
                                    sweep=sweep_run,
                                    poincare=poincare_cfg,
                                    run_cfg=run_cfg,
                                    solve_tols=solve_tols_sweep,
                                )

                        df_chunk = pd.DataFrame(df_chunk) if not isinstance(df_chunk, pd.DataFrame) else df_chunk
                        df_acc = st.session_state["sweep_acc_df"]
                        df_acc = pd.concat([df_acc, df_chunk], ignore_index=True)

                        st.session_state["sweep_acc_df"] = df_acc
                        st.session_state["sweep_last_pv"] = float(stop_here)

                        st.session_state["last_sweep_df"] = df_acc
                        st.session_state["last_sweep_meta"] = prev_meta

                        df_plot = df_acc

        if run_lya_cont:
            acc_data = st.session_state.get("lya_acc_data", None)
            last_pv = st.session_state.get("lya_last_pv", None)

            if acc_data is None or last_pv is None:
                st.warning("No previous Lyapunov sweep found. Run 'Generate Lyapunov Diagram' first.")
            else:
                prev_meta = st.session_state.get("lya_meta", {})
                now_meta = lya_meta

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
                        "Cannot continue Lyapunov: settings changed since last run. "
                        f"Changed: {', '.join(mismatches)}. Run 'Generate Lyapunov Diagram' to restart."
                    )
                else:
                    start_here = float(last_pv) + float(sweep_step)
                    stop_here = float(continue_stop_lya) if continue_stop_lya is not None else float(sweep_stop)

                    if start_here > stop_here + 1e-12:
                        st.warning("Nothing to continue: start is already beyond stop.")
                    else:
                        st.session_state["lya_boundaries"].append(float(last_pv))
                        sweep_run = SweepConfig(
                            param_name=str(sweep_param),
                            start=float(start_here),
                            stop=float(stop_here),
                            step=float(sweep_step),
                        )
                        with st.spinner("Continuing Lyapunov sweep..."):
                            param_vals, lambdas_arr, errors = _run_lyapunov_sweep(
                                system=system,
                                integration=integration_sweep,
                                initial=initial,
                                sweep=sweep_run,
                                lyapunov=lyapunov_cfg,
                                solve_tols=solve_tols_sweep,
                                warm_start=bool(warm_start),
                                parallel=parallel_enabled,
                                max_workers=int(workers),
                            )

                        prev_vals = acc_data.get("param_vals", np.array([], dtype=float))
                        prev_lambdas = acc_data.get("lambdas", np.zeros((0, len(var_names))))
                        prev_errors = acc_data.get("errors", [])

                        new_vals = np.concatenate([prev_vals, param_vals])
                        new_lambdas = np.vstack([prev_lambdas, lambdas_arr])
                        new_errors = list(prev_errors) + list(errors)

                        st.session_state["lya_acc_data"] = {
                            "param_vals": new_vals,
                            "lambdas": new_lambdas,
                            "errors": new_errors,
                            "meta": dict(prev_meta),
                        }
                        st.session_state["lya_last_pv"] = float(stop_here)

        if run_lya:
            st.session_state["lya_acc_data"] = None
            st.session_state["lya_last_pv"] = None
            st.session_state["lya_boundaries"] = []
            st.session_state["lya_meta"] = lya_meta

            with st.spinner("Computing Lyapunov sweep..."):
                param_vals, lambdas_arr, errors = _run_lyapunov_sweep(
                    system=system,
                    integration=integration_sweep,
                    initial=initial,
                    sweep=sweep_cfg,
                    lyapunov=lyapunov_cfg,
                    solve_tols=solve_tols_sweep,
                    warm_start=bool(warm_start),
                    parallel=parallel_enabled,
                    max_workers=int(workers),
                )

            st.session_state["lya_acc_data"] = {
                "param_vals": param_vals,
                "lambdas": lambdas_arr,
                "errors": errors,
                "meta": dict(lya_meta),
            }
            st.session_state["lya_last_pv"] = float(sweep_stop)

            if errors:
                st.warning(f"Lyapunov sweep had {len(errors)} failures. Showing NaNs for those points.")

        if df_plot is None:
            df_plot = st.session_state.get("sweep_acc_df", None)

        with left_col:
            st.divider()
            if df_plot is None or len(df_plot) == 0:
                st.info("No sweep data yet. Click 'Generate Bifurcation Diagram' to start.")
            else:
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
                section_label = str(section_expr).strip()
                if section_label:
                    ax.set_ylabel(f"{out_var} on section ({section_label})")
                else:
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

        with right_col:
            st.divider()
            lya_data = st.session_state.get("lya_acc_data", None)
            if lya_data is None:
                st.info("No Lyapunov sweep data yet. Click 'Generate Lyapunov Diagram'.")
                return

            param_vals = lya_data.get("param_vals", np.array([], dtype=float))
            lambdas_arr = lya_data.get("lambdas", np.zeros((0, len(var_names))))
            errors = lya_data.get("errors", [])

            if param_vals.size == 0 or lambdas_arr.size == 0:
                st.info("No Lyapunov sweep data yet. Click 'Generate Lyapunov Diagram'.")
                return

            plot_lambdas = np.array(lambdas_arr, dtype=float)
            if clip_lyapunov:
                plot_lambdas = np.maximum(plot_lambdas, float(clip_min))

            st.markdown("**Lyapunov exponents**")
            fig_lya, ax_lya = plt.subplots(figsize=(6.0, 3.2))
            fig_lya.set_dpi(140)

            n_exps = plot_lambdas.shape[1]
            for k in range(n_exps):
                ax_lya.plot(
                    param_vals,
                    plot_lambdas[:, k],
                    color=COLORS[k % len(COLORS)],
                    linestyle="-",
                    linewidth=1.1,
                    label=f"lambda{k}",
                )

            for x_sep in st.session_state.get("lya_boundaries", []):
                ax_lya.axvline(float(x_sep), color="magenta", linewidth=0.3)

            ax_lya.set_xlabel(sweep_param)
            ax_lya.set_ylabel("Lyapunov exponents")
            ax_lya.grid(True, linewidth=0.3)
            ax_lya.legend(loc="best", fontsize=8)
            st.pyplot(fig_lya, clear_figure=True)

            if clip_lyapunov:
                st.caption(f"Clipped exponents below {float(clip_min):g} for plotting.")
            if errors:
                st.caption(f"Lyapunov sweep failures: {len(errors)}")
            last_pv = st.session_state.get("lya_last_pv", None)
            if last_pv is not None:
                st.caption(f"Accumulated Lyapunov sweep up to {sweep_param} = {float(last_pv):g}")
