from typing import Dict, List, Tuple

import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from app.helpers import build_custom_rhs, parse_params
from app.params import (
    InitialConditions,
    IntegrationConfig,
    LyapunovConfig,
    SolverTolerances,
    SweepRunConfig,
    SystemConfig,
)
from app.sweep import run_sweep_chunk
from core.jacobians_fixed_systems import lorenz_jac, rossler_jac
from core.lorenz_system_rhs import lorenz_rhs
from core.rossler_system_rhs import rossler_rhs
from core.lyapunov import compute_lyapunov_spectrum
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


def _sweep_settings_fingerprint(
    system: SystemConfig,
    sweep: SweepConfig,
    poincare: PoincareConfig,
    run_cfg: SweepRunConfig,
    integration: IntegrationConfig,
    transient_frac: float,
    solve_tols: SolverTolerances,
) -> Dict[str, object]:
    return {
        "system_key": system.key,
        "sweep_param": str(sweep.param_name),
        "sweep_step": float(sweep.step),
        "section_index": int(poincare.section_index),
        "section_value": float(poincare.section_value),
        "direction": int(poincare.direction),
        "method": str(poincare.method),
        "tol": float(poincare.tol),
        "output_index": int(run_cfg.output_index),
        "tf_sweep": float(integration.tf),
        "dt_sweep": float(integration.dt),
        "transient_frac": float(transient_frac),
        "max_hits": int(run_cfg.max_hits),
        "early_stop": bool(run_cfg.early_stop),
        "chunk_time": float(run_cfg.chunk_time),
        "warm_start": bool(run_cfg.warm_start),
        "rtol": float(solve_tols.rtol),
        "atol": float(solve_tols.atol),
    }


def _frange_inclusive(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0.0:
        raise ValueError("Sweep step must be > 0.")
    n = int(np.floor((stop - start) / step + 1e-12)) + 1
    vals = start + step * np.arange(n, dtype=float)
    return vals[vals <= stop + 1e-12]


def _run_lyapunov_sweep(
    *,
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    sweep: SweepConfig,
    lyapunov: LyapunovConfig,
    solve_tols: SolverTolerances,
    warm_start: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    param_vals = _frange_inclusive(float(sweep.start), float(sweep.stop), float(sweep.step))
    y0_base = np.array(initial.y0, dtype=float).copy()
    y0_curr = y0_base.copy()

    if system.key == "lorenz":
        base_params = {
            "sigma": float(system.lorenz.sigma),
            "rho": float(system.lorenz.rho),
            "beta": float(system.lorenz.beta),
        }
    elif system.key == "rossler":
        base_params = {
            "a": float(system.rossler.a),
            "b": float(system.rossler.b),
            "c": float(system.rossler.c),
        }
    elif system.key == "custom":
        base_params = parse_params(system.custom.params_text)
    else:
        raise ValueError(f"Unknown system_key: {system.key}")

    solve_options = solve_tols.to_dict()
    var_names = list(system.custom.var_names)
    eq_lines = list(system.custom.eq_lines)

    t_transient = float(lyapunov.transient_steps) * float(integration.dt)
    total_time = float(integration.tf) - float(integration.t0)
    t_measure = total_time - t_transient
    if t_measure <= 0.0:
        raise ValueError("Not enough time for Lyapunov measurement. Increase tf or reduce transient cut.")

    if lyapunov.qr_interval <= 0.0:
        raise ValueError("Lyapunov QR interval must be > 0.")
    target_chunk = float(lyapunov.qr_interval)
    qr_every_steps = max(1, int(round(target_chunk / float(integration.dt))))

    errors: List[str] = []
    lambdas_list: List[np.ndarray] = []

    for pv in param_vals:
        params = dict(base_params)
        params[str(sweep.param_name)] = float(pv)

        if system.key == "lorenz":
            rhs = lambda tt, xx: lorenz_rhs(
                tt, xx, sigma=params["sigma"], rho=params["rho"], beta=params["beta"]
            )
            jac = lambda tt, xx: lorenz_jac(
                tt, xx, sigma=params["sigma"], rho=params["rho"], beta=params["beta"]
            )
        elif system.key == "rossler":
            rhs = lambda tt, xx: rossler_rhs(
                tt, xx, a=params["a"], b=params["b"], c=params["c"]
            )
            jac = lambda tt, xx: rossler_jac(
                tt, xx, a=params["a"], b=params["b"], c=params["c"]
            )
        else:
            rhs_custom = build_custom_rhs(var_names, eq_lines, params)
            rhs = lambda tt, xx: rhs_custom(tt, xx)
            jac = None

        try:
            res = compute_lyapunov_spectrum(
                rhs=rhs,
                x0=y0_curr,
                t0=float(integration.t0),
                dt=float(integration.dt),
                t_transient=float(t_transient),
                t_measure=float(t_measure),
                qr_every_steps=qr_every_steps,
                solve_options=solve_options,
                jac=jac,
            )
            lambdas_list.append(res.lambdas)
            if warm_start:
                y0_curr = np.array(res.x_final, dtype=float).copy()
            else:
                y0_curr = y0_base.copy()
        except Exception as exc:
            errors.append(f"{sweep.param_name}={float(pv):g}: {exc}")
            lambdas_list.append(np.full(y0_base.shape[0], np.nan))
            y0_curr = y0_base.copy()

    lambdas_arr = np.vstack(lambdas_list) if lambdas_list else np.zeros((0, y0_base.shape[0]))
    return param_vals, lambdas_arr, errors


def render_bifurcation_tab(
    tab,
    *,
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
):
    with tab:
        system_key = system.key
        t0 = float(integration.t0)
        tf = float(integration.tf)
        dt = float(integration.dt)
        var_names = list(system.custom.var_names)
        params_text = system.custom.params_text

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

            solve_tols_sweep = SolverTolerances(rtol=float(rtol_sweep), atol=float(atol_sweep))

            st.markdown("**Lyapunov diagram settings**")
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

        sweep_cfg = SweepConfig(
            param_name=str(sweep_param),
            start=float(sweep_start),
            stop=float(sweep_stop),
            step=float(sweep_step),
        )
        poincare_cfg = PoincareConfig(
            section_index=int(section_index),
            section_value=float(section_value),
            direction=int(direction),
            method=str(method),
            tol=float(tol),
            transient_steps=int(transient_steps_sweep),
        )
        integration_sweep = IntegrationConfig(
            t0=float(t0),
            tf=float(tf_sweep),
            dt=float(dt_sweep),
        )
        run_cfg = SweepRunConfig(
            output_index=int(output_index),
            warm_start=bool(warm_start),
            max_hits=int(max_hits),
            early_stop=bool(early_stop),
            chunk_time=float(chunk_time),
        )
        lyapunov_cfg = LyapunovConfig(
            transient_steps=int(transient_steps_sweep),
            qr_interval=float(qr_interval_lya),
        )

        run_new = st.button("Generate Bifurcation Diagram", type="primary", key="run_new_sweep")
        run_lya = st.button("Generate Lyapunov Diagram", type="secondary", key="run_lya_sweep")

        st.divider()

        run_cont = st.button("Continue Bifurcation", type="secondary", key="run_cont_sweep")
        run_lya_cont = st.button("Continue Lyapunov", type="secondary", key="run_cont_lya")
        reset_acc = st.button("Reset accumulated", type="secondary", key="reset_acc_sweep")

        if reset_acc:
            st.session_state["sweep_acc_df"] = None
            st.session_state["sweep_last_pv"] = None
            st.session_state["sweep_boundaries"] = []
            st.session_state["sweep_meta"] = {}
            st.session_state["lya_acc_data"] = None
            st.session_state["lya_last_pv"] = None
            st.session_state["lya_meta"] = {}
            st.session_state["lya_boundaries"] = []
            st.success("Accumulated sweep cleared.")

        df_plot = None

        have_prev_bif = (
            st.session_state.get("sweep_acc_df", None) is not None and
            st.session_state.get("sweep_last_pv", None) is not None
        )
        have_prev_lya = (
            st.session_state.get("lya_acc_data", None) is not None and
            st.session_state.get("lya_last_pv", None) is not None
        )

        continue_stop = None
        continue_stop_lya = None
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
        lya_meta["lyapunov_qr_interval"] = float(lyapunov_cfg.qr_interval)

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
                        sweep_run = SweepConfig(
                            param_name=str(sweep_param),
                            start=float(start_here),
                            stop=float(stop_here),
                            step=float(sweep_step),
                        )

                        with st.spinner("Continuing sweep..."):
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
                st.warning("No previous Lyapunov sweep found. Run 'Generate Lyapunov' first.")
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
                        f"Changed: {', '.join(mismatches)}. Run 'Generate Lyapunov' to restart."
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

        if df_plot is None or len(df_plot) == 0:
            st.info("No sweep data yet. Click 'Generate' to start.")
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

        st.divider()
        st.markdown("**Lyapunov exponents (sweep)**")
        fig_lya, ax_lya = plt.subplots(figsize=(6.0, 3.2))
        fig_lya.set_dpi(140)

        n_exps = plot_lambdas.shape[1]
        for k in range(n_exps):
            ax_lya.plot(
                param_vals,
                plot_lambdas[:, k],
                color=COLORS[k % len(COLORS)],
                linewidth=1.0,
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
