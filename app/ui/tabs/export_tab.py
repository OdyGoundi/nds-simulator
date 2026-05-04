from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.cache import solve_cached
from app.export_utils import build_static_config
from app.helpers import apply_transient_cut, build_csv_bytes
from app.params import IntegrationConfig
from app.services.export_service import build_run_bundle
from app.state import (
    DIRECT_CSV_MAX_ROWS,
    EXPORT_CHUNK_ROWS_DEFAULT,
    TRAJ_EXPORT_READY_SIG_KEY,
    TRAJ_EXPORT_SOURCE_FULL,
    TRAJ_EXPORT_SOURCE_STORED,
)
from app.ui.tabs.phase_tab import PhaseTabResult


def render_export_tab(
    tab,
    *,
    phase_result: PhaseTabResult,
    app_name: str,
    repo_root: Path,
) -> None:
    system = phase_result.system
    integration = phase_result.integration
    initial = phase_result.initial
    solve_tols = phase_result.solve_tols
    t_plot = phase_result.t_plot
    y_plot = phase_result.y_plot
    var_names = phase_result.var_names
    transient_steps = phase_result.transient_steps

    with tab:
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
            if not isinstance(df_sweep, pd.DataFrame):
                df_sweep = pd.DataFrame(df_sweep)

            csv_bytes = df_sweep.to_csv(index=False).encode("utf-8")

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
                app_name=app_name,
                repo_root=repo_root,
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
