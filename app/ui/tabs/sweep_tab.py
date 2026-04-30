from pathlib import Path

import streamlit as st

from app.params import InitialConditions, IntegrationConfig, SolverTolerances, SystemConfig
from app.services.sweep_run_service import (
    execute_cont_bif_sweep,
    execute_cont_lya_sweep,
    execute_new_bif_sweep,
    execute_new_lya_sweep,
)
from app.ui.sweep_controls import render_sweep_controls
from app.ui.sweep_plot_panel import render_sweep_plots


def render_sweep_tab(
    tab,
    *,
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    solve_tols: SolverTolerances,
    app_name: str,
    repo_root: Path,
) -> None:
    with tab:
        ctrl = render_sweep_controls(
            system=system,
            integration=integration,
            initial=initial,
            solve_tols=solve_tols,
            app_name=app_name,
            repo_root=repo_root,
        )

        df_plot = None

        if ctrl.run_new:
            df_plot = execute_new_bif_sweep(
                system=ctrl.system,
                integration_sweep=ctrl.integration_sweep,
                initial=ctrl.initial,
                sweep_param=ctrl.sweep_param,
                sweep_start=ctrl.sweep_start,
                sweep_stop=ctrl.sweep_stop,
                sweep_step=ctrl.sweep_step,
                poincare_cfg=ctrl.poincare_cfg,
                run_cfg=ctrl.run_cfg,
                solve_tols_sweep=ctrl.solve_tols_sweep,
                sweep_meta=ctrl.sweep_meta,
                ycol=ctrl.ycol,
                use_extrema=ctrl.use_extrema,
                extrema_kind=ctrl.extrema_kind,
                parallel_bif_enabled=ctrl.parallel_bif_enabled,
                workers_bif=ctrl.workers_bif,
            )

        elif ctrl.run_cont:
            acc_df = st.session_state.get("sweep_acc_df", None)
            last_pv_state = st.session_state.get("sweep_last_pv", None)
            if acc_df is None or last_pv_state is None:
                st.warning("No previous sweep found. Run 'Generate Bifurcation Diagram' first.")
                st.stop()
            df_plot = execute_cont_bif_sweep(
                system=ctrl.system,
                integration_sweep=ctrl.integration_sweep,
                initial=ctrl.initial,
                sweep_param=ctrl.sweep_param,
                sweep_step=ctrl.sweep_step,
                sweep_stop=ctrl.sweep_stop,
                continue_stop=ctrl.continue_stop,
                poincare_cfg=ctrl.poincare_cfg,
                run_cfg=ctrl.run_cfg,
                solve_tols_sweep=ctrl.solve_tols_sweep,
                sweep_meta=ctrl.sweep_meta,
                ycol=ctrl.ycol,
                use_extrema=ctrl.use_extrema,
                extrema_kind=ctrl.extrema_kind,
                parallel_bif_enabled=ctrl.parallel_bif_enabled,
                workers_bif=ctrl.workers_bif,
            )

        if ctrl.run_lya_cont:
            acc_data = st.session_state.get("lya_acc_data", None)
            last_pv_lya = st.session_state.get("lya_last_pv", None)
            if acc_data is None or last_pv_lya is None:
                st.warning("No previous Lyapunov sweep found. Run 'Generate Lyapunov Diagram' first.")
            else:
                execute_cont_lya_sweep(
                    system=ctrl.system,
                    integration_sweep=ctrl.integration_sweep,
                    initial=ctrl.initial,
                    lyapunov_cfg=ctrl.lyapunov_cfg,
                    solve_tols_sweep=ctrl.solve_tols_sweep,
                    sweep_param=ctrl.sweep_param,
                    sweep_step=ctrl.sweep_step,
                    sweep_stop=ctrl.sweep_stop,
                    continue_stop_lya=ctrl.continue_stop_lya,
                    warm_start=ctrl.warm_start,
                    parallel_enabled=ctrl.parallel_enabled,
                    workers=ctrl.workers,
                    var_names=ctrl.var_names,
                    lya_meta=ctrl.lya_meta,
                    acc_data=acc_data,
                    last_pv=float(last_pv_lya),
                )

        if ctrl.run_lya:
            execute_new_lya_sweep(
                system=ctrl.system,
                integration_sweep=ctrl.integration_sweep,
                initial=ctrl.initial,
                sweep_cfg=ctrl.sweep_cfg,
                lyapunov_cfg=ctrl.lyapunov_cfg,
                solve_tols_sweep=ctrl.solve_tols_sweep,
                sweep_stop=ctrl.sweep_stop,
                warm_start=ctrl.warm_start,
                parallel_enabled=ctrl.parallel_enabled,
                workers=ctrl.workers,
                lya_meta=ctrl.lya_meta,
            )

        if df_plot is None:
            df_plot = st.session_state.get("sweep_acc_df", None)

        render_sweep_plots(ctrl, df_plot)
