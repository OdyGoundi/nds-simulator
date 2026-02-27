#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.logic.lyapunov_sweep import _run_lyapunov_sweep
from app.params import (
    CustomSystemDefinition,
    HenonHeilesParams,
    InitialConditions,
    IntegrationConfig,
    LorenzParams,
    LyapunovConfig,
    RosslerParams,
    SolverTolerances,
    SweepRunConfig,
    SystemConfig,
)
from app.sweep import MAX_SWEEP_ROWS_BUDGET, _effective_max_hits, run_sweep_chunk
from core.poincare_sweep import PoincareConfig, SweepConfig


def _frange_count(start: float, stop: float, step: float) -> int:
    if step <= 0.0:
        raise ValueError("step must be > 0.")
    return int(math.floor((float(stop) - float(start)) / float(step) + 1e-12)) + 1


def _build_lorenz_system(rho: float) -> SystemConfig:
    return SystemConfig(
        key="lorenz",
        lorenz=LorenzParams(sigma=10.0, rho=float(rho), beta=8.0 / 3.0),
        rossler=RosslerParams(a=0.2, b=0.2, c=5.7),
        henon_heiles=HenonHeilesParams(lam=1.0),
        custom=CustomSystemDefinition(var_names=tuple(), eq_lines=tuple(), params_text=""),
    )


def _bytes_to_mib(n_bytes: float) -> float:
    return float(n_bytes) / (1024.0 * 1024.0)


def run_bifurcation_probe(
    *,
    rho_probe: np.ndarray,
    dt: float,
    tf: float,
    sweep_start: float,
    sweep_stop: float,
    sweep_step: float,
    transient_fraction: float,
    max_hits_user: int,
    chunk_time: float,
    solver_kind: str,
    rtol: float,
    atol: float,
) -> Dict[str, object]:
    n_params_full = _frange_count(sweep_start, sweep_stop, sweep_step)
    sweep_full = SweepConfig(
        param_name="rho",
        start=float(sweep_start),
        stop=float(sweep_stop),
        step=float(sweep_step),
    )
    max_hits_effective = _effective_max_hits(
        int(max_hits_user), sweep_full, int(MAX_SWEEP_ROWS_BUDGET)
    )

    integration = IntegrationConfig(
        t0=0.0, tf=float(tf), dt=float(dt), solver_kind=str(solver_kind)
    )
    initial = InitialConditions(y0=(1.0, 1.0, 1.0))
    transient_steps = int(max(0.0, transient_fraction) * ((integration.tf - integration.t0) / integration.dt))
    poincare = PoincareConfig(
        section_index=0,
        section_value=0.0,
        direction=1,
        method="crossing",
        tol=1e-6,
        transient_steps=int(transient_steps),
    )
    run_cfg = SweepRunConfig(
        output_index=1,
        warm_start=False,
        max_hits=int(max_hits_effective),
        early_stop=True,
        chunk_time=float(chunk_time),
    )
    solve_tols = SolverTolerances(rtol=float(rtol), atol=float(atol))

    rows_per_param: List[int] = []
    elapsed_per_param: List[float] = []
    probe_rows: List[Dict[str, float]] = []

    for rho in rho_probe:
        system = _build_lorenz_system(float(rho))
        sweep_one = SweepConfig(param_name="rho", start=float(rho), stop=float(rho), step=float(sweep_step))

        t0 = time.perf_counter()
        rows_obj = run_sweep_chunk(
            system=system,
            integration=integration,
            initial=initial,
            sweep=sweep_one,
            poincare=poincare,
            observable="poincare",
            extrema_kind="max",
            run_cfg=run_cfg,
            solve_tols=solve_tols,
        )
        elapsed = time.perf_counter() - t0

        if isinstance(rows_obj, pd.DataFrame):
            n_rows = int(len(rows_obj))
        else:
            n_rows = int(len(rows_obj or []))

        rows_per_param.append(n_rows)
        elapsed_per_param.append(float(elapsed))
        probe_rows.append(
            {
                "rho": float(rho),
                "stored_rows": int(n_rows),
                "elapsed_s": float(elapsed),
            }
        )

    rows_arr = np.asarray(rows_per_param, dtype=float)
    elapsed_arr = np.asarray(elapsed_per_param, dtype=float)
    avg_rows = float(np.mean(rows_arr)) if rows_arr.size else 0.0
    avg_elapsed = float(np.mean(elapsed_arr)) if elapsed_arr.size else 0.0

    projected_rows = avg_rows * float(n_params_full)
    projected_rows_capped = min(float(MAX_SWEEP_ROWS_BUDGET), projected_rows)
    projected_runtime_h = (avg_elapsed * float(n_params_full)) / 3600.0

    raw_row_bytes = 3 * 8  # rho, t_hit, y(output)
    projected_raw_data_mib = _bytes_to_mib(projected_rows_capped * raw_row_bytes)

    return {
        "probe_rows": probe_rows,
        "n_params_full": int(n_params_full),
        "max_hits_effective": int(max_hits_effective),
        "avg_rows_per_param": avg_rows,
        "max_rows_per_param_in_probe": int(np.max(rows_arr)) if rows_arr.size else 0,
        "min_rows_per_param_in_probe": int(np.min(rows_arr)) if rows_arr.size else 0,
        "nonzero_hit_fraction": float(np.mean(rows_arr > 0.0)) if rows_arr.size else 0.0,
        "avg_elapsed_s_per_param": avg_elapsed,
        "projected_total_rows_uncapped": projected_rows,
        "projected_total_rows_capped": projected_rows_capped,
        "projected_runtime_h_serial": projected_runtime_h,
        "projected_raw_data_mib": projected_raw_data_mib,
    }


def run_lyapunov_probe(
    *,
    rho_probe: np.ndarray,
    dt: float,
    tf: float,
    sweep_step: float,
    transient_fraction: float,
    qr_interval: float,
    solver_kind: str,
    rtol: float,
    atol: float,
) -> Dict[str, object]:
    integration = IntegrationConfig(
        t0=0.0, tf=float(tf), dt=float(dt), solver_kind=str(solver_kind)
    )
    initial = InitialConditions(y0=(1.0, 1.0, 1.0))
    solve_tols = SolverTolerances(rtol=float(rtol), atol=float(atol))
    transient_steps = int(max(0.0, transient_fraction) * ((integration.tf - integration.t0) / integration.dt))
    lyap_cfg = LyapunovConfig(transient_steps=int(transient_steps), qr_interval=float(qr_interval))

    probe_rows: List[Dict[str, float]] = []
    elapsed_per_param: List[float] = []
    err_count = 0

    for rho in rho_probe:
        system = _build_lorenz_system(float(rho))
        sweep_one = SweepConfig(param_name="rho", start=float(rho), stop=float(rho), step=float(sweep_step))

        t0 = time.perf_counter()
        pv, lambdas, errors = _run_lyapunov_sweep(
            system=system,
            integration=integration,
            initial=initial,
            sweep=sweep_one,
            lyapunov=lyap_cfg,
            solve_tols=solve_tols,
            warm_start=False,
            parallel=False,
            max_workers=1,
        )
        elapsed = time.perf_counter() - t0
        err_count += int(len(errors))
        elapsed_per_param.append(float(elapsed))
        probe_rows.append(
            {
                "rho": float(rho),
                "stored_rows": int(pv.size),
                "stored_lambda_values": int(lambdas.size),
                "errors": int(len(errors)),
                "elapsed_s": float(elapsed),
            }
        )

    elapsed_arr = np.asarray(elapsed_per_param, dtype=float)
    return {
        "probe_rows": probe_rows,
        "avg_elapsed_s_per_param": float(np.mean(elapsed_arr)) if elapsed_arr.size else 0.0,
        "max_elapsed_s_per_param": float(np.max(elapsed_arr)) if elapsed_arr.size else 0.0,
        "min_elapsed_s_per_param": float(np.min(elapsed_arr)) if elapsed_arr.size else 0.0,
        "total_errors": int(err_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe Lorenz sweep storage/time at tf=5000 and project to full sweep "
            "(rho 0..250 step 0.05)."
        )
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tf", type=float, default=5000.0)
    parser.add_argument("--sweep-start", type=float, default=0.0)
    parser.add_argument("--sweep-stop", type=float, default=250.0)
    parser.add_argument("--sweep-step", type=float, default=0.05)
    parser.add_argument("--bif-transient-frac", type=float, default=0.70)
    parser.add_argument("--lya-transient-frac", type=float, default=0.55)
    parser.add_argument("--bif-max-hits-user", type=int, default=200)
    parser.add_argument("--bif-chunk-time", type=float, default=2.0)
    parser.add_argument("--bif-solver-kind", type=str, default="ivp")
    parser.add_argument("--lya-solver-kind", type=str, default="rk4")
    parser.add_argument("--qr-interval", type=float, default=0.1)
    parser.add_argument("--rtol", type=float, default=3e-4)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--bif-probe-count", type=int, default=12)
    parser.add_argument("--lya-probe-count", type=int, default=6)
    args = parser.parse_args()

    n_params_full = _frange_count(args.sweep_start, args.sweep_stop, args.sweep_step)
    n_steps_per_param = int(round((args.tf - 0.0) / args.dt))
    n_steps_total = int(n_steps_per_param * n_params_full)
    transient_steps_bif = int(args.bif_transient_frac * n_steps_per_param)
    transient_steps_lya = int(args.lya_transient_frac * n_steps_per_param)
    measure_steps_lya = max(0, n_steps_per_param - transient_steps_lya)

    rho_probe_bif = np.linspace(args.sweep_start, args.sweep_stop, int(args.bif_probe_count), dtype=float)
    rho_probe_lya = np.linspace(args.sweep_start, args.sweep_stop, int(args.lya_probe_count), dtype=float)

    print("=== Lorenz Sweep Projection (tf=5000 test) ===")
    print(f"Grid params: {n_params_full} (rho {args.sweep_start}..{args.sweep_stop} step {args.sweep_step})")
    print(f"dt={args.dt}, tf={args.tf}, steps/param={n_steps_per_param:,}, total nominal steps={n_steps_total:,}")
    print(
        f"Transient steps: bif={transient_steps_bif:,} ({args.bif_transient_frac:.2f}), "
        f"lya={transient_steps_lya:,} ({args.lya_transient_frac:.2f}), "
        f"lya_measure={measure_steps_lya:,}"
    )

    full_traj_samples = n_params_full * (n_steps_per_param + 1)
    full_traj_raw_bytes = full_traj_samples * 4 * 8  # t + 3 state vars
    print(
        "If full trajectories were stored (t + x,y,z): "
        f"{full_traj_samples:,} samples, raw≈{_bytes_to_mib(full_traj_raw_bytes):,.1f} MiB"
    )

    print("\nRunning bifurcation probe...")
    bif = run_bifurcation_probe(
        rho_probe=rho_probe_bif,
        dt=args.dt,
        tf=args.tf,
        sweep_start=args.sweep_start,
        sweep_stop=args.sweep_stop,
        sweep_step=args.sweep_step,
        transient_fraction=args.bif_transient_frac,
        max_hits_user=args.bif_max_hits_user,
        chunk_time=args.bif_chunk_time,
        solver_kind=args.bif_solver_kind,
        rtol=args.rtol,
        atol=args.atol,
    )

    print(f"Bif probe points: {len(bif['probe_rows'])}")
    print(f"Effective max_hits per param (full sweep): {bif['max_hits_effective']}")
    print(
        "Rows/param (probe): "
        f"avg={bif['avg_rows_per_param']:.2f}, "
        f"min={bif['min_rows_per_param_in_probe']}, max={bif['max_rows_per_param_in_probe']}, "
        f"nonzero={100.0 * bif['nonzero_hit_fraction']:.1f}%"
    )
    print(
        "Projected stored rows (full sweep): "
        f"uncapped={bif['projected_total_rows_uncapped']:.0f}, "
        f"capped={bif['projected_total_rows_capped']:.0f} "
        f"(budget={MAX_SWEEP_ROWS_BUDGET})"
    )
    print(
        "Projected bif raw numeric storage (rho,t_hit,y): "
        f"{bif['projected_raw_data_mib']:.2f} MiB"
    )
    print(
        "Projected bif runtime (serial, same machine/profile): "
        f"{bif['projected_runtime_h_serial']:.2f} h"
    )

    print("\nRunning Lyapunov probe...")
    lya = run_lyapunov_probe(
        rho_probe=rho_probe_lya,
        dt=args.dt,
        tf=args.tf,
        sweep_step=args.sweep_step,
        transient_fraction=args.lya_transient_frac,
        qr_interval=args.qr_interval,
        solver_kind=args.lya_solver_kind,
        rtol=args.rtol,
        atol=args.atol,
    )
    print(f"Lyapunov probe points: {len(lya['probe_rows'])}")
    print(
        "Lyapunov elapsed/param (probe): "
        f"avg={lya['avg_elapsed_s_per_param']:.3f}s, "
        f"min={lya['min_elapsed_s_per_param']:.3f}s, max={lya['max_elapsed_s_per_param']:.3f}s"
    )
    print(f"Lyapunov probe errors: {lya['total_errors']}")

    lyap_rows_full = n_params_full
    lyap_values_full = n_params_full * 3
    lyap_raw_bytes = (lyap_rows_full + lyap_values_full) * 8  # rho + 3 lambdas
    lyap_runtime_h = (float(lya["avg_elapsed_s_per_param"]) * n_params_full) / 3600.0

    print(
        "Projected Lyapunov storage: "
        f"rows={lyap_rows_full:,}, lambda_values={lyap_values_full:,}, "
        f"raw≈{_bytes_to_mib(lyap_raw_bytes):.4f} MiB"
    )
    print(
        "Projected Lyapunov runtime (serial, same machine/profile): "
        f"{lyap_runtime_h:.2f} h"
    )

    report = {
        "config": {
            "dt": float(args.dt),
            "tf": float(args.tf),
            "sweep_start": float(args.sweep_start),
            "sweep_stop": float(args.sweep_stop),
            "sweep_step": float(args.sweep_step),
            "bif_transient_frac": float(args.bif_transient_frac),
            "lya_transient_frac": float(args.lya_transient_frac),
            "bif_max_hits_user": int(args.bif_max_hits_user),
            "bif_chunk_time": float(args.bif_chunk_time),
            "bif_solver_kind": str(args.bif_solver_kind),
            "lya_solver_kind": str(args.lya_solver_kind),
            "qr_interval": float(args.qr_interval),
            "rtol": float(args.rtol),
            "atol": float(args.atol),
            "bif_probe_count": int(args.bif_probe_count),
            "lya_probe_count": int(args.lya_probe_count),
        },
        "derived": {
            "n_params_full": int(n_params_full),
            "n_steps_per_param": int(n_steps_per_param),
            "n_steps_total_nominal": int(n_steps_total),
            "transient_steps_bif": int(transient_steps_bif),
            "transient_steps_lya": int(transient_steps_lya),
            "measure_steps_lya": int(measure_steps_lya),
            "full_trajectory_samples_if_stored": int(full_traj_samples),
            "full_trajectory_raw_mib_if_stored": _bytes_to_mib(full_traj_raw_bytes),
        },
        "bifurcation": bif,
        "lyapunov": {
            **lya,
            "projected_rows_full": int(lyap_rows_full),
            "projected_lambda_values_full": int(lyap_values_full),
            "projected_raw_mib": _bytes_to_mib(lyap_raw_bytes),
            "projected_runtime_h_serial": float(lyap_runtime_h),
        },
    }

    out_dir = PROJECT_ROOT / "tests" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "lorenz_sweep_projection_tf5000.json"
    out_bif_csv = out_dir / "lorenz_sweep_projection_bif_probe_tf5000.csv"
    out_lya_csv = out_dir / "lorenz_sweep_projection_lya_probe_tf5000.csv"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    pd.DataFrame(bif["probe_rows"]).to_csv(out_bif_csv, index=False)
    pd.DataFrame(lya["probe_rows"]).to_csv(out_lya_csv, index=False)

    print("\nSaved:")
    print(f"- {out_json}")
    print(f"- {out_bif_csv}")
    print(f"- {out_lya_csv}")


if __name__ == "__main__":
    main()
