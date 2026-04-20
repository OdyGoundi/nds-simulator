from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any

from app.helpers import parse_params
from app.params import InitialConditions, IntegrationConfig, SolverTolerances, SystemConfig


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_git_commit(repo_root: Optional[Path]) -> Optional[str]:
    if repo_root is None:
        return None
    head_path = repo_root / ".git" / "HEAD"
    if not head_path.exists():
        return None
    head = head_path.read_text().strip()
    if head.startswith("ref: "):
        ref_path = repo_root / ".git" / head.split(" ", 1)[1].strip()
        if ref_path.exists():
            ref = ref_path.read_text().strip()
            return ref or None
        return None
    return head or None


def build_app_block(app_name: str, repo_root: Optional[Path]) -> Dict[str, Optional[str]]:
    return {
        "name": str(app_name),
        "commit": get_git_commit(repo_root),
    }


def _safe_parse_params(params_text: str) -> Dict[str, float]:
    try:
        return parse_params(params_text)
    except Exception:
        return {}


def build_system_block(system: SystemConfig) -> Dict[str, Any]:
    if system.key == "lorenz":
        params = {
            "sigma": float(system.lorenz.sigma),
            "rho": float(system.lorenz.rho),
            "beta": float(system.lorenz.beta),
        }
    elif system.key == "rossler":
        params = {
            "a": float(system.rossler.a),
            "b": float(system.rossler.b),
            "c": float(system.rossler.c),
        }
    elif system.key == "henon_heiles":
        params = {
            "lambda": float(system.henon_heiles.lam),
        }
    else:
        params = _safe_parse_params(system.custom.params_text)

    eq_lines = list(system.custom.eq_lines) if system.key in ("custom", "henon_heiles") else []

    block = {
        "system_key": str(system.key),
        "var_names": list(system.custom.var_names),
        "eq_lines": eq_lines,
        "params": params,
    }
    if system.key == "custom":
        block["params_text"] = str(system.custom.params_text or "")
        block["auto_jacobian"] = bool(system.custom.auto_jacobian)
        block["use_jacobian"] = bool(system.custom.use_jacobian)
    return block


def build_integration_block(
    integration: IntegrationConfig,
    initial: InitialConditions,
    solve_tols: SolverTolerances,
) -> Dict[str, Any]:
    max_store_steps = getattr(integration, "max_store_steps", None)
    return {
        "t0": float(integration.t0),
        "tf": float(integration.tf),
        "dt": float(integration.dt),
        "y0": [float(v) for v in initial.y0],
        "solver_kind": str(getattr(integration, "solver_kind", "ivp")),
        "max_store_steps": (None if max_store_steps is None else int(max_store_steps)),
        "solve_options": solve_tols.to_dict(),
    }


def _jacobian_mode(system: SystemConfig) -> str:
    if system.key in ("lorenz", "rossler", "henon_heiles"):
        return "analytic"
    if system.custom.auto_jacobian and system.custom.use_jacobian:
        return "analytic"
    return "finite_difference"


def _qr_every_steps(dt: float, qr_interval: float) -> int:
    return max(1, int(round(float(qr_interval) / float(dt))))


def _measurement_window(
    t0: float,
    tf: float,
    dt: float,
    transient_steps: int,
) -> Dict[str, float]:
    total_time = float(tf) - float(t0)
    t_transient = float(transient_steps) * float(dt)
    t_measure = total_time - t_transient
    return {"t_transient": float(t_transient), "t_measure": float(t_measure)}


def build_static_config(
    *,
    app_name: str,
    repo_root: Optional[Path],
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    solve_tols: SolverTolerances,
    plot_mode: str,
    x_idx: int,
    y_idx: int,
    z_idx: Optional[int],
    phase_linewidth: float,
    transient_steps: int,
    lyapunov_transient_steps: int,
    lyapunov_transient_frac: float,
    qr_interval: float,
) -> Dict[str, Any]:
    window = _measurement_window(
        t0=integration.t0,
        tf=integration.tf,
        dt=integration.dt,
        transient_steps=lyapunov_transient_steps,
    )

    return {
        "schema_version": "1.0",
        "app": build_app_block(app_name, repo_root),
        "timestamp_utc": utc_timestamp(),
        "system": build_system_block(system),
        "integration": build_integration_block(integration, initial, solve_tols),
        "postprocess": {
            "transient_steps": int(transient_steps),
        },
        "plots": {
            "plot_mode": str(plot_mode),
            "phase_linewidth": float(phase_linewidth),
            "phase_axes": {
                "x_idx": int(x_idx),
                "y_idx": int(y_idx),
                "z_idx": None if z_idx is None else int(z_idx),
            },
        },
        "lyapunov": {
            "enabled": True,
            "settings": {
                "dt": float(integration.dt),
                "t_transient": window["t_transient"],
                "t_measure": window["t_measure"],
                "transient_steps": int(lyapunov_transient_steps),
                "transient_fraction": float(lyapunov_transient_frac),
                "qr_every_steps": _qr_every_steps(integration.dt, qr_interval),
                "qr_interval": float(qr_interval),
                "jacobian": _jacobian_mode(system),
                "fd_eps": 1e-8,
            },
        },
    }


def build_sweep_config(
    *,
    app_name: str,
    repo_root: Optional[Path],
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    solve_tols: SolverTolerances,
    sweep_param: str,
    sweep_start: float,
    sweep_stop: float,
    sweep_step: float,
    dt_sweep: float,
    tf_sweep: float,
    section_var: str,
    section_index: int,
    section_value: float,
    section_expr: str,
    direction: int,
    method: str,
    tol: float,
    output_var: str,
    output_index: int,
    observable: str,
    extrema_kind: str,
    transient_frac: float,
    transient_steps_est: int,
    warm_start: bool,
    early_stop: bool,
    max_hits: int,
    chunk_time: float,
    parallel_bif: bool,
    workers_bif: int,
    rtol_sweep: float,
    atol_sweep: float,
    qr_interval_lya: float,
    lyapunov_transient_frac: float,
    lyapunov_transient_steps: int,
    parallel_lya: bool,
    workers_lya: int,
    clip_lyapunov: bool,
    clip_min: float,
    sweep_fingerprint: Optional[Dict[str, Any]] = None,
    lya_fingerprint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    window = _measurement_window(
        t0=integration.t0,
        tf=tf_sweep,
        dt=dt_sweep,
        transient_steps=lyapunov_transient_steps,
    )

    sweep_settings = {
        "sweep_param": str(sweep_param),
        "sweep_start": float(sweep_start),
        "sweep_stop": float(sweep_stop),
        "sweep_step": float(sweep_step),
        "integration": {
            "t0": float(integration.t0),
            "tf": float(tf_sweep),
            "dt": float(dt_sweep),
            "solver_kind": str(getattr(integration, "solver_kind", "ivp")),
        },
        "solver": {"rtol": float(rtol_sweep), "atol": float(atol_sweep)},
        "mode": {
            "warm_start": bool(warm_start),
            "parallel": bool(parallel_bif),
            "parallel_workers": int(workers_bif) if parallel_bif else None,
        },
        "poincare": {
            "section_var": str(section_var),
            "section_index": int(section_index),
            "section_value": float(section_value),
            "section_expr": str(section_expr or ""),
            "direction": int(direction),
            "method": str(method),
            "tol": float(tol),
        },
        "output": {"var": str(output_var), "index": int(output_index)},
        "observable": str(observable),
        "extrema_kind": str(extrema_kind),
        "transient": {
            "fraction": float(transient_frac),
            "steps_est": int(transient_steps_est),
        },
        "run": {
            "early_stop": bool(early_stop),
            "max_hits": int(max_hits),
            "chunk_time": float(chunk_time),
        },
    }

    lyapunov_settings = {
        "integration": {
            "t0": float(integration.t0),
            "tf": float(tf_sweep),
            "dt": float(dt_sweep),
        },
        "t_transient": window["t_transient"],
        "t_measure": window["t_measure"],
        "transient_steps": int(lyapunov_transient_steps),
        "transient_fraction": float(lyapunov_transient_frac),
        "qr_every_steps": _qr_every_steps(dt_sweep, qr_interval_lya),
        "qr_interval": float(qr_interval_lya),
        "jacobian": _jacobian_mode(system),
        "fd_eps": 1e-8,
        "parallel": bool(parallel_lya),
        "parallel_workers": int(workers_lya) if parallel_lya else None,
        "clip": {
            "enabled": bool(clip_lyapunov),
            "min": float(clip_min) if clip_lyapunov else None,
        },
    }

    config = {
        "schema_version": "1.0",
        "app": build_app_block(app_name, repo_root),
        "timestamp_utc": utc_timestamp(),
        "system": build_system_block(system),
        "integration": build_integration_block(integration, initial, solve_tols),
        "sweep": {
            "enabled": True,
            "settings": sweep_settings,
        },
        "lyapunov": {
            "enabled": True,
            "settings": lyapunov_settings,
        },
    }

    if sweep_fingerprint is not None:
        config["sweep"]["fingerprint"] = dict(sweep_fingerprint)
    if lya_fingerprint is not None:
        config["lyapunov"]["fingerprint"] = dict(lya_fingerprint)

    return config
