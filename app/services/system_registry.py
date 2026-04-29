from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from core.henon_heiles_system_rhs import (
    henon_heiles_dp_dt,
    henon_heiles_dq_dt,
    henon_heiles_rhs,
)
from core.jacobians_fixed_systems import (
    henon_heiles_jac,
    lorenz_jac,
    rossler_jac,
)
from core.lorenz_system_rhs import lorenz_rhs
from core.rossler_system_rhs import rossler_rhs

from app.params import SystemConfig


@dataclass(frozen=True)
class SystemAdapter:
    key: str
    display_name: str
    dimension: int
    supports_symplectic: bool
    param_names: Tuple[str, ...]
    rhs_fn: Callable
    extract_params: Callable[[SystemConfig], Dict[str, float]]
    rhs_builder: Callable[[SystemConfig], Callable]
    jac_builder: Optional[Callable[[SystemConfig], Callable]] = None
    dq_dp_builder: Optional[Callable[[SystemConfig], Tuple[Callable, Callable]]] = None
    rhs_from_dict: Optional[Callable[[Dict[str, float]], Callable]] = None
    jac_from_dict: Optional[Callable[[Dict[str, float]], Callable]] = None


def _lorenz_params(s: SystemConfig) -> Dict[str, float]:
    return {
        "sigma": float(s.lorenz.sigma),
        "rho": float(s.lorenz.rho),
        "beta": float(s.lorenz.beta),
    }


def _rossler_params(s: SystemConfig) -> Dict[str, float]:
    return {
        "a": float(s.rossler.a),
        "b": float(s.rossler.b),
        "c": float(s.rossler.c),
    }


def _henon_heiles_params(s: SystemConfig) -> Dict[str, float]:
    return {"lambda": float(s.henon_heiles.lam)}


def _lorenz_rhs_builder(s: SystemConfig) -> Callable:
    p = s.lorenz

    def rhs(t, y):
        return lorenz_rhs(t, y, sigma=p.sigma, rho=p.rho, beta=p.beta)

    return rhs


def _rossler_rhs_builder(s: SystemConfig) -> Callable:
    p = s.rossler

    def rhs(t, y):
        return rossler_rhs(t, y, a=p.a, b=p.b, c=p.c)

    return rhs


def _henon_heiles_rhs_builder(s: SystemConfig) -> Callable:
    p = s.henon_heiles

    def rhs(t, y):
        return henon_heiles_rhs(t, y, lam=p.lam)

    return rhs


def _lorenz_jac_builder(s: SystemConfig) -> Callable:
    p = s.lorenz

    def jac(t, y):
        return lorenz_jac(t, y, sigma=p.sigma, rho=p.rho, beta=p.beta)

    return jac


def _rossler_jac_builder(s: SystemConfig) -> Callable:
    p = s.rossler

    def jac(t, y):
        return rossler_jac(t, y, a=p.a, b=p.b, c=p.c)

    return jac


def _henon_heiles_jac_builder(s: SystemConfig) -> Callable:
    lam = s.henon_heiles.lam

    def jac(t, y):
        return henon_heiles_jac(t, y, lam=lam)

    return jac


def _lorenz_rhs_from_dict(params: Dict[str, float]) -> Callable:
    def rhs(t, y):
        return lorenz_rhs(t, y, sigma=params["sigma"], rho=params["rho"], beta=params["beta"])
    return rhs


def _lorenz_jac_from_dict(params: Dict[str, float]) -> Callable:
    def jac(t, y):
        return lorenz_jac(t, y, sigma=params["sigma"], rho=params["rho"], beta=params["beta"])
    return jac


def _rossler_rhs_from_dict(params: Dict[str, float]) -> Callable:
    def rhs(t, y):
        return rossler_rhs(t, y, a=params["a"], b=params["b"], c=params["c"])
    return rhs


def _rossler_jac_from_dict(params: Dict[str, float]) -> Callable:
    def jac(t, y):
        return rossler_jac(t, y, a=params["a"], b=params["b"], c=params["c"])
    return jac


def _henon_heiles_rhs_from_dict(params: Dict[str, float]) -> Callable:
    def rhs(t, y):
        return henon_heiles_rhs(t, y, lam=params["lambda"])
    return rhs


def _henon_heiles_jac_from_dict(params: Dict[str, float]) -> Callable:
    def jac(t, y):
        return henon_heiles_jac(t, y, lam=params["lambda"])
    return jac


def _henon_heiles_dq_dp_builder(s: SystemConfig) -> Tuple[Callable, Callable]:
    lam = s.henon_heiles.lam

    def dq_dt(t, p):
        return henon_heiles_dq_dt(t, p, lam=lam)

    def dp_dt(t, q):
        return henon_heiles_dp_dt(t, q, lam=lam)

    return dq_dt, dp_dt


BUILTIN_SYSTEMS: Dict[str, SystemAdapter] = {
    "lorenz": SystemAdapter(
        key="lorenz",
        display_name="Lorenz (3D)",
        dimension=3,
        supports_symplectic=False,
        param_names=("sigma", "rho", "beta"),
        rhs_fn=lorenz_rhs,
        extract_params=_lorenz_params,
        rhs_builder=_lorenz_rhs_builder,
        jac_builder=_lorenz_jac_builder,
        rhs_from_dict=_lorenz_rhs_from_dict,
        jac_from_dict=_lorenz_jac_from_dict,
    ),
    "rossler": SystemAdapter(
        key="rossler",
        display_name="Rossler (3D)",
        dimension=3,
        supports_symplectic=False,
        param_names=("a", "b", "c"),
        rhs_fn=rossler_rhs,
        extract_params=_rossler_params,
        rhs_builder=_rossler_rhs_builder,
        jac_builder=_rossler_jac_builder,
        rhs_from_dict=_rossler_rhs_from_dict,
        jac_from_dict=_rossler_jac_from_dict,
    ),
    "henon_heiles": SystemAdapter(
        key="henon_heiles",
        display_name="Henon-Heiles (4D Hamiltonian)",
        dimension=4,
        supports_symplectic=True,
        param_names=("lambda",),
        rhs_fn=henon_heiles_rhs,
        extract_params=_henon_heiles_params,
        rhs_builder=_henon_heiles_rhs_builder,
        jac_builder=_henon_heiles_jac_builder,
        dq_dp_builder=_henon_heiles_dq_dp_builder,
        rhs_from_dict=_henon_heiles_rhs_from_dict,
        jac_from_dict=_henon_heiles_jac_from_dict,
    ),
}


def get_builtin(key: str) -> SystemAdapter:
    if key not in BUILTIN_SYSTEMS:
        raise ValueError(f"Unknown built-in system_key: {key!r}")
    return BUILTIN_SYSTEMS[key]
