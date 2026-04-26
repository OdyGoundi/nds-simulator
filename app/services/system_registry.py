from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from core.henon_heiles_system_rhs import henon_heiles_rhs
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


BUILTIN_SYSTEMS: Dict[str, SystemAdapter] = {
    "lorenz": SystemAdapter(
        key="lorenz",
        display_name="Lorenz (3D)",
        dimension=3,
        supports_symplectic=False,
        param_names=("sigma", "rho", "beta"),
        rhs_fn=lorenz_rhs,
        extract_params=_lorenz_params,
    ),
    "rossler": SystemAdapter(
        key="rossler",
        display_name="Rossler (3D)",
        dimension=3,
        supports_symplectic=False,
        param_names=("a", "b", "c"),
        rhs_fn=rossler_rhs,
        extract_params=_rossler_params,
    ),
    "henon_heiles": SystemAdapter(
        key="henon_heiles",
        display_name="Henon-Heiles (4D Hamiltonian)",
        dimension=4,
        supports_symplectic=True,
        param_names=("lambda",),
        rhs_fn=henon_heiles_rhs,
        extract_params=_henon_heiles_params,
    ),
}


def get_builtin(key: str) -> SystemAdapter:
    if key not in BUILTIN_SYSTEMS:
        raise ValueError(f"Unknown built-in system_key: {key!r}")
    return BUILTIN_SYSTEMS[key]
