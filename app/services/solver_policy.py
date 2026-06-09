from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SolverPolicy:
    kind: str
    scipy_method: Optional[str]
    auto_switch_rk4: bool
    sweep_kind: str


def resolve_solver(raw: str) -> SolverPolicy:
    kind = str(raw).lower()
    if kind == "symplectic_verlet":
        kind = "symplectic_fr"

    scipy_method: Optional[str] = None
    if kind in ("rk45", "ivp"):
        scipy_method = "RK45"
    elif kind == "dop853":
        scipy_method = "DOP853"

    auto_switch_rk4 = kind in ("rk45", "dop853", "ivp")

    if kind in ("rk4", "symplectic_fr"):
        sweep_kind = kind
    else:
        sweep_kind = "ivp"

    return SolverPolicy(
        kind=kind,
        scipy_method=scipy_method,
        auto_switch_rk4=auto_switch_rk4,
        sweep_kind=sweep_kind,
    )


def apply_to_solve_options(policy: SolverPolicy, solve_options: Dict[str, Any]) -> None:
    if policy.scipy_method is not None:
        solve_options["method"] = policy.scipy_method
