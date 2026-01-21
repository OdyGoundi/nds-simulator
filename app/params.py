from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class IntegrationConfig:
    t0: float
    tf: float
    dt: float
    solver_kind: str = "ivp"


@dataclass(frozen=True)
class InitialConditions:
    y0: Tuple[float, ...]


@dataclass(frozen=True)
class LorenzParams:
    sigma: float
    rho: float
    beta: float


@dataclass(frozen=True)
class RosslerParams:
    a: float
    b: float
    c: float


@dataclass(frozen=True)
class HenonHeilesParams:
    lam: float = 1.0


@dataclass(frozen=True)
class CustomSystemDefinition:
    var_names: Tuple[str, ...]
    eq_lines: Tuple[str, ...]
    params_text: str
    auto_jacobian: bool = False
    use_jacobian: bool = False


@dataclass(frozen=True)
class SystemConfig:
    key: str
    lorenz: LorenzParams
    rossler: RosslerParams
    henon_heiles: HenonHeilesParams
    custom: CustomSystemDefinition


@dataclass(frozen=True)
class SolverTolerances:
    rtol: float
    atol: float

    def to_dict(self) -> Dict[str, float]:
        return {"rtol": float(self.rtol), "atol": float(self.atol)}


@dataclass(frozen=True)
class SweepRunConfig:
    output_index: int
    warm_start: bool
    max_hits: int
    early_stop: bool
    chunk_time: float


@dataclass(frozen=True)
class LyapunovConfig:
    transient_steps: int
    qr_interval: float = 0.1
    keep_last_steps: Optional[int] = None
