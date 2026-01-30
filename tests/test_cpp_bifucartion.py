import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import numba_backend
from core.lorenz_system_rhs import lorenz_rhs
from core.poincare_sweep import PoincareConfig, SweepConfig, sweep_poincare

sigma = 10.0
beta = 8.0 / 3.0
rho0 = 25.0
rho1 = 125.0
step = 1.0

y0 = (1.0, 1.0, 1.0)
t0 = 0.0
tf = 80.0
dt = 0.01

section_index = 0
section_value = 0.0
output_index = 2
direction = 1
method = "crossing"
transient_steps = 0

sweep = SweepConfig(param_name="rho", start=rho0, stop=rho1, step=step)
poincare = PoincareConfig(
    section_index=section_index,
    section_value=section_value,
    direction=direction,
    method=method,
    tol=1e-3,
    transient_steps=transient_steps,
)

# Python sweep (reference)
rows_py = sweep_poincare(
    rhs=lorenz_rhs,
    y0=y0,
    t_span=(t0, tf),
    base_params={"sigma": sigma, "rho": rho0, "beta": beta},
    sweep=sweep,
    poincare=poincare,
    solver_kind="rk4",
    t_step=dt,
    output_indices=[output_index],
    warm_start=False,
)

def col(rows, idx):
    return np.array([r[f"y{idx}"] for r in rows], float) if isinstance(rows, list) else rows[f"y{idx}"].to_numpy()

vals_py = col(rows_py, output_index)
print("py hits", len(vals_py), "min/max", (vals_py.min(), vals_py.max()) if len(vals_py) else None)

if not numba_backend.numba_available():
    print("[SKIP] Numba backend not available.")
    raise SystemExit(0)

rhs_nb, _jac_nb, param_names = numba_backend.build_builtin_system("lorenz")
param_names_list = list(param_names)
if "rho" not in param_names_list:
    print("[SKIP] sweep param not found in built-in param list.")
    raise SystemExit(0)

params_base = np.array([sigma, rho0, beta], dtype=float)
param_index = param_names_list.index("rho")
method_id = 0 if method.lower() == "crossing" else 1

sweep_nb = numba_backend.build_poincare_sweep_rk4(rhs_nb)
params_out, t_hit, y_hit, count = sweep_nb(
    np.array(y0, dtype=float),
    float(t0),
    float(tf),
    float(dt),
    params_base,
    int(param_index),
    float(rho0),
    float(rho1),
    float(step),
    int(section_index),
    float(section_value),
    int(direction),
    int(method_id),
    float(poincare.tol),
    int(transient_steps),
    int(output_index),
    False,
    100,
)

y_hit = np.asarray(y_hit[:count], dtype=float)
print("numba hits", len(y_hit), "min/max", (y_hit.min(), y_hit.max()) if len(y_hit) else None)
