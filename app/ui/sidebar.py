from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import streamlit as st

from app.services import get_builtin
from app.state import (
    SOLVER_LABEL_BY_KIND,
    SYSTEM_KEY_BY_LABEL,
    SYSTEM_LABEL_BY_KEY,
    PhaseKeys,
    SidebarKeys,
    SystemParamKeys,
)
from app.state.apply_config import apply_state_values, clamp_int, to_int


def _system_key_from_label(system_label: object) -> str:
    label = str(system_label or "").strip()
    if label in SYSTEM_KEY_BY_LABEL:
        return SYSTEM_KEY_BY_LABEL[label]
    if label.startswith("Lorenz"):
        return "lorenz"
    if label.startswith("Rossler"):
        return "rossler"
    if label.startswith("Henon-Heiles"):
        return "henon_heiles"
    return "custom"


def _builtin_system_defaults(system_key: str) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        PhaseKeys.X_IDX: 0,
        PhaseKeys.Y_IDX: 1,
        PhaseKeys.Z_IDX: 2,
    }
    if system_key == "lorenz":
        defaults.update(
            {
                SidebarKeys.Y0_TEXT: "1, 1, 1",
                SidebarKeys.SOLVER_KIND_LABEL: SOLVER_LABEL_BY_KIND["rk4"],
                SystemParamKeys.SIGMA: 10.0,
                SystemParamKeys.RHO: 28.0,
                SystemParamKeys.BETA: float(8.0 / 3.0),
            }
        )
    elif system_key == "rossler":
        defaults.update(
            {
                SidebarKeys.Y0_TEXT: "1, 1, 1",
                SidebarKeys.SOLVER_KIND_LABEL: SOLVER_LABEL_BY_KIND["rk4"],
                SystemParamKeys.ROSS_A: 0.2,
                SystemParamKeys.ROSS_B: 0.2,
                SystemParamKeys.ROSS_C: 5.7,
            }
        )
    elif system_key == "henon_heiles":
        defaults.update(
            {
                SidebarKeys.Y0_TEXT: "0.1, 0.0, 0.0, 0.1",
                SidebarKeys.SOLVER_KIND_LABEL: SOLVER_LABEL_BY_KIND["symplectic_fr"],
                SystemParamKeys.HH_LAMBDA: 1.0,
            }
        )
    else:
        defaults = {}
    return defaults


def _list_text_token_count(value: object) -> int:
    return len(str(value or "").replace(",", " ").split())


def _ensure_builtin_system_sidebar_state() -> None:
    system_key = _system_key_from_label(
        st.session_state.get(SidebarKeys.SYSTEM_LABEL, SYSTEM_LABEL_BY_KEY["lorenz"])
    )
    defaults = _builtin_system_defaults(system_key)
    if not defaults:
        return

    expected_dim = 4 if system_key == "henon_heiles" else 3
    y0_token_count = _list_text_token_count(st.session_state.get(SidebarKeys.Y0_TEXT, ""))
    if y0_token_count not in (0, expected_dim):
        apply_state_values(defaults, only_missing=False)
    else:
        apply_state_values(defaults, only_missing=True)

    st.session_state[PhaseKeys.X_IDX] = clamp_int(
        to_int(st.session_state.get(PhaseKeys.X_IDX, 0), 0),
        0,
        expected_dim - 1,
    )
    st.session_state[PhaseKeys.Y_IDX] = clamp_int(
        to_int(st.session_state.get(PhaseKeys.Y_IDX, 1), 1),
        0,
        expected_dim - 1,
    )
    z_default = 2 if expected_dim > 2 else 0
    st.session_state[PhaseKeys.Z_IDX] = clamp_int(
        to_int(st.session_state.get(PhaseKeys.Z_IDX, z_default), z_default),
        0,
        expected_dim - 1,
    )


def _on_system_selection_change() -> None:
    system_key = _system_key_from_label(st.session_state.get(SidebarKeys.SYSTEM_LABEL))
    defaults = _builtin_system_defaults(system_key)
    if defaults:
        apply_state_values(defaults, only_missing=False)


@dataclass(frozen=True)
class SidebarSystemResult:
    system_label: str
    system_key: str
    n_vars: int
    solver_kind: str
    solver_kind_effective: str
    y0_text: str


def render_system_sidebar() -> SidebarSystemResult:
    _ensure_builtin_system_sidebar_state()

    with st.sidebar:
        st.header("System")
        apply_state_values(
            {SidebarKeys.SYSTEM_LABEL: SYSTEM_LABEL_BY_KEY["lorenz"]},
            only_missing=True,
        )

        system_label = st.selectbox(
            "Choose system",
            ["Lorenz (3D)", "Rossler (3D)", "Henon-Heiles (4D Hamiltonian)", "Custom (nD)"],
            key=SidebarKeys.SYSTEM_LABEL,
            on_change=_on_system_selection_change,
        )

        system_key = _system_key_from_label(system_label)
        if system_key == "custom":
            apply_state_values({SidebarKeys.N_VARS: 3}, only_missing=True)
            n_vars = st.number_input(
                "Number of equations (n)",
                min_value=1,
                max_value=12,
                step=1,
                key=SidebarKeys.N_VARS,
            )
        else:
            n_vars = get_builtin(system_key).dimension

        st.markdown("**Solver kind**")
        solver_kind_labels = [
            "RK45 (adaptive)",
            "DOP853 (non-stiff, high order)",
            "RK4 (fixed step)",
            "Symplectic Forest-Ruth (4th order)",
        ]
        solver_kind_map = {
            "RK45 (adaptive)": "rk45",
            "DOP853 (non-stiff, high order)": "dop853",
            "RK4 (fixed step)": "rk4",
            "Symplectic Forest-Ruth (4th order)": "symplectic_fr",
        }
        solver_default = "RK4 (fixed step)"
        apply_state_values({SidebarKeys.SOLVER_KIND_LABEL: solver_default}, only_missing=True)
        solver_kind_label = st.selectbox(
            "Solver kind",
            solver_kind_labels,
            key=SidebarKeys.SOLVER_KIND_LABEL,
        )
        solver_kind = solver_kind_map[solver_kind_label]
        st.markdown(
            "- RK45 adaptive: default choice, uses rtol/atol.\n"
            "- DOP853: high-order solver for non-stiff problems, uses rtol/atol.\n"
            "- RK4 fixed: fixed dt, faster but needs smaller dt for accuracy.\n"
            "- Symplectic Forest-Ruth: separable Hamiltonians, state = [q..., p...], dq/dt uses p only, dp/dt uses q only."
        )
        solver_kind_effective = solver_kind
        if solver_kind.startswith("symplectic"):
            if system_key not in ("custom", "henon_heiles"):
                st.warning("Symplectic solvers require Hamiltonian systems. Using RK45 instead.")
                solver_kind_effective = "rk45"
            elif int(n_vars) % 2 != 0:
                st.warning("Symplectic solvers require an even number of variables [q..., p...]. Using RK45 instead.")
                solver_kind_effective = "rk45"
        elif system_key == "henon_heiles":
            st.caption("Henon-Heiles is Hamiltonian: symplectic solvers are recommended.")

        st.divider()
        st.header("Initial conditions")

        if system_key == "henon_heiles":
            y0_default = "0.1, 0.0, 0.0, 0.1"
        elif int(n_vars) == 3:
            y0_default = "1, 1, 1"
        else:
            y0_default = "\n".join(["0"] * int(n_vars))
        apply_state_values({SidebarKeys.Y0_TEXT: y0_default}, only_missing=True)
        y0_text = st.text_area(
            "y0 values (comma/space/newline separated)",
            height=90,
            key=SidebarKeys.Y0_TEXT,
        )

    return SidebarSystemResult(
        system_label=str(system_label),
        system_key=str(system_key),
        n_vars=int(n_vars),
        solver_kind=str(solver_kind),
        solver_kind_effective=str(solver_kind_effective),
        y0_text=str(y0_text),
    )
