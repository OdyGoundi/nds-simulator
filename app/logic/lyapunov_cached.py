import numpy as np
import streamlit as st

from app.params import (
    InitialConditions,
    IntegrationConfig,
    LyapunovConfig,
    SolverTolerances,
    SystemConfig,
)
from app.services import compute_single_lyapunov


@st.cache_data(show_spinner=False)
def compute_lyapunov_cached(
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    lyapunov: LyapunovConfig,
    solve_tols: SolverTolerances,
) -> np.ndarray:
    return compute_single_lyapunov(system, integration, initial, lyapunov, solve_tols)
