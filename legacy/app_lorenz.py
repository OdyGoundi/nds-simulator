import sys
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Ensure project root is on sys.path (so "core" imports work no matter where you run from)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.lorenz_system_rhs import lorenz_rhs
from core.solver import integrate_system
def plot_phase_xy(y, title: str):
    """
    y: array shape (n_vars, n_steps)
    plots x vs y (indices 0 and 1) with your project's magenta/black style.
    """
    x = y[0, :]
    yy = y[1, :]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, yy, linewidth=0.7)  # default color; if you want strict magenta/black only, see note below
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, linewidth=0.3)
    ax.set_aspect("equal", adjustable="box")
    return fig


st.set_page_config(page_title="Lorenz Explorer", layout="centered")
st.title("Lorenz Explorer (live sliders)")

# --- Sliders ---
sigma = st.slider("sigma (σ)", min_value=0.1, max_value=30.0, value=10.0, step=0.1)
rho   = st.slider("rho (ρ)",   min_value=0.0, max_value=60.0, value=28.0, step=0.5)
beta  = st.slider("beta (β)",  min_value=0.1, max_value=5.0,  value=float(8.0/3.0), step=0.05)

st.divider()

# --- Integration settings (keep simple for now) ---
t0, tf = 0.0, 50.0
dt = 0.01
y0 = np.array([1.0, 1.0, 1.0], dtype=float)

def rhs(t, y):
    return lorenz_rhs(t, y, sigma=sigma, rho=rho, beta=beta)

sol = integrate_system(rhs, t_span=(t0, tf), y0=y0, t_step=dt)

if not sol.success:
    st.error(f"Solver failed: {sol.message}")
else:
    fig = plot_phase_xy(sol.y, title=f"Lorenz: σ={sigma:.2f}, ρ={rho:.2f}, β={beta:.2f}")
    st.pyplot(fig, clear_figure=True)
