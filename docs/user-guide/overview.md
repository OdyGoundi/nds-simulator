# NLDS User Guide (Overview)

Minimal steps to get value from the Streamlit app.

---

## What you can do

- Simulate Lorenz, Rossler, or custom systems.
- Plot phase portraits (2D/3D) and time series.
- Compute Lyapunov exponents for a single trajectory (Tab 1).
- Generate parameter sweeps (bifurcation + Lyapunov) with continuation.
- Speed up independent sweeps with parallel workers (local).
- Export trajectories, bifurcation sweeps, and Lyapunov sweeps.

## Quick start

1) Launch: `streamlit run app/nlds_app.py`.  
2) Pick system + integration settings in the sidebar.  
3) Adjust system parameters (sidebar) and axes (Tab 1 controls).  
4) Explore tabs: Phase portrait (incl. Lyapunov), Time series, Parameter Sweep Analysis (bifurcation + Lyapunov).  
5) Export CSVs from the Export tab (trajectory, sweep, Lyapunov).
