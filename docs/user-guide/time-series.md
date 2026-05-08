# Time Series (Tab 2)

Minimal steps to inspect trajectories over time.

---

## Steps

1) Pick one or more variables to plot in the first multiselect.  
2) Optionally choose variables for per-variable plots (second multiselect).  
3) Adjust transient cut in the sidebar to hide startup behavior.  
4) System parameters update plots automatically.

## Tips

- Tab 2 reuses the trajectory already computed by Tab 1, so changing
  the variable selection or the time window does **not** re-run the
  solver — only the integration settings in the sidebar do.
- Use smaller `dt` for smoother curves; larger `tf` for longer horizons.  
- For custom systems, ensure variable names match the equations.  
- If nothing appears, verify `y0` length matches in initial conditions block and the number of equations.
