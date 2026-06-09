# Phase Portrait (Tab 1)

Minimal steps to view phase diagrams.

---

## Steps

1) Choose axes in the Tab 1 control column (x, y, and z if 3D).  
2) Set plot mode (2D vs 3D) in the sidebar.  
3) Adjust system parameters and rerun (auto on change).  
4) Use transient cut in the sidebar to skip initial samples.

## Lyapunov exponents (Tab 1)

1) Set **Transient cut** and **QR interval (time)** in the sidebar.  
2) Make sure `final time` is long enough so measurement time stays positive.  
3) Read `lambda = [...]` under the plot.

## Poincaré map (Tab 1)

1) Generate a trajectory in Tab 1 first.  
2) Enable **Show Poincaré map** under the phase plot.  
3) Choose the section plane, plane value, crossing direction, and map axes.  
4) Adjust **Max points** if many crossings are found.

## Tips

- 2D plots use equal aspect ratio for clarity.  
- 3D plots follow the selected axes order.  
- The Poincaré map is extracted from the trajectory already shown above; it does not rerun integration.  
- Poincaré map display requires at least 3 state variables and decimates displayed hits to the selected `Max points`.  
- If a variable is flat, try different ICs or parameters.  
- For chaotic regimes, increase `final time` for more stable Lyapunov estimates.
- RK4 and Symplectic Forest-Ruth are fixed-step; use a smaller `dt` if accuracy looks off (Numba accelerates these when available).
