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

## Tips

- 2D plots use equal aspect ratio for clarity.  
- 3D plots follow the selected axes order.  
- If a variable is flat, try different ICs or parameters.  
- For chaotic regimes, increase `final time` for more stable Lyapunov estimates.
