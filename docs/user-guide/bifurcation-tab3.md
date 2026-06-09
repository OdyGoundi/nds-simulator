# Parameter Sweep Analysis (Tab 3)

Minimal recipe to generate **bifurcation (Poincare)** and **Lyapunov** diagrams.

---

## Bifurcation sweep (left column)

1) Choose sweep parameter, range, and step.  
2) Choose observable: **Poincaré crossings** or **local extrema** of one variable (max / min / both).  
3) For Poincaré crossings: set the section variable/value and crossing direction (or enter a section equation).  
4) Choose method: **crossing** (sign change + interpolation) or **slab** (tolerance band).  
5) Pick sweep mode: **Bifurcation (reset ICs)** or **Continuation (warm start)**.  
6) Optional: enable **Parallel sweep (local only)** and set **Workers**.  
7) Optional: enable **Early stop (events)** and set **Max hits**.  
8) Click **Generate Bifurcation Diagram** or **Continue Bifurcation**.

## Lyapunov sweep (right column)

1) Reuse the same sweep parameter/range from the top.  
2) Set **QR interval (time)** and **Transient fraction** for Lyapunov.  
3) Optional: enable **Parallel sweep (local only)** and set **Workers**.  
4) Click **Generate Lyapunov Diagram** or **Continue Lyapunov**.  
5) Plot shows lambda values vs parameter (magenta lines mark continuation boundaries).

## Tips

- Continuation is smooth but sequential; parallel only works in independent mode (no warm start).  
- If Lyapunov curves are noisy, increase `final time` and reduce `time step`.  
- Reset accumulated data if you change key settings (sweep param, section, solver mode).

## What The Bifurcation Plot Stores

- The app keeps a recent full-resolution sweep buffer in memory and caps its size.  
- Older dropped bifurcation rows are retained as a bounded reservoir sample instead of keeping the full history.  
- The plotted chart combines recent rows plus the reservoir sample, so the caption reports `recent + reservoir` point counts.  
- Reservoir and recent points are rendered with the same black marker style; the difference is memory handling, not visual styling.

## Solver notes

- Sweeps use the solver selected in the sidebar (same as Tab 1).  
- Event-based crossings run only for IVP solvers (RK45/DOP853) + crossing method.  
- RK4 and Symplectic Forest-Ruth use fixed-step integration; Numba is used automatically when available.  
- Expression-based sections disable the fast Numba sweep path for built-in systems.
