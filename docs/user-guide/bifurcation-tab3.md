# Parameter Sweep Analysis (Tab 3)

Minimal recipe to generate **bifurcation (Poincare)** and **Lyapunov** diagrams.

---

## Bifurcation sweep (left column)

1) Choose sweep parameter, range, and step.  
2) Set Poincare section variable/value and crossing direction (or enter a section equation).  
3) Choose method: **crossing** (sign change + interpolation) or **slab** (tolerance band).  
4) Pick sweep mode: **Bifurcation (reset ICs)** or **Continuation (warm start)**.  
5) Optional: enable **Parallel sweep (local only)** and set **Workers**.  
6) Optional: enable **Early stop (events)** and set **Max hits**.  
7) Click **Generate Bifurcation Diagram** or **Continue Bifurcation**.

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
