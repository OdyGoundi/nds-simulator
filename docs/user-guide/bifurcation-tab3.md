# Bifurcation / Poincaré Sweep (Tab 3)

Minimal recipe to generate bifurcation diagrams.

---

## Steps

1) Choose sweep parameter, range, and step (left column).  
2) Set Poincaré section variable/value and crossing direction.  
3) Configure performance (right column): `dt_sweep`, `tf_sweep`, early stop, `max_hits`, `chunk_time`, transient fraction.  
4) Click **Generate** to start from scratch, or **Continue** to extend the last sweep.  
5) View scatter plot; magenta lines mark continuation boundaries.

## Tips

- Use warm start (Continuation) for smooth parameter sweeps.  
- Reduce `tf_sweep` and `max_hits` for quick previews; increase for publication-grade diagrams.  
- Reset accumulated data if you change key settings (sweep param, section, or solver mode).
