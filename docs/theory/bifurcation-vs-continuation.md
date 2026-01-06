# Bifurcation vs Continuation

Short notes on two common sweep strategies.

---

## Bifurcation (reset initial conditions)

- Each parameter value starts from the same initial conditions.  
- Mimics many bibliography plots (independent snapshots).  
- Good when attractors jump or multiple attractors exist.

**Trade-off:** longer transients at every step.

## Continuation (warm start)

- Reuses the final state of parameter value *pᵢ* as the initial state of *pᵢ₊₁*.  
- Tracks a single branch smoothly; very fast for gradual parameter changes.  
- Reduces transient length dramatically.

**Trade-off:** can stick to one branch and miss coexisting attractors.

## When to use which

- Exploring/preview: continuation for speed.  
- Mapping multiple branches or chaotic jumps: bifurcation (reset ICs).  
- Publication-quality diagrams often combine both: continuation for bulk, selective resets around suspected jumps.
