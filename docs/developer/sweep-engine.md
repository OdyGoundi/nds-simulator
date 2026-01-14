# Sweep Engine (Tab 3)

## Flow

1) UI collects sweep + section + performance settings.  
2) Controller builds fingerprints and manages session_state.  
3) `run_sweep_chunk` executes param ranges, possibly with warm start.  
4) Lyapunov sweeps use `_run_lyapunov_sweep` (optionally parallel).  
5) Results are appended/continued and plotted (scatter for bifurcation, lines for Lyapunov).

## Modes

- **Event-based IVP** (default when method = crossing): fast, early stop.  
- **Fallback sweep**: uses dense sampling; slower but robust.
- **Lyapunov sweep**: computes spectrum per parameter value (QR-based; optional parallel).

## Performance knobs

- `chunk_time`, `early_stop`, `max_hits`: control event loop cost.  
- `tf_sweep`, `dt_sweep`, `transient_frac`: control integration effort.  
- `qr_interval`: controls Lyapunov QR frequency.  
- Warm start vs reset ICs: choose continuation vs bibliography style.  
- `parallel`, `workers`: speed up independent sweeps (no warm start).

## Data contracts

- Sweep results: dataframe with `param`, `t_hit`, `y{idx}` columns.  
- Lyapunov results: `param_vals`, `lambdas`, `errors`.  
- State caching: `sweep_acc_df`, `sweep_last_pv`, `sweep_meta`, `lya_acc_data`, `lya_last_pv`, `lya_meta`.
