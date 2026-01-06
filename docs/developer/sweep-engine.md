# Sweep Engine (Tab 3)

## Flow

1) UI collects sweep + section + performance settings.  
2) Controller builds fingerprints and manages session_state.  
3) `run_sweep_chunk` executes param ranges, possibly with warm start.  
4) Results are appended/continued and plotted as scatter.

## Modes

- **Event-based IVP** (default when method = crossing): fast, early stop.  
- **Fallback sweep**: uses dense sampling; slower but robust.

## Performance knobs

- `chunk_time`, `early_stop`, `max_hits`: control event loop cost.  
- `tf_sweep`, `dt_sweep`, `transient_frac`: control integration effort.  
- Warm start vs reset ICs: choose continuation vs bibliography style.

## Data contracts

- Sweep results: dataframe with `param`, `t_hit`, `y{idx}` columns.  
- State caching: `sweep_acc_df`, `sweep_last_pv`, `sweep_meta`.
