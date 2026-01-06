# Session State (Streamlit)

## Purpose

- Persist sweep progress across reruns.
- Avoid recomputing trajectories when UI toggles change.

## Core keys

- `sweep_acc_df`: accumulated sweep dataframe (or `None`).
- `sweep_last_pv`: last parameter value completed.
- `sweep_stop_internal`: UI stop value preserved between runs.
- `sweep_boundaries`: list of split points when continuing sweeps.
- `sweep_meta`: fingerprint of the last sweep configuration.
- `last_sweep_df`, `last_sweep_meta`: cached for export tab.

## Safety tips

- Reset these keys together (see "Reset accumulated" button).
- Do not mutate cached dataframes in-place; assign new copies instead.
- When changing sweep settings, clear `sweep_acc_df` to avoid mixing runs.
