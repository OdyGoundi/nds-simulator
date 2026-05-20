import io
from typing import Dict, List

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


def export_csv(df_or_rows, filename: str) -> None:
    """
    Export sweep output to CSV.
    Accepts either a pandas DataFrame or list-of-dicts rows.
    """
    if pd is not None and hasattr(df_or_rows, "to_csv"):
        df_or_rows.to_csv(filename, index=False)
        return

    # fallback without pandas
    rows: List[Dict[str, float]] = list(df_or_rows)
    if not rows:
        # create empty file with no content (or raise)
        with open(filename, "w", encoding="utf-8") as f:
            f.write("")
        return

    cols = list(rows[0].keys())
    with open(filename, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


def build_csv_bytes(
    t: np.ndarray,
    y: np.ndarray,
    var_names: List[str],
    *,
    chunk_rows: int = 200_000,
    start: int = 0,
    end: int | None = None,
    include_header: bool = True,
) -> bytes:
    t_arr = np.asarray(t, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim != 2:
        raise ValueError("y must be shape (n_vars, n_steps)")
    n = min(int(t_arr.size), int(y_arr.shape[1]))
    if n < 0:
        n = 0
    start_i = max(0, int(start))
    end_i = n if end is None else min(n, max(start_i, int(end)))
    chunk = max(1, int(chunk_rows))

    n_vars = int(y_arr.shape[0])
    if len(var_names) != n_vars:
        names = [f"y{i}" for i in range(n_vars)]
    else:
        names = list(var_names)

    buf = io.StringIO()
    if include_header:
        buf.write("t," + ",".join(names) + "\n")

    if end_i > start_i:
        for lo in range(start_i, end_i, chunk):
            hi = min(end_i, lo + chunk)
            block = np.empty((hi - lo, n_vars + 1), dtype=float)
            block[:, 0] = t_arr[lo:hi]
            block[:, 1:] = y_arr[:, lo:hi].T
            np.savetxt(buf, block, delimiter=",", fmt="%.18g")

    return buf.getvalue().encode("utf-8")
