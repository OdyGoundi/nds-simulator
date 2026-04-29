from __future__ import annotations

import io
import json
import zipfile
from typing import Any, Dict, List, Optional

import numpy as np

from app.export.csv_utils import build_csv_bytes
from app.state.defaults import DIRECT_CSV_MAX_ROWS, EXPORT_CHUNK_ROWS_DEFAULT


def build_zip_bytes(file_map: Dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in file_map.items():
            zf.writestr(name, data)
    return buf.getvalue()


def build_run_bundle(
    *,
    bundle_cfg: Dict[str, Any],
    static_cfg: Optional[Dict[str, Any]],
    sweep_cfg: Optional[Dict[str, Any]],
    t_traj: Optional[np.ndarray],
    y_traj: Optional[np.ndarray],
    var_names: List[str],
    traj_ready: bool,
    traj_source: str,
    df_sweep: Any,
    lya_data: Optional[Dict[str, Any]],
    direct_csv_max_rows: int = DIRECT_CSV_MAX_ROWS,
    chunk_rows: int = EXPORT_CHUNK_ROWS_DEFAULT,
) -> bytes:
    files: Dict[str, bytes] = {}

    files["config.json"] = json.dumps(bundle_cfg, indent=2).encode("utf-8")
    if static_cfg is not None:
        files["StaticParamsConfig.json"] = json.dumps(static_cfg, indent=2).encode("utf-8")
    if sweep_cfg is not None:
        files["SweepParamConfig.json"] = json.dumps(sweep_cfg, indent=2).encode("utf-8")

    traj_size = int(t_traj.size) if t_traj is not None else 0
    if traj_ready and traj_size > 0:
        if traj_size <= int(direct_csv_max_rows):
            files["trajectory.csv"] = build_csv_bytes(t_traj, y_traj, var_names)
        else:
            end_first = min(traj_size, int(chunk_rows))
            files["trajectory_part001.csv"] = build_csv_bytes(
                t_traj, y_traj, var_names, start=0, end=end_first
            )
            files["trajectory_manifest.txt"] = (
                f"Trajectory rows: {traj_size}\n"
                f"Trajectory source: {traj_source}\n"
                "Only the first chunk is included in this zip to keep memory bounded.\n"
                "Use Tab 4 chunk export to download the remaining chunks.\n"
            ).encode("utf-8")
    else:
        files["trajectory_manifest.txt"] = (
            "Trajectory export was not included in this bundle.\n"
            "If you want the full-resolution trajectory, prepare it in Tab 4 first.\n"
        ).encode("utf-8")

    if df_sweep is not None and len(df_sweep) > 0:
        import pandas as pd
        if not isinstance(df_sweep, pd.DataFrame):
            df_sweep = pd.DataFrame(df_sweep)
        files["sweep.csv"] = df_sweep.to_csv(index=False).encode("utf-8")

    if lya_data is not None:
        param_vals = np.array(lya_data.get("param_vals", []), dtype=float)
        lambdas_arr = np.array(lya_data.get("lambdas", []), dtype=float)
        if param_vals.size and lambdas_arr.size:
            meta = lya_data.get("meta", {})
            sweep_param = meta.get("sweep_param", "param")
            data: Dict[str, Any] = {str(sweep_param): param_vals}
            if lambdas_arr.ndim == 1:
                data["lambda0"] = lambdas_arr
            else:
                for k in range(lambdas_arr.shape[1]):
                    data[f"lambda{k}"] = lambdas_arr[:, k]
            import pandas as pd
            files["lyapunov_sweep.csv"] = pd.DataFrame(data).to_csv(index=False).encode("utf-8")

    return build_zip_bytes(files)
