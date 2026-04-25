from typing import Dict

import numpy as np


def parse_params(text: str) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for line in (text or "").splitlines():
        line = line.replace(" ", " ").strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"Parameter line must be name=value. Got: '{line}'")
        name, val = line.split("=", 1)
        name = name.replace(" ", " ").strip()
        val = val.replace(" ", " ").strip()
        if name.lower() == "t":
            raise ValueError("Parameter name 't' is reserved for the independent variable; use other symbols for constants.")
        params[name] = float(val)
    return params


def parse_list_of_floats(text: str, n: int, label: str) -> np.ndarray:
    raw = (text or "").strip()
    if not raw:
        raise ValueError(f"{label} is empty.")
    tokens = raw.replace(",", " ").split()
    if len(tokens) != n:
        raise ValueError(f"{label} must have exactly {n} values. Got {len(tokens)}.")
    return np.array([float(t) for t in tokens], dtype=float)
