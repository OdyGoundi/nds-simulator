from typing import Dict


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
