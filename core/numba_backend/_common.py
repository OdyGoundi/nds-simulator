from __future__ import annotations

try:
    import numba as nb  # type: ignore
except Exception:  # pragma: no cover
    nb = None


def numba_available() -> bool:
    return nb is not None


def require_numba():
    if nb is None:  # pragma: no cover
        raise RuntimeError("Numba is required for the Numba backend.")
    return nb
