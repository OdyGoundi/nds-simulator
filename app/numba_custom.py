"""Compile and cache Numba kernels for custom user-defined systems.

This module owns the JIT layer: Numba presence check, exec of generated
source, `nb.njit` decoration, and the compiled-kernel cache. The parsing
and code generation live in `app/parsing/numba_rhs_builder.py`.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import numba as nb  # type: ignore
except Exception:  # pragma: no cover
    nb = None

from app.parsing.numba_rhs_builder import (
    parse_safe_exprs,
    parse_symplectic_safe_exprs,
    render_jacobian_source,
    render_rhs_source,
    render_symplectic_dpdt_source,
    render_symplectic_dqdt_source,
)


_CacheKey = Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]
_CUSTOM_CACHE: Dict[_CacheKey, Tuple[Optional[Callable], Optional[Callable]]] = {}
_CUSTOM_SYMPLECTIC_CACHE: Dict[_CacheKey, Tuple[Callable, Callable]] = {}


def _cache_key(var_names: Sequence[str], eq_lines: Sequence[str], param_names: Sequence[str]) -> _CacheKey:
    return tuple(var_names), tuple(eq_lines), tuple(param_names)


def _require_numba():
    if nb is None:  # pragma: no cover
        raise RuntimeError("Numba is required for custom Numba compilation.")
    return nb


def _numpy_namespace() -> dict:
    """Globals for exec() — numpy aliases keep generated code self-contained."""
    return {
        "np": np,
        "math": math,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "abs": np.abs,
    }


def _compile_and_jit(src: str, *names: str) -> Tuple[Callable, ...]:
    """exec the source and JIT-decorate each named function. Returns them in order."""
    nb_mod = _require_numba()
    ns = _numpy_namespace()
    exec(src, ns)
    return tuple(nb_mod.njit(cache=False, fastmath=True)(ns[name]) for name in names)


def build_custom_numba_rhs(
    var_names: Sequence[str],
    eq_lines: Sequence[str],
    param_names: Sequence[str],
) -> Callable:
    key = _cache_key(var_names, eq_lines, param_names)
    cached = _CUSTOM_CACHE.get(key)
    if cached is not None and cached[0] is not None:
        return cached[0]

    safe_exprs, safe_vars, safe_params = parse_safe_exprs(var_names, eq_lines, param_names)
    src = render_rhs_source(safe_exprs, safe_vars, safe_params, len(param_names))
    (rhs_nb,) = _compile_and_jit(src, "rhs")

    _CUSTOM_CACHE[key] = (rhs_nb, cached[1] if cached is not None else None)
    return rhs_nb


def build_custom_numba_rhs_and_jacobian(
    var_names: Sequence[str],
    eq_lines: Sequence[str],
    param_names: Sequence[str],
) -> Tuple[Callable, Callable]:
    key = _cache_key(var_names, eq_lines, param_names)
    cached = _CUSTOM_CACHE.get(key)
    if cached is not None and cached[0] is not None and cached[1] is not None:
        return cached[0], cached[1]

    safe_exprs, safe_vars, safe_params = parse_safe_exprs(var_names, eq_lines, param_names)
    rhs_src = render_rhs_source(safe_exprs, safe_vars, safe_params, len(param_names))
    jac_src = render_jacobian_source(safe_exprs, safe_vars, safe_params, len(param_names))
    rhs_nb, jac_nb = _compile_and_jit(rhs_src + "\n\n" + jac_src, "rhs", "jac")

    _CUSTOM_CACHE[key] = (rhs_nb, jac_nb)
    return rhs_nb, jac_nb


def build_custom_numba_symplectic_functions(
    var_names: Sequence[str],
    eq_lines: Sequence[str],
    param_names: Sequence[str],
) -> Tuple[Callable, Callable]:
    key = _cache_key(var_names, eq_lines, param_names)
    cached = _CUSTOM_SYMPLECTIC_CACHE.get(key)
    if cached is not None:
        return cached

    dq_safe, dp_safe, safe_vars, safe_params, n_q = parse_symplectic_safe_exprs(
        var_names, eq_lines, param_names
    )
    n_params = len(param_names)
    dq_src = render_symplectic_dqdt_source(dq_safe, safe_vars, safe_params, n_params, n_q)
    dp_src = render_symplectic_dpdt_source(dp_safe, safe_vars, safe_params, n_params, n_q)
    dq_nb, dp_nb = _compile_and_jit(dq_src + "\n\n" + dp_src, "dq_dt", "dp_dt")

    _CUSTOM_SYMPLECTIC_CACHE[key] = (dq_nb, dp_nb)
    return dq_nb, dp_nb
