import os
from pathlib import Path
from typing import Optional

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


def _find_eigen_include() -> Optional[str]:
    candidates = []
    env_path = os.environ.get("EIGEN3_INCLUDE_DIR")
    if env_path:
        candidates.append(env_path)
    candidates.extend([
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
    ])
    for path in candidates:
        if Path(path).exists():
            return path
    return None


eigen_include = _find_eigen_include()
include_dirs = [eigen_include] if eigen_include else []


ext_modules = [
    Pybind11Extension(
        "nlds_cpp",
        [
            "src/bindings/python_bindings.cpp",
            "src/bindings/lyapunov_bindings.cpp",
            "src/bindings/rk4_bindings.cpp",
            "src/bindings/symplectic_bindings.cpp",
            "src/bindings/sweep_bindings.cpp",
        ],
        include_dirs=include_dirs,
        cxx_std=17,
        extra_compile_args=["-O3"],
    )
]


setup(
    name="nlds-cpp",
    version="0.0.1",
    description="C++ backend for NLDS simulator",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
