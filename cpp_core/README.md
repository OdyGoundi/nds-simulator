# C++ Core (pybind11)

This directory builds the optional `nlds_cpp` Python extension.  
If the extension is not available, the Python code falls back automatically.

## Build (editable)

```
cd cpp_core
python -m pip install -r requirements.txt
python -m pip install -e .
```

Eigen headers are required for the QR implementation. If Eigen is not in a
standard include path, set `EIGEN3_INCLUDE_DIR` before building.

## What is included

- RK4 integration (used by the C++ Lyapunov path)
- QR-based Lyapunov spectrum (fixed-step RK4)
- Symplectic Forest-Ruth integrator

## Notes

- The C++ Lyapunov backend is currently used only when `solver_kind="rk4"`.
- Custom systems use Python callbacks for RHS/Jacobian inside the C++ loop.
