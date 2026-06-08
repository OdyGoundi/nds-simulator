import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import numba_backend
from core.poincare_sweep import PoincareConfig, SweepConfig, sweep_poincare


def oscillator_rhs(t, y, omega=1.0):
    return np.array([y[1], -(omega * omega) * y[0], 1.0], dtype=float)


class PoincareSweepTailHitsTests(unittest.TestCase):
    def test_python_sweep_keeps_last_configured_hits(self):
        rows = sweep_poincare(
            rhs=oscillator_rhs,
            y0=(0.0, 1.0, 0.0),
            t_span=(0.0, 35.0),
            base_params={"omega": 1.0},
            sweep=SweepConfig(param_name="omega", start=1.0, stop=1.0, step=1.0),
            poincare=PoincareConfig(section_index=0, section_value=0.0, direction=+1),
            solver_kind="rk4",
            t_step=0.01,
            output_indices=[2],
            warm_start=False,
            max_hits=2,
        )

        y2 = rows["y2"].to_numpy() if hasattr(rows, "to_numpy") else np.array([r["y2"] for r in rows])

        self.assertEqual(y2.size, 2)
        np.testing.assert_allclose(y2, [4.0 * 2.0 * math.pi, 5.0 * 2.0 * math.pi], atol=2e-2)

    def test_numba_sweep_keeps_last_configured_hits(self):
        if not numba_backend.numba_available():
            self.skipTest("Numba backend not available")

        nb = numba_backend.require_numba()

        @nb.njit
        def rhs_nb(t, y, p):
            omega = p[0]
            out = np.empty(3, dtype=np.float64)
            out[0] = y[1]
            out[1] = -(omega * omega) * y[0]
            out[2] = 1.0
            return out

        sweep_nb = numba_backend.build_poincare_sweep_rk4(rhs_nb)
        params_out, t_hit, y_hit, count = sweep_nb(
            np.array([0.0, 1.0, 0.0], dtype=float),
            0.0,
            35.0,
            0.01,
            np.array([1.0], dtype=float),
            0,
            1.0,
            1.0,
            1.0,
            0,
            0.0,
            +1,
            0,
            1e-3,
            0,
            2,
            False,
            2,
            False,
        )

        y_tail = np.asarray(y_hit[:count], dtype=float)

        self.assertEqual(y_tail.size, 2)
        np.testing.assert_allclose(y_tail, [4.0 * 2.0 * math.pi, 5.0 * 2.0 * math.pi], atol=2e-2)


if __name__ == "__main__":
    unittest.main()
