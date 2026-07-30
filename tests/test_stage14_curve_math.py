import numpy as np
import pandas as pd

from scripts._stage14_utils import (
    integrate_step_curve,
    weighted_counting_process_km,
)


def test_weighted_km_no_events_has_full_rmst():
    rows = pd.DataFrame(
        {
            "strategy_raw": [0, 0],
            "weight": [1.0, 2.0],
            "event": [0, 0],
            "start": [0.0, 0.0],
            "stop": [10.0, 10.0],
            "clone_id": ["a", "b"],
        }
    )
    curve, stats = weighted_counting_process_km(rows, 0, 10.0)
    assert abs(stats["rmst"] - 10.0) < 1e-12
    assert abs(stats["survival_horizon"] - 1.0) < 1e-12


def test_conditional_rmst_normalizes_at_landmark():
    curve = pd.DataFrame(
        {
            "time": [0.0, 2.0, 5.0],
            "survival": [1.0, 0.8, 0.6],
        }
    )
    area, s_start = integrate_step_curve(curve, 2.0, 6.0, conditional=True)
    assert abs(s_start - 0.8) < 1e-12
    # 2 to 5: conditional survival 1; 5 to 6: 0.6/0.8.
    assert abs(area - 3.75) < 1e-12
