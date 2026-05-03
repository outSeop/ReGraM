from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if "matplotlib.pyplot" not in sys.modules:
    sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
    sys.modules["matplotlib.pyplot"] = types.ModuleType("matplotlib.pyplot")

from orchestration.dashboard_loader import (  # noqa: E402
    AUROC_DROP_METRICS,
    metric_has_values,
    metric_split_long_df,
)


class DashboardLoaderTests(unittest.TestCase):
    def test_metric_split_long_df_separates_overall_logical_structural(self) -> None:
        source_df = pd.DataFrame(
            [
                {
                    "baseline": "PatchCore",
                    "category": "breakfast_box",
                    "shift_family": "gaussian_noise",
                    "severity": "high",
                    "image_auroc_drop_from_clean": 1.0,
                    "image_auroc_drop_from_clean_logical": 2.0,
                    "image_auroc_drop_from_clean_structural": 3.0,
                }
            ]
        )

        split_df = metric_split_long_df(
            source_df,
            id_columns=["baseline", "category", "shift_family", "severity"],
            metric_specs=AUROC_DROP_METRICS,
            value_name="auroc_drop_from_clean",
        )

        self.assertEqual(split_df["anomaly_scope"].tolist(), ["overall", "logical", "structural"])
        self.assertEqual(split_df["auroc_drop_from_clean"].tolist(), [1.0, 2.0, 3.0])

    def test_metric_has_values_requires_non_null_metric(self) -> None:
        df = pd.DataFrame({"metric": [None, float("nan")]})

        self.assertFalse(metric_has_values(df, "metric"))
        self.assertFalse(metric_has_values(df, "missing"))

        df.loc[1, "metric"] = 1.0

        self.assertTrue(metric_has_values(df, "metric"))


if __name__ == "__main__":
    unittest.main()
