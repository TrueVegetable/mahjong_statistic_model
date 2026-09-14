import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "outputs"))

from summarize_danger_ablation import NH, _contrast_components  # noqa: E402


class DangerAblationSummaryTests(unittest.TestCase):
    def test_paired_cluster_contrast_uses_table_influence(self):
        cells = {
            0: {"left": NH(1, 1), "right": NH(1, 0)},
            1: {"left": NH(1, 0), "right": NH(1, 1)},
        }
        result, influence = _contrast_components(cells, (0, 1), "left", "right")
        self.assertEqual(result["difference"], 0.0)
        self.assertEqual(influence, {0: 0.5, 1: -0.5})
        self.assertTrue(math.isclose(result["paired_table_clustered_se"], 1.0))


if __name__ == "__main__":
    unittest.main()
