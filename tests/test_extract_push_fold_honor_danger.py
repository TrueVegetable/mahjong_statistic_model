import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "outputs/extract_push_fold_honor_danger.py"
SPEC = importlib.util.spec_from_file_location("extract_push_fold_honor_danger", SCRIPT)
extractor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(extractor)


def decision(public_honor, honor_hits, safe_types=()):
    own = [0] * 34
    for tile, copies in ((0, 4), (1, 4), (2, 4), (27, 2)):
        own[tile] = copies
    public = [0] * 34
    public[27] = public_honor
    candidates = [{"tile_type": tile, "ron_hits": honor_hits if tile == 27 else []}
                  for tile, copies in enumerate(own) if copies for _ in range(copies)]
    return {
        "threat_seats": [0], "turn": 6, "own_counts": own, "public_counts": public,
        "safe_types": [{"threat_seat": 0, "types": list(safe_types)}], "candidates": candidates,
    }


def table(trial_id, decisions):
    return {"trial_id": trial_id, "push_fold": {"decisions": decisions}}


class HonorExtractionTests(unittest.TestCase):
    def test_clusters_repeated_observations_and_excludes_genbutsu_and_other_opponents(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "observations.jsonl"
            raw.write_text("\n".join(json.dumps(row) for row in (
                table(10, [decision(0, [0]), decision(0, [0]), decision(0, [0], safe_types=(27,))]),
                table(11, [decision(0, [1])]),  # ron by a non-threat is outside the selected scope
                table(12, []),  # zero exposure must remain in the cluster denominator
            )) + "\n")
            totals, trials, invalid, per_table, _ = extractor.rows_from_raw(raw, "own-plus-public")
        primary = extractor.make_rows({key: value for key, value in totals.items() if len(key) == 2}, len(trials), False)
        cells = {(row["observer_turn"], row["visible_copies"]): row for row in primary}
        self.assertEqual((cells[6, 2]["candidate_opportunities"], cells[6, 2]["ron_hits"]), (3, 2))
        self.assertAlmostEqual(cells[6, 2]["table_clustered_se"], 0.3849001794597505)
        self.assertEqual(invalid, 0)
        self.assertEqual([row["trial_id"] for row in per_table], [10, 11, 12])
        self.assertEqual(len(per_table[0]["cells"]), 1)


if __name__ == "__main__":
    unittest.main()
