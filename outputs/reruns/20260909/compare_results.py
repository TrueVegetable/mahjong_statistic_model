#!/usr/bin/env python3
"""Validate reruns against saved aggregates without overwriting either."""
import json
import math

from rerun_experiments import HERE, OUTPUTS, SOURCES


REGROUPED_RAW_FIELDS = (
    "wait_tile_opportunities", "wait_visible_copies", "wait_live_copies",
    "eligible_hands_by_wait", "eligible_hands_by_tile_and_wait",
    "next_draw_hit_probability_sum", "next_draw_hit_probability_sum_by_tile_and_wait",
)


def fold_wait_counts(values):
    folded = {}
    for key, value in values.items():
        prefix, separator, category = key.rpartition("|")
        if not separator:
            category = key
        if category in ("sanmenchan", "chiitoitsu_tanki", "other"):
            category = "complex_and_seven_pairs"
        combined = prefix + separator + category
        folded[combined] = folded.get(combined, 0) + value
    return folded


def is_reference_regrouping(path):
    if path in ("raw.chiitoitsu_tenpai_hands",
                "average_wait_tiles_remaining_at_first_tenpai.chiitoitsu_tenpai_hands",
                "average_wait_tiles_remaining_at_first_tenpai.chiitoitsu_tenpai_rate"):
        return True
    if path.startswith(tuple("raw." + field + "." for field in REGROUPED_RAW_FIELDS)):
        return path.rsplit(".", 1)[-1].rsplit("|", 1)[-1] in ("chiitoitsu_tanki", "other")
    prefixes = ("wait_distribution.categories.", "tile_hit_rate_by_wait.categories.",
                "tile_hit_rate_by_tile_type_and_wait.by_tile_type_and_wait.")
    return path.startswith(prefixes) and path.split(".")[-2] in ("chiitoitsu_tanki", "other")


def differences(old, new, path=""):
    if isinstance(old, dict) and isinstance(new, dict):
        result = []
        for key in old:
            child = f"{path}.{key}" if path else key
            if key not in new:
                result.append({"path": child, "old": old[key], "new": "<missing>"})
            else:
                result.extend(differences(old[key], new[key], child))
        return result
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        if math.isclose(old, new, rel_tol=1e-12, abs_tol=1e-10):
            return []
    elif old == new:
        return []
    return [{"path": path, "old": old, "new": new}]


def main():
    results = []
    for name in SOURCES:
        old = json.loads((OUTPUTS / name).read_text())
        new = json.loads((HERE / name).read_text())
        assert new["trials"] == old["trials"] == sum(new["outcomes"].values())
        for key in ("seed", "start_trial", "seat_wind", "round_wind", "evaluator_backend"):
            assert old["run"][key] == new["run"][key], (name, key)
        old_policy = old["run"]["policy_parameters"]
        new_policy = new["run"]["policy_parameters"]
        assert not differences(old_policy, new_policy), name
        expected_weights = old_policy.get("future_ukeire_weights_by_shanten", {
            "1": old_policy["future_ukeire_weight"],
            "2": old_policy["future_ukeire_weight"], "3_plus": 0.0})
        assert new_policy["future_ukeire_weights_by_shanten"] == expected_weights
        changes = differences({k: v for k, v in old.items() if k != "run"},
                              {k: v for k, v in new.items() if k != "run"})
        regrouping = name == SOURCES[0]
        if regrouping:
            old_categories = old["wait_distribution"]["categories"]
            new_categories = new["wait_distribution"]["categories"]
            assert "sanmenchan" not in old_categories
            assert new_categories["sanmenchan"]["tenpai_hands_with_wait"] == 3312
            assert new_categories["other"]["tenpai_hands_with_wait"] == 2718
            assert new_categories["chiitoitsu_tanki"]["tenpai_hands_with_wait"] == 6774
            for field in REGROUPED_RAW_FIELDS:
                assert not differences(fold_wait_counts(old["raw"][field]),
                                       fold_wait_counts(new["raw"][field])), (name, field)
        unexpected = [row for row in changes
                      if row["path"] != "continued_rejected_tenpai.dama_tsumo_after_rejection"
                      and not (regrouping and is_reference_regrouping(row["path"]))]
        after = new.get("continued_rejected_tenpai")
        if after:
            assert after["tables_with_at_least_one_rejection"] == sum(
                after[key] for key in ("eligible_tenpai_after_rejection", "dama_tsumo_after_rejection",
                                       "dama_ron_after_rejection", "exhausted_after_rejection"))
            before = old.get("continued_rejected_tenpai")
            if before is not None:
                assert before["dama_tsumo_after_rejection"] == (
                    after["dama_tsumo_after_rejection"] + after["dama_ron_after_rejection"])
            else:
                assert not any(after.values())
        score = new["nondealer_ron_score"]
        assert not score["scoring_errors"]
        assert score["invalid_visible_copy_weight"] == score["invalid_live_copy_weight"] == 0
        results.append({"experiment": name, "trials": new["trials"], "outcomes": new["outcomes"],
                        "changes_to_existing_statistics": changes, "unexpected_changes": unexpected,
                        "reference_wait_regrouping": regrouping,
                        "continued_rejected_tenpai": after,
                        "unseen_copy_weighted_ron": score["unseen_copy_weighted"]})
    (HERE / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    assert not any(row["unexpected_changes"] for row in results), "Inspect comparison.json"
    print("All six reruns verified: corrected ron/tsumo split and documented reference wait regrouping; no unexplained changes.")


if __name__ == "__main__":
    main()
