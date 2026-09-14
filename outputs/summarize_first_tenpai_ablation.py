#!/usr/bin/env python3
"""Audit and summarize the four first-tenpai value-policy ablations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent / "push_fold_experiments" / "20260909" / "ablations"
SCORE_FALLBACK = Path("ablations")
SEED = 20260809


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mean_se(values: list[float]) -> dict[str, float | int | None]:
    mean = sum(values) / len(values) if values else None
    if len(values) < 2:
        return {"n": len(values), "mean": mean, "standard_error": None}
    assert mean is not None
    return {"n": len(values), "mean": mean,
            "standard_error": math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1) / len(values))}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(rollout: Mapping[str, Any], scored: Mapping[str, Any], tanyao: int, kanchan: int) -> None:
    run = rollout["run"]
    if int(rollout["trials"]) != 20000 or int(run["seed"]) != SEED or int(run["start_trial"]) != 0:
        raise ValueError("rollout does not cover the expected 20k seed range")
    policy = run["policy_parameters"]
    expected = {"future_ukeire_weight": 0.5, "improvement_weight": 0.1,
                "dora_ukeire_weight": 1.0, "dora_tiebreak_weight": 1.0,
                "dora_neighbor_weight": 1.0, "yakuhai_weight": 1.0,
                "tanyao_weight": float(tanyao)}
    for key, value in expected.items():
        if policy.get(key) != value:
            raise ValueError(f"unexpected rollout policy {key}")
    weights = policy.get("future_ukeire_weights_by_shanten", {})
    if weights.get("1") != 2.0 or weights.get("2") != 1.0 or weights.get("3_plus") != 0.0:
        raise ValueError("rollout is not extremeA 0.5/2/1")
    reversal = policy.get("shanten_reversal", {})
    if reversal.get("enabled") is not True or reversal.get("penalty") != 25.0:
        raise ValueError("rollout does not record reversal penalty 25")
    if float(policy.get("kanchan_outer_wait_weight", 0.0)) != float(kanchan):
        raise ValueError("rollout kanchan setting disagrees with filename")

    inputs = scored["inputs"]
    if int(inputs["tables"]) != 20000 or inputs["trial_id_min"] != 0 or inputs["trial_id_max"] != 19999:
        raise ValueError("scored artifact does not cover trial ids 0..19999")
    if int(inputs["expected_seed"]) != SEED or scored["diagnostics"]["scoring_errors"] != 0:
        raise ValueError("scored artifact has an invalid seed or scoring errors")
    trials = scored["trials"]
    ids = [int(item["trial_id"]) for item in trials]
    if len(ids) != 20000 or set(ids) != set(range(20000)):
        raise ValueError("scored trial records are not exactly 0..19999")
    signature = inputs["source_signature"]
    if signature["policy_parameters"] != policy:
        raise ValueError("scored source policy signature differs from rollout")
    round_winds = {"east": 27, "south": 28, "west": 29, "north": 30}
    if signature["round_wind"] != round_winds.get(run["round_wind"], run["round_wind"]):
        raise ValueError("scored round wind differs from rollout")
    if any(value is not None and (not math.isfinite(float(value)) or float(value) <= 0)
           for item in trials for value in (item["mean_ron_points"], item["mean_live_ron_points"])):
        raise ValueError("scored values must be finite and positive")

    stats = rollout["average_wait_tiles_remaining_at_first_tenpai"]
    hands = int(scored["diagnostics"]["first_tenpai_hands"])
    categories = sum(scored["diagnostics"]["waitcategory_counts"].values())
    if hands != int(stats["tenpai_hands"]) or categories != hands:
        raise ValueError("first-tenpai hand/category counts do not reconcile")
    visible = sum(int(item["visible_winning_copies"]) for item in trials)
    live = sum(int(item["live_winning_copies"]) for item in trials)
    if not math.isclose(visible / hands, float(stats["visible_remaining_copies"]), abs_tol=1e-12):
        raise ValueError("unseen winning copies do not reconcile to rollout")
    if not math.isclose(live / hands, float(stats["live_wall_remaining_copies"]), abs_tol=1e-12):
        raise ValueError("live winning copies do not reconcile to rollout")


def paired(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    a = {int(x["trial_id"]): x for x in left["trials"]}
    b = {int(x["trial_id"]): x for x in right["trials"]}
    if set(a) != set(b):
        raise ValueError("paired artifacts do not have identical trial ids")
    differences = [float(b[i]["mean_ron_points"]) - float(a[i]["mean_ron_points"])
                   for i in sorted(a) if a[i]["mean_ron_points"] is not None and b[i]["mean_ron_points"] is not None]
    return {
        "definition": "right minus left over the intersection of trials scored by both variants; this is conditional and differs from subtracting all-scored means",
        **mean_se(differences),
    }


def build(directory: Path, score_directory: Path) -> dict[str, Any]:
    variants: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    for tanyao in (0, 1):
        for kanchan in (0, 1):
            key = f"tanyao{tanyao}_kanchan{kanchan}"
            rollout_path = directory / f"first_tenpai_{key}_20k.json"
            score_path = score_directory / f"first_tenpai_value_{key}_20k.json"
            rollout, scored = load(rollout_path), load(score_path)
            validate(rollout, scored, tanyao, kanchan)
            stats = rollout["average_wait_tiles_remaining_at_first_tenpai"]
            variants[key] = {
                "tanyao_weight": tanyao, "kanchan_outer_wait_weight": kanchan,
                "primary_non_dealer_normalized_actual_wind_ron": scored["visible_copy_weighted"],
                "scored_hands": scored["diagnostics"]["visible_scored_hands"],
                "first_tenpai_hands": scored["diagnostics"]["first_tenpai_hands"],
                "mean_first_tenpai_turn": stats["mean_first_tenpai_turn"],
                "mean_visible_winning_copies": stats["visible_remaining_copies"],
                "mean_live_winning_copies": stats["live_wall_remaining_copies"],
                "waitcategory_counts": scored["diagnostics"]["waitcategory_counts"],
            }
            sources[f"rollout_{key}"] = digest(rollout_path)
            sources[f"scored_{key}"] = digest(score_path)
    pairs = {}
    for left in variants:
        for right in variants:
            if left < right:
                pairs[f"{right}__minus__{left}"] = paired(
                    load(score_directory / f"first_tenpai_value_{left}_20k.json"),
                    load(score_directory / f"first_tenpai_value_{right}_20k.json"),
                )
    sources["score_helper"] = digest(Path(__file__).resolve().parent / "score_first_tenpai_values.py")
    sources["report"] = digest(Path(__file__).resolve().parent / "hand_monte_carlo_report.py")
    return {"validation": "passed", "seed": SEED, "variants": variants,
            "paired_visible_copy_mean_differences": pairs, "source_sha256": sources}


def markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# First-tenpai value factorial summary", "", "Primary value is riichi non-dealer-normalized ron across all seats, while retaining actual seat wind for yaku. Each hand averages its unseen winning physical copies equally, using visible information to determine the unseen set.", "", "| tanyao | kanchan | scored hands | mean ron | hand SE | first-tenpai turn | unseen copies | live copies |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for value in summary["variants"].values():
        primary = value["primary_non_dealer_normalized_actual_wind_ron"]
        lines.append(f"| {value['tanyao_weight']} | {value['kanchan_outer_wait_weight']} | {value['scored_hands']} | {primary['mean_ron_points']:.4f} | {primary['hand_clustered_se']:.4f} | {value['mean_first_tenpai_turn']:.4f} | {value['mean_visible_winning_copies']:.4f} | {value['mean_live_winning_copies']:.4f} |")
    lines += ["", "Paired differences use only trials scored by both variants. They are conditional on that intersection and are not differences of the all-scored means.", ""]
    for name, value in summary["paired_visible_copy_mean_differences"].items():
        lines.append(f"- `{name}`: n={value['n']}, mean={value['mean']:.4f}, SE={value['standard_error']:.4f}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT)
    parser.add_argument("--score-directory", type=Path, default=SCORE_FALLBACK)
    args = parser.parse_args(argv)
    summary = build(args.directory, args.score_directory)
    (args.directory / "value_factorial_summary_20k.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.directory / "value_factorial_summary_20k.md").write_text(markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
