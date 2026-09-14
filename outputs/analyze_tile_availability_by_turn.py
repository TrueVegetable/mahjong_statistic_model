#!/usr/bin/env python3
"""Compare synchronized tile-availability checkpoints across sampled turns."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any

import hand_monte_carlo_report as report


def _standard_error(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "standard_error": None, "ci95": None, "n": 0}
    mean = sum(values) / len(values)
    standard_error = _standard_error(values)
    return {
        "mean": mean,
        "standard_error": standard_error,
        "ci95": (
            [mean - 1.96 * standard_error, mean + 1.96 * standard_error]
            if standard_error is not None else None
        ),
        "n": len(values),
    }


def _load(path: Path, expected_turn: int) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A running native process may have written only part of its
                # final JSONL record. Completed files always end in a newline.
                if not line.endswith("\n"):
                    break
                raise
            checkpoint = row["availability_checkpoint"]
            if int(checkpoint["target_own_turn"]) != expected_turn:
                raise ValueError(f"{path}:{line_number} has the wrong checkpoint turn")
            trial_id = int(row["trial_id"])
            if trial_id in rows:
                raise ValueError(f"duplicate trial_id {trial_id} in {path}")
            rows[trial_id] = row
    return rows


def _own_bucket(count: int) -> str:
    return str(count) if count < 3 else "3_plus"


def _tile_group(tile_type: int) -> str:
    if tile_type >= 27:
        return "honor"
    rank = tile_type % 9 + 1
    if rank in (1, 9):
        return "19"
    if rank in (2, 8):
        return "28"
    if rank in (3, 7):
        return "37"
    if rank in (4, 6):
        return "46"
    return "5"


def _location_summary(totals: dict[str, float]) -> dict[str, Any]:
    observations = int(totals["observations"])
    return {
        "observations": observations,
        "mean_unseen_copies": totals["unseen_copies"] / observations,
        "mean_live_wall_copies": totals["actual_live"] / observations,
        "live_wall_pooled_score": (
            totals["actual_live"] / totals["expected_live"]
            if totals["expected_live"] else None
        ),
        "mean_opponent_copies": totals["actual_opponent"] / observations,
        "opponent_hands_pooled_score": (
            totals["actual_opponent"] / totals["expected_opponent"]
            if totals["expected_opponent"] else None
        ),
    }


def _analyze_turn(rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    category_totals = {
        category: {"actual": 0.0, "expected": 0.0, "ratio_sum": 0.0, "n": 0}
        for category in report.AVAILABILITY_TABLE_CATEGORIES
    }
    table_ratios: dict[int, dict[str, float]] = {}
    honor_totals: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    honor_by_table: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0, 0.0])
    )
    tile_discard_totals: dict[tuple[str, str, int, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    for trial_id, row in rows.items():
        checkpoint = row["availability_checkpoint"]
        players = checkpoint["players"]
        dora_type = int(checkpoint["dora_indicator_type"])
        round_wind = int(row.get("round_wind", 27))
        per_table = {
            category: [0.0, 0.0]
            for category in report.AVAILABILITY_TABLE_CATEGORIES
        }
        for observer, player in enumerate(players):
            own = player["own_known_counts"]
            public = player["public_counts"]
            unseen = player["unseen_counts"]
            live = player["live_wall_counts"]
            opponents = player["opponent_hand_counts"]
            live_positions = int(player["live_wall_tiles"])
            opponent_positions = int(player["opponent_hand_tiles"])
            unseen_positions = int(player["unseen_tiles"])
            for tile_type in range(34):
                expected = unseen[tile_type] * live_positions / unseen_positions
                category = report.availability_table_category(tile_type, round_wind)
                totals = category_totals[category]
                totals["actual"] += live[tile_type]
                totals["expected"] += expected
                per_table[category][0] += live[tile_type]
                per_table[category][1] += expected
                if expected:
                    totals["ratio_sum"] += live[tile_type] / expected
                    totals["n"] += 1

                river_count = public[tile_type] - int(tile_type == dora_type)
                for level, identity in (
                    ("category", _tile_group(tile_type)),
                    ("tile", report.sim.tile_name(tile_type)),
                ):
                    location = tile_discard_totals[
                        (level, identity, river_count, _own_bucket(own[tile_type]))
                    ]
                    location["observations"] += 1
                    location["unseen_copies"] += unseen[tile_type]
                    location["actual_live"] += live[tile_type]
                    location["expected_live"] += expected
                    location["actual_opponent"] += opponents[tile_type]
                    location["expected_opponent"] += (
                        unseen[tile_type] * opponent_positions / unseen_positions
                    )

            for tile_type in range(27, 34):
                river_count = public[tile_type] - int(tile_type == dora_type)
                discard_bucket = "never_discarded" if river_count == 0 else "discarded"
                exact_river_bucket = f"river_{river_count}"
                buckets = [
                    (discard_bucket, "all"),
                    (discard_bucket, _own_bucket(own[tile_type])),
                    (exact_river_bucket, "all"),
                    (exact_river_bucket, _own_bucket(own[tile_type])),
                ]
                if river_count >= 2:
                    buckets.extend((
                        ("river_2_plus", "all"),
                        ("river_2_plus", _own_bucket(own[tile_type])),
                    ))
                pair_elsewhere = any(
                    int(players[seat]["own_known_counts"][tile_type]) >= 2
                    for seat in range(4) if seat != observer
                )
                triplet_elsewhere = any(
                    int(players[seat]["own_known_counts"][tile_type]) >= 3
                    for seat in range(4) if seat != observer
                )
                expected_live = unseen[tile_type] * live_positions / unseen_positions
                expected_opponent = unseen[tile_type] * opponent_positions / unseen_positions
                for key in buckets:
                    totals = honor_totals[key]
                    totals["observations"] += 1
                    totals["pair_elsewhere"] += int(pair_elsewhere)
                    totals["triplet_elsewhere"] += int(triplet_elsewhere)
                    totals["opponent_copies"] += opponents[tile_type]
                    totals["unseen_copies"] += unseen[tile_type]
                    totals["actual_live"] += live[tile_type]
                    totals["expected_live"] += expected_live
                    totals["actual_opponent"] += opponents[tile_type]
                    totals["expected_opponent"] += expected_opponent
                    table = honor_by_table[key][trial_id]
                    table[0] += 1
                    table[1] += int(pair_elsewhere)
                    table[2] += int(triplet_elsewhere)

        table_ratios[trial_id] = {
            category: actual / expected if expected else math.nan
            for category, (actual, expected) in per_table.items()
        }

    categories = {}
    for category, totals in category_totals.items():
        categories[category] = {
            "mean_score": totals["ratio_sum"] / totals["n"] if totals["n"] else None,
            "pooled_score": totals["actual"] / totals["expected"] if totals["expected"] else None,
            "actual_live_copies": totals["actual"],
            "expected_live_copies": totals["expected"],
            "valid_observations": int(totals["n"]),
        }

    honor = {}
    for key, totals in sorted(honor_totals.items()):
        observations = int(totals["observations"])
        table_pair_rates = [
            values[1] / values[0] for values in honor_by_table[key].values() if values[0]
        ]
        table_triplet_rates = [
            values[2] / values[0] for values in honor_by_table[key].values() if values[0]
        ]
        honor[f"{key[0]}__own_{key[1]}"] = {
            "observations": observations,
            "opponent_pair_probability": totals["pair_elsewhere"] / observations,
            "opponent_pair_table_clustered_se": _standard_error(table_pair_rates),
            "opponent_triplet_probability": totals["triplet_elsewhere"] / observations,
            "opponent_triplet_table_clustered_se": _standard_error(table_triplet_rates),
            "mean_opponent_copies": totals["opponent_copies"] / observations,
            "mean_unseen_copies": totals["unseen_copies"] / observations,
            "mean_live_wall_copies": totals["actual_live"] / observations,
            "live_wall_pooled_score": (
                totals["actual_live"] / totals["expected_live"]
                if totals["expected_live"] else None
            ),
            "opponent_hands_pooled_score": (
                totals["actual_opponent"] / totals["expected_opponent"]
                if totals["expected_opponent"] else None
            ),
        }

    tile_discard_condition: dict[str, dict[str, Any]] = {
        "by_category": {},
        "by_tile_type": {},
    }
    for (level, identity, river_count, own_bucket), totals in sorted(
        tile_discard_totals.items()
    ):
        tile_discard_condition[f"by_{'category' if level == 'category' else 'tile_type'}"][
            f"{identity}__river_{river_count}__own_{own_bucket}"
        ] = _location_summary(totals)

    return {
        "tables": len(rows),
        "categories": categories,
        "table_ratios": table_ratios,
        "honor_discard_condition": honor,
        "tile_discard_condition": tile_discard_condition,
    }


def _pairwise(
    early: dict[str, Any], late: dict[str, Any], early_turn: int, late_turn: int
) -> dict[str, Any]:
    common = sorted(set(early["table_ratios"]) & set(late["table_ratios"]))
    categories = {}
    for category in report.AVAILABILITY_TABLE_CATEGORIES:
        differences = [
            late["table_ratios"][trial_id][category]
            - early["table_ratios"][trial_id][category]
            for trial_id in common
            if not math.isnan(early["table_ratios"][trial_id][category])
            and not math.isnan(late["table_ratios"][trial_id][category])
        ]
        categories[category] = {
            "pooled_score_change": (
                late["categories"][category]["pooled_score"]
                - early["categories"][category]["pooled_score"]
            ),
            "mean_paired_table_ratio_change": _summary(differences),
        }
    return {
        "early_turn": early_turn,
        "late_turn": late_turn,
        "common_tables": len(common),
        "categories": categories,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoints", nargs="+", metavar="TURN=JSONL",
        help="checkpoint observation files, for example 6=turn6.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tile-discard-csv", type=Path,
        help="optional flat export of category and exact-tile discard conditions",
    )
    args = parser.parse_args()
    sources: dict[int, Path] = {}
    for item in args.checkpoints:
        turn_text, separator, path_text = item.partition("=")
        if not separator:
            parser.error(f"invalid checkpoint argument: {item}")
        sources[int(turn_text)] = Path(path_text)
    if len(sources) < 2:
        parser.error("at least two checkpoint turns are required")

    analyzed = {
        turn: _analyze_turn(_load(path, turn)) for turn, path in sorted(sources.items())
    }
    ordered_turns = sorted(analyzed)
    payload = {
        "metric": "actual live-wall copies / conditional expected live-wall copies",
        "grain": "table x observer x tile type",
        "turns": {
            str(turn): {
                key: value for key, value in result.items() if key != "table_ratios"
            }
            for turn, result in analyzed.items()
        },
        "paired_changes": [
            _pairwise(analyzed[early], analyzed[late], early, late)
            for early, late in zip(ordered_turns, ordered_turns[1:])
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if args.tile_discard_csv:
        fields = [
            "turn", "aggregation", "tile", "discarded_copies", "own_copies",
            "observations", "mean_unseen_copies", "mean_live_wall_copies",
            "live_wall_pooled_score", "mean_opponent_copies",
            "opponent_hands_pooled_score",
        ]
        args.tile_discard_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.tile_discard_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for turn, result in sorted(analyzed.items()):
                conditions = result["tile_discard_condition"]
                for aggregation, entries in conditions.items():
                    for key, values in entries.items():
                        tile, suffix = key.split("__river_", 1)
                        discarded, own = suffix.split("__own_", 1)
                        writer.writerow({
                            "turn": turn,
                            "aggregation": aggregation.removeprefix("by_"),
                            "tile": tile,
                            "discarded_copies": discarded,
                            "own_copies": own,
                            **values,
                        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
