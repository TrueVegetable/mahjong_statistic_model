#!/usr/bin/env python3
"""Post-score first-tenpai shared-table snapshots with the pinned scorer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import hand_efficiency_sim as sim
from hand_monte_carlo_report import primary_wait_category
from hand_push_fold_scoring import PushFoldScorer, _valid


def _mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def _mean_se(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1) / len(values))


def _signature(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in (
        "engine_version", "evaluator_backend", "policy", "policy_parameters", "round_wind",
    )}


def _validate_rows(rows: Iterable[Mapping[str, Any]], expected_seed: int) -> list[Mapping[str, Any]]:
    result = list(rows)
    trials: set[int] = set()
    signature: dict[str, Any] | None = None
    for row in result:
        trial = int(row["trial_id"])
        if trial in trials:
            raise ValueError("duplicate trial id")
        trials.add(trial)
        if int(row["trial_seed"]) != sim.trial_seed(expected_seed, trial):
            raise ValueError("trial seed disagrees with expected seed and trial id")
        current = _signature(row)
        if signature is None:
            signature = current
        elif signature != current:
            raise ValueError("input mixes engine, policy, or round metadata")
    if signature is not None:
        reversal = signature["policy_parameters"].get("shanten_reversal", {})
        if reversal.get("enabled") is not True:
            raise ValueError("input policy must record enabled shanten reversal")
    return result


def _score_snapshot(row: Mapping[str, Any], snapshot: Mapping[str, Any], service: PushFoldScorer) -> dict[str, Any]:
    hand = tuple(map(int, snapshot["hand_physical"]))
    if (len(hand) != 13 or len(set(hand)) != 13
            or any(tile < 0 or tile >= 136 for tile in hand)):
        raise ValueError("first-tenpai snapshot must contain 13 distinct physical tiles")
    player_wind = int(row["player_wind"])
    request = {
        "hand_physical": hand,
        "seat": player_wind - 27,
        "round_wind": int(row["round_wind"]),
        "dora_indicator": int(row["dora_indicator"]),
    }
    if not 0 <= request["seat"] < 4:
        raise ValueError("player_wind must be a seat wind")
    ura = (int(row["ura_dora_indicator"]),)
    visible: list[int] = []
    live: list[int] = []
    wait_types: list[int] = []
    for wait in snapshot["waits"]:
        tile_type = int(wait["tile_type"])
        visible_ids = [int(tile) for tile in wait["visible_remaining_tile_ids"]]
        live_ids = [int(tile) for tile in wait["live_wall_remaining_tile_ids"]]
        if len(visible_ids) != int(wait["visible_remaining"]):
            raise ValueError("visible wait count lacks matching physical tile ids")
        if len(live_ids) != int(wait["live_wall_remaining"]):
            raise ValueError("live wait count lacks matching physical tile ids")
        if any(tile // 4 != tile_type for tile in (*visible_ids, *live_ids)):
            raise ValueError("wait physical tile type disagrees with its snapshot")
        visible.extend(visible_ids)
        live.extend(live_ids)
        wait_types.append(tile_type)
    if len(set(visible)) != len(visible) or len(set(live)) != len(live):
        raise ValueError("winning physical copies are duplicated across waits")
    if not set(live).issubset(visible):
        raise ValueError("live-wall winning copies must belong to the unseen winning-copy set")
    if set(visible) & set(hand):
        raise ValueError("winning physical copies cannot be held in the 13-tile hand")

    cache: dict[int, int] = {}
    def points(tile: int) -> int:
        if tile not in cache:
            score = service._score_request(
                request, tile, riichi=True, tsumo=False, ura=ura, is_dealer=False,
            )
            if not _valid(score):
                raise ValueError(f"invalid riichi ron score for trial {row['trial_id']}: {score.error}")
            assert score.points is not None
            cache[tile] = int(score.points)
        return cache[tile]

    counts = sim.counts_from_physical(hand)
    return {
        "trial_id": int(row["trial_id"]),
        "mean_ron_points": _mean([points(tile) for tile in visible]),
        "mean_live_ron_points": _mean([points(tile) for tile in live]),
        "waitcategory": primary_wait_category(counts, wait_types),
        "visible_winning_copies": len(visible),
        "live_winning_copies": len(live),
        "score_evaluations": len(cache),
    }


def score_rows(rows: Iterable[Mapping[str, Any]], expected_seed: int, scorer=None) -> dict[str, Any]:
    valid_rows = _validate_rows(rows, expected_seed)
    service = PushFoldScorer(scorer)
    trials: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    score_evaluations = 0
    first_tenpai_hands = 0
    for row in valid_rows:
        snapshot = row.get("first_tenpai")
        if snapshot is None:
            trials.append({
                "trial_id": int(row["trial_id"]), "mean_ron_points": None,
                "mean_live_ron_points": None, "waitcategory": None,
                "visible_winning_copies": 0, "live_winning_copies": 0, "score_evaluations": 0,
            })
            continue
        value = _score_snapshot(row, snapshot, service)
        trials.append(value)
        first_tenpai_hands += 1
        categories[value["waitcategory"]] += 1
        score_evaluations += value["score_evaluations"]
    visible_means = [float(item["mean_ron_points"]) for item in trials if item["mean_ron_points"] is not None]
    live_means = [float(item["mean_live_ron_points"]) for item in trials if item["mean_live_ron_points"] is not None]
    signature = _signature(valid_rows[0]) if valid_rows else None
    return {
        "inputs": {
            "expected_seed": expected_seed,
            "trial_id_min": min((item["trial_id"] for item in trials), default=None),
            "trial_id_max": max((item["trial_id"] for item in trials), default=None),
            "tables": len(trials), "source_signature": signature,
        },
        "scoring": {
            "primary": "each unseen winning physical copy, determined from visible information, has equal weight within its first-tenpai hand",
            "live_diagnostic": "each live-wall winning physical copy has equal weight within its first-tenpai hand",
            "context": "riichi non-dealer-normalized ron across all seats; actual seat wind is retained for yaku; sampled ura included; no ippatsu",
        },
        "diagnostics": {
            "first_tenpai_hands": first_tenpai_hands,
            "visible_scored_hands": len(visible_means), "live_scored_hands": len(live_means),
            "score_evaluations": score_evaluations, "scoring_errors": 0,
            "waitcategory_counts": dict(sorted(categories.items())),
        },
        "visible_copy_weighted": {"mean_ron_points": _mean(visible_means), "hand_clustered_se": _mean_se(visible_means)},
        "live_copy_weighted_diagnostic": {"mean_ron_points": _mean(live_means), "hand_clustered_se": _mean_se(live_means)},
        "trials": trials,
    }


def load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial-metrics", type=Path)
    parser.add_argument("--expected-seed", type=int, default=20260809)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    rows = load_jsonl(args.input)
    result = score_rows(rows, args.expected_seed)
    result["inputs"]["source_path"] = str(args.input)
    with args.input.open("rb") as stream:
        result["inputs"]["source_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    result["run"] = {
        "post_scoring_seconds": time.perf_counter() - started,
        "command": "score_first_tenpai_values.py --input RAW.jsonl --output VALUES.json [--trial-metrics TRIALS.jsonl] --expected-seed 20260809",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.trial_metrics:
        args.trial_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.trial_metrics.write_text("".join(json.dumps(item) + "\n" for item in result["trials"]), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
