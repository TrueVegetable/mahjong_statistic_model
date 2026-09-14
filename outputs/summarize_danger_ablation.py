#!/usr/bin/env python3
"""Summarize paired single-riichi danger-policy ablations.

Each named run supplies both its raw JSONL observations and the aggregate JSON
created by analyze_tile_danger_by_turn.py.  The raw stream is required for
whole-hand wait classification, copy-conditioned cells, and paired cluster
standard errors; the aggregate is used as an independent topline check.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

OUTPUTS = Path(__file__).resolve().parent
sys.path.insert(0, str(OUTPUTS))

import analyze_tile_danger_by_turn as danger  # noqa: E402
from hand_monte_carlo_report import primary_wait_category  # noqa: E402


CELL_DEFINITIONS = {
    "suji_28": ("28", "suji"),
    "double_suji_46": ("46", "double_suji"),
    "half_suji_46": ("46", "half_suji"),
    "no_suji_28": ("28", "no_suji"),
}
CONTRAST_DEFINITIONS = {
    "double_suji_46_minus_suji_28": ("double_suji_46", "suji_28"),
    "half_suji_46_minus_no_suji_28": ("half_suji_46", "no_suji_28"),
}


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected nonempty NAME=PATH")
    return name, Path(raw_path)


@dataclass
class NH:
    n: int = 0
    h: int = 0

    def add(self, hit: bool) -> None:
        self.n += 1
        self.h += int(hit)


@dataclass
class RunStats:
    name: str
    raw_path: Path
    aggregate_path: Path
    raw_sha256: str = ""
    aggregate_sha256: str = ""
    trials: set[int] = field(default_factory=set)
    trial_seeds: dict[int, int] = field(default_factory=dict)
    metadata: dict[str, Any] | None = None
    wait_categories: Counter[str] = field(default_factory=Counter)
    table_wait_category: dict[int, str | None] = field(default_factory=dict)
    kanchan_ranks: Counter[int] = field(default_factory=Counter)
    table_cells: dict[int, dict[str, NH]] = field(default_factory=dict)
    table_copy_cells: dict[int, dict[tuple[str, int, int], NH]] = field(default_factory=dict)

    @property
    def trial_count(self) -> int:
        return len(self.trials)


def _rate(cell: NH) -> float | None:
    return cell.h / cell.n if cell.n else None


def _cluster_rate(cells: Mapping[int, NH], trials: Iterable[int]) -> dict[str, Any]:
    ids = tuple(trials)
    total = NH(
        sum(cells.get(trial, NH()).n for trial in ids),
        sum(cells.get(trial, NH()).h for trial in ids),
    )
    p = _rate(total)
    se = None
    if p is not None and len(ids) > 1:
        residual = sum(
            (cells.get(trial, NH()).h - p * cells.get(trial, NH()).n) ** 2
            for trial in ids
        )
        se = math.sqrt(len(ids) / (len(ids) - 1) * residual) / total.n
    return {
        "candidate_opportunities": total.n,
        "ron_hits": total.h,
        "ron_hit_probability": p,
        "table_clustered_se": se,
        "independent_tables": len(ids),
        "exposed_tables": sum(cells.get(trial, NH()).n > 0 for trial in ids),
    }


def _contrast_components(
    table_cells: Mapping[int, Mapping[str, NH]], trials: Iterable[int],
    left: str, right: str,
) -> tuple[dict[str, Any], dict[int, float]]:
    ids = tuple(trials)
    left_cells = {trial: table_cells.get(trial, {}).get(left, NH()) for trial in ids}
    right_cells = {trial: table_cells.get(trial, {}).get(right, NH()) for trial in ids}
    left_total = NH(sum(x.n for x in left_cells.values()), sum(x.h for x in left_cells.values()))
    right_total = NH(sum(x.n for x in right_cells.values()), sum(x.h for x in right_cells.values()))
    left_rate, right_rate = _rate(left_total), _rate(right_total)
    if left_rate is None or right_rate is None:
        return ({"left": left, "right": right, "difference": None,
                 "paired_table_clustered_se": None}, {trial: 0.0 for trial in ids})
    influence = {
        trial: (
            (left_cells[trial].h - left_rate * left_cells[trial].n) / left_total.n
            - (right_cells[trial].h - right_rate * right_cells[trial].n) / right_total.n
        )
        for trial in ids
    }
    se = None
    if len(ids) > 1:
        se = math.sqrt(len(ids) / (len(ids) - 1) * sum(x * x for x in influence.values()))
    return ({
        "left": left,
        "right": right,
        "left_probability": left_rate,
        "right_probability": right_rate,
        "difference": left_rate - right_rate,
        "paired_table_clustered_se": se,
        "independent_tables": len(ids),
    }, influence)


def _load_run(name: str, raw_path: Path, aggregate_path: Path) -> RunStats:
    aggregate_bytes = aggregate_path.read_bytes()
    aggregate = json.loads(aggregate_bytes)
    stats = RunStats(name, raw_path, aggregate_path)
    stats.aggregate_sha256 = hashlib.sha256(aggregate_bytes).hexdigest()
    raw_hash = hashlib.sha256()
    with raw_path.open("rb") as stream:
        for raw_line in stream:
            raw_hash.update(raw_line)
            if not raw_line.strip():
                continue
            line = raw_line.decode("utf-8")
            row = json.loads(line)
            trial = int(row["trial_id"])
            if trial in stats.trials:
                raise ValueError(f"{name}: duplicate trial_id {trial}")
            stats.trials.add(trial)
            stats.trial_seeds[trial] = int(row["trial_seed"])
            metadata = {
                "engine_version": row.get("engine_version"),
                "evaluator_backend": row.get("evaluator_backend"),
                "policy": row.get("policy"),
                "policy_parameters": row.get("policy_parameters"),
                "round_wind": row.get("round_wind"),
            }
            if stats.metadata is None:
                stats.metadata = metadata
            elif stats.metadata != metadata:
                raise ValueError(f"{name}: inconsistent raw metadata")
            payload = row.get("danger")
            if not isinstance(payload, Mapping):
                raise ValueError(f"{name}: trial {trial} lacks danger payload")
            riichi_seat = int(payload["riichi_seat"])
            stats.table_wait_category[trial] = None
            if riichi_seat >= 0:
                hand = tuple(map(int, payload["riichi_hand_counts"]))
                waits = tuple(sorted(map(int, payload["winning_types"])))
                category = primary_wait_category(hand, waits)
                stats.wait_categories[category] += 1
                stats.table_wait_category[trial] = category
                if category == "kanchan":
                    for tile in waits:
                        if tile < 27:
                            stats.kanchan_ranks[tile % 9 + 1] += 1
            local_cells: dict[str, NH] = defaultdict(NH)
            local_copy: dict[tuple[str, int, int], NH] = defaultdict(NH)
            for decision in payload.get("decisions", ()):
                if int(decision["observer_turn"]) != 12:
                    continue
                own = tuple(map(int, decision["own_counts"]))
                public = tuple(map(int, decision["public_counts"]))
                safe = tuple(map(int, decision["safe_types"]))
                for candidate in decision["candidates"]:
                    tile = int(candidate["tile_type"])
                    group = danger.tile_group(tile)
                    safety = danger.safety_class(tile, safe)
                    hit = bool(candidate["ron_hit"])
                    for cell_name, definition in CELL_DEFINITIONS.items():
                        if (group, safety) != definition:
                            continue
                        local_cells[cell_name].add(hit)
                        local_copy[(cell_name, own[tile], public[tile])].add(hit)
            stats.table_cells[trial] = dict(local_cells)
            stats.table_copy_cells[trial] = dict(local_copy)
    stats.raw_sha256 = raw_hash.hexdigest()

    if int(aggregate.get("diagnostics", {}).get("tables", -1)) != stats.trial_count:
        raise ValueError(f"{name}: aggregate table count disagrees with raw stream")
    aggregate_policy = aggregate.get("run", {}).get("policy_parameters")
    if aggregate_policy is not None and stats.metadata is not None \
            and aggregate_policy != stats.metadata["policy_parameters"]:
        raise ValueError(f"{name}: aggregate policy parameters disagree with raw stream")
    indexed = {
        (str(row.get("turn")), row.get("tile_level"), row.get("tile"), row.get("safety")): row
        for row in aggregate.get("rows", ())
        if row.get("axis") == "observer_turn"
    }
    for cell_name, (group, safety) in CELL_DEFINITIONS.items():
        row = indexed.get(("12", "category", group, safety))
        raw_n = sum(x.get(cell_name, NH()).n for x in stats.table_cells.values())
        raw_h = sum(x.get(cell_name, NH()).h for x in stats.table_cells.values())
        if row is None or int(row["candidate_opportunities"]) != raw_n or int(row["ron_hits"]) != raw_h:
            raise ValueError(f"{name}: aggregate mismatch for {cell_name}")
    return stats


def _copy_conditioned(stats: RunStats) -> list[dict[str, Any]]:
    keys = sorted({key for cells in stats.table_copy_cells.values() for key in cells})
    rows = []
    for cell_name, own_count, public_count in keys:
        table = {
            trial: cells.get((cell_name, own_count, public_count), NH())
            for trial, cells in stats.table_copy_cells.items()
        }
        row = _cluster_rate(table, stats.trials)
        row.update(cell=cell_name, own_count=own_count, public_count=public_count)
        rows.append(row)
    return rows


def summarize(runs: list[RunStats]) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one run is required")
    baseline_ids = runs[0].trials
    baseline_seeds = runs[0].trial_seeds
    for run in runs[1:]:
        if run.trials != baseline_ids:
            raise ValueError(f"{run.name}: paired runs must contain identical trial ids")
        if run.trial_seeds != baseline_seeds:
            raise ValueError(f"{run.name}: paired trial ids have different trial seeds")
    output_runs: dict[str, Any] = {}
    influences: dict[str, dict[str, dict[int, float]]] = {}
    for run in runs:
        cells = {
            name: _cluster_rate(
                {trial: row.get(name, NH()) for trial, row in run.table_cells.items()}, run.trials
            )
            for name in CELL_DEFINITIONS
        }
        contrasts = {}
        influences[run.name] = {}
        for contrast_name, (left, right) in CONTRAST_DEFINITIONS.items():
            result, influence = _contrast_components(run.table_cells, run.trials, left, right)
            contrasts[contrast_name] = result
            influences[run.name][contrast_name] = influence
        riichi_hands = sum(run.wait_categories.values())
        kanchan_wins = sum(run.kanchan_ranks.values())
        output_runs[run.name] = {
            "raw": str(run.raw_path),
            "raw_sha256": run.raw_sha256,
            "aggregate": str(run.aggregate_path),
            "aggregate_sha256": run.aggregate_sha256,
            "trials": run.trial_count,
            "metadata": run.metadata,
            "riichi_hands": riichi_hands,
            "wait_categories": {
                category: {"hands": count, "share_of_riichi_hands": count / riichi_hands if riichi_hands else None}
                for category, count in sorted(run.wait_categories.items())
            },
            "kanchan_winning_rank_shares": {
                str(rank): {"winning_types": count, "share": count / kanchan_wins if kanchan_wins else None}
                for rank, count in sorted(run.kanchan_ranks.items())
            },
            "turn12_cells": cells,
            "turn12_contrasts": contrasts,
            "turn12_copy_conditioned": _copy_conditioned(run),
        }

    paired = []
    for baseline_index, baseline in enumerate(runs):
        for run in runs[baseline_index + 1:]:
            contrast_changes = {}
            for contrast_name in CONTRAST_DEFINITIONS:
                before = output_runs[baseline.name]["turn12_contrasts"][contrast_name]["difference"]
                after = output_runs[run.name]["turn12_contrasts"][contrast_name]["difference"]
                paired_influence = {
                    trial: influences[run.name][contrast_name][trial]
                    - influences[baseline.name][contrast_name][trial]
                    for trial in baseline_ids
                }
                se = math.sqrt(
                    len(baseline_ids) / (len(baseline_ids) - 1)
                    * sum(value * value for value in paired_influence.values())
                ) if len(baseline_ids) > 1 else None
                contrast_changes[contrast_name] = {
                    "baseline_difference": before,
                    "candidate_difference": after,
                    "paired_change": None if before is None or after is None else after - before,
                    "paired_table_clustered_se": se,
                }
            categories = sorted(set(baseline.wait_categories) | set(run.wait_categories))
            wait_changes = {}
            for category in categories:
                differences = [
                    int(run.table_wait_category[trial] == category)
                    - int(baseline.table_wait_category[trial] == category)
                    for trial in sorted(baseline_ids)
                ]
                mean = sum(differences) / len(differences)
                se = math.sqrt(
                    sum((value - mean) ** 2 for value in differences)
                    / (len(differences) * (len(differences) - 1))
                ) if len(differences) > 1 else None
                wait_changes[category] = {"paired_share_change_per_table": mean, "paired_se": se}
            paired.append({
                "baseline": baseline.name,
                "candidate": run.name,
                "turn12_contrast_changes": contrast_changes,
                "wait_category_changes": wait_changes,
            })
    return {
        "definitions": {
            "turn": "observer own turn 12",
            "grain": "one distinct held candidate tile type per defender decision",
            "contrasts": CONTRAST_DEFINITIONS,
            "copy_conditioning": "candidate tile's observer-hand copy count and public copy count",
            "uncertainty": "independent-table cluster standard errors; policy comparisons pair exact trial ids",
        },
        "runs": output_runs,
        "paired_run_comparisons": paired,
        "paired_against_first_run": [row for row in paired if row["baseline"] == runs[0].name],
    }


def markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# Single-riichi danger ablation", "", "## Turn-12 ordering contrasts", "",
             "| Run | Double-suji 4/6 − suji 2/8 | SE | Half-suji 4/6 − no-suji 2/8 | SE |",
             "|---|---:|---:|---:|---:|"]
    fmt = lambda value: "—" if value is None else f"{100 * value:.3f} pp"
    for name, run in summary["runs"].items():
        first = run["turn12_contrasts"]["double_suji_46_minus_suji_28"]
        second = run["turn12_contrasts"]["half_suji_46_minus_no_suji_28"]
        lines.append(f"| {name} | {fmt(first['difference'])} | {fmt(first['paired_table_clustered_se'])} | "
                     f"{fmt(second['difference'])} | {fmt(second['paired_table_clustered_se'])} |")
    lines.extend(["", "## Paired changes in ordering contrasts", "",
                  "| Baseline | Candidate | Change in first contrast | Paired SE | Change in second contrast | Paired SE |",
                  "|---|---|---:|---:|---:|---:|"])
    for comparison in summary["paired_run_comparisons"]:
        first = comparison["turn12_contrast_changes"]["double_suji_46_minus_suji_28"]
        second = comparison["turn12_contrast_changes"]["half_suji_46_minus_no_suji_28"]
        lines.append(
            f"| {comparison['baseline']} | {comparison['candidate']} | {fmt(first['paired_change'])} | "
            f"{fmt(first['paired_table_clustered_se'])} | {fmt(second['paired_change'])} | "
            f"{fmt(second['paired_table_clustered_se'])} |"
        )
    lines.extend(["", "## Wait categories", ""])
    categories = sorted({category for run in summary["runs"].values() for category in run["wait_categories"]})
    lines.append("| Run | " + " | ".join(categories) + " |")
    lines.append("|---|" + "---:|" * len(categories))
    for name, run in summary["runs"].items():
        values = []
        for category in categories:
            item = run["wait_categories"].get(category)
            values.append("0" if item is None else f"{item['hands']} ({100 * item['share_of_riichi_hands']:.2f}%)")
        lines.append("| " + name + " | " + " | ".join(values) + " |")
    lines.extend(["", "## Kanchan winning ranks", ""])
    ranks = sorted({rank for run in summary["runs"].values() for rank in run["kanchan_winning_rank_shares"]}, key=int)
    lines.append("| Run | " + " | ".join(ranks) + " |")
    lines.append("|---|" + "---:|" * len(ranks))
    for name, run in summary["runs"].items():
        values = []
        for rank in ranks:
            item = run["kanchan_winning_rank_shares"].get(rank)
            values.append("0" if item is None else f"{item['winning_types']} ({100 * item['share']:.2f}%)")
        lines.append("| " + name + " | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--raw", action="append", type=named_path, required=True, metavar="NAME=PATH")
    result.add_argument("--aggregate", action="append", type=named_path, required=True, metavar="NAME=PATH")
    result.add_argument("--output", type=Path)
    result.add_argument("--markdown", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    raw = dict(args.raw)
    aggregates = dict(args.aggregate)
    if len(raw) != len(args.raw) or len(aggregates) != len(args.aggregate):
        raise ValueError("run names must be unique")
    if raw.keys() != aggregates.keys():
        raise ValueError("--raw and --aggregate must use the same run names")
    runs = [_load_run(name, path, aggregates[name]) for name, path in raw.items()]
    result = summarize(runs)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    if args.markdown:
        args.markdown.write_text(markdown(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
