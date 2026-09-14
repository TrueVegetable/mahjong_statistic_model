#!/usr/bin/env python3
"""Extract non-genbutsu honor danger by visible copies from push/fold JSONL.

The input is the native decision stream, not the compact aggregate.  This is
intentional: grouping cells after their table-clustered standard errors have
been calculated gives an invalid uncertainty estimate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


TURNS = (6, 9, 12, 15)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clustered_rate(values: list[int], groups: int) -> dict:
    """Return the pooled ratio and the source experiment's table-clustered SE."""
    n, h, decisions, exposed, h2, hn, n2 = values
    probability = h / n if n else None
    if not n or groups <= 1:
        se = None
    else:
        se = math.sqrt(max(0.0, groups / (groups - 1) *
                           (h2 - 2 * probability * hn + probability * probability * n2))) / n
    return {
        "candidate_opportunities": n,
        "ron_hits": h,
        "ron_hit_probability": probability,
        "table_clustered_se": se,
        "independent_tables": groups,
        "exposed_tables": exposed,
        "exposed_decisions": decisions,
    }


def add_table_moments(table_cells: dict[tuple, list[int]], totals: dict[tuple, list[int]]) -> None:
    for key, (n, h, decisions) in table_cells.items():
        cell = totals[key]
        for index, value in enumerate((n, h, decisions, 1, h * h, h * n, n * n)):
            cell[index] += value


def rows_from_raw(path: Path, visible_definition: str) -> tuple[dict[tuple, list[int]], set[int], int, list[dict], dict]:
    """Aggregate one distinct held honor type per defender decision and table."""
    totals: dict[tuple, list[int]] = defaultdict(lambda: [0] * 7)
    trial_ids: set[int] = set()
    invalid_physical_states = 0
    per_table: list[dict] = []
    first_row: dict | None = None
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            table = json.loads(line)
            if first_row is None:
                first_row = table
            trial = int(table["trial_id"])
            if trial in trial_ids:
                raise ValueError(f"duplicate trial_id {trial}")
            trial_ids.add(trial)
            local: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0])
            for decision in table["push_fold"]["decisions"]:
                threats = {int(seat) for seat in decision["threat_seats"]}
                if len(threats) != 1:
                    continue
                turn = int(decision["turn"])
                if turn not in TURNS:
                    continue
                own = list(map(int, decision["own_counts"]))
                public = list(map(int, decision["public_counts"]))
                if len(own) != 34 or len(public) != 34 or sum(own) != 14:
                    raise ValueError(f"invalid counts in trial {trial}, turn {turn}")
                safe_entries = {int(entry["threat_seat"]): set(map(int, entry["types"]))
                                for entry in decision["safe_types"]}
                if set(safe_entries) != threats:
                    raise ValueError(f"safe types do not match declared opponent in trial {trial}")
                safe = safe_entries[next(iter(threats))]
                by_type: dict[int, set[int]] = {}
                for candidate in decision["candidates"]:
                    tile = int(candidate["tile_type"])
                    hits = set(map(int, candidate["ron_hits"]))
                    if tile in by_type and hits != by_type[tile]:
                        raise ValueError(f"physical copies disagree on ron eligibility in trial {trial}, tile {tile}")
                    by_type[tile] = hits
                if set(by_type) != {tile for tile, copies in enumerate(own) if copies}:
                    raise ValueError(f"candidate types do not match held types in trial {trial}")
                touched: set[tuple] = set()
                for tile, hits in by_type.items():
                    if tile < 27 or tile in safe:
                        continue
                    if own[tile] + public[tile] > 4:
                        invalid_physical_states += 1
                        raise ValueError(f"own + public exceeds four in trial {trial}, tile {tile}")
                    visible = public[tile] if visible_definition == "public-only" else own[tile] + public[tile]
                    # A candidate is held, so the own-plus-public definition starts at one.
                    if not (0 <= visible <= 4):
                        raise ValueError(f"invalid visible count in trial {trial}, tile {tile}")
                    hit = int(bool(hits & threats))
                    # Primary visible bin plus the requested own/public diagnostic coordinate.
                    for key in ((turn, visible), (turn, visible, own[tile], public[tile])):
                        local[key][0] += 1
                        local[key][1] += hit
                        touched.add(key)
                for key in touched:
                    local[key][2] += 1
            add_table_moments(local, totals)
            # Keep coordinate-level sufficient statistics, including empty tables.  A
            # later grouping can then recompute its cluster SE from table totals.
            per_table.append({"trial_id": trial, "cells": [
                {"observer_turn": key[0], "visible_copies": key[1],
                 "own_copies": key[2], "public_copies": key[3],
                 "candidate_opportunities": values[0], "ron_hits": values[1],
                 "exposed_decisions": values[2]}
                for key, values in sorted(local.items()) if len(key) == 4
            ]})
    if first_row is None:
        raise ValueError("raw stream contains no tables")
    return totals, trial_ids, invalid_physical_states, per_table, first_row


def make_rows(totals: dict[tuple, list[int]], groups: int, coordinate: bool) -> list[dict]:
    rows = []
    for key in sorted(totals):
        turn, visible, *coordinates = key
        row = {"observer_turn": turn, "visible_copies": visible,
               "own_copies": coordinates[0] if coordinate else None,
               "public_copies": coordinates[1] if coordinate else None,
               **clustered_rate(totals[key], groups)}
        rows.append(row)
    return rows


def reconcile_baseline(primary: list[dict], baseline: Path | None, trials: set[int], first_row: dict) -> dict:
    """Check H/N against the saved unsplit honor rows; SEs are never combined."""
    if baseline is None:
        return {"status": "not_requested"}
    data = json.loads(baseline.read_text())
    if (data["tables"] != len(trials) or data["trial_id_min"] != min(trials) or
            data["trial_id_max"] != max(trials)):
        raise ValueError("raw stream trial count or range disagrees with baseline")
    raw_metadata = {key: first_row.get(key) for key in
                    ("engine_version", "evaluator_backend", "policy", "policy_parameters", "round_wind")}
    if raw_metadata != data["source_metadata"]:
        raise ValueError("raw stream engine/policy metadata disagrees with baseline")
    expected = {}
    for row in data["danger_rows"]:
        if (row["riichi_opponents"] == 1 and row["ron_scope"] == "riichi_opponents" and
                row["tile_level"] == "category" and row["tile"] == "honor" and
                row["safety_profile"] == "honor" and row["observer_turn"] != "all"):
            expected[int(row["observer_turn"])] = (row["candidate_opportunities"], row["ron_hits"])
    observed = {turn: (sum(row["candidate_opportunities"] for row in primary if row["observer_turn"] == turn),
                       sum(row["ron_hits"] for row in primary if row["observer_turn"] == turn))
                for turn in TURNS}
    comparison = {str(turn): {"expected": expected.get(turn), "observed": observed[turn]}
                  for turn in TURNS}
    failures = {turn: values for turn, values in comparison.items()
                if values["expected"] is None or tuple(values["expected"]) != values["observed"]}
    if failures:
        raise ValueError(f"subcells do not reconcile to unsplit honor baseline: {failures}")
    return {"status": "passed", "source": str(baseline), "cells": comparison}


def write_outputs(stem: Path, primary: list[dict], diagnostic: list[dict], metadata: dict,
                  per_table: list[dict], visible_bins: range) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps({**metadata, "rows": primary,
        "own_public_coordinate_rows": diagnostic}, ensure_ascii=False, indent=2) + "\n")
    fields = list(primary[0]) if primary else ["observer_turn", "visible_copies"]
    with stem.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(primary)
    lines = ["# 非现物字牌按可见张数的危险度", "",
             f"每格是面对恰好一家立直对手时的持有候选牌种荣和命中率；候选单位是每次防守决策中的一个不同持有牌种。标准误按 {metadata['tables']} 张独立牌桌聚类，包含零暴露牌桌。", "",
             "| 观察者巡目 / 可见张数 | " + " | ".join(str(n) for n in visible_bins) + " |",
             "|---|" + "|".join("---:" for _ in visible_bins) + "|"]
    for turn in TURNS:
        cells = []
        for visible in visible_bins:
            row = next((row for row in primary if row["observer_turn"] == turn and row["visible_copies"] == visible), None)
            cells.append("—" if row is None else f"{100 * row['ron_hit_probability']:.2f}%（{row['candidate_opportunities']}）")
        lines.append(f"| {turn} | " + " | ".join(cells) + " |")
    lines += ["", f"可见张数定义：{metadata['visible_copies_note']}。观察者已经立直的决策仍计入；现物字牌不计入。",
              "括号为候选观察数。JSON 同时保留 own_copies × public_copies 坐标、命中数、暴露桌数和聚类标准误。",
              "", "零命中是本组样本结果，不等于普遍安全；国士无双可能等待手中没有的字牌，不能仅凭四张均已知就排除所有荣和。"]
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    with stem.with_name(stem.name + ".trials.jsonl").open("w", encoding="utf-8") as stream:
        for row in per_table:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_jsonl", type=Path, help="native push/fold decision stream")
    parser.add_argument("--visible-copies", choices=("own-plus-public", "public-only"), default="own-plus-public")
    parser.add_argument("--output-stem", type=Path,
                        default=Path(__file__).resolve().parent / "push_fold_experiments/20260909/honor_danger_by_visible_copies")
    parser.add_argument("--baseline", type=Path,
                        default=Path(__file__).resolve().parent / "push_fold_experiments/20260909/mixed_mean_20k.json",
                        help="saved aggregate whose unsplit honor H/N must match")
    parser.add_argument("--no-baseline-check", action="store_true")
    args = parser.parse_args()
    totals, trials, invalid, per_table, first_row = rows_from_raw(args.raw_jsonl, args.visible_copies)
    if not trials:
        raise ValueError("raw stream contains no tables")
    primary = make_rows({key: value for key, value in totals.items() if len(key) == 2}, len(trials), False)
    diagnostic = make_rows({key: value for key, value in totals.items() if len(key) == 4}, len(trials), True)
    reconciliation = reconcile_baseline(primary, None if args.no_baseline_check else args.baseline, trials, first_row)
    baseline_run = {} if args.no_baseline_check else json.loads(args.baseline.read_text()).get("run", {})
    visible_bins = range(1, 5) if args.visible_copies == "own-plus-public" else range(0, 5)
    visible_note = ("弃牌前自家手牌加全部公开张数；若宝牌指示牌为该牌种，也计入公开张数"
                    if args.visible_copies == "own-plus-public" else "弃牌前全部公开张数；不含自家手牌")
    source_metadata = {key: first_row.get(key) for key in
                       ("engine_version", "evaluator_backend", "policy", "policy_parameters", "round_wind")}
    source_policy = {key: first_row["push_fold"].get(key) for key in
                     ("mode", "safety_strategy", "dora_danger_steps", "half_push_max_tier", "two_riichi_bad_wait_min_ron")}
    metadata = {
        "source": str(args.raw_jsonl), "source_sha256": sha256_file(args.raw_jsonl),
        "source_metadata": source_metadata, "source_policy": source_policy,
        "baseline": None if args.no_baseline_check else str(args.baseline),
        "baseline_sha256": None if args.no_baseline_check else sha256_file(args.baseline),
        "base_seed": baseline_run.get("native_summary", {}).get("seed"),
        "scoring_config": baseline_run.get("scoring_config"),
        "tables": len(trials), "trial_id_min": min(trials), "trial_id_max": max(trials),
        "selection": "exactly one declared opponent; riichi_opponents ron scope; non-genbutsu honor; distinct held tile type per decision; observer already riichi is retained",
        "turns": list(TURNS), "visible_copies_definition": args.visible_copies,
        "visible_copies_note": visible_note,
        "uncertainty": "pooled ratio with independent-table cluster standard error; zero-exposure tables included",
        "physical_conservation_failures": invalid,
        "unsplit_honor_baseline_reconciliation": reconciliation,
        "per_table_sufficient_statistics": args.output_stem.with_name(
            args.output_stem.name + ".trials.jsonl").name,
        "per_table_statistics_limit": "coordinate totals preserve exact pooled rates and cluster SE for any union of own/public coordinates; exposed_decisions must not be summed across coordinates because one decision can appear in more than one coordinate",
    }
    write_outputs(args.output_stem, primary, diagnostic, metadata, per_table, visible_bins)
    print(f"validated {len(trials)} tables; wrote {args.output_stem}.md/.csv/.json and trial sufficient-statistics JSONL")


if __name__ == "__main__":
    main()
