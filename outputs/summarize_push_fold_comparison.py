#!/usr/bin/env python3
"""Summarize paired push/fold table-policy experiments.

All four seats use the named policy.  Net-point differences are therefore
descriptive table outcomes, not a focal-player EV estimate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


ENDING_REASONS = ("ron", "multiple_ron", "tsumo", "exhaustive_draw")


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def _mean_se(values: Iterable[float]) -> dict[str, float | int | None]:
    items = list(values)
    mean = _mean(items)
    if len(items) < 2:
        return {"n": len(items), "mean": mean, "standard_error": None}
    assert mean is not None
    variance = sum((value - mean) ** 2 for value in items) / (len(items) - 1)
    return {"n": len(items), "mean": mean, "standard_error": math.sqrt(variance / len(items))}


def load_aggregate(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or "tables" not in value or "diagnostics" not in value:
        raise ValueError(f"{path} is not a push/fold aggregate JSON")
    return value


def load_metrics(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        required = {"trial_id", "terminal_reason", "net_points", "riichi_seats", "payments", "winner_seats", "loser"}
        if not required.issubset(row):
            raise ValueError(f"{path} has a metrics row with missing fields")
        if len(row["net_points"]) != 4:
            raise ValueError("metrics net_points must contain four seats")
    return rows


def _metric_map(rows: Iterable[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    mapped: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        trial = int(row["trial_id"])
        if trial in mapped:
            raise ValueError(f"duplicate trial id {trial} in metrics")
        mapped[trial] = row
    return mapped


def _rate(count: int, tables: int) -> float | None:
    return count / tables if tables else None


def _danger_headline(aggregate: Mapping[str, Any], threats: int) -> dict[str, Any] | None:
    matches = [
        row for row in aggregate.get("danger_rows", [])
        if int(row["riichi_opponents"]) == threats
        and str(row["observer_turn"]) == "all"
        and row["ron_scope"] == "riichi_opponents"
        and row["tile_level"] == "all"
        and row["tile"] == "all"
        and row["safety_profile"] == "all"
    ]
    if len(matches) > 1:
        raise ValueError(f"multiple all-tile all-turn danger cells for {threats} threats")
    if not matches:
        return None
    row = matches[0]
    return {
        "candidate_opportunities": row.get("candidate_opportunities"),
        "candidate_ron_hit_probability": row.get("ron_hit_probability"),
        "candidate_table_clustered_se": row.get("table_clustered_se"),
        "chosen_opportunities": row.get("chosen_opportunities"),
        "chosen_ron_hit_probability": row.get("chosen_ron_hit_probability"),
        "chosen_table_clustered_se": row.get("chosen_table_clustered_se"),
    }


def summarize_policy(name: str, aggregate: Mapping[str, Any], metrics: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    metric_rows = list(metrics)
    metric_by_trial = _metric_map(metric_rows)
    tables = int(aggregate["tables"])
    if len(metric_by_trial) != tables:
        raise ValueError(f"{name}: aggregate tables and metric trial count differ")
    diagnostics = aggregate["diagnostics"]
    ending_counts = {reason: int(diagnostics.get(reason, 0)) for reason in ENDING_REASONS}
    realized = aggregate.get("realized_win_values", {})
    return {
        "name": name,
        "tables": tables,
        "ending_counts": ending_counts,
        "ending_rates": {reason: _rate(count, tables) for reason, count in ending_counts.items()},
        # A multiple ron is one table ending and also belongs in ron incidence.
        "ron_ending_rate": _rate(ending_counts["ron"] + ending_counts["multiple_ron"], tables),
        "riichi_declarations_per_table": _rate(int(diagnostics.get("riichi_declarations", 0)), tables),
        "mean_ron_points": realized.get("ron", {}).get("mean_points"),
        "mean_tsumo_points": realized.get("tsumo", {}).get("mean_points"),
        "seat_mean_net_points": aggregate.get("seat_mean_net_points"),
        "seat_net_points_se": aggregate.get("seat_net_points_se"),
        "seat_outcomes": aggregate.get("seat_outcomes"),
        "riichi_wait_distribution": aggregate.get("riichi_wait_distribution"),
        "riichi_wait_quality": aggregate.get("riichi_wait_quality"),
        "danger_by_riichi_opponents": {
            str(threats): _danger_headline(aggregate, threats) for threats in (1, 2, 3)
        },
    }


def paired_trial_differences(left: Iterable[Mapping[str, Any]], right: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return right-minus-left paired table means and standard errors."""
    left_by_trial = _metric_map(left)
    right_by_trial = _metric_map(right)
    if set(left_by_trial) != set(right_by_trial):
        raise ValueError("paired metrics must have exactly the same trial ids")
    ids = sorted(left_by_trial)
    seat_differences = [
        [float(right_by_trial[trial]["net_points"][seat]) - float(left_by_trial[trial]["net_points"][seat])
         for trial in ids]
        for seat in range(4)
    ]

    def indicator(row: Mapping[str, Any], kind: str) -> int:
        reason = str(row["terminal_reason"])
        if kind == "ron":
            return int(reason in ("ron", "multiple_ron"))
        if kind == "tsumo":
            return int(reason == "tsumo")
        if kind == "exhaustion":
            return int(reason == "exhaustive_draw")
        raise ValueError(kind)

    return {
        "direction": "right_minus_left",
        "paired_tables": len(ids),
        "seat_net_points": [_mean_se(values) for values in seat_differences],
        "outcome_indicator_differences": {
            kind: _mean_se([
                indicator(right_by_trial[trial], kind) - indicator(left_by_trial[trial], kind)
                for trial in ids
            ]) for kind in ("ron", "tsumo", "exhaustion")
        },
        "riichi_declarations": _mean_se([
            len(right_by_trial[trial]["riichi_seats"]) - len(left_by_trial[trial]["riichi_seats"])
            for trial in ids
        ]),
    }


def _named_paths(items: Iterable[str] | None, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError(f"{label} must use NAME=PATH")
        name, raw_path = item.split("=", 1)
        if not name or not raw_path or name in result:
            raise ValueError(f"invalid or duplicate {label} name")
        result[name] = Path(raw_path)
    return result


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Push/fold policy comparison",
        "",
        "All four seats use each named policy. Net-point differences describe paired table outcomes, not focal-player EV. Payments include dealer-aware ron and tsumo, riichi deposits, multiple-ron settlement, and exhaustive-draw transfers; ippatsu and honba are excluded.",
        "",
        "## Policy summaries",
        "",
        "| policy | tables | ron endings | multiple ron | tsumo | exhaustive draw | riichi/table | mean ron | mean tsumo |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in result["policies"]:
        rates = policy["ending_rates"]
        lines.append(
            f"| {policy['name']} | {policy['tables']} | {_fmt(policy['ron_ending_rate'])} | "
            f"{_fmt(rates['multiple_ron'])} | {_fmt(rates['tsumo'])} | {_fmt(rates['exhaustive_draw'])} | "
            f"{_fmt(policy['riichi_declarations_per_table'])} | {_fmt(policy['mean_ron_points'], 1)} | {_fmt(policy['mean_tsumo_points'], 1)} |"
        )
    lines += ["", "## All-tile danger by declared opponents", ""]
    for policy in result["policies"]:
        lines += [f"### {policy['name']}", "", "| opponents | candidate rate (SE) | chosen rate (SE) |", "| ---: | ---: | ---: |"]
        for threats, cell in policy["danger_by_riichi_opponents"].items():
            if cell is None:
                lines.append(f"| {threats} | — | — |")
            else:
                lines.append(
                    f"| {threats} | {_fmt(cell['candidate_ron_hit_probability'])} ({_fmt(cell['candidate_table_clustered_se'])}) | "
                    f"{_fmt(cell['chosen_ron_hit_probability'])} ({_fmt(cell['chosen_table_clustered_se'])}) |"
                )
        lines.append("")
    if result["paired_differences"]:
        lines += ["## Paired differences", "", "Each value is right minus left, with a paired-table standard error.", ""]
        for name, difference in result["paired_differences"].items():
            lines += [f"### {name}", "", f"Paired tables: {difference['paired_tables']}", ""]
            lines.append("| measure | mean difference | SE |")
            lines.append("| --- | ---: | ---: |")
            for seat, value in enumerate(difference["seat_net_points"]):
                lines.append(f"| seat {seat} net points | {_fmt(value['mean'])} | {_fmt(value['standard_error'])} |")
            for kind, value in difference["outcome_indicator_differences"].items():
                lines.append(f"| {kind} indicator | {_fmt(value['mean'])} | {_fmt(value['standard_error'])} |")
            value = difference["riichi_declarations"]
            lines.append(f"| riichi declarations | {_fmt(value['mean'])} | {_fmt(value['standard_error'])} |")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="NAME=AGGREGATE_JSON")
    parser.add_argument("--metrics", action="append", required=True, metavar="NAME=METRICS_JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Markdown output path")
    parser.add_argument("--json", type=Path, help="Optional machine-readable summary path")
    args = parser.parse_args(argv)
    inputs = _named_paths(args.input, "--input")
    metric_paths = _named_paths(args.metrics, "--metrics")
    if set(inputs) != set(metric_paths):
        parser.error("--input and --metrics must name exactly the same policies")

    aggregates = {name: load_aggregate(path) for name, path in inputs.items()}
    metrics = {name: load_metrics(metric_paths[name]) for name in inputs}
    policies = [summarize_policy(name, aggregates[name], metrics[name]) for name in inputs]
    paired = {}
    names = list(inputs)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            paired[f"{right}__minus__{left}"] = paired_trial_differences(metrics[left], metrics[right])
    result = {"policies": policies, "paired_differences": paired}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(result), encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
