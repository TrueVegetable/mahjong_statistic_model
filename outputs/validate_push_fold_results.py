#!/usr/bin/env python3
"""Validate the four paired 20k push/fold result aggregates and trial ledgers."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_DIRECTORY = Path("outputs/push_fold_experiments/20260909")
RUNS = {
    "all_push": ("all_push", "mean_safety"),
    "full_fold": ("full_fold", "mean_safety"),
    "mixed_mean": ("mixed", "mean_safety"),
    "mixed_minimax": ("mixed", "minimax_safety"),
}
TERMINALS = ("ron", "multiple_ron", "tsumo", "exhaustive_draw")
COUNT_FIELDS = ("candidate_opportunities", "ron_hits", "chosen_opportunities", "chosen_ron_hits")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: float | None, right: float | None, tolerance: float = 1e-10) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


class Validation:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.failures: list[str] = []

    def check(self, condition: bool, name: str, details: Any = None) -> None:
        row = {"name": name, "passed": bool(condition)}
        if details is not None:
            row["details"] = details
        self.checks.append(row)
        if not condition:
            self.failures.append(name)


def expected_policy(parameters: Mapping[str, Any]) -> bool:
    by_shanten = parameters.get("future_ukeire_weights_by_shanten", {})
    reversal = parameters.get("shanten_reversal", {})
    return (
        close(parameters.get("future_ukeire_weight"), 0.5)
        and close(by_shanten.get("1"), 2)
        and close(by_shanten.get("2"), 1)
        and close(by_shanten.get("3_plus"), 0)
        and close(parameters.get("improvement_weight"), 0.1)
        and close(parameters.get("dora_ukeire_weight"), 1)
        and close(parameters.get("dora_tiebreak_weight"), 1)
        and close(parameters.get("dora_neighbor_weight"), 1)
        and close(parameters.get("yakuhai_weight"), 1)
        and close(parameters.get("tanyao_weight"), 1)
        and close(parameters.get("kanchan_outer_wait_weight", 0), 0)
        and reversal.get("enabled") is True
        and close(reversal.get("penalty"), 25)
        and close(reversal.get("recovered_ukeire_weight"), 0.5)
        and reversal.get("max_turn") == 9
        and reversal.get("candidate_limit") == 2
        and reversal.get("max_per_hand") == 1
    )


def expected_scoring(config: Mapping[str, Any]) -> bool:
    return config == {
        "cheap_max_total_han": 1,
        "valuable_bad_wait_points": 5200,
        "two_riichi_bad_wait_min_ron": 7700,
        "enable_haneman_lock": True,
    }


def load_trials(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows = []
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            if raw.strip():
                rows.append(json.loads(raw))
    return rows, digest.hexdigest()


def trial_reconciliation(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    terminal = Counter()
    seats = [Counter() for _ in range(4)]
    net_sum = [0.0] * 4
    net_sum2 = [0.0] * 4
    declarations = 0
    carried = 0
    values: dict[str, list[int]] = {"ron": [], "tsumo": []}
    conservation_failures = []
    for row in rows:
        reason = str(row["terminal_reason"])
        terminal[reason] += 1
        net = list(map(int, row["net_points"]))
        pot = int(row["carried_riichi_pot"])
        if len(net) != 4 or sum(net) + pot != 0:
            conservation_failures.append(int(row["trial_id"]))
        declared = set(map(int, row["riichi_seats"]))
        winners = set(map(int, row["winner_seats"]))
        loser = int(row["loser"])
        tenpai = set(map(int, row["tenpai_seats"]))
        declarations += len(declared)
        carried += pot
        for seat, value in enumerate(net):
            net_sum[seat] += value
            net_sum2[seat] += value * value
            seats[seat]["tables"] += 1
            seats[seat]["riichi"] += seat in declared
            seats[seat]["deal_in"] += loser == seat
            seats[seat]["ron"] += seat in winners and loser >= 0
            seats[seat]["tsumo"] += seat in winners and loser < 0
            seats[seat]["exhaustive_draw"] += not winners
            seats[seat]["exhaustive_draw_tenpai"] += not winners and seat in tenpai
        for payment in row["payments"]:
            values["tsumo" if payment["is_tsumo"] else "ron"].append(int(payment["points"]))
    count = len(rows)
    means = [value / count for value in net_sum]
    standard_errors = [
        math.sqrt(max(0.0, (square - total * total / count) / (count - 1) / count))
        for total, square in zip(net_sum, net_sum2)
    ] if count > 1 else []
    return {
        "terminal": terminal,
        "seat_outcomes": [dict(value) for value in seats],
        "seat_mean_net_points": means,
        "seat_net_points_se": standard_errors,
        "riichi_declarations": declarations,
        "carried_riichi_pot": carried,
        "realized_win_values": {
            kind: {"wins": len(points), "mean_points": sum(points) / len(points) if points else None}
            for kind, points in values.items()
        },
        "conservation_failures": conservation_failures,
    }


def row_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    materialized = list(rows)
    return {field: sum(int(row[field]) for row in materialized) for field in COUNT_FIELDS}


def same_counts(row: Mapping[str, Any], counts: Mapping[str, int]) -> bool:
    return all(int(row[field]) == counts[field] for field in COUNT_FIELDS)


def validate_rate_rows(validation: Validation, name: str, rows: list[Mapping[str, Any]]) -> None:
    valid = True
    for row in rows:
        n, h = int(row["candidate_opportunities"]), int(row["ron_hits"])
        cn, ch = int(row["chosen_opportunities"]), int(row["chosen_ron_hits"])
        valid &= 0 <= h <= n and 0 <= ch <= cn <= n
        valid &= close(row["ron_hit_probability"], h / n if n else None)
        valid &= close(row["chosen_ron_hit_probability"], ch / cn if cn else None)
        valid &= int(row["independent_tables"]) == 20000
    validation.check(valid, f"{name}: row rates and denominators")


def validate_danger(validation: Validation, name: str, aggregate: Mapping[str, Any]) -> None:
    rows = list(aggregate["danger_rows"])
    validate_rate_rows(validation, f"{name}: primary danger", rows)
    index = {
        (int(row["riichi_opponents"]), str(row["observer_turn"]), row["ron_scope"],
         row["tile_level"], row["tile"], row["safety_profile"]): row
        for row in rows
    }
    tile_names = {
        row["tile"] for row in rows
        if int(row["riichi_opponents"]) == 1 and str(row["observer_turn"]) == "all"
        and row["ron_scope"] == "riichi_opponents" and row["tile_level"] == "tile"
        and row["safety_profile"] == "all"
    }
    validation.check(len(tile_names) == 34, f"{name}: all 34 exact tile identities covered",
                     {"tile_count": len(tile_names)})
    available_threats = sorted({
        int(row["riichi_opponents"]) for row in rows
        if str(row["observer_turn"]) == "all"
        and row["ron_scope"] == "riichi_opponents"
        and row["tile_level"] == "all" and row["tile"] == "all"
        and row["safety_profile"] == "all"
    })
    expected_threats = [1] if name == "full_fold" else [1, 2, 3]
    validation.check(
        available_threats == expected_threats,
        f"{name}: policy-consistent threat-count coverage",
        {"actual": available_threats, "expected": expected_threats},
    )
    for threats in available_threats:
        for scope in ("riichi_opponents", "any_opponent"):
            base = index[(threats, "all", scope, "all", "all", "all")]
            exact = [row for row in rows if int(row["riichi_opponents"]) == threats
                     and str(row["observer_turn"]) == "all" and row["ron_scope"] == scope
                     and row["tile_level"] == "tile" and row["safety_profile"] == "all"]
            categories = [row for row in rows if int(row["riichi_opponents"]) == threats
                          and str(row["observer_turn"]) == "all" and row["ron_scope"] == scope
                          and row["tile_level"] == "category" and row["safety_profile"] == "all"]
            turns = [row for row in rows if int(row["riichi_opponents"]) == threats
                     and str(row["observer_turn"]) != "all" and row["ron_scope"] == scope
                     and row["tile_level"] == "all" and row["tile"] == "all"
                     and row["safety_profile"] == "all"]
            profiles = [row for row in rows if int(row["riichi_opponents"]) == threats
                        and str(row["observer_turn"]) == "all" and row["ron_scope"] == scope
                        and row["tile_level"] == "all" and row["tile"] == "all"
                        and row["safety_profile"] != "all"]
            validation.check(same_counts(base, row_counts(exact)),
                             f"{name}: {threats} threats {scope} exact-tile reconciliation")
            validation.check(same_counts(base, row_counts(categories)),
                             f"{name}: {threats} threats {scope} category reconciliation")
            validation.check(same_counts(base, row_counts(turns)),
                             f"{name}: {threats} threats {scope} turn reconciliation")
            validation.check(same_counts(base, row_counts(profiles)),
                             f"{name}: {threats} threats {scope} safety-profile reconciliation")
        declared = index[(threats, "all", "riichi_opponents", "all", "all", "all")]
        any_hit = index[(threats, "all", "any_opponent", "all", "all", "all")]
        validation.check(
            int(declared["candidate_opportunities"]) == int(any_hit["candidate_opportunities"])
            and int(declared["chosen_opportunities"]) == int(any_hit["chosen_opportunities"])
            and int(any_hit["ron_hits"]) >= int(declared["ron_hits"])
            and int(any_hit["chosen_ron_hits"]) >= int(declared["chosen_ron_hits"]),
            f"{name}: {threats} threats any-hit dominates declared-hit with common denominators",
        )
    for threats in available_threats:
        pure_safe = [
            row for row in rows if int(row["riichi_opponents"]) == threats
            and row["ron_scope"] == "riichi_opponents"
            and row["safety_profile"] != "all"
            and set(str(row["safety_profile"]).split("+")) == {"genbutsu"}
        ]
        validation.check(
            bool(pure_safe)
            and all(int(row["ron_hits"]) == 0 and int(row["chosen_ron_hits"]) == 0
                    for row in pure_safe),
            f"{name}: {threats} threats pure-genbutsu profiles have zero hits",
            {"rows": len(pure_safe)},
        )

    physical = list(aggregate["physical_conditioned_danger_rows"])
    validate_rate_rows(validation, f"{name}: physical danger", physical)
    for threats in available_threats:
        tier_zero = [
            row for row in physical if int(row["riichi_opponents"]) == threats
            and set(str(row["adjusted_tier_profile"]).split("+")) == {"0"}
        ]
        validation.check(
            bool(tier_zero)
            and all(int(row["ron_hits"]) == 0 and int(row["chosen_ron_hits"]) == 0
                    for row in tier_zero),
            f"{name}: {threats} threats physical all-tier-zero profiles have zero hits",
            {"rows": len(tier_zero)},
        )
        physical_counts = row_counts(row for row in physical if int(row["riichi_opponents"]) == threats)
        declared = index[(threats, "all", "riichi_opponents", "all", "all", "all")]
        validation.check(
            physical_counts["chosen_opportunities"] == int(declared["chosen_opportunities"])
            and physical_counts["chosen_ron_hits"] == int(declared["chosen_ron_hits"]),
            f"{name}: {threats} threats physical chosen totals reconcile to primary",
        )


def validate_run(
    validation: Validation, name: str, aggregate: Mapping[str, Any], trials: list[Mapping[str, Any]],
    expected_mode: str, expected_strategy: str,
) -> None:
    prefix = f"{name}"
    ids = [int(row["trial_id"]) for row in trials]
    validation.check(len(ids) == 20000 and len(set(ids)) == 20000 and set(ids) == set(range(20000)),
                     f"{prefix}: exact unique trial ids 0..19999")
    validation.check(aggregate["tables"] == 20000 and aggregate["trial_id_min"] == 0
                     and aggregate["trial_id_max"] == 19999, f"{prefix}: aggregate trial bounds")
    source = aggregate["source_policy"]
    validation.check(
        source == {"mode": expected_mode, "safety_strategy": expected_strategy,
                   "dora_danger_steps": 1, "half_push_max_tier": 5,
                   "two_riichi_bad_wait_min_ron": 7700},
        f"{prefix}: source push-fold policy",
    )
    metadata = aggregate["source_metadata"]
    validation.check(metadata["evaluator_backend"] == "necessary_tiles_masks"
                     and metadata["policy"] == "shanten_then_value_ukeire"
                     and metadata["round_wind"] == 27
                     and expected_policy(metadata["policy_parameters"]),
                     f"{prefix}: extremeA source metadata")
    validation.check(expected_scoring(aggregate["run"]["scoring_config"]),
                     f"{prefix}: scoring configuration")
    native = aggregate["run"]["native_summary"]
    validation.check(native["seed"] == 20260809 and native["start_trial"] == 0
                     and native["trials"] == 20000, f"{prefix}: native seed and trial range")
    reconciled = trial_reconciliation(trials)
    validation.check(not reconciled["conservation_failures"], f"{prefix}: every trial conserves points plus pot",
                     {"failures": reconciled["conservation_failures"][:10]})
    terminal_total = sum(int(aggregate["diagnostics"].get(key, 0)) for key in TERMINALS)
    validation.check(terminal_total == 20000, f"{prefix}: terminal outcome counts sum to tables")
    validation.check(all(int(aggregate["diagnostics"].get(key, 0)) == reconciled["terminal"].get(key, 0)
                         for key in TERMINALS), f"{prefix}: terminal outcomes reconcile to trials")
    wait_total = sum(map(int, aggregate["riichi_wait_distribution"].values()))
    validation.check(wait_total == int(aggregate["diagnostics"]["riichi_declarations"])
                     == reconciled["riichi_declarations"],
                     f"{prefix}: declarations reconcile to waits and per-trial seats")
    validation.check(int(aggregate["diagnostics"]["carried_riichi_pot"])
                     == reconciled["carried_riichi_pot"], f"{prefix}: carried pot reconciliation")
    validation.check(aggregate["seat_outcomes"] == reconciled["seat_outcomes"],
                     f"{prefix}: seat outcomes reconcile")
    validation.check(all(close(a, b) for a, b in zip(
        aggregate["seat_mean_net_points"], reconciled["seat_mean_net_points"])),
        f"{prefix}: seat mean points reconcile")
    validation.check(all(close(a, b) for a, b in zip(
        aggregate["seat_net_points_se"], reconciled["seat_net_points_se"])),
        f"{prefix}: seat point standard errors reconcile")
    values_ok = all(
        int(aggregate["realized_win_values"][kind]["wins"])
        == reconciled["realized_win_values"][kind]["wins"]
        and close(aggregate["realized_win_values"][kind]["mean_points"],
                  reconciled["realized_win_values"][kind]["mean_points"])
        for kind in ("ron", "tsumo")
    )
    validation.check(values_ok, f"{prefix}: realized payment counts and means reconcile")
    validate_danger(validation, name, aggregate)


def relevant_source_hashes(root: Path) -> dict[str, str]:
    paths = [
        root / "outputs/hand_efficiency_sim.cpp",
        root / "outputs/hand_push_fold_native.inc",
        root / "outputs/run_push_fold_experiments.py",
        root / "outputs/hand_push_fold_scoring.py",
        root / "outputs/hand_push_fold_policy.py",
    ]
    paths.extend(sorted((root / "tests").glob("test_*push_fold*.py")))
    return {str(path.relative_to(root)): sha256(path) for path in paths if path.exists()}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    result.add_argument("--output", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    directory = args.directory
    output = args.output or directory / "validation.json"
    root = Path(__file__).resolve().parents[1]
    validation = Validation()
    aggregates: dict[str, dict[str, Any]] = {}
    ledgers: dict[str, list[dict[str, Any]]] = {}
    input_hashes: dict[str, dict[str, str]] = {}
    paired_ids = None
    for name, (mode, strategy) in RUNS.items():
        aggregate_path = directory / f"{name}_20k.json"
        ledger_path = directory / f"{name}_20k.trials.jsonl"
        aggregates[name] = json.loads(aggregate_path.read_text(encoding="utf-8"))
        ledgers[name], ledger_hash = load_trials(ledger_path)
        input_hashes[name] = {
            "aggregate": sha256(aggregate_path),
            "trial_metrics": ledger_hash,
            "recorded_observations": aggregates[name]["run"]["observations_sha256"],
        }
        current_ids = [int(row["trial_id"]) for row in ledgers[name]]
        if paired_ids is None:
            paired_ids = current_ids
        validation.check(current_ids == paired_ids, f"{name}: exact paired trial order")
        validate_run(validation, name, aggregates[name], ledgers[name], mode, strategy)

    control_path = Path("/private/tmp/push-fold-control-equivalence.json")
    control = json.loads(control_path.read_text(encoding="utf-8"))
    validation.check(control.get("result") == "byte_identical"
                     and all(item.get("cmp_exit_code") == 0 for item in control.get("comparisons", ())),
                     "pre-fix control equivalence evidence")
    manifest = {
        "status": "passed" if not validation.failures else "failed",
        "directory": str(directory),
        "expected_design": {
            "tables_per_run": 20000,
            "seed": 20260809,
            "trial_ids": [0, 19999],
            "paired_runs": list(RUNS),
            "stopping_rule": "continue to ron, tsumo, or exhaustive draw",
        },
        "checks": validation.checks,
        "failures": validation.failures,
        "input_hashes": input_hashes,
        "source_hashes": relevant_source_hashes(root),
        "provenance": {
            "control_equivalence_evidence": control,
            "control_equivalence_evidence_sha256": sha256(control_path),
            "late_guard_note": (
                "The retained all_push and full_fold controls were generated before the late two-riichi "
                "intended-action guard fix. The attached 128-table before/current fallback comparison is "
                "byte-identical for both controls."
            ),
            "legacy_stop_at_note": (
                "Saved production native summaries report config.stop_at='tenpai' from the legacy summary "
                "field. Push/fold observations and trial ledgers continue to actual table terminal; the "
                "current native source changes this output-only field to 'table_terminal'."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if validation.failures:
        raise SystemExit("validation failed: " + ", ".join(validation.failures))
    print(f"validated {len(RUNS)} paired 20k runs; wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
