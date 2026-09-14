#!/usr/bin/env python3
"""Run the native score-service table and summarize actual joint discard danger."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Iterable

import analyze_tile_danger_by_turn as danger
import hand_efficiency_sim as sim
from hand_push_fold_scoring import PushFoldScorer, ScoringConfig, SCORE_RESPONSE_FIELDS
from hand_push_fold_policy import safety_tier
from hand_monte_carlo_report import primary_wait_category


def run_native(args: argparse.Namespace, observations: Path) -> dict[str, Any]:
    """Keep native rollouts cheap; answer only explicit tenpai value queries."""
    command = danger.native_command(args, observations)
    command.remove("--tile-danger")
    command.extend([
        "--push-fold", "--score-service", "--push-fold-mode", args.mode,
        "--safety-strategy", args.safety_strategy,
        "--dora-danger-steps", str(args.dora_danger_steps),
        "--half-push-max-tier", str(args.half_push_max_tier),
        "--two-riichi-bad-wait-min-ron", str(args.two_riichi_bad_wait_min_ron),
    ])
    scoring_config = ScoringConfig(
        cheap_max_total_han=args.cheap_max_total_han,
        valuable_bad_wait_points=None if args.valuable_dama_points == 0 else args.valuable_dama_points,
        two_riichi_bad_wait_min_ron=args.two_riichi_bad_wait_min_ron,
        enable_haneman_lock=args.haneman_lock,
    )
    service = PushFoldScorer()
    observations.parent.mkdir(parents=True, exist_ok=True)
    queries = 0
    summary = None
    started = time.perf_counter()
    with tempfile.TemporaryFile(mode="w+t", dir="/private/tmp") as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=errors, text=True, bufsize=1)
        assert process.stdout is not None and process.stdin is not None
        try:
            for line in process.stdout:
                if not line.strip():
                    continue
                message = json.loads(line)
                if message.get("kind") == "score_request":
                    values = service.score_request(message, scoring_config)
                    if len(values) != len(SCORE_RESPONSE_FIELDS) or not all(math.isfinite(v) for v in values):
                        raise ValueError("invalid scoring callback response")
                    process.stdin.write(" ".join(str(v) for v in values) + "\n")
                    process.stdin.flush()
                    queries += 1
                else:
                    if summary is not None:
                        raise ValueError("native emitted more than one summary")
                    summary = message
            process.stdin.close()
            return_code = process.wait()
            if return_code:
                errors.seek(0)
                raise RuntimeError(f"native failed ({return_code}): {errors.read()}")
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            process.stdout.close()
            if not process.stdin.closed:
                process.stdin.close()
    if summary is None:
        raise ValueError("native omitted summary")
    return {
        "native_command": command, "native_summary": summary,
        "scoring_config": asdict(scoring_config), "score_requests": queries,
        "native_and_decision_scoring_seconds": time.perf_counter() - started,
    }


def _winner_score(row: dict[str, Any], winner: dict[str, Any], tsumo: bool) -> sim.HandScore:
    """Score the realized event directly, including its actual payment breakdown."""
    from mahjong.hand_calculating.hand import HandCalculator
    from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
    hand = sorted(map(int, winner["hand_physical"]))
    win = int(winner["winning_tile"])
    if len(hand) != 14 or len(set(hand)) != 14 or win not in hand:
        raise ValueError("winner must have 14 distinct physical tiles including the winning tile")
    riichi = bool(winner["is_riichi"])
    seat = int(winner["seat"])
    last_tile = bool(winner.get("is_haitei", False))
    context = HandConfig(
        is_tsumo=tsumo, is_riichi=riichi,
        is_ippatsu=bool(winner.get("is_ippatsu", False)),
        is_haitei=last_tile and tsumo, is_houtei=last_tile and not tsumo,
        player_wind=27 + seat, round_wind=int(row["round_wind"]),
        options=OptionalRules(has_aka_dora=True),
    )
    result = HandCalculator.estimate_hand_value(
        hand, win, config=context, dora_indicators=[int(row["dora_indicator"])],
        ura_dora_indicators=[int(row["ura_dora_indicator"])] if riichi else None,
    )
    if result.error:
        raise ValueError(f"invalid realized win trial={row['trial_id']} seat={seat}: {result.error}")
    return sim.HandScore(int(result.cost["total"]), result.han, result.fu,
                         tuple(str(yaku) for yaku in result.yaku), dict(result.cost))


def settle_table(row: dict[str, Any]) -> dict[str, Any]:
    state = row["push_fold"]
    terminal = state["terminal"]
    declared = list(map(int, state["riichi_seats"]))
    if len(declared) != len(set(declared)):
        raise ValueError("duplicate riichi declaration")
    net = [-1000 * int(seat in declared) for seat in range(4)]
    sticks = int(terminal["riichi_sticks"])
    if sticks != len(declared):
        raise ValueError("riichi pot does not match declarations")
    winners = terminal.get("winners", [])
    payments = []
    loser = int(terminal.get("loser", -1))
    if winners:
        for winner in winners:
            seat = int(winner["seat"])
            score = _winner_score(row, winner, tsumo=loser < 0)
            if loser >= 0:
                net[loser] -= int(score.points)
                net[seat] += int(score.points)
            else:
                if len(winners) != 1:
                    raise ValueError("multiple tsumo winners")
                for payer in range(4):
                    if payer == seat:
                        continue
                    amount = int(score.cost["main"] if seat == 0 or payer == 0 else score.cost["additional"])
                    net[payer] -= amount
                    net[seat] += amount
            payments.append({"seat": seat, "is_tsumo": loser < 0, "points": score.points,
                             "han": score.han, "fu": score.fu, "cost": dict(score.cost)})
        recipient = min((int(w["seat"]) for w in winners),
                        key=lambda seat: (seat - loser) % 4) if loser >= 0 else int(winners[0]["seat"])
        net[recipient] += 1000 * sticks
        carried_pot = 0
    else:
        tenpai = set(map(int, terminal["tenpai_seats"]))
        if 0 < len(tenpai) < 4:
            for seat in range(4):
                net[seat] += 3000 // len(tenpai) if seat in tenpai else -(3000 // (4 - len(tenpai)))
        carried_pot = 1000 * sticks
    if sum(net) + carried_pot != 0:
        raise ValueError("point transfers do not conserve points plus carried pot")
    return {"trial_id": int(row["trial_id"]), "terminal_reason": terminal["reason"],
            "net_points": net, "carried_riichi_pot": carried_pot, "payments": payments,
            "winner_seats": [int(w["seat"]) for w in winners], "loser": loser,
            "riichi_seats": declared, "tenpai_seats": terminal.get("tenpai_seats", [])}


def clustered_rate(values: list[int], groups: int) -> dict[str, Any]:
    n, h, cn, ch, decisions, exposed, h2, hn, n2, ch2, chcn, cn2 = values
    def standard_error(a: int, b: int, a2: int, ab: int, b2: int) -> float | None:
        if not b or groups <= 1:
            return None
        p = a / b
        return math.sqrt(max(0, groups / (groups - 1) * (a2 - 2 * p * ab + p * p * b2))) / b
    return {"candidate_opportunities": n, "ron_hits": h,
            "ron_hit_probability": h / n if n else None,
            "table_clustered_se": standard_error(h, n, h2, hn, n2),
            "independent_tables": groups, "exposed_tables": exposed,
            "exposed_decisions": decisions, "chosen_opportunities": cn,
            "chosen_ron_hits": ch, "chosen_ron_hit_probability": ch / cn if cn else None,
            "chosen_table_clustered_se": standard_error(ch, cn, ch2, chcn, cn2)}


def aggregate_rows(rows: Iterable[dict[str, Any]], trial_metrics: Path | None = None,
                   base_seed: int | None = None) -> dict[str, Any]:
    """Cluster all repeated candidate observations by their physical table."""
    totals: dict[tuple, list[int]] = defaultdict(lambda: [0] * 12)
    physical_totals: dict[tuple, list[int]] = defaultdict(lambda: [0] * 12)
    diagnostics: Counter = Counter()
    modes: Counter = Counter()
    seen = set()
    points = [0.0] * 4
    points2 = [0.0] * 4
    win_values: dict[str, list[int]] = {"ron": [], "tsumo": []}
    seat_outcomes = [Counter() for _ in range(4)]
    declarations: Counter = Counter()
    wait_quality = defaultdict(lambda: [0, 0, 0, 0])
    source_policy = None
    source_metadata = None
    with ExitStack() as stack:
        metric_stream = None
        if trial_metrics:
            trial_metrics.parent.mkdir(parents=True, exist_ok=True)
            metric_stream = stack.enter_context(trial_metrics.open("w", encoding="utf-8"))
        for row in rows:
            trial = int(row["trial_id"])
            if trial in seen:
                raise ValueError("duplicate trial id")
            seen.add(trial)
            if base_seed is not None and row["trial_seed"] != sim.trial_seed(base_seed, trial):
                raise ValueError("trial seed disagrees with base seed and trial id")
            metadata = {key: row.get(key) for key in ("engine_version", "evaluator_backend", "policy", "policy_parameters", "round_wind")}
            if source_metadata is None:
                source_metadata = metadata
            elif source_metadata != metadata:
                raise ValueError("mixed engine/policy/round metadata in observations")
            state = row["push_fold"]
            signature = {key: state[key] for key in ("mode", "safety_strategy", "dora_danger_steps",
                         "half_push_max_tier", "two_riichi_bad_wait_min_ron")}
            if source_policy is None:
                source_policy = signature
            elif source_policy != signature:
                raise ValueError("mixed policy configurations in observations")
            settled = settle_table(row)
            diagnostics[settled["terminal_reason"]] += 1
            diagnostics["riichi_declarations"] += len(settled["riichi_seats"])
            diagnostics["carried_riichi_pot"] += settled["carried_riichi_pot"]
            for seat, value in enumerate(settled["net_points"]):
                points[seat] += value
                points2[seat] += value * value
                seat_outcomes[seat]["tables"] += 1
                seat_outcomes[seat]["riichi"] += seat in settled["riichi_seats"]
                seat_outcomes[seat]["deal_in"] += settled["loser"] == seat
                seat_outcomes[seat]["ron"] += seat in settled["winner_seats"] and settled["loser"] >= 0
                seat_outcomes[seat]["tsumo"] += seat in settled["winner_seats"] and settled["loser"] < 0
                seat_outcomes[seat]["exhaustive_draw"] += not settled["winner_seats"]
                seat_outcomes[seat]["exhaustive_draw_tenpai"] += not settled["winner_seats"] and seat in settled["tenpai_seats"]
            for event in state.get("riichi_events", []):
                counts = sim.counts_from_physical(event["hand_physical"])
                category = primary_wait_category(counts, event["winning_types"])
                declarations[category] += 1
                quality = wait_quality[category]
                for index, value in enumerate((1, event["unseen_wait_copies"], event["live_wait_copies"], int(event["furiten"]))):
                    quality[index] += value
            for payment in settled["payments"]:
                win_values["tsumo" if payment["is_tsumo"] else "ron"].append(payment["points"])
            local: dict[tuple, list[int]] = defaultdict(lambda: [0] * 5)
            physical_local: dict[tuple, list[int]] = defaultdict(lambda: [0] * 5)
            dora_type = sim.dora_type_from_indicator(int(row["dora_indicator"]) // 4)
            for decision in state["decisions"]:
                modes[decision["mode"]] += 1
                threats = set(map(int, decision["threat_seats"]))
                if not threats:
                    continue
                own, public = decision["own_counts"], decision["public_counts"]
                if sum(own) != 14 or any(a + b > 4 for a, b in zip(own, public)):
                    raise ValueError("invalid defender visible state")
                candidates = decision["candidates"]
                safe_by_threat = {int(entry["threat_seat"]): set(map(int, entry["types"]))
                                  for entry in decision["safe_types"]}
                if set(safe_by_threat) != threats:
                    raise ValueError("safe sets must cover precisely the declared opponents")
                if sum(bool(c["chosen"]) for c in candidates) != 1:
                    raise ValueError("exactly one physical candidate must be selected")
                by_type: dict[int, dict[str, Any]] = {}
                physical_touched = set()
                for candidate in candidates:
                    tile = int(candidate["tile_type"])
                    hits = set(map(int, candidate["ron_hits"]))
                    physical = int(candidate["physical_id"])
                    red = physical in (16, 52, 88)
                    is_dora = red or tile == dora_type
                    expected = [safety_tier(tile, safe_by_threat[threat], is_dora=is_dora,
                                dora_steps=state["dora_danger_steps"]) for threat in decision["threat_seats"]]
                    if expected != candidate["safety_tiers"]:
                        raise ValueError("native adjusted safety disagrees with Python reference")
                    if candidate["known_copies"] != own[tile] + public[tile]:
                        raise ValueError("known-copy tie break does not match visible counts")
                    physical_key = (len(threats), danger.tile_group(tile), own[tile], public[tile],
                                    red, tile == dora_type, "+".join(map(str, sorted(expected))),
                                    decision["mode"], decision["mode_reason"] == "riichi_lock")
                    cell = physical_local[physical_key]
                    chosen = int(candidate["chosen"])
                    hit = int(bool(hits & threats))
                    for i, value in enumerate((1, hit, chosen, chosen * hit)):
                        cell[i] += value
                    physical_touched.add(physical_key)
                    for threat, safe in safe_by_threat.items():
                        if tile in safe and threat in hits:
                            raise ValueError("genbutsu cannot ron the corresponding riichi opponent")
                    if tile in by_type:
                        if hits != by_type[tile]["hits"]:
                            raise ValueError("physical copies disagree on structural ron eligibility")
                        by_type[tile]["chosen"] |= bool(candidate["chosen"])
                    else:
                        by_type[tile] = {"hits": hits, "chosen": bool(candidate["chosen"])}
                if set(by_type) != {tile for tile, count in enumerate(own) if count}:
                    raise ValueError("candidate set must cover all held tile types")
                touched = set()
                for tile, candidate in by_type.items():
                    chosen = int(candidate["chosen"])
                    safety_profile = "+".join(sorted(danger.safety_class(tile, safe)
                                                     for safe in safe_by_threat.values()))
                    for scope, hit in (("riichi_opponents", bool(candidate["hits"] & threats)),
                                       ("any_opponent", bool(candidate["hits"]))):
                        for axis_turn in (str(decision["turn"]), "all"):
                            for level, identity in (("all", "all"), ("category", danger.tile_group(tile)),
                                                    ("tile", sim.tile_name(tile))):
                                for safety in ("all", safety_profile):
                                    key = (len(threats), axis_turn, scope, level, identity, safety)
                                    cell = local[key]
                                    for i, v in enumerate((1, int(hit), chosen, chosen * int(hit))):
                                        cell[i] += v
                                    touched.add(key)
                for key in touched:
                    local[key][4] += 1
                for key in physical_touched:
                    physical_local[key][4] += 1
                diagnostics["observed_defender_decisions"] += 1
            for table_local, target_totals in ((local, totals), (physical_local, physical_totals)):
                for key, (n, h, cn, ch, decisions) in table_local.items():
                    for i, v in enumerate((n, h, cn, ch, decisions, 1, h * h, h * n, n * n,
                                           ch * ch, ch * cn, cn * cn)):
                        target_totals[key][i] += v
            if metric_stream:
                metric_stream.write(json.dumps(settled, separators=(",", ":")) + "\n")
    groups = len(seen)
    result_rows = []
    for key in sorted(totals, key=lambda k: (k[0], -1 if k[1] == "all" else int(k[1]), *k[2:])):
        count, turn, scope, level, tile, safety = key
        result_rows.append({"riichi_opponents": count, "observer_turn": turn, "ron_scope": scope,
                            "tile_level": level, "tile": tile, "safety_profile": safety,
                            **clustered_rate(totals[key], groups)})
    return {
        "tables": groups, "trial_id_min": min(seen) if seen else None,
        "trial_id_max": max(seen) if seen else None,
        "diagnostics": dict(diagnostics), "decision_modes": dict(modes),
        "source_policy": source_policy,
        "source_metadata": source_metadata,
        "seat_outcomes": [dict(counts) for counts in seat_outcomes],
        "riichi_wait_distribution": dict(declarations),
        "riichi_wait_quality": {category: {"hands": n, "mean_unseen_winning_copies": unseen / n,
                                "mean_live_winning_copies": live / n, "furiten_hands": furiten}
                               for category, (n, unseen, live, furiten) in wait_quality.items()},
        "seat_mean_net_points": [v / groups for v in points] if groups else [],
        "seat_net_points_se": [math.sqrt(max(0, (v2 - v * v / groups) / (groups - 1) / groups))
                               for v, v2 in zip(points, points2)] if groups > 1 else [],
        "realized_win_values": {kind: {"wins": len(values), "mean_points": sum(values) / len(values) if values else None}
                                for kind, values in win_values.items()},
        "danger_rows": result_rows,
        "physical_conditioned_danger_rows": [dict(zip(("riichi_opponents", "category", "own_copies",
                   "public_copies", "red", "indicator_dora", "adjusted_tier_profile", "mode", "own_riichi"), key),
                   **clustered_rate(physical_totals[key], groups)) for key in sorted(physical_totals)],
    }


def parser() -> argparse.ArgumentParser:
    p = danger.parser()
    p.description = __doc__
    p.add_argument("--mode", choices=("all_push", "full_fold", "mixed"), default="mixed")
    p.add_argument("--safety-strategy", choices=("mean_safety", "minimax_safety"), default="mean_safety")
    p.add_argument("--dora-danger-steps", type=int, default=1)
    p.add_argument("--half-push-max-tier", type=int, default=5)
    p.add_argument("--cheap-max-total-han", type=int, default=1)
    p.add_argument("--valuable-dama-points", type=int, choices=(0, 5200, 8000), default=5200)
    p.add_argument("--two-riichi-bad-wait-min-ron", type=int, default=7700)
    p.add_argument("--haneman-lock", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trial-metrics", type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.trials <= 0 or min(args.start_trial, args.seed, args.dora_danger_steps,
                                args.half_push_max_tier, args.cheap_max_total_han,
                                args.two_riichi_bad_wait_min_ron, args.kanchan_outer_wait_weight) < 0:
        p.error("invalid negative parameter or nonpositive trials")
    if args.input and args.observations:
        p.error("--input and --observations are mutually exclusive")
    if args.native_summary and not args.input:
        p.error("--native-summary requires --input")
    with ExitStack() as stack:
        if args.input:
            path = args.input
            run = {"source": "existing_observations"}
            if args.native_summary:
                supplied = json.loads(args.native_summary.read_text())
                original_run = supplied.get("run", {})
                run["native_summary"] = original_run.get("native_summary", supplied)
                if "scoring_config" in original_run:
                    run["scoring_config"] = original_run["scoring_config"]
        else:
            directory = stack.enter_context(tempfile.TemporaryDirectory(dir="/private/tmp", prefix="push-fold-"))
            path = args.observations or Path(directory) / "observations.jsonl"
            run = run_native(args, path)
        started = time.perf_counter()
        source_seed = run.get("native_summary", {}).get("seed") if args.input else args.seed
        result = aggregate_rows(danger.load_jsonl(path), args.trial_metrics,
                                base_seed=source_seed)
        if run.get("native_summary") is not None:
            summary = run["native_summary"]
            expected = {key: summary.get(key) for key in ("engine_version", "evaluator_backend", "policy", "policy_parameters")}
            expected["round_wind"] = summary["config"]["round_wind"]
            if not summary["config"].get("push_fold") or expected != result["source_metadata"]:
                raise ValueError("observations disagree with push/fold native summary metadata")
            if (result["tables"], result["trial_id_min"], result["trial_id_max"]) != (
                summary["trials"], summary["start_trial"], summary["start_trial"] + summary["trials"] - 1
            ):
                raise ValueError("observations disagree with native summary trial range")
            source_policy = result["source_policy"]
            for key, value in source_policy.items():
                config_key = "push_fold_mode" if key == "mode" else key
                if summary["config"][config_key] != value:
                    raise ValueError("source push/fold policy disagrees with native summary")
        if not args.input and (result["tables"], result["trial_id_min"], result["trial_id_max"]) != (
            args.trials, args.start_trial, args.start_trial + args.trials - 1
        ):
            raise ValueError("incomplete native trial range")
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        result["run"] = {
            **run, "aggregation_seconds": time.perf_counter() - started,
            "observations_sha256": checksum, **(result["source_policy"] or {}),
            "candidate_unit": "distinct held tile type per defender decision; physical red/ordinary copies deduplicated",
            "threat_count": "declared opponents, excluding observer",
            "multi_ron": "multiple ron; riichi pot goes to winner closest after discarder",
            "stopping_rule": "continue through all declarations until any legal ron/tsumo or exhaustive draw; no calls or abortive draws",
            "scoring": "actual dealer-aware ron/tsumo; realized ura for riichi; ippatsu excluded; haitei/houtei included; no honba; exhaustive draw payments and riichi deposits included",
            "physical_conditioned_unit": "one ordinary and one red representative per held tile type when present; separate from primary 34-type denominator",
            "input_metadata": "source policy fields validated across rows; weights/seed/backend require original native summary" if args.input else "native command and summary included",
            "comparison_scope": "all four players use the table policy; not a focal-agent EV comparison",
        }
        encoded = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
        else:
            print(encoded, end="")
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", newline="") as stream:
                fields = list(result["danger_rows"][0]) if result["danger_rows"] else ["riichi_opponents", "observer_turn"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(result["danger_rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
