#!/usr/bin/env python3
"""Generate and aggregate post-riichi discard-danger observations by turn.

The primary unit is one distinct candidate tile type in a defender decision,
not a physical copy or the discard actually selected by the policy.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

OUTPUTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUTPUTS_DIR))
import hand_efficiency_sim as sim  # noqa: E402

TILE_COUNT = 34
SAFETY = ("genbutsu", "honor", "double_suji", "half_suji", "suji", "no_suji")
MODEL_METADATA = {
    "grain": "one distinct held candidate tile type per defender decision",
    "copy_weighting": "none; physical copies never duplicate a candidate observation",
    "turn_axes": "observer_turn, riichi_player_turn, turns_since_riichi, and all_turns; turns_since_riichi is riichi_player_turn minus declaration turn",
    "safety": "genbutsu uses observed safe_types only; honors are separate; suited suji uses same-suit rank plus or minus three, with both anchors required for double_suji on 4/5/6",
    "furiten": "target permanent furiten only",
    "ron_scope": "target riichi ron only; non-target dama ron omitted",
    "stopping_rule": "second legal riichi declaration or target ron/tsumo, defender tsumo, exhaustion",
    "minimum_live_tiles_for_riichi": 4,
    "opponent_policy": "attack throughout; no folding, cheap-bad-wait rejection, or valuable-dama filtering",
    "safe_information": "target river including declaration; all safely passed post-declaration discards; no other players' pre-declaration rivers",
    "scoring": "binary target ron eligibility; no point-value scoring",
    "uncertainty": "pooled ratio with independent-table cluster standard error; zero-exposure tables included in G",
}


def tile_group(tile: int) -> str:
    if tile >= 27:
        return "honor"
    rank = tile % 9 + 1
    if rank in (1, 9):
        return "19"
    if rank in (2, 8):
        return "28"
    if rank in (3, 7):
        return "37"
    return "5" if rank == 5 else "46"


def safety_class(tile: int, safe_types: Iterable[int]) -> str:
    safe = set(map(int, safe_types))
    if tile in safe:
        return "genbutsu"
    if tile >= 27:
        return "honor"
    rank, suit = tile % 9 + 1, tile // 9
    anchors = {rank - 3, rank + 3}
    known = {anchor for anchor in anchors if 1 <= anchor <= 9 and suit * 9 + anchor - 1 in safe}
    if rank in (4, 5, 6):
        return "double_suji" if len(known) == 2 else "half_suji" if known else "no_suji"
    return "suji" if known else "no_suji"


def structural_waits(counts: Iterable[int]) -> set[int]:
    hand = list(map(int, counts))
    waits = set()
    for tile in range(TILE_COUNT):
        if hand[tile] == 4:
            continue
        hand[tile] += 1
        if sim.shanten(hand) == -1:
            waits.add(tile)
        hand[tile] -= 1
    return waits


def _validate_decision(
    danger: Mapping[str, Any], decision: Mapping[str, Any], waits: set[int], previous_safe: set[int] | None,
    previous_public: list[int] | None,
) -> set[int]:
    target = int(danger["riichi_seat"])
    observer = int(decision["observer_seat"])
    if target not in range(4) or observer not in range(4) or observer == target:
        raise ValueError("decision must have a distinct valid defender and riichi seat")
    if int(decision["draw_index"]) <= int(danger["riichi_draw_index"]):
        raise ValueError("decision predates riichi")
    if int(decision["riichi_player_turn"]) < int(danger["riichi_turn"]):
        raise ValueError("riichi player turn predates declaration")
    if int(decision["turns_since_riichi"]) != int(decision["riichi_player_turn"]) - int(danger["riichi_turn"]):
        raise ValueError("inconsistent turns_since_riichi")
    own, public = list(map(int, decision["own_counts"])), list(map(int, decision["public_counts"]))
    if len(own) != TILE_COUNT or len(public) != TILE_COUNT or sum(own) != 14:
        raise ValueError("invalid own counts")
    if sum(public) != int(decision["draw_index"]):
        raise ValueError("public count must equal draw index")
    target_counts = list(map(int, danger["riichi_hand_counts"]))
    if any(a < 0 or b < 0 or a + b + target_counts[tile] > 4
           for tile, (a, b) in enumerate(zip(own, public))):
        raise ValueError("invalid physical counts")
    candidates = decision["candidates"]
    types = [int(x["tile_type"]) for x in candidates]
    held = [i for i, n in enumerate(own) if n]
    if len(types) != len(set(types)) or set(types) != set(held):
        raise ValueError("candidates must be distinct held tile types")
    if sum(bool(x["chosen"]) for x in candidates) != 1:
        raise ValueError("exactly one candidate must be chosen")
    safe = set(map(int, decision["safe_types"]))
    if any(tile not in range(TILE_COUNT) for tile in safe):
        raise ValueError("invalid safe tile")
    river_counts = list(map(int, decision["riichi_river_counts"]))
    if len(river_counts) != TILE_COUNT or any(
        count < 0 or count > public[tile] for tile, count in enumerate(river_counts)
    ):
        raise ValueError("target river must be part of public counts")
    river = {tile for tile, count in enumerate(river_counts) if count}
    if previous_safe is None:
        if safe != river:
            raise ValueError("first safe types must equal riichi river")
    else:
        assert previous_public is not None
        if any(after < before for before, after in zip(previous_public, public)):
            raise ValueError("public discards cannot disappear")
        newly_public = {tile for tile, count in enumerate(public) if count > previous_public[tile]}
        if safe != previous_safe | newly_public:
            raise ValueError("safe types must equal prior safe types plus new public discards")
    furiten = bool(waits & river)
    if furiten != bool(danger["furiten_at_riichi"]):
        raise ValueError("locked target furiten disagrees with its complete river")
    for candidate in candidates:
        tile, hit = int(candidate["tile_type"]), bool(candidate["ron_hit"])
        if tile not in range(TILE_COUNT):
            raise ValueError("invalid candidate tile")
        if hit != (tile in waits and not furiten):
            raise ValueError("ron label disagrees with riichi hand")
        if tile in safe and hit:
            raise ValueError("genbutsu cannot ron")
    return safe


def _axis_values(decision: Mapping[str, Any]) -> dict[str, str]:
    return {"observer_turn": str(decision["observer_turn"]), "riichi_player_turn": str(decision["riichi_player_turn"]), "turns_since_riichi": str(decision["turns_since_riichi"]), "all_turns": "all"}


def aggregate_rows(
    rows: Iterable[Mapping[str, Any]], *, base_seed: int | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # Totals: n, h, chosen n, chosen h, exposed decisions, observed tables,
    # and table-cluster moments h², hn, n². Zero-exposure tables enter G below.
    totals: dict[tuple[str, ...], list[float]] = defaultdict(lambda: [0] * 9)
    diagnostics = defaultdict(int)
    trial_ids: set[int] = set()
    metadata_fields = ("engine_version", "evaluator_backend", "policy", "policy_parameters", "round_wind")
    source_metadata = None
    for row in rows:
        trial = int(row["trial_id"])
        if trial < 0 or trial in trial_ids:
            raise ValueError("trial_id must be unique and nonnegative")
        trial_ids.add(trial)
        if base_seed is not None and int(row["trial_seed"]) != sim.trial_seed(base_seed, trial):
            raise ValueError("trial seed disagrees with base seed and trial id")
        metadata = {key: row.get(key) for key in metadata_fields}
        if source_metadata is None:
            source_metadata = metadata
        elif metadata != source_metadata:
            raise ValueError("inconsistent input engine, policy, backend, or round wind")
        if expected_metadata is not None:
            expected = {key: expected_metadata.get(key) for key in metadata_fields}
            expected["round_wind"] = expected_metadata["config"]["round_wind"]
            if metadata != expected:
                raise ValueError("observations disagree with native summary metadata")
        danger = row.get("danger")
        if danger is None:
            raise ValueError("row lacks danger payload")
        diagnostics[f"terminal_{danger.get('terminal_reason', 'unknown')}"] += 1
        if int(danger["riichi_seat"]) < 0:
            if danger.get("decisions"):
                raise ValueError("pre-riichi row has decisions")
            diagnostics["no_riichi_tables"] += 1
            continue
        diagnostics["riichi_tables"] += 1
        hand_counts = list(map(int, danger["riichi_hand_counts"]))
        if len(hand_counts) != TILE_COUNT or sum(hand_counts) != 13 or any(n < 0 or n > 4 for n in hand_counts):
            raise ValueError("invalid riichi hand counts")
        waits = structural_waits(hand_counts)
        if set(map(int, danger["winning_types"])) != waits:
            raise ValueError("winning types disagree with riichi hand")
        previous_safe: set[int] | None = None
        previous_public: list[int] | None = None
        local: dict[tuple[str, ...], list[float]] = defaultdict(lambda: [0, 0, 0, 0, 0])
        for decision in danger.get("decisions", ()):
            previous_safe = _validate_decision(danger, decision, waits, previous_safe, previous_public)
            previous_public = list(map(int, decision["public_counts"]))
            axes = _axis_values(decision)
            seen_keys: set[tuple[str, ...]] = set()
            for c in decision["candidates"]:
                tile, hit, chosen = int(c["tile_type"]), int(bool(c["ron_hit"])), bool(c["chosen"])
                safety = safety_class(tile, decision["safe_types"])
                for axis, value in axes.items():
                    for level, identity in (("category", tile_group(tile)), ("tile", sim.tile_name(tile)), ("all", "all")):
                        for safe in (safety, "all"):
                            key = (axis, value, level, identity, safe)
                            bucket = local[key]
                            bucket[0] += 1
                            bucket[1] += hit
                            bucket[2] += chosen
                            bucket[3] += chosen * hit
                            seen_keys.add(key)
            for key in seen_keys:
                local[key][4] += 1
            diagnostics["decisions"] += 1
        for key, values in local.items():
            n, h, cn, ch, exposed = values
            total = totals[key]
            for index, value in enumerate((n, h, cn, ch, exposed, 1, h * h, h * n, n * n)):
                total[index] += value
    axis_order = {"observer_turn": 0, "riichi_player_turn": 1, "turns_since_riichi": 2, "all_turns": 3}
    keys = sorted(totals, key=lambda key: (
        axis_order[key[0]], -1 if key[1] == "all" else int(key[1]), *key[2:]
    ))
    output = []
    for key in keys:
        n, h, cn, ch, exposed, observed, h2, hn, n2 = totals[key]
        groups = len(trial_ids)
        p = h / n
        se = None
        if groups > 1 and n:
            ss = h2 - 2 * p * hn + p * p * n2
            se = math.sqrt(max(0.0, groups / (groups - 1) * ss)) / n
        axis, value, level, identity, safety = key
        output.append({
            "axis": axis, "turn": value, "tile_level": level, "tile": identity,
            "safety": safety, "candidate_opportunities": n, "ron_hits": h,
            "pooled_ron_hit_probability": p, "table_clusters": groups,
            "table_clusters_with_observations": observed,
            "table_clustered_standard_error": se, "exposed_decisions": exposed,
            "chosen_opportunities": cn, "chosen_ron_hits": ch,
            "chosen_ron_hit_probability": ch / cn if cn else None,
        })
    return {"diagnostics": {**diagnostics, "tables": len(trial_ids),
            "trial_id_min": min(trial_ids) if trial_ids else None,
            "trial_id_max": max(trial_ids) if trial_ids else None,
            "input_signatures": int(source_metadata is not None)},
            "source_metadata": source_metadata, "rows": output}


def load_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def native_command(args: argparse.Namespace, observations: Path) -> list[str]:
    command = [
        str(args.native_binary), "simulate-table", "--tile-danger",
        "--trials", str(args.trials), "--seed", str(args.seed),
        "--start-trial", str(args.start_trial), "--round-wind", args.round_wind,
        "--policy", args.policy, "--observations", str(observations),
    ]
    options = (
        ("--future-ukeire-weight", args.future_ukeire_weight),
        ("--future-ukeire-weight-1-shanten", args.future_ukeire_weight_1_shanten),
        ("--future-ukeire-weight-2-shanten", args.future_ukeire_weight_2_shanten),
        ("--improvement-weight", args.improvement_weight),
        ("--dora-ukeire-weight", args.dora_ukeire_weight),
        ("--dora-tiebreak-weight", args.dora_tiebreak_weight),
        ("--dora-neighbor-weight", args.dora_neighbor_weight),
        ("--yakuhai-weight", args.yakuhai_weight),
        ("--tanyao-weight", args.tanyao_weight),
        ("--shanten-reversal-penalty", args.shanten_reversal_penalty),
        ("--shanten-reversal-future-weight", args.shanten_reversal_future_weight),
        ("--shanten-reversal-max-turn", args.shanten_reversal_max_turn),
        ("--shanten-reversal-candidate-limit", args.shanten_reversal_candidate_limit),
    )
    for option, value in options:
        if value is not None:
            command.extend((option, str(value)))
    if args.allow_shanten_reversal:
        command.append("--allow-shanten-reversal")
    if args.kanchan_outer_wait_weight:
        command.extend(("--kanchan-outer-wait-weight", str(args.kanchan_outer_wait_weight)))
    return command


def _check_trial_range(result: Mapping[str, Any], start: int, trials: int) -> None:
    diagnostics = result["diagnostics"]
    if (diagnostics["tables"] != trials or diagnostics["trial_id_min"] != start
            or diagnostics["trial_id_max"] != start + trials - 1):
        raise ValueError("observations do not contain the requested complete trial range")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path)
    p.add_argument("--native-summary", type=Path)
    p.add_argument("--observations", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--csv", type=Path)
    p.add_argument(
        "--native-binary", type=Path,
        default=Path("/private/tmp/hand-efficiency-build/hand_efficiency_sim_cpp"),
    )
    p.add_argument("--trials", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument("--start-trial", type=int, default=0)
    p.add_argument("--round-wind", choices=("east", "south", "west", "north"), default="east")
    sim._add_policy_arguments(p, default_policy="value")
    p.add_argument("--no-shanten-reversal", dest="allow_shanten_reversal", action="store_false")
    p.set_defaults(
        future_ukeire_weight_1_shanten=2.0, future_ukeire_weight_2_shanten=1.0,
        allow_shanten_reversal=True, shanten_reversal_penalty=25.0,
    )
    return p


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.trials <= 0 or args.start_trial < 0 or args.seed < 0:
        p.error("trials must be positive; seed and start-trial must be nonnegative")
    if args.native_summary and not args.input:
        p.error("--native-summary requires --input")
    if args.input and args.observations:
        p.error("--input and --observations are mutually exclusive")

    summary = None
    command = None
    native_seconds = None
    with ExitStack() as stack:
        if args.input:
            path = args.input
            if args.native_summary:
                summary = json.loads(args.native_summary.read_text(encoding="utf-8"))
        else:
            if not args.native_binary.is_file():
                p.error("native binary not found")
            if args.observations:
                path = args.observations
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                directory = stack.enter_context(tempfile.TemporaryDirectory(
                    prefix="tile-danger-", dir="/private/tmp",
                ))
                path = Path(directory) / "danger.jsonl"
            command = native_command(args, path)
            started = time.perf_counter()
            completed = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE)
            native_seconds = time.perf_counter() - started
            summary = json.loads(completed.stdout)

        if summary is not None and not summary.get("config", {}).get("tile_danger"):
            raise ValueError("native summary is not from tile-danger mode")
        started = time.perf_counter()
        result = aggregate_rows(
            load_jsonl(path), base_seed=summary["seed"] if summary else None,
            expected_metadata=summary,
        )
        aggregate_seconds = time.perf_counter() - started
        if summary is not None:
            _check_trial_range(result, summary["start_trial"], summary["trials"])
            for reason, count in summary["outcomes"].items():
                if result["diagnostics"].get(f"terminal_{reason}", 0) != count:
                    raise ValueError("terminal counts disagree with native summary")
        if not args.input:
            _check_trial_range(result, args.start_trial, args.trials)
        with path.open("rb") as source:
            source_hash = hashlib.file_digest(source, "sha256").hexdigest()
        result["run"] = {
            "schema_version": 1,
            "source": "input_jsonl" if args.input else "native",
            "observations": str(path) if args.input or args.observations else None,
            "observations_retained_by_runner": bool(args.input or args.observations),
            "observations_sha256": source_hash,
            "seed": summary["seed"] if summary else None,
            "start_trial": result["diagnostics"]["trial_id_min"],
            "trials": result["diagnostics"]["tables"],
            **(result["source_metadata"] or {}),
            **MODEL_METADATA,
            "native_command": command,
            "native_seconds": native_seconds,
            "aggregation_seconds": aggregate_seconds,
            "native_summary": summary,
        }
        _write(result, args)
    return 0


def _write(result: Mapping[str, Any], args: argparse.Namespace) -> None:
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = list(result["rows"][0]) if result["rows"] else ["axis", "turn", "tile_level", "tile", "safety"]
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result["rows"])

if __name__ == "__main__":
    raise SystemExit(main())
