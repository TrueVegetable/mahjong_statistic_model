#!/usr/bin/env python3
"""Validate, compare, and summarize legal-wait iishanten simulator JSON.

This tool deliberately knows no hand-generation algorithm.  It enforces the
shared output contract used by the Python reference and C++ production engines,
then compares independently sampled estimates with their combined standard
errors.  It is therefore useful during an engine rewrite and after large runs.

Required engine CLI (both engines):

  --trials N --seed N --side-mode full|heads_up --shape all|SHAPE
  --hero-strategy all|STRATEGY --side-riichi-probability P
  --current-category all|genbutsu|suji37|nosuji2378|nosuji456
  --json [--weights PATH] [--current-passed-tile TILE]

``--json`` writes precisely one UTF-8 JSON document to stdout.  Its top level
is ``{engine_version, engine, seed, trials, config, results, calibration}``.
The exact count state is measured after South/hero's ninth discard: rivers are
dealer/hero/side1(West)/side2(North) = 9/9/8/8, the live wall is 36 tiles, and
hero has 88 unseen positions.  The next actions are West, North, then dealer.
The resolution timeline starts by checking hero's current discard.  A dealer
hit is ippatsu ron; safe West/North discards become dealer genbutsu, and only
then does the dealer take the still-ippatsu next draw.  A West/North deal-in to
the dealer ends the hand with zero hero point EV.  Dealer ippatsu ends after
the dealer completes the following discard.
The formal schema lives beside this file; semantic checks below intentionally
cover the relationships JSON Schema cannot express.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"
SCHEMA_PATH = OUTPUTS / "iishanten_validation_schema.json"

# koba (2019 Tenhou Houou East-South, 171,650 hanchan and 1,300,316 riichi;
# after the riichi declaration discard, excluding furiten and kokushi riichi).
# These are candidate-tile-as-winning-tile contributions, not hand-form shares.
KOBA_TARGETS: dict[str, float] = {
    "honor.tanki": 0.0056,
    "honor.shanpon": 0.0120,
    "honor.total": 0.0176,
    "one_nine.tanki": 0.0047,
    "one_nine.shanpon": 0.0101,
    "one_nine.ryanmen": 0.0461,
    "one_nine.total": 0.0609,
    "two_eight.tanki": 0.0048,
    "two_eight.shanpon": 0.0089,
    "two_eight.kanchan": 0.0118,
    "two_eight.ryanmen": 0.0466,
    "two_eight.total": 0.0720,
    "three_seven.tanki": 0.0038,
    "three_seven.shanpon": 0.0066,
    "three_seven.kanchan": 0.0118,
    "three_seven.penchan": 0.0097,
    "three_seven.ryanmen": 0.0474,
    "three_seven.total": 0.0794,
    "four_five_six.tanki": 0.0044,
    "four_five_six.shanpon": 0.0067,
    "four_five_six.kanchan": 0.0083,
    "four_five_six.ryanmen": 0.0762,
    "four_five_six.total": 0.0956,
}
BASE_KOBA_CELLS = tuple(key for key in KOBA_TARGETS if not key.endswith(".total"))
WAIT_KINDS = ("ryanmen", "kanchan", "penchan", "shanpon", "tanki")
EPSILON = 1e-9
SEAT_MAP = {"dealer": "east", "hero": "south", "side1": "west", "side2": "north"}
RIVER_LENGTHS = {"dealer": 9, "hero": 9, "side1": 8, "side2": 8}
ACTION_ORDER = ["hero_current_discard_check", "west_draw_discard", "north_draw_discard", "dealer_draw"]
# The `heads_up` label is retained for CLI compatibility, but the ablation is
# now four-seat forced fold: West and North still draw and discard.
HEADS_UP_ACTION_ORDER = ACTION_ORDER


class ContractError(ValueError):
    """One or more engine outputs do not meet the shared contract."""


@dataclass(frozen=True)
class Comparison:
    cell: tuple[str, str, str]
    metric: str
    python_value: float
    cpp_value: float
    combined_se: float
    z_score: float
    passed: bool


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _number(value: Any, path: str, errors: list[str], *, nonnegative: bool = False) -> float | None:
    if not _is_number(value):
        errors.append(f"{path} must be a finite number")
        return None
    result = float(value)
    if nonnegative and result < 0:
        errors.append(f"{path} must be non-negative")
    return result


def _integer(value: Any, path: str, errors: list[str], *, nonnegative: bool = False) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append(f"{path} must be an integer")
        return None
    if nonnegative and value < 0:
        errors.append(f"{path} must be non-negative")
    return value


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: top-level JSON value must be an object")
    return value


def parse_json_stdout(stdout: str, command: Sequence[str]) -> dict[str, Any]:
    """Require the new --json contract, while accepting a final JSON line for diagnostics."""
    text = stdout.strip()
    candidates = [text] if text else []
    candidates.extend(line.strip() for line in text.splitlines() if line.lstrip().startswith("{"))
    for candidate in reversed(candidates):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    joined = " ".join(command)
    raise ContractError(f"{joined} did not emit a JSON object for --json")


def _result_key(result: dict[str, Any]) -> tuple[str, str, str]:
    return (str(result.get("shape")), str(result.get("strategy")), str(result.get("current_category")))


def validate_document(document: dict[str, Any], *, expected_engine: str | None = None, strict_calibration: bool = True) -> list[str]:
    """Return contract errors.  Empty output means the document is valid."""
    errors: list[str] = []
    required_top = ("engine_version", "engine", "seed", "trials", "config", "results", "calibration")
    for key in required_top:
        if key not in document:
            errors.append(f"top-level field missing: {key}")
    if errors:
        return errors
    if document["engine"] not in {"python", "cpp"}:
        errors.append("engine must be 'python' or 'cpp'")
    elif expected_engine and document["engine"] != expected_engine:
        errors.append(f"engine is {document['engine']!r}, expected {expected_engine!r}")
    if not isinstance(document["engine_version"], str) or not document["engine_version"]:
        errors.append("engine_version must be a non-empty string")
    seed = _integer(document["seed"], "seed", errors, nonnegative=True)
    trials = _integer(document["trials"], "trials", errors, nonnegative=True)
    if trials == 0:
        errors.append("trials must be positive")
    if seed is None:
        pass

    config = document["config"]
    if not isinstance(config, dict):
        errors.append("config must be an object")
    else:
        for name, expected in (("live_wall_tiles", 36), ("hero_unseen_positions", 88)):
            if config.get(name) != expected:
                errors.append(f"config.{name} must be {expected}, got {config.get(name)!r}")
        expected_order = HEADS_UP_ACTION_ORDER if config.get("side_mode") == "heads_up" else ACTION_ORDER
        for name, expected in (("seat_map", SEAT_MAP), ("river_lengths", RIVER_LENGTHS), ("action_order", expected_order)):
            if config.get(name) != expected:
                errors.append(f"config.{name} must be {expected!r}, got {config.get(name)!r}")
        if "discard_count_each" in config:
            errors.append("config.discard_count_each is obsolete; use the exact 9/9/8/8 river_lengths")

    results = document["results"]
    if not isinstance(results, list) or not results:
        errors.append("results must be a non-empty array")
    else:
        seen: set[tuple[str, str, str]] = set()
        for index, result in enumerate(results):
            _validate_result(result, index, trials, errors)
            if isinstance(result, dict):
                key = _result_key(result)
                if key in seen:
                    errors.append(f"results[{index}] duplicates cell {key}")
                seen.add(key)

    _validate_calibration(document["calibration"], errors, strict=strict_calibration)
    return errors


def _validate_result(result: Any, index: int, trials: int | None, errors: list[str]) -> None:
    prefix = f"results[{index}]"
    if not isinstance(result, dict):
        errors.append(f"{prefix} must be an object")
        return
    required = (
        "shape", "strategy", "current_category", "ev", "se", "fold_ev", "fold_se", "ev_difference",
        "paired_ev_difference_se", "break_even_danger", "event_counts", "event_rates", "route_counts",
        "wait_kind_counts", "current_hit_rates", "current_dealer_ippatsu_contribution", "continuation_ev_after_safe",
    )
    for field in required:
        if field not in result:
            errors.append(f"{prefix}.{field} is required")
    if any(field not in result for field in required):
        return
    if result["shape"] not in {"ryanmen_ryanmen", "ryanmen_ryankan", "ryanmen_kanchan"}:
        errors.append(f"{prefix}.shape is not a shared shape key")
    if result["strategy"] not in {"suji37", "nosuji2378", "allout", "fixed05", "fixed10", "fixed15", "fold"}:
        errors.append(f"{prefix}.strategy is not a shared strategy key")
    if result["current_category"] not in {"genbutsu", "suji37", "nosuji2378", "nosuji456", "fold"}:
        errors.append(f"{prefix}.current_category is not a shared category key")
    values: dict[str, float | None] = {}
    for field in ("ev", "se", "fold_ev", "fold_se", "ev_difference", "paired_ev_difference_se", "break_even_danger", "current_dealer_ippatsu_contribution", "continuation_ev_after_safe"):
        values[field] = _number(result[field], f"{prefix}.{field}", errors, nonnegative=field.endswith("_se") or field == "se")
    # The algebraic crossing point may be below zero or above one.  That means
    # no physical danger probability makes the branches equal, so preserve it.
    if values["ev"] is not None and values["fold_ev"] is not None and values["ev_difference"] is not None:
        if not math.isclose(values["ev"] - values["fold_ev"], values["ev_difference"], abs_tol=1e-6):
            errors.append(f"{prefix}.ev_difference must equal ev - fold_ev")

    counts = result["event_counts"]
    if not isinstance(counts, dict):
        errors.append(f"{prefix}.event_counts must be an object")
    else:
        if not counts:
            errors.append(f"{prefix}.event_counts must not be empty")
        count_sum = 0
        for name, count in counts.items():
            parsed = _integer(count, f"{prefix}.event_counts.{name}", errors, nonnegative=True)
            if parsed is not None:
                count_sum += parsed
            if trials and parsed is not None and parsed > trials:
                errors.append(f"{prefix}.event_counts.{name} exceeds trials")
        if trials and count_sum != trials:
            errors.append(f"{prefix}.event_counts must partition trials ({trials}), got {count_sum}")
    rates = result["event_rates"]
    if not isinstance(rates, dict):
        errors.append(f"{prefix}.event_rates must be an object")
    else:
        for name, rate in rates.items():
            parsed = _number(rate, f"{prefix}.event_rates.{name}", errors, nonnegative=True)
            if parsed is not None and parsed > 1:
                errors.append(f"{prefix}.event_rates.{name} exceeds one")
            if isinstance(counts, dict) and name in counts and trials and parsed is not None:
                if not math.isclose(parsed, counts[name] / trials, abs_tol=max(1e-8, 0.5 / trials)):
                    errors.append(f"{prefix}.event_rates.{name} disagrees with event_counts")
        if not rates:
            errors.append(f"{prefix}.event_rates must not be empty")
    route_counts = result["route_counts"]
    if not isinstance(route_counts, dict):
        errors.append(f"{prefix}.route_counts must be an object")
    else:
        for name, count in route_counts.items():
            parsed = _integer(count, f"{prefix}.route_counts.{name}", errors, nonnegative=True)
            if trials and parsed is not None and parsed > trials:
                errors.append(f"{prefix}.route_counts.{name} exceeds trials")

    wait_counts = result["wait_kind_counts"]
    if not isinstance(wait_counts, dict):
        errors.append(f"{prefix}.wait_kind_counts must be an object")
    else:
        for kind in WAIT_KINDS:
            _integer(wait_counts.get(kind), f"{prefix}.wait_kind_counts.{kind}", errors, nonnegative=True)
    hits = result["current_hit_rates"]
    if not isinstance(hits, dict):
        errors.append(f"{prefix}.current_hit_rates must be an object")
    else:
        parsed_hits: dict[str, float | None] = {}
        for name in ("dealer", "initial_side", "any"):
            parsed_hits[name] = _number(hits.get(name), f"{prefix}.current_hit_rates.{name}", errors, nonnegative=True)
            if parsed_hits[name] is not None and parsed_hits[name] > 1:
                errors.append(f"{prefix}.current_hit_rates.{name} exceeds one")
        if all(parsed_hits[name] is not None for name in parsed_hits):
            if parsed_hits["any"] + EPSILON < max(parsed_hits["dealer"], parsed_hits["initial_side"]):
                errors.append(f"{prefix}.current_hit_rates.any must cover dealer and initial_side")


def _validate_calibration(calibration: Any, errors: list[str], *, strict: bool) -> None:
    if not isinstance(calibration, dict):
        errors.append("calibration must be an object")
        return
    residuals = calibration.get("residuals")
    if not isinstance(residuals, dict):
        errors.append("calibration.residuals must be an object")
        return
    squared_error = 0.0
    for cell, target in KOBA_TARGETS.items():
        item = residuals.get(cell)
        if not isinstance(item, dict):
            errors.append(f"calibration.residuals.{cell} is required")
            continue
        values: dict[str, float | None] = {}
        for field in ("target", "fitted", "difference", "standard_error"):
            values[field] = _number(item.get(field), f"calibration.residuals.{cell}.{field}", errors, nonnegative=field == "standard_error")
        if values["target"] is not None and not math.isclose(values["target"], target, abs_tol=1e-9):
            errors.append(f"calibration.residuals.{cell}.target must be {target:.4%}")
        if values["fitted"] is not None and not 0 <= values["fitted"] <= 1:
            errors.append(f"calibration.residuals.{cell}.fitted must be in [0, 1]")
        if values["difference"] is not None and values["fitted"] is not None and values["target"] is not None:
            if not math.isclose(values["difference"], values["fitted"] - values["target"], abs_tol=1e-9):
                errors.append(f"calibration.residuals.{cell}.difference must equal fitted - target")
        if cell in BASE_KOBA_CELLS and values["difference"] is not None:
            squared_error += values["difference"] ** 2
    if len(BASE_KOBA_CELLS):
        actual_rmse = math.sqrt(squared_error / len(BASE_KOBA_CELLS))
        reported = _number(calibration.get("weighted_rmse"), "calibration.weighted_rmse", errors, nonnegative=True)
        if reported is not None and not math.isclose(reported, actual_rmse, abs_tol=1e-9):
            errors.append("calibration.weighted_rmse must equal the equal-cell RMSE of the 18 basic koba cells")
        acceptance = calibration.get("acceptance")
        if acceptance is not None:
            if not isinstance(acceptance, dict):
                errors.append("calibration.acceptance must be an object")
            else:
                rmse_target = _number(acceptance.get("basic_rmse_target"), "calibration.acceptance.basic_rmse_target", errors, nonnegative=True)
                total_target = _number(acceptance.get("main_total_error_target"), "calibration.acceptance.main_total_error_target", errors, nonnegative=True)
                observed_rmse = _number(acceptance.get("observed_basic_rmse"), "calibration.acceptance.observed_basic_rmse", errors, nonnegative=True)
                observed_total = _number(acceptance.get("observed_max_total_error"), "calibration.acceptance.observed_max_total_error", errors, nonnegative=True)
                if observed_rmse is not None and not math.isclose(observed_rmse, actual_rmse, abs_tol=1e-9):
                    errors.append("calibration.acceptance.observed_basic_rmse disagrees with residuals")
                total_errors = [abs(residuals[key]["difference"]) for key in residuals if key.endswith(".total") and isinstance(residuals[key], dict) and isinstance(residuals[key].get("difference"), (int, float))]
                actual_total = max(total_errors, default=0.0)
                if observed_total is not None and not math.isclose(observed_total, actual_total, abs_tol=1e-9):
                    errors.append("calibration.acceptance.observed_max_total_error disagrees with residuals")
                if isinstance(acceptance.get("met"), bool) and None not in (rmse_target, total_target):
                    expected_met = actual_rmse <= rmse_target + EPSILON and actual_total <= total_target + EPSILON
                    if acceptance["met"] != expected_met:
                        errors.append("calibration.acceptance.met disagrees with reported residuals")


def ensure_valid(document: dict[str, Any], **kwargs: Any) -> None:
    errors = validate_document(document, **kwargs)
    if errors:
        raise ContractError("\n".join(errors))


def build_engine_command(
    executable: Sequence[str], args: argparse.Namespace,
) -> list[str]:
    command = list(executable) + [
        "--trials", str(args.trials), "--seed", str(args.seed), "--side-mode", args.side_mode,
        "--shape", args.shape, "--hero-strategy", args.hero_strategy,
        "--side-riichi-probability", str(args.side_riichi_probability),
        "--current-category", args.current_category, "--json",
    ]
    if args.weights:
        command += ["--weights", str(args.weights)]
    if args.current_passed_tile is not None:
        command += ["--current-passed-tile", str(args.current_passed_tile)]
    return command


def run_engine(executable: Sequence[str], args: argparse.Namespace, expected_engine: str) -> dict[str, Any]:
    command = build_engine_command(executable, args)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise ContractError(f"{' '.join(command)} failed ({completed.returncode}):\n{completed.stderr.strip()}")
    document = parse_json_stdout(completed.stdout, command)
    ensure_valid(document, expected_engine=expected_engine)
    return document


def compile_cpp(source: Path, compiler: str, extra_flags: Sequence[str]) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    if not source.is_file():
        raise ContractError(f"C++ source not found: {source}")
    work = tempfile.TemporaryDirectory(prefix="iishanten-cpp-")
    binary = Path(work.name) / "iishanten_wall_sim"
    command = [compiler, "-O3", "-std=c++20", *extra_flags, str(source), "-o", str(binary)]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    if completed.returncode:
        work.cleanup()
        raise ContractError(f"C++ compilation failed:\n{completed.stderr.strip()}")
    return binary, work


def _binomial_se(rate: float, trials: int) -> float:
    return math.sqrt(max(0.0, rate * (1.0 - rate)) / trials)


def compare_documents(python_doc: dict[str, Any], cpp_doc: dict[str, Any], sigma: float = 3.0) -> list[Comparison]:
    ensure_valid(python_doc, expected_engine="python")
    ensure_valid(cpp_doc, expected_engine="cpp")
    python_rows = {_result_key(row): row for row in python_doc["results"]}
    cpp_rows = {_result_key(row): row for row in cpp_doc["results"]}
    if python_rows.keys() != cpp_rows.keys():
        missing_python = sorted(cpp_rows.keys() - python_rows.keys())
        missing_cpp = sorted(python_rows.keys() - cpp_rows.keys())
        raise ContractError(f"result cells differ; missing from Python={missing_python}, missing from C++={missing_cpp}")
    comparisons: list[Comparison] = []
    for cell in sorted(python_rows):
        left, right = python_rows[cell], cpp_rows[cell]
        comparisons.append(_comparison(cell, "ev", left["ev"], left["se"], right["ev"], right["se"], sigma))
        shared_events = sorted(set(left["event_rates"]) & set(right["event_rates"]))
        if not shared_events:
            raise ContractError(f"{cell}: engines have no shared event-rate key")
        for event in shared_events:
            comparisons.append(_comparison(
                cell, f"event_rate.{event}", left["event_rates"][event], _binomial_se(left["event_rates"][event], python_doc["trials"]),
                right["event_rates"][event], _binomial_se(right["event_rates"][event], cpp_doc["trials"]), sigma,
            ))
        comparisons.append(_comparison(
            cell, "ev_difference", left["ev_difference"], left["paired_ev_difference_se"],
            right["ev_difference"], right["paired_ev_difference_se"], sigma,
        ))
    return comparisons


def _comparison(cell: tuple[str, str, str], metric: str, left: float, left_se: float, right: float, right_se: float, sigma: float) -> Comparison:
    combined = math.hypot(float(left_se), float(right_se))
    difference = abs(float(left) - float(right))
    z_score = 0.0 if difference == 0 else math.inf if combined == 0 else difference / combined
    return Comparison(cell, metric, float(left), float(right), combined, z_score, z_score <= sigma + EPSILON)


def format_comparisons(comparisons: Iterable[Comparison]) -> str:
    lines = ["cell\tmetric\tpython\tcpp\tcombined_se\tz\tstatus"]
    for item in comparisons:
        cell = "/".join(item.cell)
        lines.append(f"{cell}\t{item.metric}\t{item.python_value:.8g}\t{item.cpp_value:.8g}\t{item.combined_se:.8g}\t{item.z_score:.4g}\t{'PASS' if item.passed else 'FAIL'}")
    return "\n".join(lines)


def collect_result_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        document = load_json(path)
        rows.extend(_rows_from_document(document, str(path)))
    return rows


def _rows_from_document(document: dict[str, Any], source: str) -> list[dict[str, Any]]:
    ensure_valid(document, strict_calibration=False)
    return [
        {"source": source, "engine": document["engine"], "engine_version": document["engine_version"], "seed": document["seed"], "trials": document["trials"], **result}
        for result in document["results"]
    ]


def markdown_summary(rows: Iterable[dict[str, Any]]) -> str:
    header = "| 引擎 | 牌形 | 策略 | 当前牌类别 | EV | 标准误 | 弃和 EV | 配对 EV 差 | 配对差标准误 | 庄家命中 | 初始听牌旁家命中 | 任一家命中 | 安全通过后继续 EV |"
    separator = "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, separator]
    for row in sorted(rows, key=lambda item: (item["shape"], item["strategy"], item["current_category"], item["engine"])):
        hit = row["current_hit_rates"]
        lines.append(
            f"| {row['engine']} | {row['shape']} | {row['strategy']} | {row['current_category']} | "
            f"{row['ev']:.2f} | {row['se']:.2f} | {row['fold_ev']:.2f} | {row['ev_difference']:.2f} | "
            f"{row['paired_ev_difference_se']:.2f} | {hit['dealer']:.3%} | {hit['initial_side']:.3%} | "
            f"{hit['any']:.3%} | {row['continuation_ev_after_safe']:.2f} |"
        )
    return "\n".join(lines)


def csv_summary(rows: Iterable[dict[str, Any]]) -> str:
    import io
    output = io.StringIO()
    fields = ["source", "engine", "engine_version", "seed", "trials", "shape", "strategy", "current_category", "ev", "se", "fold_ev", "fold_se", "ev_difference", "paired_ev_difference_se", "break_even_danger", "current_dealer_ippatsu_contribution", "continuation_ev_after_safe", "current_hit_dealer", "current_hit_initial_side", "current_hit_any"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, "current_hit_dealer": row["current_hit_rates"]["dealer"], "current_hit_initial_side": row["current_hit_rates"]["initial_side"], "current_hit_any": row["current_hit_rates"]["any"]})
    return output.getvalue()


@dataclass(frozen=True)
class ScanFinding:
    path: Path
    line: int
    rule: str
    text: str


def scan_repository(root: Path = PROJECT_ROOT) -> list[ScanFinding]:
    """Find stale model language and verify the exact South-seat timepoint."""
    targets = [root / "outputs" / "statistical-mahjong-the-whys-cn.md", root / "outputs" / "iishanten_wall_sim.py", root / "outputs" / "iishanten_wall_sim.cpp"]
    findings: list[ScanFinding] = []
    old_source = re.compile(r"\b(?:Q_DEALER_GOOD_WAIT|SIDE_GOOD_WAIT_PROBABILITY)\b")
    old_text = re.compile(r"初始危险标签.*(?:边际|独立)|名义八张好形|筋只降低行为模型|筋.*不会被误当成绝对安全")
    stale_live_wall = re.compile(r"(?:\bLIVE_WALL_TILES\s*=\s*34\b|[\"']live_wall_tiles[\"']\s*:\s*34\b|活牌山[^。\n]{0,24}(?<!\d)34(?!\d)|(?<!\d)34(?!\d)[^。\n]{0,24}活牌山)")
    stale_unseen = re.compile(r"(?:\bUNKNOWN_POSITIONS\s*=\s*86\b|[\"']hero_unseen_positions[\"']\s*:\s*86\b|未见[^。\n]{0,24}(?<!\d)86(?!\d)|(?<!\d)86(?!\d)[^。\n]{0,24}未见)")
    stale_equal_rivers = re.compile(r"四家各(?:有)?(?:九|9)张(?:弃牌|牌河|河牌)|四家(?:牌河|河牌).*各(?:九|9)张")
    stale_next_dealer = re.compile(r"(?:当前弃牌|南家弃牌|我方弃牌)[^。\n]{0,80}(?:庄家(?:即将|随后|马上)下一摸|庄家下一摸)")
    for path in targets:
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            if "押降" in line:
                findings.append(ScanFinding(path, number, "rare-term", line.strip()))
            if old_source.search(line):
                findings.append(ScanFinding(path, number, "legacy-wait-prior", line.strip()))
            if old_text.search(line):
                findings.append(ScanFinding(path, number, "legacy-independent-danger-model", line.strip()))
            if stale_live_wall.search(line):
                findings.append(ScanFinding(path, number, "stale-34-live-wall", line.strip()))
            if stale_unseen.search(line):
                findings.append(ScanFinding(path, number, "stale-86-unseen-positions", line.strip()))
            if stale_equal_rivers.search(line):
                findings.append(ScanFinding(path, number, "stale-four-equal-rivers", line.strip()))
            if stale_next_dealer.search(line) and not ("西家" in line and "北家" in line and line.find("西家") < line.find("北家") < line.find("庄家", line.find("北家"))):
                findings.append(ScanFinding(path, number, "stale-north-seat-action-order", line.strip()))
    document = root / "outputs" / "statistical-mahjong-the-whys-cn.md"
    if document.is_file():
        text = document.read_text(encoding="utf-8")
        if not re.search(r"活牌山[^。\n]{0,30}(?<!\d)36(?!\d)|(?<!\d)36(?!\d)[^。\n]{0,30}活牌山", text):
            findings.append(ScanFinding(document, 0, "missing-36-live-wall", "正文必须明确南家精确时点的活牌山为 36 张"))
        if not re.search(r"未见[^。\n]{0,30}(?<!\d)88(?!\d)|(?<!\d)88(?!\d)[^。\n]{0,30}未见", text):
            findings.append(ScanFinding(document, 0, "missing-88-unseen-positions", "正文必须明确南家精确时点的我方未见位置为 88 个"))
        river_checks = (
            re.search(r"庄家[^。\n]{0,20}(?:河|弃牌)[^。\n]{0,12}(?:九|9)\s*张|庄家[^。\n]{0,20}(?:九|9)\s*张[^。\n]{0,12}(?:河|弃牌)", text),
            re.search(r"(?:南家|我方)[^。\n]{0,20}(?:河|弃牌)[^。\n]{0,12}(?:九|9)\s*张|(?:南家|我方)[^。\n]{0,20}(?:九|9)\s*张[^。\n]{0,12}(?:河|弃牌)", text),
            re.search(r"西家[^。\n]{0,20}(?:河|弃牌)[^。\n]{0,12}(?:八|8)\s*张|西家[^。\n]{0,20}(?:八|8)\s*张[^。\n]{0,12}(?:河|弃牌)", text),
            re.search(r"北家[^。\n]{0,20}(?:河|弃牌)[^。\n]{0,12}(?:八|8)\s*张|北家[^。\n]{0,20}(?:八|8)\s*张[^。\n]{0,12}(?:河|弃牌)", text),
        )
        if not all(river_checks):
            findings.append(ScanFinding(document, 0, "missing-9-9-8-8-river-state", "正文必须明确庄家/南家/西家/北家河长为 9/9/8/8"))
        mathematical_approximation = re.search(r"数学[^。\n]{0,160}(?:近似|约)[^。\n]{0,80}(?<!\d)36(?!\d)[^。\n]{0,80}(?<!\d)88(?!\d)", text)
        exact_simulation = re.search(r"(?:模拟|实体)[^。\n]{0,200}(?:南家|精确|时点)[^。\n]{0,160}(?<!\d)36(?!\d)[^。\n]{0,80}(?<!\d)88(?!\d)", text)
        if not mathematical_approximation or not exact_simulation:
            findings.append(ScanFinding(document, 0, "missing-36-88-source-distinction", "正文必须区分数学层近似 36/88 与南家实体模拟精确时点 36/88"))
        if not re.search(r"(?:当前弃牌|候选牌)[^。\n]{0,100}(?:类别|现物|筋)[^。\n]{0,100}庄家[^。\n]{0,80}(?:九|9)\s*张[^。\n]{0,30}(?:河|弃牌)|庄家[^。\n]{0,80}(?:九|9)\s*张[^。\n]{0,30}(?:河|弃牌)[^。\n]{0,100}(?:当前弃牌|候选牌)", text):
            findings.append(ScanFinding(document, 0, "missing-current-category-dealer-river-source", "正文必须说明当前牌类别只由庄家自己的 9 张牌河构造"))
        if not re.search(r"西家[^。\n]{0,100}(?:摸|打)[^。\n]{0,100}北家[^。\n]{0,100}(?:摸|打)[^。\n]{0,100}庄家[^。\n]{0,80}摸", text):
            findings.append(ScanFinding(document, 0, "missing-south-seat-action-order", "正文必须说明当前弃牌后按西家、北家、庄家下一摸推进"))
        if not re.search(r"候选牌.*胜张|成为胜张.*贡献|不是.*每手.*等待形式", text):
            findings.append(ScanFinding(document, 0, "missing-koba-interpretation", "正文必须说明 koba 表是候选牌成为胜张的贡献，不是每手等待形式比例"))
    return findings


def print_scan(findings: Iterable[ScanFinding]) -> None:
    for item in findings:
        location = f"{item.path}:{item.line}" if item.line else str(item.path)
        print(f"{location}\t{item.rule}\t{item.text}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="action", required=True)
    validate = subparsers.add_parser("validate", help="validate one or more JSON result files")
    validate.add_argument("json_files", nargs="+", type=Path)
    validate.add_argument("--allow-calibration-residual", action="store_true", help="validate structure but do not enforce the 0.25 pp RMSE threshold")
    summarize = subparsers.add_parser("summarize", help="turn formal-run JSON into Markdown or CSV; never edits the article")
    summarize.add_argument("json_files", nargs="+", type=Path)
    summarize.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    summarize.add_argument("--output", type=Path)
    scan = subparsers.add_parser("scan", help="scan simulator/article consistency without editing files")
    scan.add_argument("--root", type=Path, default=PROJECT_ROOT)
    scan.add_argument("--strict", action="store_true", help="return non-zero when a finding exists")
    cross = subparsers.add_parser("cross-validate", help="compile/run C++ and compare it with Python using the common JSON contract")
    cross.add_argument("--python", dest="python_script", type=Path, default=OUTPUTS / "iishanten_wall_sim.py")
    cross.add_argument("--cpp-source", type=Path, default=OUTPUTS / "iishanten_wall_sim.cpp")
    cross.add_argument("--cpp-binary", type=Path)
    cross.add_argument("--compiler", default=shutil.which("c++") or "c++")
    cross.add_argument("--cxx-flag", action="append", default=[])
    cross.add_argument("--trials", type=int, default=10_000)
    cross.add_argument("--seed", type=int, default=20260809)
    cross.add_argument("--side-mode", choices=("full", "heads_up"), default="full")
    cross.add_argument("--shape", default="all")
    cross.add_argument("--hero-strategy", default="all")
    cross.add_argument("--side-riichi-probability", type=float, default=0.5)
    cross.add_argument("--current-category", choices=("all", "genbutsu", "suji37", "nosuji2378", "nosuji456"), default="all")
    cross.add_argument("--current-passed-tile", type=int)
    cross.add_argument("--weights", type=Path)
    cross.add_argument("--sigma", type=float, default=3.0)
    cross.add_argument("--report", type=Path, help="optional tab-separated comparison report")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.action == "validate":
            for path in args.json_files:
                document = load_json(path)
                errors = validate_document(document, strict_calibration=not args.allow_calibration_residual)
                if errors:
                    print(f"{path}: INVALID", file=sys.stderr)
                    print("\n".join(errors), file=sys.stderr)
                    return 1
                print(f"{path}: valid ({len(document['results'])} result cells)")
            return 0
        if args.action == "summarize":
            content = markdown_summary(collect_result_rows(args.json_files)) if args.format == "markdown" else csv_summary(collect_result_rows(args.json_files))
            if args.output:
                args.output.write_text(content, encoding="utf-8")
            else:
                print(content, end="" if args.format == "csv" else "\n")
            return 0
        if args.action == "scan":
            findings = scan_repository(args.root)
            print_scan(findings)
            return 1 if args.strict and findings else 0
        if not args.python_script.is_file():
            raise ContractError(f"Python engine not found: {args.python_script}")
        temporary: tempfile.TemporaryDirectory[str] | None = None
        if args.cpp_binary:
            cpp_executable = [str(args.cpp_binary)]
        else:
            binary, temporary = compile_cpp(args.cpp_source, args.compiler, args.cxx_flag)
            cpp_executable = [str(binary)]
        try:
            python_doc = run_engine([sys.executable, str(args.python_script)], args, "python")
            cpp_doc = run_engine(cpp_executable, args, "cpp")
            comparisons = compare_documents(python_doc, cpp_doc, args.sigma)
        finally:
            if temporary:
                temporary.cleanup()
        report = format_comparisons(comparisons)
        if args.report:
            args.report.write_text(report + "\n", encoding="utf-8")
        print(report)
        return 0 if all(item.passed for item in comparisons) else 1
    except ContractError as exc:
        print(f"cross-validation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
