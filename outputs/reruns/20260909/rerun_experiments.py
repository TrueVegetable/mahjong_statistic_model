#!/usr/bin/env python3
"""Reproduce six saved experiments from their recorded policy settings."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]
SOURCES = [
    "hand_monte_carlo_value_reversal25_wait_matrix_100k.json",
    "shared_four_player_extremeA_dora1_continue_rejected_20k.json",
    "shared_four_player_extremeA_valuable_dama_3han40fu_20k.json",
    "shared_four_player_extremeA_valuable_dama_4han40fu_20k.json",
    "shared_four_player_extremeA_valuable_dama_3han40fu_haneman_20k.json",
    "shared_four_player_extremeA_valuable_dama_4han40fu_haneman_20k.json",
]


def command(source, binary, target, trials):
    run = json.loads(source.read_text())["run"]
    policy = run["policy_parameters"]
    reversal = policy["shanten_reversal"]
    argv = [sys.executable, str(OUTPUTS / "hand_monte_carlo_report.py"),
            "--engine", "native", "--native-binary", str(binary),
            "--policy", "value", "--trials", str(trials or run["trials"]),
            "--seed", str(run["seed"]), "--start-trial", str(run["start_trial"]),
            "--seat-wind", run["seat_wind"], "--round-wind", run["round_wind"],
            "--output", str(target), "--pretty"]
    for key in ("future_ukeire_weight", "improvement_weight", "dora_ukeire_weight",
                "dora_tiebreak_weight", "dora_neighbor_weight", "yakuhai_weight",
                "tanyao_weight"):
        argv.extend(["--" + key.replace("_", "-"), str(policy[key])])
    weights = policy.get("future_ukeire_weights_by_shanten", {})
    for shanten in ("1", "2"):
        argv.extend([f"--future-ukeire-weight-{shanten}-shanten",
                     str(weights.get(shanten, policy["future_ukeire_weight"]))])
    if reversal["enabled"]:
        argv.append("--allow-shanten-reversal")
    for key, flag in (("penalty", "penalty"), ("recovered_ukeire_weight", "future-weight"),
                      ("max_turn", "max-turn"), ("candidate_limit", "candidate-limit")):
        argv.extend(["--shanten-reversal-" + flag, str(reversal[key])])
    assert reversal["max_per_hand"] == 1
    if run.get("continue_rejected_tenpai"):
        argv.extend(["--shared-four-player-table", "--reject-cheap-bad-wait-riichi",
                     "--continue-rejected-tenpai"])
    if run.get("valuable_bad_wait_dama_line"):
        argv.extend(["--valuable-bad-wait-dama-line", run["valuable_bad_wait_dama_line"]])
    if run.get("haneman_or_higher_dama"):
        argv.append("--haneman-or-higher-dama")
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--benchmark", type=int, default=0,
                        help="Use this many trials per experiment; write under /private/tmp")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    target_dir = Path("/private/tmp/hand-fix-benchmark") if args.benchmark else HERE
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"stage": "benchmark" if args.benchmark else "full", "experiments": [],
                "source_sha256": {name: hashlib.sha256((OUTPUTS / name).read_bytes()).hexdigest()
                                  for name in SOURCES},
                "implementation_sha256": {name: hashlib.sha256((OUTPUTS / name).read_bytes()).hexdigest()
                                          for name in ("hand_efficiency_sim.py", "hand_efficiency_sim.cpp",
                                                       "hand_monte_carlo_report.py")}}

    def execute(name):
        target = target_dir / name
        argv = command(OUTPUTS / name, args.binary.resolve(), target, args.benchmark)
        started = time.monotonic()
        with (target_dir / (name + ".log")).open("w") as log:
            subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT)
        result = json.loads(target.read_text())
        entry = {"source": name, "command": argv, "elapsed_seconds": time.monotonic() - started,
                 "trials": result["trials"], "outcomes": result["outcomes"]}
        print(json.dumps(entry), flush=True)
        return entry

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(execute, name) for name in SOURCES]
        for future in as_completed(futures):
            manifest["experiments"].append(future.result())
            (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
