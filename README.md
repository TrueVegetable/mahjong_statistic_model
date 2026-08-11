# Statistical Mahjong: Why?

This repository contains a small, physically explicit riichi-mahjong simulator used to study attack, folding, wait quality, and expected value after a dealer riichi. The model uses legal local wait templates, real tile counts, rivers, visible tiles, furiten, suji/genbutsu information, side-player behavior, and shared Python/C++ simulation rules.

## Main files

- `outputs/iishanten_wall_sim.py` — readable Python reference model, calibration code, and tests.
- `outputs/iishanten_wall_sim.cpp` — faster C++ simulation engine.
- `outputs/statistical-mahjong-the-whys-cn.md` — Chinese research article.
- `outputs/iishanten_calibrated_weights.json` — calibrated wait-template weights.
- `tests/` — deterministic and physical-state checks.

## Run

Run the Python reference simulator:

```sh
python3 outputs/iishanten_wall_sim.py --trials 10000 --json
```

Build and run the C++ engine:

```sh
c++ -std=c++17 -O3 outputs/iishanten_wall_sim.cpp -o outputs/iishanten_wall_sim_cpp
./outputs/iishanten_wall_sim_cpp --trials 10000 --json
```

Run the checks:

```sh
python3 -m unittest discover -s tests -v
./outputs/iishanten_wall_sim_cpp --self-test
```

The simulator is intentionally a local-hand approximation rather than a complete mahjong hand parser. Results should therefore be interpreted as model-based comparisons, with Monte Carlo uncertainty and calibration residuals reported alongside the estimates.
