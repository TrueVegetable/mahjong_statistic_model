#!/usr/bin/env python3
"""Extract publication tables from the validated mixed-mean experiment."""
import csv
import hashlib
import json
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent / "push_fold_experiments/20260909"
DEFINITIONS = (
    ("现物", "all", "all", "genbutsu"),
    ("筋一、九", "category", "19", "suji"),
    ("非现物字牌", "category", "honor", "honor"),
    ("双筋四、六", "category", "46", "double_suji"),
    ("双筋五", "category", "5", "double_suji"),
    ("筋二、八", "category", "28", "suji"),
    ("筋三、七", "category", "37", "suji"),
    ("无筋一、九", "category", "19", "no_suji"),
    ("半筋四、六", "category", "46", "half_suji"),
    ("半筋五", "category", "5", "half_suji"),
    ("无筋二、八", "category", "28", "no_suji"),
    ("无筋三、七", "category", "37", "no_suji"),
    ("无筋四、六", "category", "46", "no_suji"),
    ("无筋五", "category", "5", "no_suji"),
)


def main():
    source = DIRECTORY / "mixed_mean_20k.json"
    data = json.loads(source.read_text())
    assert data["tables"] == 20000
    assert data["source_policy"]["mode"] == "mixed"
    assert data["source_policy"]["safety_strategy"] == "mean_safety"
    index = {}
    for row in data["danger_rows"]:
        if row["riichi_opponents"] != 1 or row["ron_scope"] != "riichi_opponents":
            continue
        key = (row["observer_turn"], row["tile_level"], row["tile"], row["safety_profile"])
        assert key not in index
        index[key] = row
    lines = ["| 持有牌类 | 第 6 巡 | 第 9 巡 | 第 12 巡 | 第 15 巡 |",
             "|---|---:|---:|---:|---:|"]
    rows = []
    for label, level, tile, safety in DEFINITIONS:
        cells = []
        for turn in (6, 9, 12, 15):
            row = index[str(turn), level, tile, safety]
            n, h = row["candidate_opportunities"], row["ron_hits"]
            assert n > 0 and 0 <= h <= n
            assert abs(row["ron_hit_probability"] - h / n) < 1e-12
            assert row["independent_tables"] == 20000
            cells.append(f"{100*h/n:.2f}%（{n}）")
            rows.append({"label": label, **row})
        lines.append("| " + " | ".join((label, *cells)) + " |")
    table = "\n".join(lines)
    stem = DIRECTORY / "tile_danger_mixed_mean_by_turn"
    metadata = {
        "source": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "policy": data["source_policy"], "source_metadata": data["source_metadata"],
        "selection": "one declared opponent; observer's own turn; held distinct tile type per decision; declared-opponent ron only",
        "uncertainty": "table-clustered; 20000 independent tables including zero exposure",
        "rows": rows,
    }
    stem.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    with stem.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = """# 混合攻守策略的逐巡持牌危险度

主表采用平均安全等级优先版本，种子 20260809，20000 桌。每格为面对恰好一家立直对手时，持有候选牌种的荣和命中率；括号是候选观察数，同桌重复决策按桌聚类。自家已经立直的观察者仍包含在内。现物优先单列，其余类别均不含现物。表格保留明示宝牌与赤牌的实际混合比例，没有把“宝牌降低一级”的行为规则当作概率修正。

""" + table + """

第 6 巡双筋五仅有 143 次候选观察，标准误为 2.17 个百分点；双筋四、六为 380 次，标准误为 0.94 个百分点。第 12 巡双筋四、六、半筋四、六、无筋四、六的标准误分别为 0.289、0.189、0.231 个百分点。完整命中数、出现桌数和聚类标准误见同名 JSON/CSV。

这是持牌候选的条件危险度。全巡合并的策略所选弃牌命中率约为 1.29%，采用另一分母，不能替代主表。后巡单元还取决于尚未和牌以及当时仍只有一家立直对手的选择条件。字牌汇总混合了自家持有数和公开张数，不能直接套用到所有未现字牌。

两家立直时，每张牌有两套现物与筋状态；应按安全关系对记录至少一家荣和的联合命中率，不能把本表加倍。全部 34 种精确牌和其他巡数见 [mixed_mean_20k.csv](mixed_mean_20k.csv)。最危险一家优先版本与其他策略见 [comparison.md](comparison.md)。

非现物字牌进一步按自家手牌加公开张数细分，见[字牌可见张数表](honor_danger_by_visible_copies.md)。该表来自同一组牌桌，各巡命中数与候选数均与本表字牌行相符。
"""
    stem.with_suffix(".md").write_text(report)
    Path("/private/tmp/push-fold-paper-danger-table.md").write_text(table + "\n")
    print(f"Validated {len(rows)} cells; wrote {stem.name}.md/.json/.csv")


if __name__ == "__main__":
    main()
