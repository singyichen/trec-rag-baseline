#!/usr/bin/env python
"""用 ir_measures 一次評多個 run 檔並排成比較表。

用法:
    python src/evaluate.py runs/run.bm25.rag24.test.txt [更多 run 檔...] \
        --qrels data/qrels.rag24.test-umbrela-all.txt

之後做 dense / rerank 實驗時,把所有 run 檔一起丟進來,直接看哪個方法贏。
"""

import argparse
from pathlib import Path

import ir_measures
from ir_measures import nDCG, R


METRICS = [nDCG @ 20, nDCG @ 100, R @ 100]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="TREC run 檔(可多個)")
    parser.add_argument(
        "--qrels",
        default="data/qrels.rag24.test-umbrela-all.txt",
        help="qrels 檔(預設:2024 UMBRELA)",
    )
    args = parser.parse_args()

    qrels_path = Path(args.qrels)
    if not qrels_path.exists():
        raise SystemExit(f"找不到 qrels:{qrels_path}(先跑 scripts/download_data.sh)")
    qrels = list(ir_measures.read_trec_qrels(str(qrels_path)))

    header = ["run"] + [str(m) for m in METRICS]
    print("\t".join(header))
    for run_path in args.runs:
        run = list(ir_measures.read_trec_run(run_path))
        scores = ir_measures.calc_aggregate(METRICS, qrels, run)
        row = [Path(run_path).name] + [f"{scores[m]:.4f}" for m in METRICS]
        print("\t".join(row))


if __name__ == "__main__":
    main()
