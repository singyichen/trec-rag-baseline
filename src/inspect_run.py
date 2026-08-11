#!/usr/bin/env python
"""看某一題撈回的前 N 個段落原文——肉眼檢查檢索品質用。

用法:
    python src/inspect_run.py --run runs/run.bm25.rag24.test.txt \
        --qid 2024-145979 --top 5

需要本機已有 msmarco-v2.1-doc-segmented 預建索引(run_bm25.sh 跑過就有)。
"""

import argparse
import json
from collections import defaultdict

from pyserini.search.lucene import LuceneSearcher


def load_run(path: str) -> dict[str, list[tuple[str, float]]]:
    run: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, docid, _, score, _ = line.split()
            run[qid].append((docid, float(score)))
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--qid", required=True)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    run = load_run(args.run)
    if args.qid not in run:
        raise SystemExit(f"run 檔裡沒有 qid={args.qid}")

    searcher = LuceneSearcher.from_prebuilt_index("msmarco-v2.1-doc-segmented")
    for rank, (docid, score) in enumerate(run[args.qid][: args.top], 1):
        doc = searcher.doc(docid)
        text = ""
        if doc is not None:
            raw = json.loads(doc.raw())
            text = (raw.get("segment") or raw.get("body") or "")[:500]
        print(f"--- rank {rank}  {docid}  score={score:.4f} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
