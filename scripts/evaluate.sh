#!/usr/bin/env bash
# 用 trec_eval 對 run 檔評分(track 建議指標:nDCG@20、nDCG@100、Recall@100)。
# 用法:bash scripts/evaluate.sh [run檔,預設 runs/run.bm25.rag24.test.txt]
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="${1:-runs/run.bm25.rag24.test.txt}"
QRELS="data/qrels.rag24.test-umbrela-all.txt"

[ -f "$QRELS" ] || { echo "找不到 $QRELS,先跑 bash scripts/download_data.sh"; exit 1; }
[ -f "$RUN" ] || { echo "找不到 $RUN,先跑 bash scripts/run_bm25.sh"; exit 1; }

python -m pyserini.eval.trec_eval -c \
  -m ndcg_cut.20 -m ndcg_cut.100 -m recall.100 \
  "$QRELS" "$RUN"
