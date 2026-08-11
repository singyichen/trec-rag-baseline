#!/usr/bin/env bash
# 開發用:RAGgy dev 集(topics 與 qrels 都內建於 pyserini,不用另外下載)。
# 與官方文件相同的參數:--max-passage 把段落分數聚合回文件層級。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs

source scripts/env.sh
THREADS="${THREADS:-8}"

python -m pyserini.search.lucene \
  --threads "$THREADS" --batch-size 128 \
  --index msmarco-v2.1-doc-segmented \
  --topics rag24.raggy-dev \
  --output runs/run.bm25.raggy-dev.txt \
  --bm25 --hits 10000 --max-passage-hits 1000 --max-passage

python -m pyserini.eval.trec_eval -c -M 100 -m ndcg_cut.10 \
  rag24.raggy-dev runs/run.bm25.raggy-dev.txt
