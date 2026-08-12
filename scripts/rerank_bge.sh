#!/usr/bin/env bash
# 用 BAAI/bge-reranker-v2-m3 對 BM25 top-100 run 重排。
#
# ⚠️ 第一次執行會從 HuggingFace 下載模型(~2.3GB)到 ~/.cache/huggingface/。
# 中斷可直接重跑,會自動跳過已完成的題目。
#
# 用法:bash scripts/rerank_bge.sh [額外參數原樣傳給 src/rerank.py]
# 例:  bash scripts/rerank_bge.sh --max-queries 3   # smoke 測試
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/env.sh

RUN="runs/run.bm25.rag24.test.txt"
[ -f "$RUN" ] || { echo "找不到 $RUN,先跑 bash scripts/run_bm25.sh"; exit 1; }

INDEX="${INDEX:-$HOME/indexes/lucene-inverted.msmarco-v2.1-doc-segmented.20240418.4f9675}"
[ -d "$INDEX" ] || { echo "找不到索引 $INDEX(可用 INDEX=... 覆寫)"; exit 1; }

python src/rerank.py --index "$INDEX" "$@"

echo "=== 完成 ==="
echo "run 檔:runs/run.bm25-bge.rag24.test.txt"
