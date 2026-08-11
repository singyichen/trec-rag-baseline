#!/usr/bin/env bash
# 對 TREC RAG 2024 測試題目跑 BM25。
#
# ⚠️ 第一次執行會自動下載 msmarco-v2.1-doc-segmented 預建索引(數十 GB),
#    下載位置預設在 ~/.cache/pyserini/。請確認硬碟空間 ≥ 60 GB。
#
# topics 用 pyserini 內建註冊名 rag24.test(等同 data/topics.rag24.test.txt)。
# 指令依官方文件:
# https://github.com/castorini/pyserini/blob/master/docs/experiments-msmarco-v2.1.md
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs

source scripts/env.sh
THREADS="${THREADS:-8}"

python3 -m pyserini.search.lucene \
  --threads "$THREADS" --batch-size 128 \
  --index msmarco-v2.1-doc-segmented \
  --topics rag24.test \
  --output runs/run.bm25.rag24.test.txt \
  --bm25 --hits 100

echo "=== 完成 ==="
echo "run 檔:runs/run.bm25.rag24.test.txt"
head -3 runs/run.bm25.rag24.test.txt
