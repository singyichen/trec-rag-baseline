#!/usr/bin/env bash
# 下載 TREC RAG 2024 的 topics 與 UMBRELA qrels(檔案都很小)。
# 來源:https://trec-rag.github.io/annoucements/umbrela-qrels/
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

BASE="https://trec-rag.github.io/assets/txt"

curl -fL --retry 3 -o data/topics.rag24.test.txt \
  "$BASE/topics.rag24.test.txt"
curl -fL --retry 3 -o data/qrels.rag24.test-umbrela-all.txt \
  "$BASE/qrels.rag24.test-umbrela-all.txt"

echo "=== 下載完成 ==="
wc -l data/topics.rag24.test.txt data/qrels.rag24.test-umbrela-all.txt
echo
echo "topics 前 3 行(格式:qid<TAB>問句):"
head -3 data/topics.rag24.test.txt
