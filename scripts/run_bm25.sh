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
# HITS / OUTPUT 可覆寫,預設維持原行為(top-100、原檔名)。
# 例:HITS=1000 OUTPUT=runs/run.bm25-1000.rag24.test.txt bash scripts/run_bm25.sh
HITS="${HITS:-100}"
OUTPUT="${OUTPUT:-runs/run.bm25.rag24.test.txt}"

# 用本地題目檔(scripts/download_data.sh 下載的),
# 不用內建名稱 rag24.test —— 後者會連 raw.githubusercontent.com 抓檔,某些網路環境連不上。
TOPICS="data/topics.rag24.test.txt"
[ -f "$TOPICS" ] || { echo "找不到 $TOPICS,先跑 bash scripts/download_data.sh"; exit 1; }

# INDEX 可覆寫:預設用 pyserini 自動下載;若自動下載連不上,
# 手動下載解壓後把資料夾路徑傳進來,例如:
#   INDEX=~/indexes/lucene-inverted.msmarco-v2.1-doc-segmented.20240418.4f9675 bash scripts/run_bm25.sh
if [[ -z "${INDEX:-}" ]]; then
  LOCAL_INDEX="$HOME/indexes/lucene-inverted.msmarco-v2.1-doc-segmented.20240418.4f9675"
  if [[ -d "$LOCAL_INDEX" ]]; then INDEX="$LOCAL_INDEX"; else INDEX="msmarco-v2.1-doc-segmented"; fi
fi

python -m pyserini.search.lucene \
  --threads "$THREADS" --batch-size 128 \
  --index "$INDEX" \
  --topics "$TOPICS" \
  --output "$OUTPUT" \
  --bm25 --hits "$HITS"

echo "=== 完成 ==="
echo "run 檔:$OUTPUT"
head -3 "$OUTPUT"
