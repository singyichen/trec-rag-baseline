#!/usr/bin/env bash
# 離線端到端測試:不需要下載任何索引(0 網路流量)。
# 建 5 筆小語料 → Lucene 建索引 → BM25 檢索 → 輸出 TREC run → ir_measures 評分。
# 全部通過代表 Java / pyserini / 評分流程都正常,可以放心跑正式流程。
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

rm -rf .e2e && mkdir -p .e2e/corpus

cat > .e2e/corpus/docs.jsonl <<'EOF'
{"id": "d1", "contents": "Retrieval augmented generation combines a retriever with a large language model to produce grounded answers."}
{"id": "d2", "contents": "BM25 is a classic lexical ranking function based on term frequency and inverse document frequency."}
{"id": "d3", "contents": "The Labor Standards Act regulates working hours, overtime pay, and severance in Taiwan."}
{"id": "d4", "contents": "Dense retrieval encodes queries and passages into vectors and searches by similarity."}
{"id": "d5", "contents": "TREC is a long running evaluation conference for information retrieval systems."}
EOF

printf 'q1\twhat is retrieval augmented generation\nq2\thow does bm25 rank documents\n' > .e2e/topics.tsv
printf 'q1 0 d1 2\nq1 0 d4 1\nq2 0 d2 2\n' > .e2e/qrels.txt

python -m pyserini.index.lucene \
  --collection JsonCollection --input .e2e/corpus --index .e2e/index \
  --generator DefaultLuceneDocumentGenerator --threads 1 --storeRaw

python -m pyserini.search.lucene \
  --index .e2e/index --topics .e2e/topics.tsv \
  --output .e2e/run.txt --bm25 --hits 5

echo "--- TREC run 檔 ---"
cat .e2e/run.txt
echo "--- 評分 ---"
python src/evaluate.py .e2e/run.txt --qrels .e2e/qrels.txt
echo
echo "OK:索引 → 檢索 → run 檔 → 評分,整條 pipeline 正常。"
