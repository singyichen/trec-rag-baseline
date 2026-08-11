#!/usr/bin/env bash
# 冒煙測試:用小索引(約 2.9 GB)驗證 Java + Pyserini 環境正常,
# 不需要先下載 60GB 的完整索引。
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/env.sh

python - <<'PY'
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
hits = searcher.search('what is retrieval augmented generation', k=3)
print('\n=== 冒煙測試結果(msmarco-v1-passage)===')
for i, h in enumerate(hits, 1):
    print(f'{i}. docid={h.docid}  score={h.score:.4f}')
assert len(hits) > 0
print('\nOK:Java 與 Pyserini 都正常,可以進行完整流程。')
PY
