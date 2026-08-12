# 實驗分數紀錄

每跑一個 run 就記一列。同一組 topics + qrels 的分數才能互相比較。

| 日期 | run 檔 | 方法 | nDCG@20 | nDCG@100 | Recall@100 | 備註 |
|---|---|---|---|---|---|---|
| 2026-08-12 | run.bm25.rag24.test.txt | BM25(k=100) | 0.3198 | 0.2563 | 0.1395 | 對照組;與 Anserini 官方 regression 完全一致 ✅ |
