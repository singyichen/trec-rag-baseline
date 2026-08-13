# 實驗分數紀錄

每跑一個 run 就記一列。同一組 topics + qrels 的分數才能互相比較。

| 日期 | run 檔 | 方法 | nDCG@20 | nDCG@100 | Recall@100 | 備註 |
|---|---|---|---|---|---|---|
| 2026-08-12 | run.bm25.rag24.test.txt | BM25(k=100) | 0.3198 | 0.2563 | 0.1395 | 對照組;與 Anserini 官方 regression 完全一致 ✅ |
| 2026-08-12 | run.bm25-bge.rag24.test.txt | BM25 top-100 + bge-reranker-v2-m3 | 0.4707 | 0.2986 | 0.1395 | Recall@100 與 BM25 相同(同一候選集);nDCG@20 +0.15 |
| 2026-08-14 | run.bm25-1000.rag24.test.txt | BM25(k=1000,候選池) | — | — | — | Recall@1000 = 0.3467(池天花板) |
| 2026-08-14 | run.dense100.rag24.test.txt | BM25-1000 → bge-m3 dense top-100 | 0.4679 | 0.3761 | 0.1960 | stage-1;Recall +40% vs BM25 top-100 |
| 2026-08-14 | run.rrf100.rag24.test.txt | BM25-1000 → RRF(BM25+dense) top-100 | 0.4447 | 0.3787 | 0.2038 | stage-1;Recall 最高,選入精排 |
| 2026-08-14 | run.final.rag24.test.txt | RRF top-100 + bge-reranker-v2-m3 | 0.5400 | 0.4050 | 0.2038 | 里程碑 3 ✅;vs 里程碑 2 +0.07 |
