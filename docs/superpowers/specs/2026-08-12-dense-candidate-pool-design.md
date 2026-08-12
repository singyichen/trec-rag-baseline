# 里程碑 3:加深候選池 + bge-m3 dense 混合檢索 — 設計文件

日期:2026-08-12
狀態:已與使用者確認(含 RRF 融合)

## 目標

解決 Recall 瓶頸(BM25 top-100 的 Recall@100 = 0.1395):
BM25 取 top-1000 → bge-m3 dense 分數篩回 top-100(純 dense 與 RRF 融合兩版)
→ 既有 cross-encoder 精排 → 評分。

成功標準:
- stage-1 新 run 的 Recall@100 > 0.1395(預期 0.20+)
- 最終 nDCG@20 > 0.4707(預期 0.50+)
- 四個 run(bm25-1000 / dense100 / rrf100 / 最終)分數記入 `notes/baseline-scores.md`

## 前提(已驗證)

- 本地 Lucene 索引可取回 `title`/`segment` 原文(里程碑 2 已用)
- `torch 2.13`(MPS)+ `transformers 5.15` 已裝,**零新依賴**
- bge-m3(BAAI/bge-m3,~2.3GB)查詢端不需 instruction 前綴;dense 向量
  = last_hidden_state 的 CLS(第 0 個 token)+ L2 normalize
- 工作量:301 題 × 1000 候選,跨題去重後約 25–30 萬 segment;
  M1 Pro MPS 嵌入約 3 小時

## 架構

### 1. `scripts/run_bm25.sh`(小改,預設行為不變)

加 `HITS`(預設 100)與 `OUTPUT`(預設原檔名)環境變數覆寫。
產出 top-1000 用:
`HITS=1000 OUTPUT=runs/run.bm25-1000.rag24.test.txt bash scripts/run_bm25.sh`

### 2. `src/embed_select.py`(新)

從 `rerank.py` import 共用函式(`load_topics`、`load_run`、`passage_text`、
`format_trec_lines`),新增:

- `unique_docids(run) -> list[str]`:跨題去重,保留首次出現順序
- `rrf_scores(rankings, k=60) -> dict[str, float]`:
  rankings 為多個「docid 依名次排序清單」,分數 = Σ 1/(k + rank),rank 從 1 起
- 嵌入快取:向量按批次(4096 筆/塊)存 `embeddings/chunk_XXXXX.npy`,
  docid 順序寫 `embeddings/docids.txt`(一次性,重跑時驗證一致);
  **續跑** = 跳過已存在且列數完整的塊
- 主流程:
  1. 讀 run(bm25-1000)、topics;建 unique docid 清單
  2. 從索引取原文(`passage_text`),bge-m3 嵌入(fp16/MPS、truncate 512、
     batch 預設 16),分塊寫入快取
  3. 嵌入 301 個 query(量小不快取)
  4. 每題:dense 分數 = query·doc 內積 → 取 top-100 → 寫
     `runs/run.dense100.rag24.test.txt`(tag `bm25-1000-bge-m3-dense`)
  5. 每題:RRF 融合 BM25 名次與 dense 名次 → 取 top-100 → 寫
     `runs/run.rrf100.rag24.test.txt`(tag `bm25-1000-bge-m3-rrf`)
- CLI:`--run --topics --index --cache-dir --dense-output --rrf-output
  --batch-size --max-queries`(smoke 模式)
- MPS 不可用退回 CPU 並警告;索引取不到的 docid 警告後排除於候選之外

### 3. 精排與評分(全部用現成件)

- Recall 較好的 stage-1 run 餵 `src/rerank.py --run <該檔> --output
  runs/run.final.rag24.test.txt`
- `scripts/evaluate.sh` 評 nDCG@20/100、Recall@100;
  bm25-1000 另用 `-m recall.1000` 手動評一次(診斷用)

## 錯誤處理

- 快取 docids.txt 與現有 run 不一致(換了 run 檔重跑):報錯退出,
  提示刪 `embeddings/` 重來,不靜默混用
- 嵌入中斷:重跑自動從最後完整塊續起
- smoke 模式用獨立 cache 子目錄,避免污染全量快取

## 測試與驗收

1. 單元測試:`unique_docids` 去重保序、`rrf_scores` 數學正確
   (手算小例)、選取 top-100 排序正確
2. smoke(3 題):dense100/rrf100 每題 100 列、docid ⊆ 該題 bm25-1000、
   分數降冪;實測嵌入速度回報
3. 全量:四個 run 全評分,記入 notes,dashboard 同步

## 明確不做(YAGNI)

- 不下載 dense 預建索引(硬碟不足)、不做全語料嵌入
- 不裝 FlagEmbedding / sentence-transformers
- 不做加權線性融合調參(RRF 免調參,夠用;調參留待之後)
- 不用 bge-m3 的 sparse / ColBERT 輸出(先驗證 dense 主線)

## 其他

- `embeddings/` 加入 `.gitignore`
- 預估總時程:BM25-1000 ~15 分 + 嵌入 ~3 小時 + 精排 ~50 分,
  長任務以 nohup + 監控背景執行
