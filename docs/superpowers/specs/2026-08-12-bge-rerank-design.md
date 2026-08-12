# 里程碑 2:BGE reranker 重排 BM25 top-100 — 設計文件

日期:2026-08-12
狀態:已與使用者確認(選項 A:直接用 transformers)

## 目標

用 cross-encoder `BAAI/bge-reranker-v2-m3` 對現有 BM25 top-100 run
(`runs/run.bm25.rag24.test.txt`)重排,產出新 run 檔並用既有流程評分。

成功標準:
- 新 run 檔通過 `scripts/evaluate.sh` 評分,三指標可與 baseline 同表比較
- nDCG@20 顯著高於 BM25 的 0.3198(預期 ≥ 0.45)
- 結果記入 `notes/baseline-scores.md`

## 前提(已驗證)

- 本地索引 `~/indexes/lucene-inverted.msmarco-v2.1-doc-segmented.20240418.4f9675`
  為完整版,`LuceneSearcher.doc(docid).raw()` 可取回 JSON,含
  `title`、`headings`、`segment` 欄位
- `.venv` 已有 `torch 2.13`(MPS 可用)與 `transformers 5.15`,零新依賴
- 硬體 M1 Pro / 16GB;工作量 301 題 × 100 候選 = 30,100 對

## 架構

兩個新檔案,沿用專案既有慣例(shell wrapper + Python 核心):

### `src/rerank.py`

流程:
1. 讀 topics(tsv:qid \t query)與 run 檔(TREC 6 欄),彙整每題的候選 docid 清單
2. 若輸出檔已存在,先讀出已完成的 qid,跳過(**可續跑**)
3. 逐題:
   - 從 Lucene 索引取回每個候選的原文,passage 文字 = `title + " " + segment`
   - 以 batch(預設 16)餵入 reranker,fp16、MPS、`max_length=512`(pair 整體截斷)
   - 依分數重排,以 TREC 格式**立即 append 寫出**該題 100 列(增量寫出,
     中斷不丟已完成進度)
4. 分數直接用模型 raw logit(排序用途不需 sigmoid)

參數:
- `--run`(預設 `runs/run.bm25.rag24.test.txt`)
- `--topics`(預設 `data/topics.rag24.test.txt`)
- `--index`(預設本地索引路徑,可覆寫)
- `--output`(預設 `runs/run.bm25-bge.rag24.test.txt`)
- `--batch-size`(預設 16)
- `--max-queries N`:smoke 模式,只跑前 N 題,用來驗證流程與實測速度

run tag 欄用 `bm25-bge-v2-m3`。

### `scripts/rerank_bge.sh`

- source `scripts/env.sh`(啟 venv、設 JAVA_HOME)
- 檢查 run 檔與索引存在,不存在給中文指引訊息
- 轉傳參數給 `src/rerank.py`
- 檔頭註明:首次執行會下載 ~2.3GB 模型至 HuggingFace cache

## 錯誤處理

- 索引取不到某 docid:警告並給該候選極低分(沉底),不中斷整批
- MPS 不可用:自動退回 CPU 並警告(會慢很多但能跑)
- 中斷/當機:重跑同指令自動續接(靠已完成 qid 檢查)

## 測試與驗收

1. smoke:`--max-queries 3`,確認輸出格式正確(每題 100 列、分數遞減、
   docid 集合與 BM25 該題一致)並回報實測速度
2. 全量跑完:`scripts/evaluate.sh runs/run.bm25-bge.rag24.test.txt`
3. 分數列入 `notes/baseline-scores.md`,與 BM25 baseline 同表比較

## 明確不做(YAGNI)

- 不裝 FlagEmbedding
- 不做多模型/多超參掃描(之後的實驗再說)
- 不平行多進程(單機 MPS 是瓶頸,平行無益)
