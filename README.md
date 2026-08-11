# TREC RAG Baseline — Pyserini BM25

里程碑第 1 項:「跑通 Pyserini BM25 baseline(2024 資料)」的起始專案。

目標:對 TREC RAG 2024 的題目跑出 BM25 檢索結果(TREC run 格式),
並用官方 qrels 算出 nDCG@20 / nDCG@100 / Recall@100 —— 這個數字就是你之後
所有實驗(dense retrieval、reranking)的對照組。

## 環境需求

| 項目 | 說明 |
|---|---|
| Python 3.10 / 3.11 | 建議用 venv 或 conda 隔離 |
| Java 21 | Pyserini 底層是 Lucene。`java -version` 確認;macOS 可 `brew install openjdk@21` |
| 硬碟空間 | **完整索引預留 ≥ 60 GB**(msmarco-v2.1-doc-segmented 預建索引數十 GB);冒煙測試只需約 3 GB |
| 記憶體 | 16 GB 以上較順 |

## 安裝

```bash
cd trec-rag-baseline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "from pyserini.search.lucene import LuceneSearcher; print('pyserini OK')"
```

> 已知陷阱:pyserini 1.6 在 import 時會初始化 OpenAI client,環境沒有
> `OPENAI_API_KEY` 會直接報錯。專案的腳本都已自動帶入占位值;若你直接在
> Python/REPL 裡用 pyserini,記得先 `export OPENAI_API_KEY=dummy`。

## 使用順序

```bash
# 步驟 -1:離線端到端測試(0 下載,幾秒鐘;已在雲端沙箱驗證通過)
bash scripts/local_e2e_test.sh

# 步驟 0:冒煙測試(下載約 2.9 GB 的小索引,驗證 Java/Pyserini 都正常)
bash scripts/smoke_test.sh

# 步驟 1:下載 2024 題目與 UMBRELA qrels 到 data/(檔案很小)
bash scripts/download_data.sh

# 步驟 2:跑 BM25(第一次會自動下載完整預建索引,需要時間與空間!)
bash scripts/run_bm25.sh

# 步驟 3:評分
bash scripts/evaluate.sh
```

跑完步驟 3 你會看到三個數字(nDCG@20、nDCG@100、Recall@100)。
把它們記進 `notes/baseline-scores.md` —— 里程碑第 1 項完成,第 4 項(內部評分流程)也完成一半。

## 各檔案說明

```
scripts/
  smoke_test.sh       用小索引(MS MARCO v1 passage)驗證環境,不佔 60GB
  download_data.sh    下載 topics.rag24.test.txt 與 qrels(UMBRELA)到 data/
  run_bm25.sh         官方語料 BM25 檢索 → runs/run.bm25.rag24.test.txt
  run_bm25_dev.sh     RAGgy dev 集(開發用,qrels 內建於 pyserini,可先玩這個)
  evaluate.sh         用 trec_eval 對 run 檔評分
src/
  evaluate.py         ir_measures 版評分(輸出同 evaluate.sh,方便之後程式化比較多個 run)
  inspect_run.py      看某一題撈回的前 N 個段落原文(檢查檢索品質用)
runs/                 檢索結果(TREC run 格式,6 欄)
data/                 topics 與 qrels
notes/
  baseline-scores.md  記錄每次實驗的分數(建議每個 run 一列,附 git commit)
  paper-notes-template.md  論文筆記模板(里程碑第 5 項)
```

## TREC run 格式(6 欄,空白分隔)

```
qid Q0 docid rank score run_tag
2024-145979 Q0 msmarco_v2.1_doc_00_12345#3_67890 1 21.53 bm25-baseline
```

之後不管你用什麼方法(dense、rerank、RAG),只要輸出這個格式,評分流程完全共用。

## 下一步(對應里程碑第 2 項)

1. 在同一組 topics 上加 dense retrieval(如 BGE-M3 / Arctic-embed),輸出另一個 run 檔
2. 對 BM25 top-100 加 reranker(bge-reranker / RankZephyr),再輸出一個 run 檔
3. `python src/evaluate.py runs/*.txt --qrels data/qrels.rag24.test-umbrela-all.txt` 一次比較全部

## 參考

- 官方 Pyserini MS MARCO v2.1 實驗文件:https://github.com/castorini/pyserini/blob/master/docs/experiments-msmarco-v2.1.md
- TREC RAG 官網(topics/qrels 公告):https://trec-rag.github.io/
- Ragnarök(官方 RAG baseline 框架):https://arxiv.org/abs/2406.16828
