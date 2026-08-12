# 閱讀筆記:Beyond RAG — LlamaIndex Workshop 導讀(2026-07)

- 來源:https://blog.aihao.tw/2026/07/26/beyond-rag-llamaindex-workshop/
  (Jerry Liu @ AI Engineer Singapore 2026 工作坊投影片導讀)
- 記錄日期:2026-08-12
- 視角提醒:文章講的是**企業 RAG 產品**;我們是 **TREC 評測賽道**,
  適用性要逐項過濾。

## 對本專案不適用(勿被帶偏)

- 核心論點「瓶頸在文件解析」:TREC 用官方預切 segment,無 PDF 可解析。
  spatial text / LlamaParse / LiteParse / ParseBench / 結構化分塊 → 全部略過。
- 向量庫 upsert、活的知識庫、權限治理:持續運營系統的需求,批次實驗無關
  (同理:不需要 pgvector,見 2026-08-12 討論——候選池向量用 npy 檔即可)。

## 驗證我們現有做法

- 「Reranking ROI 第一,nDCG@10 典型 +4~14 分」→ 里程碑 2 實測 nDCG@20
  +15.1 分(0.3198→0.4707),打到區間上緣。
- 「bi-encoder 撈廣 → cross-encoder 精排」→ 即里程碑 3 架構。
- 評測黃金法則「無法量測的改動是 Demo」→ baseline-scores.md + 固定 qrels
  + trec_eval 流程已符合。

## 可採用(按優先序)

### 近期:檢索任務

1. **LLM query 改寫/分解 + RRF 融合**(文章 ROI 排序第 2,里程碑 4 候選)
   - 每題 LLM 產 2–3 個改寫(多面向分解、縮寫展開),各自檢索後 RRF 融合
   - `embed_select.py` 的 `rrf_scores()` 可直接重用,增量成本低
   - 主攻長尾難題(BM25 與 dense 都撈不到的)
2. **更強 reranker 對照組**:Voyage Rerank-2.5(有免費額度)/ Cohere
   Rerank-v3.5,拿現成 stage-1 run 換 reranker 即可對照,半天實驗
3. **CRAG 簡化版(信心分流)**:reranker 最高分過低的題 = 檢索失敗訊號,
   只對這些題做改寫重撈;文章「該加」三條件(長尾失敗/多跳/代價高)全中

### 中期:RAG 生成任務(2026 Q4)

4. **三指標對映**:Faithfulness/Relevancy/Recall ↔ TREC 的 Support /
   AutoNuggetizer / UMBRELA——生成 pipeline 的內部評測按這三軸設計
5. **引用帶出處 + 信心分數為一級輸出**:每句標 segment 出處;
   metadata 三層(檢索過濾/合成引用/稽核追溯)作 pipeline 設計參考

### 流程觀念

6. 「做完就停,多數系統只需 1–3 項」:檢索側做到 rerank + dense +
   query 改寫即近邊際遞減點,之後分數從生成側與 LLM listwise 重排來
7. HyDE / Multi-Query 已被 agentic 動態改寫吸收(2024 技巧,不必補課)
8. Token 成本預算、版本化一切(parser/索引/prompt/schema):
   進生成階段後採用
