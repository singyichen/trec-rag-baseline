# Dense 候選池(里程碑 3)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** BM25 top-1000 → bge-m3 dense 篩 top-100(純 dense 與 RRF 兩版)→ 既有 cross-encoder 精排,Recall@100 > 0.1395、最終 nDCG@20 > 0.4707。

**Architecture:** `run_bm25.sh` 加 HITS/OUTPUT 覆寫產出 top-1000;新 `src/embed_select.py` 做候選嵌入(分塊快取、可續跑)與兩種 top-100 選取;精排與評分全用現成件(`rerank.py`、`evaluate.sh`)。

**Tech Stack:** pyserini、transformers 5.x(AutoModel bge-m3)、torch 2.13(MPS)、numpy、pytest。

## Global Constraints

- 零新依賴(numpy 隨 pyserini 已裝);模型 `BAAI/bge-m3`,CLS pooling + L2 normalize,fp16/MPS(CPU fallback 警告),truncate 512,batch 預設 16
- tag:dense 版 `bm25-1000-bge-m3-dense`、RRF 版 `bm25-1000-bge-m3-rrf`;RRF k=60
- 嵌入快取:4096 筆/塊 → `<cache>/chunk_XXXXX.npy`,docid 順序存 `<cache>/docids.txt`;不一致即報錯退出;索引缺 docid → 零向量(選取時以 norm==0 偵測並排除)
- `embeddings/` 進 `.gitignore`;smoke 用 `--cache-dir embeddings/smoke`
- 共用函式一律從 `rerank` import,不複製
- 腳本註解訊息繁體中文;commit 結尾 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 1: run_bm25.sh 參數化 + 產出 BM25 top-1000 run

**Files:**
- Modify: `scripts/run_bm25.sh`
- Modify: `.gitignore`(加 `embeddings/`)

**Interfaces:**
- Produces: `runs/run.bm25-1000.rag24.test.txt`(301 題 × 1000 列);
  `HITS`/`OUTPUT` 環境變數(預設 100 / 原檔名,舊行為不變)

- [ ] **Step 1: 修改 run_bm25.sh**

把固定的 output/hits 換成:

```bash
HITS="${HITS:-100}"
OUTPUT="${OUTPUT:-runs/run.bm25.rag24.test.txt}"
```

`python -m pyserini.search.lucene` 的參數改為 `--output "$OUTPUT" --bm25 --hits "$HITS"`,
結尾的 echo 與 head 改用 `$OUTPUT`。

- [ ] **Step 2: .gitignore 加 embeddings/**

- [ ] **Step 3: 語法檢查與預設行為確認**

Run: `bash -n scripts/run_bm25.sh && grep -E 'HITS|OUTPUT' scripts/run_bm25.sh`
Expected: 無語法錯誤;預設值 100 與原檔名

- [ ] **Step 4: 產出 top-1000 run(背景,約 15–40 分鐘)**

Run: `HITS=1000 OUTPUT=runs/run.bm25-1000.rag24.test.txt bash scripts/run_bm25.sh`
Expected: 完成後 `wc -l runs/run.bm25-1000.rag24.test.txt` = 301000

- [ ] **Step 5: Commit**

```bash
git add scripts/run_bm25.sh .gitignore
git commit -m "feat: run_bm25.sh 支援 HITS/OUTPUT 覆寫"
```

---

### Task 2: 純函式(unique_docids、rrf_scores、top_k)+ 單元測試

**Files:**
- Create: `src/embed_select.py`(本 task 只含純函式)
- Modify: `tests/test_rerank.py` → 新增 `tests/test_embed_select.py`

**Interfaces:**
- Consumes: 無(純函式)
- Produces(Task 3 使用,簽名一字不差):
  - `unique_docids(run) -> list[str]`  # run: dict[str, list[str]];跨題去重、保留首次出現順序
  - `rrf_scores(rankings, k=60) -> dict[str, float]`  # rankings: list[list[str]],分數 = Σ 1/(k+rank),rank 從 1 起
  - `top_k(pairs, k) -> list[tuple[str, float]]`  # 依分數降冪(穩定)取前 k

- [ ] **Step 1: 寫失敗測試**

`tests/test_embed_select.py`:

```python
import pytest

from embed_select import unique_docids, rrf_scores, top_k


def test_unique_docids_dedup_preserves_order():
    run = {"q1": ["a", "b", "c"], "q2": ["b", "d", "a"]}
    assert unique_docids(run) == ["a", "b", "c", "d"]


def test_rrf_scores_math():
    scores = rrf_scores([["a", "b"], ["a", "b"]], k=1)
    assert scores["a"] == pytest.approx(2 * (1 / 2))
    assert scores["b"] == pytest.approx(2 * (1 / 3))


def test_rrf_scores_disjoint_docs():
    scores = rrf_scores([["a"], ["b"]], k=1)
    assert scores["a"] == pytest.approx(1 / 2)
    assert scores["b"] == pytest.approx(1 / 2)


def test_top_k_sorts_desc_and_truncates():
    pairs = [("a", 1.0), ("b", 3.0), ("c", 2.0)]
    assert top_k(pairs, 2) == [("b", 3.0), ("c", 2.0)]
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `source scripts/env.sh && python -m pytest tests/test_embed_select.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'embed_select'`

- [ ] **Step 3: 最小實作**

`src/embed_select.py`:

```python
"""bge-m3 候選池嵌入與 top-100 選取(dense 與 RRF 融合兩版)。

讀 BM25 top-1000 run,從 Lucene 索引取回候選原文,bge-m3 嵌入
(分塊快取、可續跑),對每題產出 dense top-100 與 RRF top-100 兩個 run 檔。
"""


def unique_docids(run):
    """跨題去重 docid,保留首次出現順序。"""
    seen = set()
    out = []
    for docids in run.values():
        for d in docids:
            if d not in seen:
                seen.add(d)
                out.append(d)
    return out


def rrf_scores(rankings, k=60):
    """Reciprocal Rank Fusion:分數 = Σ 1/(k + rank),rank 從 1 起。"""
    scores = {}
    for ranking in rankings:
        for rank, docid in enumerate(ranking, start=1):
            scores[docid] = scores.get(docid, 0.0) + 1.0 / (k + rank)
    return scores


def top_k(pairs, k):
    """依分數降冪(穩定排序)取前 k 個 (docid, score)。"""
    return sorted(pairs, key=lambda x: -x[1])[:k]
```

- [ ] **Step 4: 跑測試確認通過**

Run: `source scripts/env.sh && python -m pytest tests/ -v`
Expected: 11 passed(7 舊 + 4 新)

- [ ] **Step 5: Commit**

```bash
git add src/embed_select.py tests/test_embed_select.py
git commit -m "feat: embed_select 純函式(去重/RRF/top-k)+ 單元測試"
```

---

### Task 3: 嵌入快取 + 主流程 + smoke(3 題)

**Files:**
- Modify: `src/embed_select.py`(追加,不動 Task 2 的函式)

**Interfaces:**
- Consumes: Task 2 純函式;`rerank.py` 的 `load_topics`/`load_run`/`passage_text`/`format_trec_lines`
- Produces: CLI — `python src/embed_select.py [--run --topics --index --cache-dir --dense-output --rrf-output --batch-size --max-queries]`

- [ ] **Step 1: 追加實作(embed_select.py)**

```python
EMB_MODEL = "BAAI/bge-m3"
DENSE_TAG = "bm25-1000-bge-m3-dense"
RRF_TAG = "bm25-1000-bge-m3-rrf"
CHUNK = 4096
SELECT_K = 100


def encode_texts(model, tokenizer, texts, device, batch_size):
    """CLS pooling + L2 normalize,回傳 (n, dim) float32 numpy。"""
    import numpy as np
    import torch

    vecs = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=512, return_tensors="pt").to(device)
            out = model(**inputs).last_hidden_state[:, 0]
            out = torch.nn.functional.normalize(out, dim=-1)
            vecs.append(out.float().cpu().numpy())
    return np.concatenate(vecs)


def main():
    import argparse
    import json
    import os
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/run.bm25-1000.rag24.test.txt")
    parser.add_argument("--topics", default="data/topics.rag24.test.txt")
    parser.add_argument("--index", default=os.path.expanduser(
        "~/indexes/lucene-inverted.msmarco-v2.1-doc-segmented.20240418.4f9675"))
    parser.add_argument("--cache-dir", default="embeddings")
    parser.add_argument("--dense-output", default="runs/run.dense100.rag24.test.txt")
    parser.add_argument("--rrf-output", default="runs/run.rrf100.rag24.test.txt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-queries", type=int, default=None,
                        help="smoke 模式:只跑前 N 題(建議搭配獨立 --cache-dir)")
    args = parser.parse_args()

    import numpy as np
    import torch
    from pyserini.search.lucene import LuceneSearcher
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    from rerank import load_topics, load_run, passage_text, format_trec_lines

    topics = load_topics(args.topics)
    run = load_run(args.run)
    if args.max_queries is not None:
        run = {q: run[q] for q in list(run)[: args.max_queries]}

    docids = unique_docids(run)
    os.makedirs(args.cache_dir, exist_ok=True)

    # docid 順序是快取對齊的依據;不一致代表 run 檔換了,必須重建快取
    ids_path = os.path.join(args.cache_dir, "docids.txt")
    if os.path.exists(ids_path):
        cached = [l.rstrip("\n") for l in open(ids_path, encoding="utf-8")]
        if cached != docids:
            sys.exit(f"❌ {ids_path} 與目前 run 檔不一致,請刪除 {args.cache_dir}/ 後重跑")
    else:
        with open(ids_path, "w", encoding="utf-8") as f:
            f.writelines(d + "\n" for d in docids)

    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        print("⚠️ MPS 不可用,退回 CPU(會慢很多)", file=sys.stderr)
        device, dtype = "cpu", torch.float32

    print(f"載入 {EMB_MODEL}(device={device})…")
    tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL)
    model = AutoModel.from_pretrained(EMB_MODEL, dtype=dtype).to(device).eval()
    searcher = LuceneSearcher(args.index)

    # 分塊嵌入(可續跑):缺 docid 給零向量,選取時以 norm==0 排除
    n_chunks = (len(docids) + CHUNK - 1) // CHUNK
    for ci in tqdm(range(n_chunks), desc="embed"):
        path = os.path.join(args.cache_dir, f"chunk_{ci:05d}.npy")
        chunk_ids = docids[ci * CHUNK : (ci + 1) * CHUNK]
        if os.path.exists(path) and np.load(path, mmap_mode="r").shape[0] == len(chunk_ids):
            continue
        texts, missing_rows = [], []
        for j, docid in enumerate(chunk_ids):
            doc = searcher.doc(docid)
            if doc is None:
                missing_rows.append(j)
                texts.append("")
            else:
                texts.append(passage_text(json.loads(doc.raw())))
        vecs = encode_texts(model, tokenizer, texts, device, args.batch_size)
        if missing_rows:
            vecs[missing_rows] = 0.0
            print(f"⚠️ chunk {ci}:{len(missing_rows)} 個 docid 取不到,以零向量排除",
                  file=sys.stderr)
        np.save(path, vecs)

    mat = np.concatenate(
        [np.load(os.path.join(args.cache_dir, f"chunk_{ci:05d}.npy"))
         for ci in range(n_chunks)])
    row = {d: i for i, d in enumerate(docids)}
    valid = np.linalg.norm(mat, axis=1) > 0

    qids = list(run.keys())
    qvecs = encode_texts(model, tokenizer, [topics[q] for q in qids],
                         device, args.batch_size)

    with open(args.dense_output, "w", encoding="utf-8") as fd, \
         open(args.rrf_output, "w", encoding="utf-8") as fr:
        for qi, qid in enumerate(tqdm(qids, desc="select")):
            cand = [d for d in run[qid] if valid[row[d]]]
            sims = mat[[row[d] for d in cand]] @ qvecs[qi]
            dense_pairs = list(zip(cand, sims.tolist()))
            fd.writelines(format_trec_lines(
                qid, top_k(dense_pairs, SELECT_K), DENSE_TAG))

            dense_ranking = [d for d, _ in sorted(dense_pairs, key=lambda x: -x[1])]
            fused = rrf_scores([run[qid], dense_ranking])
            fr.writelines(format_trec_lines(
                qid, top_k(list(fused.items()), SELECT_K), RRF_TAG))

    print(f"完成:{args.dense_output}、{args.rrf_output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 單元測試仍全過**

Run: `source scripts/env.sh && python -m pytest tests/ -q`
Expected: 11 passed

- [ ] **Step 3: smoke — 3 題、獨立快取(首次會下載 bge-m3 ~2.3GB)**

Run:
```bash
source scripts/env.sh && python src/embed_select.py --max-queries 3 \
  --cache-dir embeddings/smoke \
  --dense-output runs/run.dense.smoke.txt --rrf-output runs/run.rrf.smoke.txt
```
Expected: embed 進度條(3000 docids ≈ 1 塊)+ select 3 題;記下嵌入速度(筆/秒)

- [ ] **Step 4: 驗證 smoke 輸出**

```bash
source scripts/env.sh && python - <<'EOF'
from collections import Counter
bm25 = {}
for l in open("runs/run.bm25-1000.rag24.test.txt"):
    p = l.split()
    bm25.setdefault(p[0], set()).add(p[2])
for name in ["runs/run.dense.smoke.txt", "runs/run.rrf.smoke.txt"]:
    lines = [l.split() for l in open(name)]
    counts = Counter(l[0] for l in lines)
    assert all(v == 100 for v in counts.values()), (name, counts)
    for qid in counts:
        ours = [l for l in lines if l[0] == qid]
        assert {l[2] for l in ours} <= bm25[qid], (name, qid, "docid 不在 bm25-1000 候選內")
        scores = [float(l[4]) for l in ours]
        assert scores == sorted(scores, reverse=True), (name, qid, "分數未降冪")
    print(name, "OK", dict(counts))
EOF
```
Expected: 兩檔皆 OK、每題 100 列

- [ ] **Step 5: 驗證續跑(重跑同指令,embed 全跳過、幾秒完成)**

Run: 重複 Step 3 指令
Expected: embed 進度條瞬間完成(塊已存在),select 重寫輸出

- [ ] **Step 6: 清理 smoke 產物並 commit**

```bash
rm -rf embeddings/smoke runs/run.dense.smoke.txt runs/run.rrf.smoke.txt
git add src/embed_select.py
git commit -m "feat: embed_select 主流程(bge-m3 分塊快取嵌入、dense/RRF 選取)"
```

---

### Task 4: 全量嵌入 + 兩個 stage-1 run 評分

**Files:**
- Modify: `notes/baseline-scores.md`(加 bm25-1000 / dense100 / rrf100 三列)

**Interfaces:**
- Consumes: Task 3 CLI、`scripts/evaluate.sh`
- Produces: `runs/run.dense100.rag24.test.txt`、`runs/run.rrf100.rag24.test.txt`;
  「較優 stage-1 run」的決定(Recall@100 高者,平手取 nDCG@20 高者)

- [ ] **Step 1: 全量跑(~3 小時,nohup 背景 + 監控)**

Run: `source scripts/env.sh && nohup python src/embed_select.py > <scratchpad>/embed_full.log 2>&1 &`
監控:輪詢 `embeddings/chunk_*.npy` 數量與 log 尾行,異常(Traceback/提早退出)即通報
Expected: `runs/run.dense100.rag24.test.txt` 與 `runs/run.rrf100.rag24.test.txt` 各 30100 行

- [ ] **Step 2: 評分三個 run**

```bash
bash scripts/evaluate.sh runs/run.dense100.rag24.test.txt
bash scripts/evaluate.sh runs/run.rrf100.rag24.test.txt
# 診斷:候選池天花板
python -m pyserini.eval.trec_eval -c -m recall.1000 \
  data/qrels.rag24.test-umbrela-all.txt runs/run.bm25-1000.rag24.test.txt
```
Expected: dense100 或 rrf100 的 Recall@100 > 0.1395(若兩者皆 ≤,停下調查而非照記)

- [ ] **Step 3: 記錄到 notes/baseline-scores.md(分數以實際輸出為準)**

```markdown
| 2026-08-12 | run.bm25-1000.rag24.test.txt | BM25(k=1000,候選池) | — | — | <recall100> | Recall@1000 = <r1000>(池天花板) |
| 2026-08-12 | run.dense100.rag24.test.txt | BM25-1000 → bge-m3 dense top-100 | <n20> | <n100> | <r100> | stage-1 |
| 2026-08-12 | run.rrf100.rag24.test.txt | BM25-1000 → RRF(BM25+dense) top-100 | <n20> | <n100> | <r100> | stage-1 |
```

- [ ] **Step 4: Commit**

```bash
git add notes/baseline-scores.md
git commit -m "exp: dense/RRF stage-1 runs 評分"
```

---

### Task 5: cross-encoder 精排 + 最終評分 + 記錄同步

**Files:**
- Modify: `notes/baseline-scores.md`(加最終列)
- Modify: `docs/dashboard.html`(同步實測)

**Interfaces:**
- Consumes: Task 4 較優 run、`src/rerank.py`(`--run <較優run> --output runs/run.final.rag24.test.txt`)、`scripts/evaluate.sh`

- [ ] **Step 1: 精排(~50 分鐘,nohup 背景 + 監控)**

Run: `source scripts/env.sh && nohup python src/rerank.py --run <較優run檔> --output runs/run.final.rag24.test.txt > <scratchpad>/final_rerank.log 2>&1 &`
Expected: 30100 行

- [ ] **Step 2: 評分**

Run: `bash scripts/evaluate.sh runs/run.final.rag24.test.txt`
Expected: nDCG@20 > 0.4707(若更低,停下調查)

- [ ] **Step 3: notes 加最終列(分數以實際輸出為準)**

```markdown
| 2026-08-12 | run.final.rag24.test.txt | <較優stage-1> + bge-reranker-v2-m3 | <n20> | <n100> | <r100> | 里程碑 3 |
```

- [ ] **Step 4: dashboard 同步(天梯圖 Dense 列轉實測、tiles、檢核清單勾選)**

- [ ] **Step 5: 全測試 + Commit + push**

```bash
source scripts/env.sh && python -m pytest tests/ -q
git add notes/baseline-scores.md docs/dashboard.html
git commit -m "Milestone 3: dense 候選池 + 精排(nDCG@20 <實際分數>)"
git push
```
