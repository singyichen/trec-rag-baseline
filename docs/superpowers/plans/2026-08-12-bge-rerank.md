# BGE Reranker(里程碑 2)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 `BAAI/bge-reranker-v2-m3` 對 BM25 top-100 run 重排,產出可評分的新 run 檔,nDCG@20 預期 ≥ 0.45。

**Architecture:** 單一 Python 腳本 `src/rerank.py`(純函式解析/格式化 + cross-encoder 打分主流程,逐題增量寫出、可續跑),外加 `scripts/rerank_bge.sh` wrapper。從本地 Lucene 索引取回 segment 原文,不需下載語料。

**Tech Stack:** pyserini(LuceneSearcher)、transformers 5.x、torch 2.13(MPS)、pytest(新增 dev 依賴)。

## Global Constraints

- 零新執行期依賴:只能用已裝的 `pyserini`、`torch`、`transformers`、`tqdm`;測試新增 `pytest`
- 模型:`BAAI/bge-reranker-v2-m3`,fp16(MPS)/fp32(CPU fallback),`max_length=512`,batch 預設 16
- run tag 欄固定 `bm25-bge-v2-m3`;輸出預設 `runs/run.bm25-bge.rag24.test.txt`
- topics 檔為 tab 分隔、**CRLF 行尾**,解析必須 strip `\r`
- 索引取不到 docid:警告 + 給 `-1e9` 分(沉底),不中斷
- 逐題 append 寫出;重跑時跳過輸出檔中已有的 qid(續跑)
- 腳本註解與訊息用繁體中文,風格對齊 `scripts/run_bm25.sh`
- commit 訊息結尾加 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 1: 純函式(解析、續跑、passage 組字、TREC 格式化)+ 單元測試

**Files:**
- Create: `src/rerank.py`(本 task 只含純函式)
- Create: `tests/conftest.py`
- Create: `tests/test_rerank.py`
- Modify: `requirements.txt`(加 `pytest`)

**Interfaces:**
- Produces(Task 2 會用到,簽名必須一字不差):
  - `load_topics(path) -> dict[str, str]`  # qid -> query
  - `load_run(path) -> dict[str, list[str]]`  # qid -> docids(保留原排序;dict 保留 qid 出現順序)
  - `completed_qids(path) -> set[str]`  # 輸出檔已有的 qid;檔案不存在回空集合
  - `passage_text(raw: dict) -> str`  # title + segment;title 缺或為 "-" 時只用 segment
  - `format_trec_lines(qid, scored, tag) -> list[str]`  # scored: list[(docid, score)],依分數降冪(穩定排序),rank 從 1 起,6 欄位空格分隔,含結尾 \n

- [ ] **Step 1: 安裝 pytest 並加入 requirements**

```bash
source scripts/env.sh && pip install pytest
```

`requirements.txt` 加一行:`pytest`

- [ ] **Step 2: 寫失敗測試**

`tests/conftest.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

`tests/test_rerank.py`:

```python
from rerank import (
    load_topics, load_run, completed_qids, passage_text, format_trec_lines,
)


def test_load_topics_strips_crlf(tmp_path):
    p = tmp_path / "topics.txt"
    p.write_bytes(b"2024-1\thello world?\r\n2024-2\tsecond query\r\n")
    topics = load_topics(str(p))
    assert topics == {"2024-1": "hello world?", "2024-2": "second query"}


def test_load_run_groups_and_preserves_order(tmp_path):
    p = tmp_path / "run.txt"
    p.write_text(
        "q1 Q0 docA 1 9.5 bm25\n"
        "q1 Q0 docB 2 8.0 bm25\n"
        "q2 Q0 docC 1 7.0 bm25\n"
    )
    run = load_run(str(p))
    assert list(run.keys()) == ["q1", "q2"]
    assert run["q1"] == ["docA", "docB"]
    assert run["q2"] == ["docC"]


def test_completed_qids_missing_file(tmp_path):
    assert completed_qids(str(tmp_path / "nope.txt")) == set()


def test_completed_qids_reads_existing(tmp_path):
    p = tmp_path / "out.txt"
    p.write_text("q1 Q0 docA 1 0.9 tag\nq3 Q0 docB 1 0.8 tag\n")
    assert completed_qids(str(p)) == {"q1", "q3"}


def test_passage_text_with_title():
    assert passage_text({"title": "T", "segment": "S"}) == "T S"


def test_passage_text_dash_or_missing_title():
    assert passage_text({"title": "-", "segment": "S"}) == "S"
    assert passage_text({"segment": "S"}) == "S"


def test_format_trec_lines_sorts_desc_and_ranks():
    lines = format_trec_lines("q1", [("a", 1.0), ("b", 3.0), ("c", 2.0)], "tag")
    assert lines == [
        "q1 Q0 b 1 3.000000 tag\n",
        "q1 Q0 c 2 2.000000 tag\n",
        "q1 Q0 a 3 1.000000 tag\n",
    ]
```

- [ ] **Step 3: 跑測試確認失敗**

Run: `source scripts/env.sh && python -m pytest tests/ -v`
Expected: FAIL,`ImportError`/`ModuleNotFoundError`(rerank 尚不存在)

- [ ] **Step 4: 最小實作**

`src/rerank.py`:

```python
"""用 BAAI/bge-reranker-v2-m3 對 BM25 top-100 run 重排。

從本地 Lucene 索引取回 segment 原文,cross-encoder 打分後重排,
逐題增量寫出 TREC 格式 run 檔;中斷後重跑會跳過已完成的題目。
"""


def load_topics(path):
    """讀 tab 分隔 topics 檔(qid \\t query),行尾可能是 CRLF。"""
    topics = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, query = line.split("\t", 1)
            topics[qid] = query
    return topics


def load_run(path):
    """讀 TREC run 檔,回傳 qid -> docid 清單(保留原排序)。"""
    run = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            run.setdefault(parts[0], []).append(parts[2])
    return run


def completed_qids(path):
    """輸出檔中已完成的 qid 集合;檔案不存在回空集合。"""
    try:
        with open(path, encoding="utf-8") as f:
            return {line.split()[0] for line in f if line.strip()}
    except FileNotFoundError:
        return set()


def passage_text(raw):
    """組 passage 文字:title + segment;title 缺或為 '-' 時只用 segment。"""
    title = raw.get("title", "")
    segment = raw.get("segment", "")
    if title and title != "-":
        return f"{title} {segment}"
    return segment


def format_trec_lines(qid, scored, tag):
    """依分數降冪排序(穩定),輸出 TREC 6 欄格式(含換行)。"""
    ordered = sorted(scored, key=lambda x: -x[1])
    return [
        f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n"
        for rank, (docid, score) in enumerate(ordered, start=1)
    ]
```

- [ ] **Step 5: 跑測試確認通過**

Run: `source scripts/env.sh && python -m pytest tests/ -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add src/rerank.py tests/ requirements.txt
git commit -m "feat: rerank 純函式(解析/續跑/格式化)+ 單元測試"
```

---

### Task 2: 打分主流程 + CLI(整合真實索引與模型)

**Files:**
- Modify: `src/rerank.py`(追加,不動 Task 1 的函式)

**Interfaces:**
- Consumes: Task 1 的五個純函式(簽名見 Task 1)
- Produces: CLI — `python src/rerank.py [--run --topics --index --output --batch-size --max-queries]`;
  `score_pairs(model, tokenizer, query, passages, device, batch_size) -> list[float]`

- [ ] **Step 1: 實作打分與主流程(追加到 src/rerank.py)**

```python
MODEL_NAME = "BAAI/bge-reranker-v2-m3"
TAG = "bm25-bge-v2-m3"
MISSING_DOC_SCORE = -1e9


def score_pairs(model, tokenizer, query, passages, device, batch_size):
    """對 (query, passage) 逐 batch 打分,回傳 raw logit 清單。"""
    import torch

    scores = []
    with torch.inference_mode():
        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            inputs = tokenizer(
                [query] * len(batch), batch, padding=True,
                truncation=True, max_length=512, return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits.view(-1).float()
            scores.extend(logits.cpu().tolist())
    return scores


def main():
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/run.bm25.rag24.test.txt")
    parser.add_argument("--topics", default="data/topics.rag24.test.txt")
    parser.add_argument("--index", default=os.path.expanduser(
        "~/indexes/lucene-inverted.msmarco-v2.1-doc-segmented.20240418.4f9675"))
    parser.add_argument("--output", default="runs/run.bm25-bge.rag24.test.txt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-queries", type=int, default=None,
                        help="smoke 模式:只跑前 N 題")
    args = parser.parse_args()

    import json

    import torch
    from pyserini.search.lucene import LuceneSearcher
    from tqdm import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    topics = load_topics(args.topics)
    run = load_run(args.run)
    done = completed_qids(args.output)

    qids = list(run.keys())
    if args.max_queries is not None:
        qids = qids[: args.max_queries]
    todo = [q for q in qids if q not in done]
    print(f"共 {len(qids)} 題,已完成 {len(qids) - len(todo)} 題,待跑 {len(todo)} 題")
    if not todo:
        return

    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        print("⚠️ MPS 不可用,退回 CPU(會慢很多)", file=sys.stderr)
        device, dtype = "cpu", torch.float32

    print(f"載入 {MODEL_NAME}(device={device})…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, dtype=dtype).to(device).eval()
    searcher = LuceneSearcher(args.index)

    with open(args.output, "a", encoding="utf-8") as out:
        for qid in tqdm(todo, desc="rerank"):
            docids = run[qid]
            texts, present = [], []
            missing = []
            for docid in docids:
                doc = searcher.doc(docid)
                if doc is None:
                    missing.append(docid)
                    continue
                texts.append(passage_text(json.loads(doc.raw())))
                present.append(docid)
            if missing:
                print(f"⚠️ {qid}:索引取不到 {len(missing)} 個 docid,沉底處理",
                      file=sys.stderr)
            scores = score_pairs(model, tokenizer, topics[qid], texts,
                                 device, args.batch_size)
            scored = list(zip(present, scores))
            scored += [(d, MISSING_DOC_SCORE) for d in missing]
            out.writelines(format_trec_lines(qid, scored, TAG))
            out.flush()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 單元測試仍全過(純函式未被改壞)**

Run: `source scripts/env.sh && python -m pytest tests/ -v`
Expected: 7 passed

- [ ] **Step 3: Smoke — 跑 2 題(首次會下載 ~2.3GB 模型)**

Run:
```bash
source scripts/env.sh && python src/rerank.py --max-queries 2 \
  --output runs/run.smoke.txt
```
Expected: 進度條跑完 2 題不報錯;記下每題耗時(推估全量時間用)

- [ ] **Step 4: 驗證 smoke 輸出格式**

```bash
source scripts/env.sh && python - <<'EOF'
from collections import Counter
lines = [l.split() for l in open("runs/run.smoke.txt")]
counts = Counter(l[0] for l in lines)
assert all(v == 100 for v in counts.values()), counts
bm25 = {}
for l in open("runs/run.bm25.rag24.test.txt"):
    p = l.split()
    bm25.setdefault(p[0], set()).add(p[2])
for qid in counts:
    ours = {l[2] for l in lines if l[0] == qid}
    assert ours == bm25[qid], f"{qid} docid 集合不一致"
    scores = [float(l[4]) for l in lines if l[0] == qid]
    assert scores == sorted(scores, reverse=True), f"{qid} 分數未降冪"
print("smoke 輸出格式 OK:", dict(counts))
EOF
```
Expected: `smoke 輸出格式 OK: {...每題 100}`

- [ ] **Step 5: 驗證續跑(重跑同指令應跳過已完成)**

Run: 重複 Step 3 指令
Expected: 印出「已完成 2 題,待跑 0 題」後直接結束

- [ ] **Step 6: 清掉 smoke 檔並 commit**

```bash
rm runs/run.smoke.txt
git add src/rerank.py
git commit -m "feat: rerank 主流程(MPS cross-encoder 打分、增量寫出、可續跑)"
```

---

### Task 3: shell wrapper

**Files:**
- Create: `scripts/rerank_bge.sh`

**Interfaces:**
- Consumes: Task 2 的 CLI(參數原樣轉傳)

- [ ] **Step 1: 寫 wrapper**

`scripts/rerank_bge.sh`:

```bash
#!/usr/bin/env bash
# 用 BAAI/bge-reranker-v2-m3 對 BM25 top-100 run 重排。
#
# ⚠️ 第一次執行會從 HuggingFace 下載模型(~2.3GB)到 ~/.cache/huggingface/。
# 中斷可直接重跑,會自動跳過已完成的題目。
#
# 用法:bash scripts/rerank_bge.sh [額外參數原樣傳給 src/rerank.py]
# 例:  bash scripts/rerank_bge.sh --max-queries 3   # smoke 測試
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/env.sh

RUN="runs/run.bm25.rag24.test.txt"
[ -f "$RUN" ] || { echo "找不到 $RUN,先跑 bash scripts/run_bm25.sh"; exit 1; }

INDEX="${INDEX:-$HOME/indexes/lucene-inverted.msmarco-v2.1-doc-segmented.20240418.4f9675}"
[ -d "$INDEX" ] || { echo "找不到索引 $INDEX(可用 INDEX=... 覆寫)"; exit 1; }

python src/rerank.py --index "$INDEX" "$@"

echo "=== 完成 ==="
echo "run 檔:runs/run.bm25-bge.rag24.test.txt"
```

- [ ] **Step 2: 驗證 wrapper(smoke 已完成的題目會被跳過,幾秒內結束)**

Run:
```bash
bash scripts/rerank_bge.sh --max-queries 2 --output runs/run.smoke.txt
rm runs/run.smoke.txt
```
Expected: 印出「待跑 2 題」正常跑完(或若沿用舊 smoke 檔則直接跳過);結尾印「=== 完成 ===」

- [ ] **Step 3: Commit**

```bash
git add scripts/rerank_bge.sh
git commit -m "feat: rerank_bge.sh wrapper"
```

---

### Task 4: 全量跑 + 評分 + 記錄

**Files:**
- Modify: `notes/baseline-scores.md`(加一列)

**Interfaces:**
- Consumes: Task 3 wrapper、既有 `scripts/evaluate.sh`

- [ ] **Step 1: 全量跑(301 題,預估 30–90 分鐘,背景執行)**

Run: `bash scripts/rerank_bge.sh`
Expected: 進度條到 301/301;`runs/run.bm25-bge.rag24.test.txt` 共 30,100 行

- [ ] **Step 2: 行數與格式檢查**

Run: `wc -l runs/run.bm25-bge.rag24.test.txt`
Expected: `30100`

- [ ] **Step 3: 評分**

Run: `bash scripts/evaluate.sh runs/run.bm25-bge.rag24.test.txt`
Expected: 三指標輸出;nDCG@20 ≥ 0.45(若 < 0.40 停下來調查而不是照記)

- [ ] **Step 4: 記錄到 notes/baseline-scores.md**

在表格加一列(分數以實際輸出為準):

```markdown
| 2026-08-12 | run.bm25-bge.rag24.test.txt | BM25 top-100 + bge-reranker-v2-m3 | <ndcg20> | <ndcg100> | 0.1395 | Recall@100 與 BM25 相同(同一候選集) |
```

註:重排不改變 top-100 集合,Recall@100 理論上必等於 0.1395,若不同表示有 bug。

- [ ] **Step 5: Commit + push**

```bash
git add notes/baseline-scores.md
git commit -m "Milestone 2: BGE rerank BM25 top-100(nDCG@20 <實際分數>)"
git push
```
