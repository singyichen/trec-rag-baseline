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
