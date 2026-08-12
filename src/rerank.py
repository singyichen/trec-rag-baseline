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
