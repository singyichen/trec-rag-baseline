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
