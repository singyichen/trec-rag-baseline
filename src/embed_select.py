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
