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
