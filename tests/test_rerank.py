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
