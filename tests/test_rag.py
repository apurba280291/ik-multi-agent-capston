import numpy as np
import pytest

from src.rag.documents import Document, build_chunks, chunk_document, load_articles, load_glossary
from src.rag.retriever import Retriever
from src.rag.vector_store import VectorStore


def test_knowledge_base_size_and_categories(retriever, settings):
    articles = load_articles(settings.resolve_path("src/data/knowledge_base"))
    assert len(articles) >= 50  # project requirement: 50-100+ articles
    assert {"stocks", "bonds", "funds", "tax", "retirement", "portfolio"} <= set(retriever.categories)
    assert "glossary" in retriever.categories
    assert all(a.text and a.title and a.reference for a in articles)


def test_glossary_loading(settings, tmp_path):
    assert len(load_glossary(settings.resolve_path("src/data/glossary.json"))) >= 40
    assert load_glossary(tmp_path / "missing.json") == []
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert load_glossary(bad) == []


@pytest.mark.parametrize(
    "query,expected",
    [
        ("what is an ETF", "ETF"),
        ("difference between roth and traditional ira", "IRA"),
        ("what is dividend yield", "Dividend"),
        ("how does rebalancing work", "Rebalancing"),
        ("tax loss harvesting wash sale", "Wash-Sale"),
        ("what is a 529 plan", "529"),
    ],
)
def test_retrieval_relevance(retriever, query, expected):
    hits = retriever.search(query)
    assert hits, query
    assert any(expected.lower() in h.title.lower() for h in hits[:3])


def test_scores_sorted_and_one_chunk_per_doc(retriever):
    hits = retriever.search("diversification and asset allocation", k=6)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert len({(h.title, h.category) for h in hits}) == len(hits)


def test_category_filter(retriever):
    hits = retriever.search("what is a roth", k=5, category="tax")
    assert hits and all(h.category == "tax" for h in hits)
    multi = retriever.search("what is a roth", k=5, category=["tax", "glossary"])
    assert {h.category for h in multi} <= {"tax", "glossary"}


def test_source_attribution(retriever):
    src = retriever.search("what is a bond")[0].to_source()
    assert src.title and src.category and src.reference.startswith("http") and src.score > 0


def test_irrelevant_and_empty_queries(retriever):
    assert retriever.search("") == []
    assert retriever.search("   ") == []
    assert retriever.search("zxqv wlkj") == []  # no vocabulary overlap
    assert retriever.search("what is a stock", min_score=0.99) == []


def test_empty_retriever():
    r = Retriever([])
    assert r.search("anything") == [] and r.backend == "none" and r.document_count == 0


def test_chunking_overlap():
    doc = Document("d", "T", " ".join(f"w{i}" for i in range(300)), "c")
    chunks = chunk_document(doc, 100, 20)
    assert len(chunks) == 4 and chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]
    assert len(chunk_document(Document("d", "T", "short text", "c"))) == 1
    assert len(build_chunks([doc, doc], 100, 20)) == 8


def test_vector_store_backends():
    vecs = np.eye(3, dtype="float32")
    for use_faiss in (True, False):
        vs = VectorStore(3, use_faiss=use_faiss)
        assert vs.search(vecs[0], 2) == []
        vs.add(vecs)
        res = vs.search(np.array([0, 1, 0], dtype="float32"), 2)
        assert res[0][0] == 1 and res[0][1] == pytest.approx(1.0) and len(res) == 2 and len(vs) == 3
        assert vs.search(vecs[0], 0) == []
    assert VectorStore(3, use_faiss=False).backend == "numpy"


def test_load_articles_handles_header_only_file(tmp_path):
    (tmp_path / "x.md").write_text("category: x\nreference: http://r\n", encoding="utf-8")
    assert load_articles(tmp_path) == []
