import numpy as np
import pytest

from src.core.config import PROJECT_ROOT, Settings, load_settings
from src.rag.documents import Chunk
from src.rag.embedders import Embedder, TfidfEmbedder
from src.rag.evaluation import evaluate_retrieval, load_queries
from src.rag.retriever import Retriever


class ConceptEmbedder(Embedder):
    """Stand-in for a neural model: synonyms share a concept axis; unknown words (codes, form numbers) all land on one
    shared 'misc' axis, so it cannot tell two rare identifiers apart - the weakness lexical matching must cover."""

    name = "concepts"
    CONCEPTS = {"cancel": 0, "revoke": 0, "undo": 0, "money": 1, "cash": 1, "savings": 1, "dividend": 2, "payout": 2}
    MISC = 3

    def _vec(self, text):
        v = np.zeros(4, dtype="float32")
        for word in text.lower().replace(".", " ").split():
            v[self.CONCEPTS.get(word, self.MISC)] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def fit_transform(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, text):
        return self._vec(text)


def chunks():
    texts = {
        "revoke": "You may revoke the account within seven days of opening it",
        "credit": "Form 8880 claims the saver credit for retirement contributions",
        "form5498": "Form 5498 reports contributions made to the account for the year",
        "dividend": "A dividend is a payout of company profit to shareholders",
    }
    return [Chunk(f"{k}-0", k, k, t, "c") for k, t in texts.items()]


def test_hybrid_matches_by_meaning_and_by_exact_identifier():
    r = Retriever(chunks(), min_score=0.0, embedder=ConceptEmbedder(), lexical=TfidfEmbedder(), lexical_weight=0.5)
    assert r.mode == "hybrid"
    assert r.search("undo my account", k=1)[0].title == "revoke"  # no shared word: meaning must carry it
    assert r.search("form 8880", k=1)[0].title == "credit"        # rare identifier: exact words must carry it
    assert r.search("form 5498", k=1)[0].title == "form5498"
    # Semantic-only cannot separate the two form chunks (both look like 'misc' to it), which is why the lexical half exists.
    semantic_only = Retriever(chunks(), min_score=0.0, embedder=ConceptEmbedder())
    assert semantic_only.mode == "concepts"
    scores = {h.title: round(h.score, 6) for h in semantic_only.search("form 8880", k=4)}
    assert scores["credit"] == scores["form5498"]


def test_hybrid_scores_are_the_weighted_blend():
    pure_sem = Retriever(chunks(), min_score=0.0, embedder=ConceptEmbedder())
    pure_lex = Retriever(chunks(), min_score=0.0, embedder=TfidfEmbedder())
    both = Retriever(chunks(), min_score=0.0, embedder=ConceptEmbedder(), lexical=TfidfEmbedder(), lexical_weight=0.25)
    q = "revoke the account"
    sem = {h.title: h.score for h in pure_sem.search(q, k=4)}
    lex = {h.title: h.score for h in pure_lex.search(q, k=4)}
    for h in both.search(q, k=4):
        assert h.score == pytest.approx(0.75 * sem.get(h.title, 0.0) + 0.25 * lex.get(h.title, 0.0), abs=1e-4)


def test_hybrid_works_with_hnsw_and_category_filter():
    flat = Retriever(chunks(), min_score=0.0, embedder=ConceptEmbedder(), lexical=TfidfEmbedder())
    hnsw = Retriever(chunks(), min_score=0.0, embedder=ConceptEmbedder(), lexical=TfidfEmbedder(),
                     index_type="hnsw", index_options={"hnsw_m": 8, "ef_search": 32})
    assert hnsw.backend == "faiss-hnsw"
    for q in ("undo my account", "form 8880", "dividend payout"):
        assert [h.title for h in hnsw.search(q, k=2)] == [h.title for h in flat.search(q, k=2)]
    assert flat.search("form 8880", k=3, category="nope") == []


def test_hybrid_falls_back_to_tfidf_when_semantic_half_is_unavailable(monkeypatch):
    from src.rag import embedders

    def boom(*args, **kwargs):
        raise ImportError("sentence-transformers is not installed")

    monkeypatch.setattr(embedders, "SentenceTransformerEmbedder", boom)
    primary, lexical, weight = embedders.create_channels(Settings({"rag": {"embedder": "hybrid"}}))
    assert primary.name == "tfidf" and lexical is None and weight == 0.0
    cfg = Settings({"rag": {"embedder": "hybrid", "pdf_dir": "", "min_score": 0.07, "hybrid_min_score": 0.5}})
    r = Retriever.from_settings(cfg)
    assert r.mode == "tfidf" and r.min_score == 0.07  # the TF-IDF floor applies, not the hybrid one
    assert r.search("what is an etf", k=1)


def test_hybrid_factory_blends_semantic_with_tfidf_and_clamps_weight(monkeypatch):
    from src.rag import embedders

    monkeypatch.setattr(embedders, "SentenceTransformerEmbedder", lambda model, **kw: ConceptEmbedder())
    primary, lexical, weight = embedders.create_channels(Settings({"rag": {"embedder": "hybrid", "hybrid_lexical_weight": 0.4}}))
    assert primary.name == "concepts" and lexical.name == "tfidf" and weight == 0.4
    assert embedders.create_channels(Settings({"rag": {"embedder": "hybrid", "hybrid_lexical_weight": 9}}))[2] == 1.0


def test_hybrid_end_to_end_if_sentence_transformers_available(tmp_path):
    pytest.importorskip("sentence_transformers")
    cfg = load_settings()
    cfg.set("rag.embedder", "hybrid")
    cfg.set("rag.pdf_cache_dir", str(tmp_path))
    r = Retriever.from_settings(cfg)
    if r.mode != "hybrid":
        pytest.skip("neural model could not be loaded (offline?)")
    rep = evaluate_retrieval(r, load_queries("retrieval_eval.json"), k=3)
    assert rep.hit_at_k >= 0.85 and rep.by_kind["paraphrase"]["hit@3"] >= 0.6 and rep.by_kind["exact"]["hit@3"] >= 0.85
    if (PROJECT_ROOT / "src" / "data" / "pdfs" / "tax" / "p590a.pdf").exists():
        pdf = evaluate_retrieval(r, load_queries("pdf_eval.json"), k=3)
        assert pdf.hit_at_k >= 0.7 and pdf.by_kind["exact"]["hit@3"] >= 0.8  # semantic-only scores 0.71 / 0.60


def test_embedding_cache_skips_recomputation_and_detects_changes(tmp_path):
    from src.rag.embedders import cached_matrix

    calls = []

    def compute(texts):
        calls.append(len(texts))
        return np.ones((len(texts), 3), dtype="float32")

    first = cached_matrix(tmp_path, "model-a", ["x", "y"], compute)
    again = cached_matrix(tmp_path, "model-a", ["x", "y"], compute)
    assert calls == [2] and np.array_equal(first, again)           # second call served from disk
    cached_matrix(tmp_path, "model-a", ["x", "y", "z"], compute)    # corpus changed
    cached_matrix(tmp_path, "model-b", ["x", "y"], compute)         # model changed
    assert calls == [2, 3, 2]
    assert cached_matrix(None, "m", ["x"], compute).shape == (1, 3) and calls[-1] == 1  # cache disabled


# --- router rescue: exact identifiers and semantic matches count as on-topic ----------------------------
def test_has_identifier_finds_codes_in_the_knowledge_base_but_ignores_years_and_unknowns():
    r = Retriever(chunks(), min_score=0.0)
    assert r.has_identifier("what is form 8880 used for")
    assert r.has_identifier("FORM 5498")                    # case-insensitive
    assert not r.has_identifier("what is form 9999")        # code not in the KB
    assert not r.has_identifier("revenue in 2025")          # bare years are ignored
    assert not r.has_identifier("how do i fix a flat tire")  # no code at all


class StubKB:
    def __init__(self, mode="hybrid", identifier=False, top=None):
        self.mode, self._identifier, self._top = mode, identifier, top
        self.searches = []

    def has_identifier(self, text):
        return self._identifier

    def search(self, text, k=1, min_score=0.0):
        self.searches.append(min_score)
        return [object()] if self._top is not None and self._top >= min_score else []


def test_router_rescues_keyword_free_questions_the_knowledge_base_recognises():
    from src.workflow.router import Router

    text = "what is the cutoff for funding it"  # no finance keyword, no ticker
    assert Router().route(text).off_topic                                        # no retriever: refused, as before
    by_identifier = Router(retriever=StubKB(identifier=True)).route(text)
    assert not by_identifier.off_topic and by_identifier.method == "knowledge-base" and by_identifier.intents == ["finance_qa"]
    kb = StubKB(top=0.4)
    assert not Router(retriever=kb, rescue_score=0.3).route(text).off_topic and kb.searches == [0.3]
    assert Router(retriever=StubKB(top=0.2), rescue_score=0.3).route(text).off_topic     # recognised too weakly
    assert Router(retriever=StubKB(mode="tfidf", top=0.9)).route(text).off_topic          # TF-IDF scores are not comparable: identifier test only
    assert not Router(retriever=StubKB(mode="tfidf", identifier=True)).route(text).off_topic


def test_rescue_does_not_change_routing_of_ordinary_questions():
    from src.workflow.router import Router

    kb = StubKB(identifier=True, top=1.0)
    d = Router(retriever=kb).route("I own 10 VTI and 5 BND, analyze my portfolio")
    assert d.intents == ["portfolio_analysis"] and kb.searches == []  # keyword path never consults the knowledge base
