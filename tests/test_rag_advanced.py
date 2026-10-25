import numpy as np
import pytest

from src.core.config import PROJECT_ROOT, Settings, load_settings
from src.rag.documents import Document, build_chunks, chunk_document_recursive, recursive_split
from src.rag.embedders import Embedder, TfidfEmbedder, create_embedder
from src.rag.evaluation import RetrievalReport, evaluate_retrieval, evaluate_routing, load_queries
from src.rag.retriever import Retriever
from src.rag.vector_store import VectorStore
from src.workflow.router import Router


# --- recursive chunking -----------------------------------------------------------------------------
def test_recursive_split_prefers_paragraph_then_sentence_boundaries():
    text = "Alpha one two three.\n\nBeta four five six.\n\nGamma seven eight nine."
    assert recursive_split(text, 5) == ["Alpha one two three.", "Beta four five six.", "Gamma seven eight nine."]
    sentences = "One two three. Four five six. Seven eight nine."
    assert recursive_split(sentences, 6) == ["One two three. Four five six.", "Seven eight nine."]


def test_recursive_split_respects_limit_and_loses_nothing():
    text = " ".join(f"w{i}" for i in range(250))  # no punctuation: falls through to word level
    parts = recursive_split(text, 60)
    assert all(len(p.split()) <= 60 for p in parts)
    assert " ".join(parts).split() == text.split()
    assert recursive_split("   ", 10) == [] and recursive_split("short text", 10) == ["short text"]


def test_recursive_chunks_overlap_and_ids():
    doc = Document("d", "T", ". ".join(f"sentence number {i} has some words" for i in range(40)) + ".", "c", "http://r")
    chunks = chunk_document_recursive(doc, 50, 10)
    assert len(chunks) > 1 and [c.chunk_id for c in chunks] == [f"d-{i}" for i in range(len(chunks))]
    assert chunks[1].text.split()[:10] == chunks[0].text.split()[-10:]
    assert all(c.title == "T" and c.reference == "http://r" for c in chunks)


def test_build_chunks_strategy_switch_and_unknown_falls_back():
    doc = Document("d", "T", " ".join(f"w{i}" for i in range(300)), "c")
    assert len(build_chunks([doc], 100, 20, "fixed")) == 4
    assert len(build_chunks([doc], 100, 20, "recursive")) >= 3
    assert len(build_chunks([doc], 100, 20, "bogus")) == 4  # warns, uses fixed


# --- HNSW index -------------------------------------------------------------------------------------
def test_hnsw_matches_flat_on_random_vectors():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(500, 32)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    flat, hnsw = VectorStore(32), VectorStore(32, index_type="hnsw", ef_search=128)
    flat.add(vecs)
    hnsw.add(vecs)
    assert hnsw.backend == "faiss-hnsw" and hnsw.approximate and not flat.approximate
    overlap = []
    for q in vecs[:25]:
        exact = {i for i, _ in flat.search(q, 10)}
        approx = {i for i, _ in hnsw.search(q, 10)}
        overlap.append(len(exact & approx) / 10)
    assert np.mean(overlap) >= 0.9  # recall@10 vs exact search


def test_vector_store_rejects_unknown_index_and_numpy_ignores_hnsw():
    with pytest.raises(ValueError):
        VectorStore(4, index_type="ivf")
    assert VectorStore(4, use_faiss=False, index_type="hnsw").backend == "numpy"


def test_retriever_hnsw_same_answers_as_flat(settings):
    flat = Retriever.from_settings(settings)
    cfg = Settings({**settings.raw, "rag": {**settings.raw["rag"], "index_type": "hnsw"}})
    hnsw = Retriever.from_settings(cfg)
    assert hnsw.backend == "faiss-hnsw"
    for q in ("what is an etf", "roth ira", "bond prices and interest rates", "how does rebalancing work"):
        assert [h.title for h in hnsw.search(q, k=3)] == [h.title for h in flat.search(q, k=3)]
    assert all(h.category == "tax" for h in hnsw.search("what is a roth", k=3, category="tax"))  # filter still works


# --- embedders --------------------------------------------------------------------------------------
class BagOfLettersEmbedder(Embedder):
    """Tiny deterministic 'neural' stand-in: proves the retriever works with any Embedder."""

    name = "fake"

    @staticmethod
    def _vec(text):
        v = np.zeros(26, dtype="float32")
        for ch in text.lower():
            if "a" <= ch <= "z":
                v[ord(ch) - 97] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def fit_transform(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, text):
        return self._vec(text)


def test_retriever_accepts_custom_embedder(retriever):
    r = Retriever(retriever.chunks, embedder=BagOfLettersEmbedder(), min_score=0.0)
    assert r.embedder.name == "fake" and r.search("what is a bond", k=3)


def test_embedder_factory_defaults_and_falls_back(monkeypatch):
    assert create_embedder(Settings({})).name == "tfidf"
    assert create_embedder(Settings({"rag": {"embedder": "bogus"}})).name == "tfidf"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert create_embedder(Settings({"rag": {"embedder": "openai"}})).name == "tfidf"  # no key -> fallback
    assert isinstance(TfidfEmbedder(), Embedder)


# --- evals ------------------------------------------------------------------------------------------
class StubRetriever:
    def __init__(self, table):
        self.table = table

    def search(self, q, k=3):
        class H:
            def __init__(self, title):
                self.title = title

        return [H(t) for t in self.table.get(q, [])][:k]


def test_retrieval_metrics_are_computed_correctly():
    queries = [
        {"q": "a", "expected": ["X"], "kind": "direct"},   # rank 1
        {"q": "b", "expected": ["Y"], "kind": "direct"},   # rank 2
        {"q": "c", "expected": ["Z"], "kind": "paraphrase"},  # miss
        {"q": "d", "expected": [], "kind": "out_of_scope"},  # correctly empty
        {"q": "e", "expected": [], "kind": "out_of_scope"},  # false positive
    ]
    table = {"a": ["X", "o1", "o2"], "b": ["o1", "Y", "o2"], "c": ["o1", "o2", "o3"], "e": ["junk"]}
    rep: RetrievalReport = evaluate_retrieval(StubRetriever(table), queries, k=3)
    assert rep.hit_at_1 == pytest.approx(1 / 3) and rep.hit_at_k == pytest.approx(2 / 3)
    assert rep.mrr == pytest.approx((1 + 0.5 + 0) / 3)
    assert rep.precision_at_k == pytest.approx((1 / 3 + 1 / 3 + 0) / 3)
    assert rep.no_answer_rate == 0.5
    assert {m["q"] for m in rep.misses} == {"c", "e"}
    assert rep.by_kind["direct"]["hit@1"] == 0.5


def test_routing_metrics():
    queries = [{"q": "a", "expected": ["x"]}, {"q": "b", "expected": ["x", "y"]}, {"q": "c", "expected": ["z"]}]
    got = {"a": ["x"], "b": ["x"], "c": ["z", "w"]}
    rep = evaluate_routing(lambda q: got[q], queries)
    assert rep.exact_match == pytest.approx(1 / 3)
    assert rep.agent_recall == pytest.approx(3 / 4) and rep.agent_precision == pytest.approx(3 / 4)


def test_eval_sets_are_well_formed(retriever):
    titles = {c.title.lower() for c in retriever.chunks}
    for item in load_queries("retrieval_eval.json"):
        assert item["kind"] in {"direct", "paraphrase", "exact", "out_of_scope"}
        assert bool(item["expected"]) == (item["kind"] != "out_of_scope")
        assert all(t.lower() in titles for t in item["expected"]), item  # labels must name real articles
    agents = {"market_analysis", "news_synthesizer", "portfolio_analysis", "goal_planning", "tax_education", "finance_qa"}
    assert all(set(i["expected"]) <= agents for i in load_queries("routing_eval.json"))


# --- regression guards on the real eval sets (current TF-IDF baseline; raise these as quality improves) ----
def test_retrieval_quality_floor(retriever):
    rep = evaluate_retrieval(retriever, load_queries("retrieval_eval.json"), k=3)
    assert rep.by_kind["direct"]["hit@3"] >= 0.95
    assert rep.hit_at_k >= 0.75  # paraphrase queries drag this down: TF-IDF is lexical, see README


def test_routing_quality_floor():
    router = Router()
    rep = evaluate_routing(lambda q: router.route(q).intents, load_queries("routing_eval.json"))
    assert rep.exact_match >= 0.75 and rep.agent_recall >= 0.9


def test_401k_with_parentheses_is_not_off_topic():
    d = Router().route("How does a 401(k) match work?")
    assert not d.off_topic and "tax_education" in d.intents


# --- neural score floor and calibration ---------------------------------------------------------------
def test_neural_embedders_get_their_own_score_floor(monkeypatch):
    from src.rag import retriever as retriever_mod

    base = {"knowledge_base_dir": "src/data/knowledge_base", "glossary_file": "src/data/glossary.json", "pdf_dir": ""}
    tfidf = Retriever.from_settings(Settings({"rag": {**base, "min_score": 0.07}}))
    assert tfidf.min_score == 0.07
    monkeypatch.setattr(retriever_mod, "create_channels", lambda s: (BagOfLettersEmbedder(), None, 0.0))
    neural = Retriever.from_settings(Settings({"rag": {**base, "min_score": 0.07}}))
    assert neural.min_score == 0.30  # default neural floor
    neural = Retriever.from_settings(Settings({"rag": {**base, "neural_min_score": 0.5}}))
    assert neural.min_score == 0.5
    monkeypatch.setattr(retriever_mod, "create_channels", lambda s: (BagOfLettersEmbedder(), TfidfEmbedder(), 0.5))
    hybrid = Retriever.from_settings(Settings({"rag": {**base, "hybrid_min_score": 0.11}}))
    assert hybrid.min_score == 0.11 and hybrid.mode == "hybrid"


def test_calibrate_floor_finds_gap_or_reports_overlap():
    from src.rag.evaluation import calibrate_floor

    class Scored:
        def __init__(self, table):
            self.table = table

        def search(self, q, k=3, min_score=0.0):
            class H:
                def __init__(self, title, score):
                    self.title, self.score = title, score

            return [H(t, s) for t, s in self.table.get(q, [])]

    queries = [
        {"q": "a", "expected": ["X"], "kind": "direct"},
        {"q": "b", "expected": ["Y"], "kind": "direct"},
        {"q": "c", "expected": [], "kind": "out_of_scope"},
    ]
    gap = calibrate_floor(Scored({"a": [("X", 0.6)], "b": [("o", 0.5), ("Y", 0.4)], "c": [("z", 0.2)]}), queries)
    assert gap.separable and gap.suggested_floor == pytest.approx(0.3) and gap.relevant == [0.4, 0.6]
    overlap = calibrate_floor(Scored({"a": [("X", 0.3)], "b": [("Y", 0.4)], "c": [("z", 0.35)]}), queries)
    assert not overlap.separable


def test_neural_retrieval_floor_if_sentence_transformers_available():
    pytest.importorskip("sentence_transformers")
    cfg = load_settings()
    cfg.set("rag.pdf_dir", "")
    cfg.set("rag.embedder", "sentence_transformers")
    r = Retriever.from_settings(cfg)
    if r.embedder.name != "sentence_transformers":
        pytest.skip("model could not be loaded (offline?)")
    rep = evaluate_retrieval(r, load_queries("retrieval_eval.json"), k=3)
    assert rep.hit_at_k >= 0.85 and rep.by_kind["paraphrase"]["hit@3"] >= 0.6  # TF-IDF: 0.82 / 0.42
    assert rep.no_answer_rate >= 0.75  # TF-IDF: 0.50
