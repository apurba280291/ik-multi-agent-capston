import json
import re
from pathlib import Path

import pytest

from src.core.config import PROJECT_ROOT, Settings, load_settings
from src.rag.evaluation import evaluate_retrieval, load_queries
from src.rag.pdf_loader import clean_page_text, extract_pages, load_pdfs, strip_repeated_lines
from src.rag.retriever import Retriever

pypdf = pytest.importorskip("pypdf")
REAL_PDF = PROJECT_ROOT / "src" / "data" / "pdfs" / "tax" / "p590a.pdf"


def make_pdf(pages: list[str]) -> bytes:
    """Smallest valid multi-page PDF with one line of Helvetica text per page (no external libraries)."""
    objs: list[bytes] = []
    n = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    font_id = 3 + 2 * n
    for i, text in enumerate(pages):
        content_id = 4 + 2 * i
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 10 Tf 40 700 Td ({safe}) Tj ET".encode()
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


LONG = " ".join(["Investors should diversify their holdings across many asset classes to manage risk."] * 3)


def test_strip_repeated_lines_removes_running_headers_but_keeps_content():
    topics = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"]
    pages = [f"Publication 590-A  Page {i} of 9\nbody text about {t} with enough words\nConfidential draft" for i, t in enumerate(topics, 1)]
    cleaned = strip_repeated_lines(pages)
    assert all("Publication" not in p and "Confidential" not in p for p in cleaned)  # repeated modulo page digits
    assert all(f"body text about {t}" in cleaned[i] for i, t in enumerate(topics))
    short = ["a\nb", "a\nc"]  # fewer than 3 pages: nothing to compare, left untouched
    assert strip_repeated_lines(short) == short


def test_clean_page_text_fixes_kerning_and_hyphenation():
    assert clean_page_text("Y ou can make a T able of the W hat they invest-\nment plan") == "You can make a Table of the What they investment plan"
    assert clean_page_text("A b test\x0c  spaced   out") == "A b test spaced out"  # unrelated single letters untouched


def test_load_pdfs_pages_categories_titles_and_cache(tmp_path):
    pdf_dir, cache = tmp_path / "pdfs", tmp_path / "cache"
    (pdf_dir / "tax").mkdir(parents=True)
    (pdf_dir / "tax" / "guide.pdf").write_bytes(make_pdf([LONG, "tiny", LONG + " More text for page three here."]))
    (pdf_dir / "loose.pdf").write_bytes(make_pdf([LONG]))
    (pdf_dir / "sources.json").write_text(json.dumps({"guide.pdf": {"title": "Tax Guide", "reference": "http://example.com/g"}}))
    docs = load_pdfs(pdf_dir, cache)
    by_id = {d.doc_id: d for d in docs}
    assert set(by_id) == {"guide-p1", "guide-p3", "loose-p1"}  # near-empty page 2 skipped
    assert by_id["guide-p3"].title == "Tax Guide (p. 3)" and by_id["guide-p3"].category == "tax"
    assert by_id["guide-p3"].reference == "http://example.com/g"
    assert by_id["loose-p1"].category == "documents" and by_id["loose-p1"].title == "loose (p. 1)"
    assert list(cache.glob("*.json"))  # extraction cached
    assert [d.doc_id for d in load_pdfs(pdf_dir, cache)] == [d.doc_id for d in docs]  # served from cache


def test_cache_is_used_instead_of_reparsing(tmp_path, monkeypatch):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(make_pdf([LONG]))
    first = extract_pages(pdf, tmp_path / "c")
    monkeypatch.setattr(pypdf, "PdfReader", lambda *a, **k: pytest.fail("PDF was parsed again"))
    assert extract_pages(pdf, tmp_path / "c") == first


def test_unreadable_or_missing_pdfs_do_not_crash(tmp_path):
    (tmp_path / "bad.pdf").write_bytes(b"this is not a pdf")
    assert load_pdfs(tmp_path) == []
    assert load_pdfs(tmp_path / "nope") == []


def test_pdf_pages_are_retrievable_with_page_citations(tmp_path, settings):
    pdf_dir = tmp_path / "pdfs" / "tax"
    pdf_dir.mkdir(parents=True)
    pages = [
        "Contributions to a Roth IRA must be made by the tax filing due date. " * 4,
        "Required minimum distributions from a traditional IRA begin at a set age. " * 4,
    ]
    (pdf_dir / "ira.pdf").write_bytes(make_pdf(pages))
    cfg = Settings({"rag": {**settings.raw["rag"], "pdf_dir": str(tmp_path / "pdfs"), "pdf_cache_dir": str(tmp_path / "c")}})
    r = Retriever.from_settings(cfg)
    hit = r.search("required minimum distributions", k=1)[0]
    assert hit.title == "ira (p. 2)" and hit.category == "tax"
    assert "tax" in r.categories


# --- the real downloaded PDF (IRS Publication 590-A) -------------------------------------------------
real = pytest.mark.skipif(not REAL_PDF.exists(), reason="IRS Publication 590-A PDF not present")


@real
def test_real_pdf_loads_with_boilerplate_removed(tmp_path):
    docs = load_pdfs(REAL_PDF.parent.parent, tmp_path)
    assert len(docs) >= 55 and all(d.category == "tax" for d in docs)
    # The print-proof banner repeats on every page and is stripped; the one-off banner on the cover page (p. 1) cannot be
    # detected by repetition and stays (known limitation).
    rest = [d.text for d in docs if not d.doc_id.endswith("-p1")]
    assert not any("MUST be removed before printing" in t or "Fileid" in t or re.search(r"\bY ou\b", t) for t in rest)


@real
def test_real_pdf_quality_floor(tmp_path):
    cfg = load_settings()
    cfg.set("rag.pdf_cache_dir", str(tmp_path))
    r = Retriever.from_settings(cfg)
    assert {"tax", "glossary"} <= set(r.categories)
    inherit = r.search("what happens if I inherit an IRA", k=3)
    assert any(h.title.endswith("(p. 21)") for h in inherit)
    rep = evaluate_retrieval(r, load_queries("pdf_eval.json"), k=3)
    assert rep.hit_at_k >= 0.3  # current TF-IDF baseline is ~0.42; neural embeddings should beat it


@real
def test_form_number_question_reaches_the_pdf_through_the_whole_assistant(tmp_path):
    from src.workflow.assistant import FinanceAssistant

    cfg = load_settings()
    cfg.set("rag.embedder", "tfidf")
    cfg.set("rag.pdf_cache_dir", str(tmp_path))
    bot = FinanceAssistant(cfg)
    res = bot.chat("What is form 8880 used for?", session_id="forms")  # no finance keyword in the question
    assert res.agents and any("Publication 590-A" in s["title"] for s in res.sources)
    assert bot.chat("what is the capital of france", session_id="geo").agents == []  # still off-topic
