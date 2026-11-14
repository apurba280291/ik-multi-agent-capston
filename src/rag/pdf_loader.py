"""Load PDFs into the knowledge base, one ``Document`` per page so answers can cite page numbers.

Layout under ``rag.pdf_dir`` (default ``src/data/pdfs``):

    pdfs/tax/p590a.pdf        -> category "tax" (the sub-folder name; files directly in pdfs/ get "documents")
    pdfs/sources.json         -> optional {"p590a.pdf": {"title": "IRS Publication 590-A", "reference": "https://..."}}

Extraction needs ``pypdf`` (optional dependency). Extracted page text is cached on disk, keyed by file size and
modification time, so a large PDF is only parsed again when it changes. Scanned PDFs have no text layer and yield nothing
(they would need OCR).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path

from src.rag.documents import Document
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

MIN_PAGE_WORDS = 20  # blank, cover and image-only pages are skipped
DEFAULT_CATEGORY = "documents"


def strip_repeated_lines(pages: list[str], min_share: float = 0.4, min_pages: int = 3) -> list[str]:
    """Remove running headers, footers and print-proof banners: lines that recur (ignoring digits) on many pages."""
    if len(pages) < min_pages:
        return pages

    def norm(line: str) -> str:
        return re.sub(r"\d+", "#", " ".join(line.split())).lower()

    seen: Counter[str] = Counter()
    for page in pages:
        seen.update({norm(line) for line in page.splitlines() if line.strip()})
    threshold = max(2, min_share * len(pages))
    repeated = {line for line, n in seen.items() if n >= threshold}
    return ["\n".join(line for line in page.splitlines() if norm(line) not in repeated) for page in pages]


# Some PDFs kern a capital from the rest of the word ("Y ou", "T able"); re-join only the common, unambiguous cases.
_KERNED = re.compile(r"\b([YTW]) (ou|our|ours|o|able|ax|axes|aking|his|hese|hat|he|hen|here|hich|hy|ith|ill|e|ould|as)\b")


def clean_page_text(text: str) -> str:
    text = text.replace("\x0c", " ").replace("­", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across a line break
    return _KERNED.sub(r"\1\2", " ".join(text.split()))


def _cache_file(cache_dir: Path, pdf: Path) -> Path:
    stat = pdf.stat()
    key = hashlib.sha1(f"{pdf.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:20]
    return cache_dir / f"{pdf.stem}-{key}.json"


def extract_pages(pdf: Path, cache_dir: Path | None = None) -> list[str]:
    """Cleaned text of each page (index 0 = page 1); ``[]`` if the PDF cannot be read."""
    cache = _cache_file(cache_dir, pdf) if cache_dir else None
    if cache and cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf is not installed; skipping %s (pip install pypdf)", pdf.name)
        return []
    logging.getLogger("pypdf").setLevel(logging.CRITICAL)  # font-encoding warnings are noise for text extraction
    try:
        reader = PdfReader(str(pdf))
        raw = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # corrupt, encrypted, ...
        log.warning("could not read %s: %s", pdf.name, exc)
        return []
    pages = [clean_page_text(t) for t in strip_repeated_lines(raw)]
    if cache:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(pages), encoding="utf-8")
        except OSError:
            pass
    return pages


def _load_sources(pdf_dir: Path) -> dict[str, dict[str, str]]:
    try:
        data = json.loads((pdf_dir / "sources.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def load_pdfs(pdf_dir: Path, cache_dir: Path | None = None) -> list[Document]:
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_dir():
        return []
    sources = _load_sources(pdf_dir)
    docs: list[Document] = []
    for pdf in sorted(pdf_dir.rglob("*.pdf")):
        category = pdf.parent.name if pdf.parent != pdf_dir else DEFAULT_CATEGORY
        meta = sources.get(pdf.name, {})
        title = meta.get("title") or pdf.stem.replace("_", " ").replace("-", " ")
        pages = extract_pages(pdf, cache_dir)
        kept = 0
        for number, text in enumerate(pages, start=1):
            if len(text.split()) < MIN_PAGE_WORDS:
                continue
            docs.append(Document(f"{pdf.stem}-p{number}", f"{title} (p. {number})", text, category, meta.get("reference", "")))
            kept += 1
        log.info("pdf %s: %d pages -> %d documents (category %s)", pdf.name, len(pages), kept, category)
    return docs
