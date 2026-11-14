"""Knowledge-base loading and chunking."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

GLOSSARY_REFERENCE = "https://www.investor.gov/introduction-investing/investing-basics/glossary"


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    category: str
    reference: str = ""


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    category: str
    reference: str = ""


def load_articles(kb_dir: Path) -> list[Document]:
    """Parse ``category:``/``reference:`` headers and ``## Title`` article sections."""
    docs: list[Document] = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        header, _, body = raw.partition("\n## ")
        category = re.search(r"^category:\s*(.+)$", header, re.M)
        reference = re.search(r"^reference:\s*(.+)$", header, re.M)
        cat = category.group(1).strip() if category else path.stem
        ref = reference.group(1).strip() if reference else ""
        if not body:
            continue
        for i, section in enumerate(("## " + body).split("\n## ")):
            section = section.removeprefix("## ")
            title, _, text = section.partition("\n")
            text = " ".join(text.split())
            if title.strip() and text:
                docs.append(Document(f"{cat}-{i}", title.strip(), text, cat, ref))
    return docs


def load_glossary(path: Path) -> list[Document]:
    try:
        entries = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("glossary unavailable: %s", exc)
        return []
    return [
        Document(f"glossary-{i}", term, f"{term}: {definition}", "glossary", GLOSSARY_REFERENCE)
        for i, (term, definition) in enumerate(entries.items())
    ]


def chunk_document(doc: Document, chunk_words: int = 140, overlap: int = 30) -> list[Chunk]:
    words = doc.text.split()
    if len(words) <= chunk_words:
        return [Chunk(f"{doc.doc_id}-0", doc.doc_id, doc.title, doc.text, doc.category, doc.reference)]
    step = max(1, chunk_words - overlap)
    chunks = []
    for n, start in enumerate(range(0, len(words), step)):
        piece = words[start : start + chunk_words]
        chunks.append(Chunk(f"{doc.doc_id}-{n}", doc.doc_id, doc.title, " ".join(piece), doc.category, doc.reference))
        if start + chunk_words >= len(words):
            break
    return chunks


# Recursive splitting tries coarse boundaries first and only falls back to finer ones when a piece is still too big.
_SPLITTERS = (
    (re.compile(r"\n\s*\n"), "\n\n"),         # paragraphs
    (re.compile(r"\n"), "\n"),                # lines
    (re.compile(r"(?<=[.!?])\s+"), " "),      # sentences
    (re.compile(r"\s+"), " "),                # words (last resort)
)


def recursive_split(text: str, max_words: int, level: int = 0) -> list[str]:
    """Split ``text`` into pieces of at most ``max_words`` words along paragraph > line > sentence > word boundaries.

    Pieces from one level are greedily merged up to the limit; a piece that is still too long is split again
    with the next, finer separator.
    """
    text = text.strip()
    if not text:
        return []
    if len(text.split()) <= max_words or level >= len(_SPLITTERS):
        return [text]
    pattern, joiner = _SPLITTERS[level]
    pieces = [p for p in pattern.split(text) if p.strip()]
    if len(pieces) <= 1:
        return recursive_split(text, max_words, level + 1)
    out: list[str] = []
    current: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal current, size
        if current:
            out.append(joiner.join(current))
        current, size = [], 0

    for piece in pieces:
        n = len(piece.split())
        if n > max_words:
            flush()
            out.extend(recursive_split(piece, max_words, level + 1))
            continue
        if size + n > max_words:
            flush()
        current.append(piece)
        size += n
    flush()
    return out


def chunk_document_recursive(doc: Document, chunk_words: int = 140, overlap: int = 30) -> list[Chunk]:
    """Boundary-aware chunks; each chunk after the first starts with the last ``overlap`` words of the previous one."""
    parts = recursive_split(doc.text, chunk_words)
    chunks = []
    for n, part in enumerate(parts):
        if n and overlap > 0:
            tail = " ".join(parts[n - 1].split()[-overlap:])
            part = f"{tail} {part}"
        chunks.append(Chunk(f"{doc.doc_id}-{n}", doc.doc_id, doc.title, part, doc.category, doc.reference))
    return chunks


STRATEGIES = ("fixed", "recursive")


def build_chunks(docs: list[Document], chunk_words: int = 140, overlap: int = 30, strategy: str = "fixed") -> list[Chunk]:
    """``fixed``: sliding word windows. ``recursive``: split on paragraph/sentence/word boundaries, then add overlap."""
    if strategy not in STRATEGIES:
        log.warning("unknown chunk strategy %r; using 'fixed'", strategy)
        strategy = "fixed"
    split = chunk_document_recursive if strategy == "recursive" else chunk_document
    return [c for d in docs for c in split(d, chunk_words, overlap)]
