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


def build_chunks(docs: list[Document], chunk_words: int = 140, overlap: int = 30) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, chunk_words, overlap)]
