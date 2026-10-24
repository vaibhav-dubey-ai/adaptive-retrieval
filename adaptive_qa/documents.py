"""User-uploaded document support: extraction, chunking, persistence, retrieval.

Responsibilities:
- Accept PDF / TXT / Markdown / DOCX uploads, validate type and size.
- Extract readable text while PRESERVING page numbers for PDFs.
- Warn clearly when a PDF has no extractable text (scanned image; OCR is a
  documented future extension, not silently skipped here).
- Chunk documents with configurable size/overlap; each chunk remembers its
  source document and page range.
- Persist upload, metadata and chunks on disk under ``data/documents/`` so
  they survive server restarts.
- Provide a BM25 retriever restricted to the ACTIVE document collection
  (searching only the selected docs for Option B; never the benchmark
  corpus, which REMAINS accessible via the existing experiment pipeline).

Duplicate uploads are returned with their existing doc_id instead of
re-indexed (content is deduplicated by SHA-256).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import PROJECT_ROOT
from .retrieval.bm25 import BM25Retriever
from .types import RetrievalHit

DOCUMENT_ROOT = PROJECT_ROOT / "data" / "documents"

ALLOWED_EXTENSIONS = {".pdf": "pdf", ".txt": "text", ".md": "markdown", ".docx": "docx"}
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB


class DocumentUploadError(ValueError):
    """Malformed or unsupported upload (message is safe to show to the user)."""


@dataclass
class ExtractedDocument:
    doc_id: str
    kind: str
    pages: List[Dict[str, Any]]  # [{"page": int|None, "text": str}, ...]
    needs_ocr: bool
    total_chars: int


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_text(filename: str, data: bytes) -> ExtractedDocument:
    """Dispatch to the right extractor; raise DocumentUploadError on problems."""
    suffix = Path(filename or "").suffix.lower()
    kind = ALLOWED_EXTENSIONS.get(suffix)
    if kind is None:
        raise DocumentUploadError(
            f"Unsupported file type '{suffix}'. Supported: PDF, TXT, Markdown, DOCX."
        )
    if not data:
        raise DocumentUploadError("Uploaded file is empty (0 bytes).")
    if kind == "text" or kind == "markdown":
        pages = [_decode_text(data)]
        needs_ocr = False
    elif kind == "pdf":
        pages, needs_ocr = _extract_pdf(data)
    else:  # docx
        pages, needs_ocr = _extract_docx(data)
    total = sum(len(p["text"]) for p in pages)
    # NOTE: zero extractable text is NOT an exception here; it is surfaced as
    # needs_ocr=True so the caller (e.g. save_document) can explain the
    # likely scanned-PDF situation clearly instead of a vague parse error.
    doc_id = "udoc_" + sha256_hex(kind.encode() + data)[:12]
    return ExtractedDocument(
        doc_id=doc_id, kind=kind, pages=pages, needs_ocr=needs_ocr, total_chars=total
    )


def _decode_text(data: bytes) -> Dict[str, Any]:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return {"page": None, "text": data.decode(enc)}
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise DocumentUploadError("Text file could not be decoded with common encodings.")


def _extract_pdf(data: bytes) -> tuple:
    try:
        from pypdf import PdfReader
        import io
    except ImportError as exc:
        raise DocumentUploadError(
            "PDF support requires the 'pypdf' package (pip install pypdf)."
        ) from exc
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            pages.append({"page": i, "text": text})
    except DocumentUploadError:
        raise
    except Exception as exc:
        raise DocumentUploadError(f"Malformed PDF: {exc}") from exc
    needs_ocr = all(not p["text"].strip() for p in pages)
    return pages, needs_ocr


def _extract_docx(data: bytes) -> tuple:
    try:
        import docx as _docx  # python-docx
        import io
    except ImportError as exc:
        raise DocumentUploadError(
            "DOCX support requires 'python-docx' (pip install python-docx)."
        ) from exc
    try:
        d = _docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs]
        for table in d.tables:
            for row in table.rows:
                parts.append("\t".join(c.text for c in row.cells))
    except DocumentUploadError:
        raise
    except Exception as exc:
        raise DocumentUploadError(f"Malformed DOCX: {exc}") from exc
    text = "\n".join(parts)
    needs_ocr = not text.strip()
    return [{"page": None, "text": text}], needs_ocr


def chunk_pages(
    pages: List[Dict[str, Any]],
    doc_id: str,
    target_chars: int = 900,
    overlap_chars: int = 150,
) -> List[Dict[str, Any]]:
    """Chunk page texts into overlapping windows, keeping page attribution.

    A chunk never spans page boundaries (so page numbers stay truthful);
    documents without pages (TXT/MD/DOCX) produce one logical page-less stream.
    """
    if target_chars <= 0:
        raise ValueError("chunk size must be positive")
    if overlap_chars >= target_chars:
        raise ValueError("chunk overlap must be smaller than the chunk size")
    chunks: List[Dict[str, Any]] = []
    for page in pages:
        text = (page["text"] or "").replace("\x00", " ").strip()
        if not text:
            continue
        step = max(1, target_chars - overlap_chars)
        i = 0
        idx = 0
        while i < len(text):
            piece = text[i : i + target_chars]
            if piece.strip():
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}:{idx:04d}",
                        "doc_id": doc_id,
                        "chunk_index": idx,
                        "page": page.get("page"),
                        "text": piece.strip(),
                    }
                )
                idx += 1
            i += step
            if i + len(piece) >= len(text):  # consumed
                break
    return chunks


class DocumentStore:
    """Persistent store under data/documents/<doc_id>/{meta.json, chunks.jsonl, source.bin}."""

    def __init__(self, root: Path = DOCUMENT_ROOT, max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES) -> None:
        self.root = Path(root)
        self.max_upload_bytes = int(max_upload_bytes)

    # ------------------------------------------------------------- storage
    def _doc_dir(self, doc_id: str) -> Path:
        return self.root / doc_id

    def save_document(self, filename: str, data: bytes, *, max_chars: Optional[int] = None) -> Dict[str, Any]:
        if not filename:
            raise DocumentUploadError("Filename required.")
        if len(data) > self.max_upload_bytes:
            raise DocumentUploadError(
                f"File too large ({len(data)} bytes > limit {self.max_upload_bytes})."
            )
        digest = sha256_hex(data)
        existing = self.find_by_digest(digest)
        if existing is not None:
            return {**existing, "duplicate": True}
        if not data:
            raise DocumentUploadError("Uploaded file is empty (0 bytes).")
        extracted = extract_text(filename, data)
        self.root.mkdir(parents=True, exist_ok=True)
        ddir = self._doc_dir(extracted.doc_id)
        ddir.mkdir(parents=True, exist_ok=False)
        (ddir / "source.bin").write_bytes(data)
        chunks = chunk_pages(extracted.pages, extracted.doc_id)
        chunks = [c for c in chunks if c["text"]]
        if extracted.needs_ocr:
            raise DocumentUploadError(
                "No extractable text found; the file may be a scanned image or blank. "
                "OCR is not implemented — provide a copy with selectable text."
            )
        if not chunks:
            raise DocumentUploadError("Document produced no readable chunks.")
        meta = {
            "doc_id": extracted.doc_id,
            "filename": filename,
            "kind": extracted.kind,
            "pages": [p["page"] for p in extracted.pages if p.get("page") is not None],
            "n_pages": len([p for p in extracted.pages if p.get("page") is not None]),
            "n_chunks": len(chunks),
            "total_chars": extracted.total_chars,
            "needs_ocr": extracted.needs_ocr,
            "sha256": digest,
            "size_bytes": len(data),
            "active": True,  # newly created docs start as the active collection
            "uploaded_at": int(__import__("time").time()),
            "warning": (
                "PDF contains no extractable text layer (likely a scanned image); OCR is not implemented. Document-specific retrieval will return nothing until a searchable copy is provided."
                if extracted.needs_ocr
                else ""
            ),
        }
        (ddir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        with (ddir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        return {**meta, "duplicate": False}

    def list_documents(self) -> List[Dict[str, Any]]:
        if not self.root.exists():
            return []
        docs: List[Dict[str, Any]] = []
        for d in self.root.iterdir():
            meta_path = d / "meta.json"
            if meta_path.exists():
                try:
                    docs.append(json.loads(meta_path.read_text(encoding="utf-8")))
                except Exception:
                    continue
        docs.sort(key=lambda m: m.get("uploaded_at", 0), reverse=True)
        return docs

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        meta_path = self._doc_dir(doc_id) / "meta.json"
        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def find_by_digest(self, digest: str) -> Optional[Dict[str, Any]]:
        for m in self.list_documents():
            if m.get("sha256") == digest:
                return {**m, "duplicate": True}
        return None

    def delete(self, doc_id: str) -> bool:
        ddir = self._doc_dir(doc_id)
        if not ddir.exists():
            return False
        import shutil

        shutil.rmtree(ddir, ignore_errors=True)
        return True

    def set_active(self, doc_id: str, active: bool) -> bool:
        meta = self.get(doc_id)
        if meta is None:
            return False
        meta["active"] = bool(active)
        (self._doc_dir(doc_id) / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return True

    # ------------------------------------------------------------ retrieval
    def load_chunks(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load chunks of ONE document, or of every ACTIVE document when None."""
        ids = [doc_id] if doc_id else [m["doc_id"] for m in self.list_documents() if m.get("active")]
        out: List[Dict[str, Any]] = []
        for did in ids:
            cpath = self._doc_dir(did) / "chunks.jsonl"
            if not cpath.exists():
                continue
            for line in cpath.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
        return out

    def active_documents(self) -> List[Dict[str, Any]]:
        return [m for m in self.list_documents() if m.get("active")]

    def build_retriever(self, doc_id: Optional[str] = None) -> "CollectionRetriever":
        """BM25 over the active collection (current BM25 implementation kept)."""
        chunks = self.load_chunks(doc_id)
        bars: List[Any] = []
        for c in chunks:
            bars.append(
                type(
                    "Chunk",
                    (),
                    {
                        "doc_id": c["chunk_id"],
                        "title": f"{self.get(c['doc_id']).get('filename', c['doc_id'])}"
                        + (f" (page {c['page']})" if c.get("page") is not None else ""),
                        "text": c["text"],
                    },
                )
            )
        inner = BM25Retriever(bars)
        meta_map = {
            c["chunk_id"]: c for c in chunks
        }
        meta_by_doc = {m["doc_id"]: m for m in self.list_documents()}
        return CollectionRetriever(inner, meta_map, meta_by_doc, n_chunks=len(chunks))


class CollectionRetriever:
    """Scope-limited retriever: only the collection's own chunks are searched."""

    def __init__(self, inner: BM25Retriever, meta_map: Dict[str, Any], meta_by_doc: Dict[str, Any], n_chunks: int) -> None:
        self._inner = inner
        self._meta = meta_map
        self._docs = meta_by_doc
        self.n_chunks = n_chunks

    def retrieve(self, question: str, top_k: int) -> List[RetrievalHit]:
        # NOTE: RetrievalHit is a frozen dataclass; no replay hit subclass is
        # created. Upload-specific source info is joined in ``source_meta``
        # from ``self._meta`` using the chunk id.
        return self._inner.retrieve(question, top_k)

    def source_meta(self, hit: RetrievalHit) -> Dict[str, Any]:
        """Join upload provenance onto a retrieval hit (for evidence display)."""
        cm = self._meta.get(hit.doc_id) or {}
        dn = self._docs.get(cm.get("doc_id")) or {}
        return {
            "filename": dn.get("filename"),
            "page": cm.get("page"),
            "score": hit.score,
            "text": hit.text,
            "doc_id": cm.get("doc_id"),
        }
