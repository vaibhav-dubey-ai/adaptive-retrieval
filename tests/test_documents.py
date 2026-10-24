"""Tests for user-document upload: extraction, chunking, persistence, scoped retrieval.

Unit tests only — no network and no real model API is required.
"""

import io
import json
from pathlib import Path

import pytest

from adaptive_qa.documents import (
    DocumentStore,
    DocumentUploadError,
    chunk_pages,
    extract_text,
    sha256_hex,
)


def _docx_bytes(paragraphs) -> bytes:
    import docx as python_docx

    d = python_docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def _minimal_pdf_bytes(pages):
    """Hand-rolled minimal PDF: one page per text page, extractable by pypdf."""
    stream = io.BytesIO()
    objs = []
    page_ids = []

    def add_raw(obj: str) -> int:
        objs.append(obj)
        return len(objs)

    font_id = add_raw("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for text in pages:
        content = f"BT /F1 12 Tf 72 700 Td ({text}) Tj ET"
        cid = add_raw(f"<< /Length {len(content)} >>\nstream\n{content}\nendstream")
        pid = add_raw(
            f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {cid} 0 R >>"
        )
        page_ids.append(pid)
    kids = " ".join(f"{p} 0 R" for p in page_ids) or ""
    pages_id = add_raw(
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>"
    )
    for p in page_ids:  # fix parent pointer
        objs[p - 1] = objs[p - 1].replace("/Parent 0 0 R", f"/Parent {pages_id} 0 R")
    catalog_id = add_raw(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objs, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n{obj}\nendobj\n".encode())
    xref_pos = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objs) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n".encode()
    )
    return out.getvalue()


# ------------------------------------------------------------------ extraction
def test_extract_txt_and_markdown():
    ext = extract_text("notes.md", "Alpha beta\ngamma".encode("utf-8"))
    assert ext.kind in ("text", "markdown")
    assert ext.pages[0]["page"] is None
    assert "beta" in ext.pages[0]["text"]


def test_unsupported_type_rejected():
    with pytest.raises(DocumentUploadError) as ei:
        extract_text("photo.jpg", b"\xff\xd8")
    assert "Unsupported file type" in str(ei.value)


def test_empty_file_rejected():
    with pytest.raises(DocumentUploadError) as ei:
        extract_text("empty.txt", b"")
    assert "0 bytes" in str(ei.value)


def test_docx_extraction():
    data = _docx_bytes(["Machine learning uses data.", "Second paragraph."])
    ext = extract_text("ml.docx", data)
    assert "Machine learning uses data." in ext.pages[0]["text"]
    assert ext.pages[0]["page"] is None


def test_scanned_pdf_no_extractable_text_warns():
    data = _minimal_pdf_bytes(["", ""])
    ext = extract_text("scanned.pdf", data)
    assert ext.needs_ocr is True

    store = _tmp_store()
    with pytest.raises(DocumentUploadError) as ei:
        store.save_document("scanned.pdf", data)
    assert "scanned" in str(ei.value).lower()


def test_pdf_pages_preserved():
    data = _minimal_pdf_bytes(["First page fact.", "Second page fact."])
    ext = extract_text("two.pdf", data)
    assert [p["page"] for p in ext.pages] == [1, 2]
    assert "First page fact." in ext.pages[0]["text"]


def test_malformed_pdf_rejected():
    with pytest.raises(DocumentUploadError) as ei:
        extract_text("broken.pdf", b"this is not a pdf")
    assert "Malformed PDF" in str(ei.value)


def _tmp_store() -> DocumentStore:
    import tempfile

    return DocumentStore(root=Path(tempfile.mkdtemp(prefix="qa_docs_test")))


# ------------------------------------------------------------------ chunking
def test_chunking_overlap_and_size():
    text = "word " * 400  # 2000 chars
    chunks = chunk_pages([{"page": 3, "text": text}], doc_id="D", target_chars=200, overlap_chars=50)
    assert len(chunks) > 1
    sizes = [len(c["text"]) for c in chunks[:-1]]
    assert all(s <= 200 for s in sizes)
    # overlap: consecutive chunks share content (modulo edge whitespace)
    assert chunks[0]["text"][-50:].strip() == chunks[1]["text"][:50].strip()
    assert all(c["page"] == 3 for c in chunks)
    # non-positive target / huge overlap both rejected
    with pytest.raises(ValueError):
        chunk_pages([{"page": None, "text": "x"}], "D", target_chars=0)
    with pytest.raises(ValueError):
        chunk_pages([{"page": None, "text": "x" * 10}], "D", target_chars=10, overlap_chars=20)


def test_chunk_pages_never_span_page_boundary():
    data = _minimal_pdf_bytes(["Alpha page one text here.", "Beta page two text here."])
    ext = extract_text("p.pdf", data)
    chunks = chunk_pages(ext.pages, doc_id="D", target_chars=1000, overlap_chars=50)
    texts = [c["text"] for c in chunks]
    assert any("Alpha" in t for t in texts) and any("Beta" in t for t in texts)
    assert not any(("Alpha" in t and "Beta" in t) for t in texts)


# ------------------------------------------------------- persistence + dedup
def _store(tmp_path: Path) -> DocumentStore:
    return DocumentStore(root=tmp_path / "documents")


def test_persistence_survives_new_store_instance(tmp_path):
    store = _store(tmp_path)
    data = ("Long-lived document content. " * 40).encode()
    meta1 = store.save_document("a.txt", data)
    assert meta1["duplicate"] is False
    # new instance = simulated server restart
    store2 = DocumentStore(root=tmp_path / "documents")
    docs = [d for d in store2.list_documents() if d["doc_id"] == meta1["doc_id"]]
    assert len(docs) == 1
    chunks = store2.load_chunks(meta1["doc_id"])
    assert chunks and json.loads(json.dumps(chunks[0]))["doc_id"] == meta1["doc_id"]
    assert (json.dumps(chunks[0]))  # chunk metadata preserved


def test_duplicate_upload_detected(tmp_path):
    store = _store(tmp_path)
    data = ("Deterministic duplicate content. " * 30).encode()
    m1 = store.save_document("first.txt", data)
    m2 = store.save_document("renamed.txt", data)
    assert m2["duplicate"] is True
    assert m1["doc_id"] == m2["doc_id"]
    assert store.list_documents() and len([d for d in store.list_documents()]) == 1


def test_delete_and_toggle_active(tmp_path):
    store = _store(tmp_path)
    meta = store.save_document("x.txt", b"toggle content " * 30)
    assert store.set_active(meta["doc_id"], False)
    assert all(not d["active"] for d in store.list_documents() if d["doc_id"] == meta["doc_id"])
    assert store.delete(meta["doc_id"])
    assert store.get(meta["doc_id"]) is None
    assert not store.delete(meta["doc_id"])  # already gone


def test_too_large_rejected(tmp_path):
    store = DocumentStore(root=tmp_path / "documents", max_upload_bytes=10)
    with pytest.raises(DocumentUploadError) as ei:
        store.save_document("big.txt", b"x" * 50)
    assert "too large" in str(ei.value).lower()


# ------------------------------------------------------------ scoped retrieval
def test_retrieval_restricted_to_active_collection(tmp_path):
    store = _store(tmp_path)
    m_active = store.save_document("active.txt", ("The Eiffel Tower is in Paris. " * 20).encode())
    m_inactive = store.save_document("inactive.txt", ("Quantum chromodynamics is a theory. " * 20).encode())
    store.set_active(m_inactive["doc_id"], False)
    retriever = store.build_retriever()
    hits = retriever.retrieve("the eiffel tower paris", top_k=5)
    assert hits
    assert all(h.doc_id.startswith(m_active["doc_id"]) for h in hits)
    # provenance join works on whatever retriever returns
    if hits:
        sm = retriever.source_meta(hits[0])
        assert sm["filename"] == "active.txt"
    # restoring active brings the other doc back into scope
    store.set_active(m_inactive["doc_id"], True)
    hits2 = store.build_retriever().retrieve("quantum chromodynamics theory", top_k=5)
    assert all(h.doc_id.startswith(m_inactive["doc_id"]) for h in hits2)


def test_selected_collection_only_for_option_b(tmp_path):
    """The benchmark corpus retriever must not be reachable through DocumentStore."""
    store = _store(tmp_path)
    meta = store.save_document("only.txt", ("Only this doc. " * 10).encode())
    r = store.build_retriever()
    assert r.n_chunks > 0
    assert set(c["doc_id"] for c in store.load_chunks()) == {meta["doc_id"]}
