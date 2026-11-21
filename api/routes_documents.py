"""Document routes: upload, list, activate/deactivate, delete."""

from __future__ import annotations

from typing import Any, Dict
from fastapi import File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from adaptive_qa.live.documents import DocumentUploadError
from fastapi import APIRouter
from api import deps

router = APIRouter()


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a PDF/TXT/MD/DOCX, persist it, chunk + index it."""
    data = await file.read()
    try:
        meta = deps._docs().save_document(file.filename or "", data)
    except DocumentUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # Any collection change invalidates the cached retrievers + preflight.
    deps._invalidate_collection()
    return JSONResponse(
        {
            "document": meta,
            "message": (
                "Document already existed (deduplicated by content)."
                if meta.get("duplicate")
                else "Uploaded, chunked and indexed into the active collection."
            ),
        }
    )


@router.get("/documents")
def list_documents() -> JSONResponse:
    docs = deps._docs().list_documents()
    return JSONResponse({"documents": docs})


@router.post("/documents/{doc_id}/active")
def set_document_active(doc_id: str, payload: Dict[str, Any] = None) -> JSONResponse:
    active = bool(((payload or {}).get("active")) or ((payload or {}).get("value")))
    if not deps._docs().set_active(doc_id, active):
        raise HTTPException(status_code=404, detail="document not found")
    deps._invalidate_collection()
    return JSONResponse({"doc_id": doc_id, "active": active})


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> JSONResponse:
    if not deps._docs().delete(doc_id):
        raise HTTPException(status_code=404, detail="document not found")
    deps._invalidate_collection()
    return JSONResponse({"deleted": doc_id})
