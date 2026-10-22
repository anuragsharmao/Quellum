"""Anonymous-session API for documents, conversations and messages. Every handler receives the session id from the cookie and passes it
to the service layer, which scopes every query by it. Ids in URLs only identify resources."""
from __future__ import annotations

import re

from fastapi import APIRouter, Depends, File, Request, Response, UploadFile
from pydantic import BaseModel, Field

from ..config import MAX_QUESTION_CHARS
from ..services import chat, conversations as convs, documents as docs
from ..services.db import iso
from ..services.errors import NotFound, Unavailable
from ..services.sessions import SessionCtx
from .session import require_session

router = APIRouter(prefix="/api")
_ID = re.compile(r"^[0-9a-f]{32}$")
NO_STORE = {"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"}


class TextDocBody(BaseModel):
    text: str = Field(..., min_length=1)
    name: str = Field("Text Source", max_length=150)


class CreateConversationBody(BaseModel):
    documentId: str = Field(..., max_length=64)
    title: str = Field("", max_length=200)


class PatchConversationBody(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class SendMessageBody(BaseModel):
    content: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS * 2)


def _id(value: str) -> str:
    if not _ID.match(value or ""):
        raise NotFound("Not found.")
    return value


def _embedder(request: Request):
    e = request.app.state.embedder
    if e is None:
        raise Unavailable("Document search is not available on this server.")
    return e


@router.get("/status")
def api_status():
    return dict(service="Quellum", status="running")


@router.get("/session")
def session(request: Request, ctx: SessionCtx = Depends(require_session)):
    s = request.app.state.settings
    st = request.app.state
    return dict(status="ok", retentionDays=s.expiration_days,
                capabilities=dict(generation=st.generator is not None, hallucinationAnalysis=bool(st.detector and st.detector.ready),
                                  retrieval=st.embedder is not None, maxUploadMb=s.max_upload_mb))


# ---------------------------------------------------------------- documents
@router.post("/documents")
def upload_pdf(request: Request, file: UploadFile = File(...), ctx: SessionCtx = Depends(require_session)):
    st = request.app.state
    st.limiter.check(f"upload:{ctx.id}", st.settings.uploads_per_hour, 3600)
    data = file.file.read(st.settings.max_upload_mb * 1024 * 1024 + 1)            # never read more than the limit + 1 byte
    meta = docs.ingest_pdf(st.db, st.settings, _embedder(request), ctx.id, file.filename or "document.pdf", data)
    return dict(status="ok", document=meta)


@router.post("/documents/text")
def create_text_document(body: TextDocBody, request: Request, ctx: SessionCtx = Depends(require_session)):
    st = request.app.state
    st.limiter.check(f"upload:{ctx.id}", st.settings.uploads_per_hour, 3600)
    return dict(status="ok", document=docs.ingest_text(st.db, st.settings, _embedder(request), ctx.id, body.text, body.name))


@router.get("/documents/{doc_id}")
def get_document(doc_id: str, request: Request, ctx: SessionCtx = Depends(require_session)):
    return dict(status="ok", document=docs.get_document_meta(request.app.state.db, ctx.id, _id(doc_id), with_text=True))


@router.get("/documents/{doc_id}/file")
def get_document_file(doc_id: str, request: Request, ctx: SessionCtx = Depends(require_session)):
    st = request.app.state
    data, name = docs.read_pdf_bytes(st.db, st.settings, ctx.id, _id(doc_id))
    return Response(content=data, media_type="application/pdf", headers={**NO_STORE, "Content-Disposition": "inline"})


# ---------------------------------------------------------------- conversations
@router.post("/conversations", status_code=201)
def create_conversation(body: CreateConversationBody, request: Request, ctx: SessionCtx = Depends(require_session)):
    st = request.app.state
    return dict(status="ok", conversation=convs.create(st.db, st.settings, ctx.id, _id(body.documentId), body.title))


@router.get("/conversations")
def list_conversations(request: Request, include: str = "", ctx: SessionCtx = Depends(require_session)):
    items = convs.list_conversations(request.app.state.db, ctx.id, include_messages=(include == "messages"))
    return dict(status="ok", conversations=items, retentionDays=request.app.state.settings.expiration_days)


@router.get("/conversations/{cid}")
def get_conversation(cid: str, request: Request, ctx: SessionCtx = Depends(require_session)):
    return dict(status="ok", conversation=convs.get(request.app.state.db, ctx.id, _id(cid)))


@router.patch("/conversations/{cid}")
def rename_conversation(cid: str, body: PatchConversationBody, request: Request, ctx: SessionCtx = Depends(require_session)):
    st = request.app.state
    return dict(status="ok", conversation=convs.rename(st.db, st.settings, ctx.id, _id(cid), body.title))


@router.delete("/conversations/{cid}")
def delete_conversation(cid: str, request: Request, ctx: SessionCtx = Depends(require_session)):
    st = request.app.state
    convs.delete(st.db, st.settings, ctx.id, _id(cid))
    return dict(status="ok", deleted=True)


@router.post("/conversations/{cid}/messages")
def send_message(cid: str, body: SendMessageBody, request: Request, ctx: SessionCtx = Depends(require_session)):
    st = request.app.state
    st.limiter.check(f"msg:{ctx.id}", st.settings.messages_per_10min, 600)
    det = st.detector
    analyzer = (lambda q, c, a: det.analyze(q, c, a)) if det is not None and det.ready else None
    msgs = chat.send_message(st.db, st.settings, ctx.id, _id(cid), body.content, embedder=_embedder(request), generator=st.generator, analyzer=analyzer)
    conv = convs.get(st.db, ctx.id, cid)
    return dict(status="ok", messages=msgs, conversation=dict(id=cid, updatedAt=conv["updatedAt"], expiresAt=conv["expiresAt"]))
