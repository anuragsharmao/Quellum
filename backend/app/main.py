from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.conversations import router as conversations_router
from .api.routes import router
from .config import db_path, get_settings
from .services.cleanup import run_cleanup
from .services.db import Database
from .services.detector import Detector, ModelLoadError
from .services.embedding import build_embedder
from .services.errors import AppError
from .services.generation import build_generator
from .services.ratelimit import RateLimiter

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("quellum")
settings = get_settings()

if settings.cookie_samesite not in ("lax", "strict", "none"):
    raise RuntimeError("QUELLUM_COOKIE_SAMESITE must be lax, strict or none")
if settings.cookie_samesite == "none" and not settings.cookie_secure:
    raise RuntimeError("SameSite=None cookies require QUELLUM_COOKIE_SECURE=true (HTTPS)")


async def _cleanup_loop(app: FastAPI):
    while True:
        try:
            await asyncio.to_thread(run_cleanup, app.state.db, app.state.settings)
        except Exception:  # noqa: BLE001
            log.exception("scheduled cleanup crashed; will retry next interval")
        await asyncio.sleep(max(settings.cleanup_interval_minutes, 1) * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    st = app.state
    st.settings, st.limiter = settings, RateLimiter()
    st.db = Database(db_path(settings))
    st.detector, st.load_error = None, None
    try:
        det = Detector(settings)
        det.load()
        st.detector = det
        log.info("detector ready: %d models", len(det.nets))
    except ModelLoadError as e:
        log.error("detector load failed: %s", e)
        st.load_error = e.code
    except Exception:  # noqa: BLE001
        log.exception("detector load failed")
        st.load_error = "unexpected_error"
    shared = getattr(getattr(st.detector, "sbert", None), "model", None)
    try:
        st.embedder = build_embedder(settings.embedder, shared)
        if settings.embedder != "hashing" and shared is None:
            import sentence_transformers  # noqa: F401  (fail early if retrieval cannot work)
    except Exception:  # noqa: BLE001
        log.error("embedder %r is unavailable: uploads will be refused (set QUELLUM_EMBEDDER=hashing for a dependency-free dev embedder)", settings.embedder)
        st.embedder = None
    st.generator = build_generator(settings)
    if st.generator is None:
        log.warning("answer generation is NOT configured (QUELLUM_LLM_*): questions will be refused with 503, never faked")
    task = asyncio.create_task(_cleanup_loop(app)) if settings.cleanup_interval_minutes > 0 else None
    yield
    if task:
        task.cancel()


app = FastAPI(title="Quellum", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=True,
                   allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["Content-Type"])


@app.middleware("http")
async def guards(request: Request, call_next):
    """CSRF defence in depth (cookies are SameSite already): state-changing requests from a browser must come from an allowed origin.
    Also rejects oversized bodies before they are parsed."""
    if request.method in ("POST", "PATCH", "PUT", "DELETE"):
        origin = request.headers.get("origin")
        if origin and origin not in settings.cors_origins and urlparse(origin).netloc != request.headers.get("host", ""):
            return JSONResponse(status_code=403, content=dict(status="error", code="forbidden_origin", detail="Cross-site request refused."))
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            length = 0
        if length > (settings.max_upload_mb + 1) * 1024 * 1024:
            return JSONResponse(status_code=413, content=dict(status="error", code="too_large", detail="Request body too large."))
    return await call_next(request)


@app.exception_handler(AppError)
async def _app_error(_: Request, exc: AppError):
    headers = {"Retry-After": "60"} if exc.status == 429 else None
    return JSONResponse(status_code=exc.status, content=dict(status="error", code=exc.code, detail=exc.message), headers=headers)


@app.exception_handler(RequestValidationError)
async def _validation(_: Request, exc: RequestValidationError):
    errs = [dict(field=".".join(str(p) for p in e["loc"][1:]), message=e["msg"]) for e in exc.errors()]
    return JSONResponse(status_code=422, content=dict(status="error", code="invalid_request", detail="Invalid request", errors=errs))


@app.exception_handler(Exception)
async def _unexpected(_: Request, exc: Exception):
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content=dict(status="error", code="internal_error", detail="Something went wrong."))


app.include_router(conversations_router)
app.include_router(router)


@app.get("/healthz")
def healthz(request: Request):
    """Liveness for platform health checks: the process is up and the database answers (the detector may still be loading or off)."""
    ok = request.app.state.db.ping()
    return JSONResponse(status_code=200 if ok else 503, content=dict(status="ok" if ok else "degraded", storage=ok))


@app.get("/health")
def health(request: Request):
    st = request.app.state
    det_ok = st.detector is not None and st.detector.ready
    body = dict(status="ok" if (det_ok and st.db.ping()) else "degraded", storage=st.db.ping(),
                detector=dict(ready=det_ok, models_loaded=len(st.detector.nets) if det_ok else 0, error=None if det_ok else st.load_error),
                generation_configured=st.generator is not None, retrieval_available=st.embedder is not None)
    return JSONResponse(status_code=200 if body["status"] == "ok" else 503, content=body)


if settings.serve_frontend and settings.frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(settings.frontend_dir), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return dict(service="Quellum hallucination verification API", status="running", docs="/docs")
