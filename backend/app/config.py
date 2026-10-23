"""Central configuration. Only paths and request limits live here; model hyper-parameters come from artifacts/config.json,
which scripts/export_artifacts.py copies out of the research run's results.json."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]

# Load .env BEFORE any setting below is read (values are read once, at import time). Real environment variables always win.
# backend/.env is read first, then the repository-root .env as a fallback.
try:
    from dotenv import load_dotenv
    for _env_file in (BACKEND_DIR / ".env", BACKEND_DIR.parent / ".env"):
        if _env_file.is_file():
            load_dotenv(_env_file, override=False, encoding="utf-8-sig")
except ImportError:  # python-dotenv missing: fall back to a tiny parser so the keys are still picked up
    for _env_file in (BACKEND_DIR / ".env", BACKEND_DIR.parent / ".env"):
        if not _env_file.is_file():
            continue
        for _line in _env_file.read_text(encoding="utf-8-sig").splitlines():
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _v = _line.split("=", 1)
            _v = _v.split(" #", 1)[0].strip().strip('"').strip("'")
            os.environ.setdefault(_k.strip(), _v)

# request limits: application policy, not part of the research model
MAX_QUESTION_CHARS = int(os.environ.get("QUELLUM_MAX_QUESTION_CHARS", 2_000))
MAX_CONTEXT_CHARS = int(os.environ.get("QUELLUM_MAX_CONTEXT_CHARS", 30_000))
MAX_ANSWER_CHARS = int(os.environ.get("QUELLUM_MAX_ANSWER_CHARS", 6_000))
MAX_SENTENCES = int(os.environ.get("QUELLUM_MAX_SENTENCES", 60))


def _bool(name, default):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _int_tuple(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as e:
        raise RuntimeError(f"{name} must be a comma-separated list of integers") from e


_PROD = os.environ.get("QUELLUM_ENV", "development").lower() == "production"


@dataclass(frozen=True)
class Settings:
    # ---- environment / storage
    env: str = "production" if _PROD else "development"
    data_dir: Path = Path(os.environ.get("QUELLUM_DATA_DIR", BACKEND_DIR / "data"))
    frontend_dir: Path = Path(os.environ.get("QUELLUM_FRONTEND_DIR", BACKEND_DIR.parent / "frontend"))
    serve_frontend: bool = _bool("QUELLUM_SERVE_FRONTEND", True)
    # ---- anonymous sessions / cookies
    cookie_name: str = os.environ.get("QUELLUM_COOKIE_NAME", "quellum_sid")
    cookie_secure: bool = _bool("QUELLUM_COOKIE_SECURE", _PROD)        # True on HTTPS; browsers require it for SameSite=None
    cookie_samesite: str = os.environ.get("QUELLUM_COOKIE_SAMESITE", "lax").lower()   # "lax" same-site, "none" cross-site (needs secure)
    # ---- retention
    expiration_days: float = float(os.environ.get("QUELLUM_EXPIRATION_DAYS", 30))
    orphan_doc_ttl_hours: float = float(os.environ.get("QUELLUM_ORPHAN_DOC_TTL_HOURS", 24))
    tombstone_days: float = float(os.environ.get("QUELLUM_TOMBSTONE_DAYS", 90))
    cleanup_interval_minutes: float = float(os.environ.get("QUELLUM_CLEANUP_INTERVAL_MINUTES", 60))   # 0 disables the in-process scheduler
    # ---- limits
    max_upload_mb: int = int(os.environ.get("QUELLUM_MAX_UPLOAD_MB", 20))
    max_pdf_pages: int = int(os.environ.get("QUELLUM_MAX_PDF_PAGES", 300))
    max_text_chars: int = int(os.environ.get("QUELLUM_MAX_TEXT_CHARS", 200_000))
    max_conversations_per_session: int = int(os.environ.get("QUELLUM_MAX_CONVERSATIONS", 100))
    max_messages_per_conversation: int = int(os.environ.get("QUELLUM_MAX_MESSAGES", 200))
    max_storage_mb_per_session: int = int(os.environ.get("QUELLUM_MAX_STORAGE_MB", 100))
    uploads_per_hour: int = int(os.environ.get("QUELLUM_UPLOADS_PER_HOUR", 20))
    messages_per_10min: int = int(os.environ.get("QUELLUM_MESSAGES_PER_10MIN", 30))
    new_sessions_per_hour_per_ip: int = int(os.environ.get("QUELLUM_NEW_SESSIONS_PER_HOUR", 60))
    # ---- retrieval / generation
    embedder: str = os.environ.get("QUELLUM_EMBEDDER", "sbert")        # "sbert" (all-MiniLM-L6-v2) or "hashing" (dependency-free, weaker)
    chunk_words: int = int(os.environ.get("QUELLUM_CHUNK_WORDS", 140))
    chunk_overlap: int = int(os.environ.get("QUELLUM_CHUNK_OVERLAP", 30))
    top_k: int = int(os.environ.get("QUELLUM_TOP_K", 5))
    llm_provider: str = os.environ.get("QUELLUM_LLM_PROVIDER", "groq").lower()   # groq | openai_compatible | anthropic | none
    llm_model: str = os.environ.get("QUELLUM_LLM_MODEL", "llama-3.3-70b-versatile")
    llm_base_url: str = os.environ.get("QUELLUM_LLM_BASE_URL", "https://api.openai.com/v1")
    llm_api_key: str = os.environ.get("QUELLUM_LLM_API_KEY", "")
    groq_api_key: str = os.environ.get("GROQ_API_KEY", os.environ.get("QUELLUM_GROQ_API_KEY", ""))
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", os.environ.get("QUELLUM_GEMINI_API_KEY", ""))
    llm_fallback_provider: str = os.environ.get("QUELLUM_LLM_FALLBACK_PROVIDER", "gemini").lower()
    llm_fallback_model: str = os.environ.get("QUELLUM_LLM_FALLBACK_MODEL", "gemini-3.8-flash")
    llm_timeout_s: float = float(os.environ.get("QUELLUM_LLM_TIMEOUT_S", 60))
    llm_max_prompt_chars: int = int(os.environ.get("QUELLUM_LLM_MAX_PROMPT_CHARS", 12_000))
    # Seed ids map to models/seed_{id}.weights.h5 (0-4) or seed{id}.weights.h5 (5-9).
    detector_seeds: tuple = _int_tuple("QUELLUM_DETECTOR_SEEDS", tuple(range(8)))
    # ---- detector backend paths (previous fields)
    research_dir: Path = Path(os.environ.get("QUELLUM_RESEARCH_DIR", BACKEND_DIR / "research"))
    models_dir: Path = Path(os.environ.get("QUELLUM_MODELS_DIR", BACKEND_DIR / "models"))
    artifacts_dir: Path = Path(os.environ.get("QUELLUM_ARTIFACTS_DIR", BACKEND_DIR / "artifacts"))
    cors_origins: tuple = tuple(o.strip() for o in os.environ.get(
        "QUELLUM_CORS_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://localhost:5500,"
        "http://127.0.0.1:5173,http://127.0.0.1:3000,http://127.0.0.1:5500"
    ).split(",") if o.strip())
    torch_threads: int = int(os.environ.get("QUELLUM_TORCH_THREADS", 0))   # 0 = library default
    tf_threads: int = int(os.environ.get("QUELLUM_TF_THREADS", 0))


def get_settings() -> Settings:
    return Settings()


def db_path(s: Settings) -> Path:
    return s.data_dir / "quellum.sqlite3"


def files_dir(s: Settings) -> Path:
    return s.data_dir / "files"
