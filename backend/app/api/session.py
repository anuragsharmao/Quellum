"""FastAPI dependency: resolves the anonymous session from the HttpOnly cookie, creating one (and setting the cookie) on first contact."""
from __future__ import annotations

from fastapi import Request, Response

from ..config import Settings
from ..services import sessions
from ..services.sessions import SessionCtx


def set_session_cookie(response: Response, s: Settings, token: str):
    response.set_cookie(key=s.cookie_name, value=token, max_age=int(s.expiration_days * 86400), httponly=True,
                        secure=s.cookie_secure, samesite=s.cookie_samesite, path="/")


def require_session(request: Request, response: Response) -> SessionCtx:
    st = request.app.state
    s: Settings = st.settings
    ctx = sessions.resolve_session(st.db, s, request.cookies.get(s.cookie_name))
    if ctx is None:
        ip = request.client.host if request.client else "unknown"
        st.limiter.check(f"newsess:{ip}", s.new_sessions_per_hour_per_ip, 3600)
        ctx = sessions.create_session(st.db, s)
    if ctx.new_token:
        set_session_cookie(response, s, ctx.new_token)
    return ctx
