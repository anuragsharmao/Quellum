class AppError(Exception):
    status = 400
    code = "bad_request"

    def __init__(self, message: str = ""):
        super().__init__(message or self.code)
        self.message = message or self.code


class InvalidInput(AppError):
    status, code = 422, "invalid_input"


class NotFound(AppError):
    status, code = 404, "not_found"


class Expired(AppError):
    status, code = 410, "expired"


class LimitExceeded(AppError):
    status, code = 413, "limit_exceeded"


class RateLimited(AppError):
    status, code = 429, "rate_limited"


class GenerationNotConfigured(AppError):
    status, code = 503, "generation_not_configured"


class GenerationFailed(AppError):
    status, code = 502, "generation_failed"


class Unavailable(AppError):
    status, code = 503, "unavailable"
