"""Small in-process sliding-window limiter. Correct for ONE server process; use Redis (or the platform's limiter) if you scale out."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from .errors import RateLimited


class RateLimiter:
    def __init__(self, clock=time.time):
        self.clock, self._hits, self._lock = clock, defaultdict(deque), threading.Lock()

    def check(self, key: str, limit: int, window_s: float):
        if limit <= 0:
            return
        now = self.clock()
        with self._lock:
            q = self._hits[key]
            while q and q[0] <= now - window_s:
                q.popleft()
            if len(q) >= limit:
                raise RateLimited("Too many requests. Please wait a moment and try again.")
            q.append(now)
            if len(self._hits) > 50_000:                     # bound memory
                for k in [k for k, v in self._hits.items() if not v][:10_000]:
                    self._hits.pop(k, None)
