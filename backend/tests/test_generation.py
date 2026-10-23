from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_settings
from app.services.errors import GenerationFailed
from app.services.generation import FallbackGenerator, GeminiGenerator, build_generator, build_user_prompt


class Response:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


def test_prompt_limit_retains_question():
    prompt = build_user_prompt("What is stated?", ["A long passage " * 50], max_chars=120)
    assert len(prompt) <= 120
    assert prompt.endswith("Question: What is stated?\n\nAnswer:")


def test_groq_failure_uses_gemini_once(monkeypatch):
    import requests

    settings = replace(get_settings(), llm_provider="groq", groq_api_key="groq-key", gemini_api_key="gemini-key")
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        if "groq.com" in url:
            raise requests.Timeout()
        return Response({"candidates": [{"content": {"parts": [{"text": "Grounded answer"}]}}]})

    monkeypatch.setattr(requests, "post", post)
    answer = build_generator(settings).generate("Question?", ["Relevant text."])

    assert answer == "Grounded answer"
    assert len(calls) == 2
    assert calls[0][1]["headers"]["Authorization"] == "Bearer groq-key"
    assert calls[1][1]["headers"]["x-goog-api-key"] == "gemini-key"


def test_successful_groq_does_not_spend_gemini_quota(monkeypatch):
    import requests

    settings = replace(get_settings(), llm_provider="groq", groq_api_key="groq-key", gemini_api_key="gemini-key")
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        return Response({"choices": [{"message": {"content": "Primary answer"}}]})

    monkeypatch.setattr(requests, "post", post)
    assert build_generator(settings).generate("Question?", ["Relevant text."]) == "Primary answer"
    assert calls == ["https://api.groq.com/openai/v1/chat/completions"]


def test_gemini_can_serve_when_groq_key_is_unavailable(monkeypatch):
    import requests

    settings = replace(get_settings(), llm_provider="groq", groq_api_key="", gemini_api_key="gemini-key")
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs:
                        Response({"candidates": [{"content": {"parts": [{"text": "Fallback answer"}]}}]}))
    generator = build_generator(settings)
    assert isinstance(generator, GeminiGenerator)
    assert generator.generate("Question?", ["Relevant text."]) == "Fallback answer"


def test_fallback_failure_is_reported(monkeypatch):
    class Failed:
        def generate(self, question, passages):
            raise GenerationFailed("provider unavailable")

    with pytest.raises(GenerationFailed):
        FallbackGenerator(Failed(), Failed()).generate("Question?", ["Relevant text."])


# ---- retired-model recovery -------------------------------------------------------------------
class HttpResp:
    def __init__(self, status, body):
        import requests
        self.status_code, self._body, self.text = status, body, str(body)
        self._requests = requests

    def raise_for_status(self):
        if self.status_code >= 400:
            raise self._requests.HTTPError(response=self)

    def json(self):
        return self._body


def test_groq_retired_model_switches_to_live_model(monkeypatch):
    import requests
    from app.services.generation import GroqGenerator
    settings = replace(get_settings(), groq_api_key="k", llm_model="llama-3.1-8b-instant")
    used = []

    def post(url, **kw):
        used.append(kw["json"]["model"])
        if kw["json"]["model"] == "llama-3.1-8b-instant":
            return HttpResp(404, {"error": {"code": "model_not_found"}})
        return HttpResp(200, {"choices": [{"message": {"content": "OK answer"}}]})

    def get(url, **kw):
        return HttpResp(200, {"data": [{"id": "whisper-large-v3"}, {"id": "meta-llama/llama-guard-4-12b"}, {"id": "llama-3.3-70b-versatile"}]})

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", get)
    g = GroqGenerator(settings)
    assert g.generate("Q?", ["text"]) == "OK answer"
    assert used == ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"] and g.model == "llama-3.3-70b-versatile"


def test_gemini_retired_model_switches_to_newest_flash(monkeypatch):
    import requests
    from app.services.generation import GeminiGenerator, pick_gemini_model
    models = [{"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
              {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
              {"name": "models/gemini-3.8-flash-image", "supportedGenerationMethods": ["generateContent"]},
              {"name": "models/gemini-3.8-flash-lite", "supportedGenerationMethods": ["generateContent"]},
              {"name": "models/text-embedding", "supportedGenerationMethods": ["embedContent"]}]
    assert pick_gemini_model(models) == "gemini-3.8-flash"
    settings = replace(get_settings(), gemini_api_key="k", llm_fallback_model="gemini-2.5-flash")

    def post(url, **kw):
        if "gemini-2.5-flash" in url:
            return HttpResp(404, {"error": {"message": "no longer available"}})
        return HttpResp(200, {"candidates": [{"content": {"parts": [{"text": "G answer"}]}}]})

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", lambda url, **kw: HttpResp(200, {"models": models}))
    assert GeminiGenerator(settings).generate("Q?", ["t"]) == "G answer"


def test_gpt_oss_gets_low_reasoning_and_big_budget():
    from app.services.generation import GroqGenerator
    g = GroqGenerator(replace(get_settings(), groq_api_key="k", llm_model="openai/gpt-oss-120b"))
    body = g._payload("Q?", ["t"])
    assert body["reasoning_effort"] == "low" and body["max_tokens"] >= 2048
    g2 = GroqGenerator(replace(get_settings(), groq_api_key="k", llm_model="some-llama"))
    assert "reasoning_effort" not in g2._payload("Q?", ["t"])
