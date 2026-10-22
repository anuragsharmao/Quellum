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
