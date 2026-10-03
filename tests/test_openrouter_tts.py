"""Unit tests for the OpenRouter TTS adapter and its factory wiring."""
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest

from config import Secrets, TTSConfig
from tts import build_tts
from tts.base import TTSError
from tts.openrouter import _ENDPOINT, OpenRouterTTS


class _FakeResponse:
    def __init__(self, status_code=200, chunks=(b"abc",), headers=None):
        self.status_code = status_code
        self._chunks = chunks
        self.headers = headers or {}

    async def aread(self):
        return b"error body"

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _FakeStream:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def stream(self, method, url, json=None, headers=None):
        self.calls.append({"method": method, "url": url, "json": json, "headers": headers})
        return _FakeStream(self.response)

    async def aclose(self):
        pass


def _install(monkeypatch, response):
    import tts.openrouter as module

    client = _FakeClient(response)
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: client)
    return client


async def _collect(provider, text, direction=""):
    return [chunk async for chunk in provider.synthesize(text, direction=direction)]


def _provider(**kwargs):
    kwargs.setdefault("api_key", "sk-or-secret")
    kwargs.setdefault("model", "openai/gpt-4o-mini-tts-2025-12-15")
    return OpenRouterTTS(**kwargs)


# ─────────────────────────────────────────────────────────────────────
# Construction / configuration errors
# ─────────────────────────────────────────────────────────────────────
def test_factory_builds_openrouter_provider():
    cfg = TTSConfig(
        provider="openrouter",
        model="openai/gpt-4o-mini-tts-2025-12-15",
        voice_id="alloy",
    )
    provider = build_tts(cfg, Secrets(openrouter_api_key="sk-or-test"))
    try:
        assert provider.name == "openrouter"
    finally:
        asyncio.run(provider.aclose())


def test_missing_api_key_is_a_useful_error():
    with pytest.raises(TTSError, match="OPENROUTER_API_KEY"):
        OpenRouterTTS(api_key="", model="openai/gpt-4o-mini-tts-2025-12-15")


def test_missing_model_is_a_useful_error():
    with pytest.raises(TTSError, match="missing model"):
        OpenRouterTTS(api_key="sk-or-secret", model="")


def test_explicit_timeout_is_used_for_the_client():
    provider = _provider(timeout=180.0)
    try:
        assert provider._client.timeout.read == 180.0
    finally:
        asyncio.run(provider.aclose())


def test_unsupported_output_format_is_rejected():
    with pytest.raises(TTSError, match="unsupported output_format"):
        OpenRouterTTS(api_key="sk-or-secret", model="some/model", output_format="wav")


# ─────────────────────────────────────────────────────────────────────
# Request shape and streaming
# ─────────────────────────────────────────────────────────────────────
def test_synthesize_posts_expected_body_and_yields_audio(monkeypatch):
    client = _install(monkeypatch, _FakeResponse(chunks=(b"aa", b"bb")))
    provider = _provider(voice_id="alloy", output_format="mp3", speed=1.25)
    try:
        chunks = asyncio.run(_collect(provider, "hello there"))
    finally:
        asyncio.run(provider.aclose())

    assert chunks == [b"aa", b"bb"]
    call = client.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == _ENDPOINT
    assert call["json"] == {
        "model": "openai/gpt-4o-mini-tts-2025-12-15",
        "input": "hello there",
        "response_format": "mp3",
        "voice": "alloy",
        "speed": 1.25,
    }
    assert call["headers"]["Authorization"] == "Bearer sk-or-secret"
    assert call["headers"]["Content-Type"] == "application/json"


def test_pcm_is_the_default_and_speed_is_omitted_when_zero(monkeypatch):
    client = _install(monkeypatch, _FakeResponse(chunks=(b"pcm",)))
    provider = _provider()
    try:
        asyncio.run(_collect(provider, "hi"))
    finally:
        asyncio.run(provider.aclose())

    body = client.calls[0]["json"]
    assert body["response_format"] == "pcm"
    assert "speed" not in body
    assert "voice" not in body


def test_blank_text_makes_no_request(monkeypatch):
    client = _install(monkeypatch, _FakeResponse(chunks=(b"x",)))
    provider = _provider()
    try:
        chunks = asyncio.run(_collect(provider, "   "))
    finally:
        asyncio.run(provider.aclose())

    assert chunks == []
    assert client.calls == []


# ─────────────────────────────────────────────────────────────────────
# Performance direction (documented mechanism only)
# ─────────────────────────────────────────────────────────────────────
def test_direction_is_carried_as_a_documented_audio_tag(monkeypatch):
    client = _install(monkeypatch, _FakeResponse(chunks=(b"aa",)))
    provider = _provider(
        model="google/gemini-3.1-flash-tts-preview", voice_id="Charon"
    )
    try:
        assert provider.supports_direction is True
        asyncio.run(
            _collect(provider, "שלום", direction="deadpan, matter-of-fact")
        )
    finally:
        asyncio.run(provider.aclose())

    body = client.calls[0]["json"]
    assert body["input"] == "[deadpan, matter-of-fact] שלום"
    assert body["voice"] == "Charon"
    assert provider.last_direction == "deadpan, matter-of-fact"
    assert provider.last_request_input == body["input"]


def test_bracketed_direction_is_normalized_to_one_tag(monkeypatch):
    client = _install(monkeypatch, _FakeResponse(chunks=(b"aa",)))
    provider = _provider(model="google/gemini-3.1-flash-tts-preview")
    try:
        asyncio.run(_collect(provider, "hello", direction="  [deadpan]  "))
    finally:
        asyncio.run(provider.aclose())

    assert client.calls[0]["json"]["input"] == "[deadpan] hello"


def test_direction_is_rejected_without_a_documented_mechanism(monkeypatch):
    client = _install(monkeypatch, _FakeResponse(chunks=(b"aa",)))
    provider = _provider()  # openai model: instructions are vendor-specific
    try:
        assert provider.supports_direction is False
        with pytest.raises(TTSError, match="direction is not supported"):
            asyncio.run(_collect(provider, "hello", direction="warm tone"))
    finally:
        asyncio.run(provider.aclose())

    assert client.calls == []


def test_direction_with_bracket_characters_is_rejected(monkeypatch):
    client = _install(monkeypatch, _FakeResponse(chunks=(b"aa",)))
    provider = _provider(model="google/gemini-3.1-flash-tts-preview")
    try:
        with pytest.raises(TTSError, match="single English phrase"):
            asyncio.run(_collect(provider, "hello", direction="[deadpan] warm"))
    finally:
        asyncio.run(provider.aclose())

    assert client.calls == []


def test_no_direction_keeps_the_original_payload(monkeypatch):
    client = _install(monkeypatch, _FakeResponse(chunks=(b"aa",)))
    provider = _provider(model="google/gemini-3.1-flash-tts-preview")
    try:
        asyncio.run(_collect(provider, "hello"))
    finally:
        asyncio.run(provider.aclose())

    assert client.calls[0]["json"]["input"] == "hello"
    assert provider.last_direction == ""
    assert provider.last_request_input == "hello"


def test_generation_id_is_captured_from_the_response(monkeypatch):
    response = _FakeResponse(
        chunks=(b"aa",), headers={"x-generation-id": "gen-tts-42"}
    )
    _install(monkeypatch, response)
    provider = _provider()
    try:
        assert provider.last_generation_id == ""
        asyncio.run(_collect(provider, "hello"))
        assert provider.last_generation_id == "gen-tts-42"
    finally:
        asyncio.run(provider.aclose())


def test_missing_generation_header_stays_empty(monkeypatch):
    _install(monkeypatch, _FakeResponse(chunks=(b"aa",)))
    provider = _provider()
    try:
        asyncio.run(_collect(provider, "hello"))
        assert provider.last_generation_id == ""
    finally:
        asyncio.run(provider.aclose())


def test_model_payload_carries_catalogue_pricing():
    import tts.openrouter as module

    model = module._model_from_payload(
        {
            "id": "google/gemini-3.1-flash-tts-preview",
            "name": "Gemini TTS",
            "supported_voices": ["Charon", "Puck"],
            "pricing": {"prompt": "0.000001", "completion": "0.00002"},
        }
    )
    assert model.prompt_price == 0.000001
    assert model.completion_price == 0.00002
    assert model.to_dict()["prompt_price"] == 0.000001
    restored = module.SpeechModel.from_dict(model.to_dict())
    assert restored == model


def test_missing_pricing_is_none_not_an_error():
    import tts.openrouter as module

    model = module._model_from_payload({"id": "m", "supported_voices": []})
    assert model.prompt_price is None
    assert model.completion_price is None


# ─────────────────────────────────────────────────────────────────────
# Failure modes
# ─────────────────────────────────────────────────────────────────────
def test_http_error_raises_tts_error(monkeypatch):
    _install(monkeypatch, _FakeResponse(status_code=400, chunks=(b"bad",)))
    provider = _provider()
    try:
        with pytest.raises(TTSError, match="openrouter 400"):
            asyncio.run(_collect(provider, "hello"))
    finally:
        asyncio.run(provider.aclose())


def test_empty_audio_response_raises(monkeypatch):
    _install(monkeypatch, _FakeResponse(chunks=()))
    provider = _provider()
    try:
        with pytest.raises(TTSError, match="empty audio response"):
            asyncio.run(_collect(provider, "hello"))
    finally:
        asyncio.run(provider.aclose())
