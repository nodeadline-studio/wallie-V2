"""Tests for the non-interactive OpenRouter TTS smoke-test CLI (mocked HTTP)."""
import importlib.util
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

_SCRIPT = pathlib.Path(__file__).parent.parent / "scripts" / "smoke_openrouter_tts.py"
_spec = importlib.util.spec_from_file_location("smoke_openrouter_tts", _SCRIPT)
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)


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


def test_output_format_is_inferred_from_suffix():
    assert smoke._output_format(pathlib.Path("a.mp3"), None) == "mp3"
    assert smoke._output_format(pathlib.Path("a.pcm"), None) == "pcm"
    assert smoke._output_format(pathlib.Path("a.wav"), None) == "pcm"
    assert smoke._output_format(pathlib.Path("a.wav"), "mp3") == "mp3"


def test_main_makes_one_request_and_writes_artifact(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    client = _install(monkeypatch, _FakeResponse(chunks=(b"ID3", b"audio")))
    out = tmp_path / "out" / "smoke.mp3"

    code = smoke.main(
        [
            "--model", "openai/gpt-4o-mini-tts-2025-12-15",
            "--voice", "alloy",
            "--language", "ru",
            "--text", "Привет, это проверка.",
            "--out", str(out),
        ]
    )

    assert code == 0
    assert out.read_bytes() == b"ID3audio"
    assert len(client.calls) == 1
    body = client.calls[0]["json"]
    assert body["model"] == "openai/gpt-4o-mini-tts-2025-12-15"
    assert body["voice"] == "alloy"
    assert body["response_format"] == "mp3"
    assert body["input"] == "Привет, это проверка."
    captured = capsys.readouterr()
    assert "sk-or-secret" not in captured.out
    assert "sk-or-secret" not in captured.err


def test_missing_key_fails_clearly(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = _install(monkeypatch, _FakeResponse(chunks=(b"x",)))
    out = tmp_path / "smoke.pcm"

    code = smoke.main(["--model", "m", "--text", "hi", "--out", str(out)])

    assert code == 2
    assert not out.exists()
    assert client.calls == []
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_empty_text_fails_clearly(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    client = _install(monkeypatch, _FakeResponse(chunks=(b"x",)))
    out = tmp_path / "smoke.pcm"

    code = smoke.main(["--model", "m", "--text", "   ", "--out", str(out)])

    assert code == 2
    assert client.calls == []
    assert "must not be empty" in capsys.readouterr().err


def test_http_failure_returns_one(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    _install(monkeypatch, _FakeResponse(status_code=500, chunks=(b"boom",)))
    out = tmp_path / "smoke.mp3"

    code = smoke.main(["--model", "m", "--text", "hi", "--out", str(out)])

    assert code == 1
    assert not out.exists()
    assert "error:" in capsys.readouterr().err
