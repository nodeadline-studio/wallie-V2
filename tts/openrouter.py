"""OpenRouter TTS adapter (OpenAI-compatible ``/audio/speech``).

OpenRouter returns a single raw audio byte stream (not JSON) for a POST to
``https://openrouter.ai/api/v1/audio/speech``. Configure the exact speech model
and one of its voices; discover both via
``GET https://openrouter.ai/api/v1/models?output_modalities=speech``. No model is
assumed or silently substituted here.

The endpoint's ``response_format`` supports ``pcm`` (24 kHz mono PCM16, which is
what Wallie's live audio player consumes) and ``mp3``. The API key is sent as a
Bearer token and is never logged.

Because the sample rate of a raw PCM response is fixed by the endpoint's
contract — not by caller configuration — :attr:`OpenRouterTTS.output_audio_format`
reports the real format so a downstream writer never guesses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping

import httpx

from .base import AudioFormat, TTSError, TTSProvider

_ENDPOINT = "https://openrouter.ai/api/v1/audio/speech"
_MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"
_ALLOWED_FORMATS = ("pcm", "mp3")

# Documented OpenRouter speech contract for response_format=pcm.
_PCM_CONTRACT = AudioFormat(
    encoding="pcm_s16le", sample_rate=24000, channels=1, sample_width=2
)

# The requested vendor preference for production narration. ElevenLabs is a
# preference, never an automatic fallback: when it is absent the caller must
# decide explicitly rather than having another vendor substituted silently.
PREFERRED_SPEECH_VENDOR = "elevenlabs"


class ProviderSelectionError(TTSError):
    """A required speech model/voice could not be selected from the catalogue."""


@dataclass(frozen=True)
class SpeechModel:
    """One speech model offered by OpenRouter, with its supported voices.

    ``prompt_price``/``completion_price`` are USD per token from the catalogue
    ``pricing`` block (``None`` when the catalogue omits them). They only feed
    pre-flight cost estimates; the provider never charges from them.
    """

    id: str
    name: str = ""
    supported_voices: tuple[str, ...] = ()
    prompt_price: float | None = None
    completion_price: float | None = None

    @property
    def vendor(self) -> str:
        return self.id.split("/", 1)[0] if "/" in self.id else ""

    def supports_voice(self, voice: str) -> bool:
        if not voice:
            return True
        return not self.supported_voices or voice in self.supported_voices

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "vendor": self.vendor,
            "supported_voices": list(self.supported_voices),
            "prompt_price": self.prompt_price,
            "completion_price": self.completion_price,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SpeechModel":
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            supported_voices=tuple(str(v) for v in (data.get("supported_voices") or ())),
            prompt_price=_opt_price(data.get("prompt_price")),
            completion_price=_opt_price(data.get("completion_price")),
        )


def _opt_price(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _model_from_payload(item: Mapping[str, Any]) -> SpeechModel:
    pricing = item.get("pricing") or {}
    return SpeechModel(
        id=str(item.get("id", "")),
        name=str(item.get("name", "")),
        supported_voices=tuple(
            str(voice) for voice in (item.get("supported_voices") or ())
        ),
        prompt_price=_opt_price(pricing.get("prompt")),
        completion_price=_opt_price(pricing.get("completion")),
    )


async def fetch_speech_models(
    api_key: str = "",
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = 20.0,
) -> list[SpeechModel]:
    """Discover the speech models OpenRouter currently offers.

    The catalogue endpoint is a free GET; ``api_key`` is optional and is only
    sent when provided. Raises :class:`TTSError` on a network or HTTP failure.
    """
    owned = client is None
    active = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5.0))
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = await active.get(
            _MODELS_ENDPOINT,
            params={"output_modalities": "speech"},
            headers=headers,
        )
        if response.status_code >= 400:
            body = (await response.aread())[:200]
            raise TTSError(f"openrouter models {response.status_code}: {body!r}")
        payload = response.json()
    except httpx.HTTPError as exc:
        raise TTSError(f"openrouter models network error: {exc}") from exc
    finally:
        if owned:
            await active.aclose()
    return [_model_from_payload(item) for item in (payload.get("data") or [])]


def find_speech_model(
    models: list[SpeechModel] | tuple[SpeechModel, ...], model_id: str
) -> SpeechModel | None:
    """Return the catalogue entry for ``model_id``, or ``None``."""
    wanted = (model_id or "").strip()
    return next((model for model in models if model.id == wanted), None)


def select_preferred_speech_model(
    models: list[SpeechModel] | tuple[SpeechModel, ...],
    *,
    vendor: str = PREFERRED_SPEECH_VENDOR,
) -> SpeechModel:
    """Select the first speech model from ``vendor`` or raise a clear blocker.

    Never substitutes another vendor: the raised
    :class:`ProviderSelectionError` names the vendors that *are* available so an
    operator can make the choice explicitly.
    """
    matches = [model for model in models if model.vendor == vendor]
    if not matches:
        available = sorted({model.vendor for model in models if model.vendor})
        raise ProviderSelectionError(
            f"no {vendor!r} speech model is currently offered by OpenRouter "
            f"(available vendors: {', '.join(available) or 'none'}). "
            "OpenRouter speech will not silently substitute another vendor; "
            "set an explicit tts.model to authorize a different model."
        )
    return matches[0]


class OpenRouterTTS(TTSProvider):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        voice_id: str = "",
        output_format: str = "pcm",
        sample_rate: int = 24000,
        speed: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise TTSError("openrouter: missing OPENROUTER_API_KEY")
        if not model:
            raise TTSError(
                "openrouter: missing model (set tts.model to an OpenRouter speech "
                "model; discover them at /api/v1/models?output_modalities=speech)"
            )
        fmt = (output_format or "pcm").strip().lower()
        if fmt not in _ALLOWED_FORMATS:
            raise TTSError(
                f"openrouter: unsupported output_format {output_format!r}; "
                f"use one of {list(_ALLOWED_FORMATS)}"
            )
        self.name = "openrouter"
        self.model = model
        self.sample_rate = sample_rate
        self.channels = 1
        self._api_key = api_key
        self._model = model
        self._voice_id = voice_id
        self._output_format = fmt
        self._speed = float(speed) if speed else 0.0
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(float(timeout), connect=5.0)
        )
        # The OpenAI-compatible response carries OpenRouter's generation id in
        # ``x-generation-id``; it is the only handle for the reported per-request
        # cost (``GET /api/v1/generation?id=...``). Empty when the header is
        # absent, so a caller records "not reported" instead of guessing.
        self.last_generation_id = ""

    @property
    def output_audio_format(self) -> AudioFormat:
        """The real format of :meth:`synthesize` output for this configuration.

        PCM is fixed by the endpoint contract (24 kHz mono PCM16); MP3 is
        self-describing. The caller's ``sample_rate`` is an informational hint
        and never overrides the contract.
        """
        if self._output_format == "pcm":
            return _PCM_CONTRACT
        return AudioFormat(encoding=self._output_format)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        if not text.strip():
            return
        payload: dict = {
            "model": self._model,
            "input": text,
            "response_format": self._output_format,
        }
        if self._voice_id:
            payload["voice"] = self._voice_id
        if self._speed:
            payload["speed"] = self._speed
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with self._client.stream(
                "POST", _ENDPOINT, json=payload, headers=headers
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise TTSError(f"openrouter {resp.status_code}: {body[:200]!r}")
                self.last_generation_id = str(
                    resp.headers.get("x-generation-id", "") or ""
                )
                received = False
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        received = True
                        yield chunk
                if not received:
                    raise TTSError("openrouter: empty audio response")
        except httpx.HTTPError as e:
            raise TTSError(f"openrouter network error: {e}") from e

    async def aclose(self) -> None:
        await self._client.aclose()
