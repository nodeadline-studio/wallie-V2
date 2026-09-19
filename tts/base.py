"""TTS provider protocol."""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol


class TTSError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioFormat:
    """The exact shape of the bytes a provider returns.

    ``encoding`` is an FFmpeg codec name (``pcm_s16le``, ``mp3``, ...). PCM
    encodings declare ``sample_rate``/``channels``/``sample_width`` so a
    consumer never has to infer them from a configurable number.
    """

    encoding: str
    sample_rate: int | None = None
    channels: int | None = None
    sample_width: int | None = None

    def is_pcm(self) -> bool:
        return self.encoding.startswith("pcm_")


class TTSProvider(Protocol):
    name: str
    sample_rate: int  # PCM sample rate of the bytes returned
    channels: int = 1

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        ...

    async def aclose(self) -> None:
        ...
