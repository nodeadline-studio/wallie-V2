"""Non-interactive smoke test for the OpenRouter TTS provider.

Makes exactly ONE short speech request, writes the returned audio to ``--out``,
and prints the result. The API key is read from ``OPENROUTER_API_KEY`` and is
never printed. Nothing runs on import.

Russian smoke test (paid — run only when intended):

    OPENROUTER_API_KEY=sk-or-... python scripts/smoke_openrouter_tts.py \\
        --model openai/gpt-4o-mini-tts-2025-12-15 \\
        --voice alloy \\
        --language ru \\
        --text "Привет, это короткая проверка озвучки." \\
        --out /tmp/wallie-openrouter-smoke-ru.mp3

``--model``/``--voice`` must come from OpenRouter's currently listed speech
models (GET https://openrouter.ai/api/v1/models?output_modalities=speech).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tts.base import TTSError
from tts.openrouter import OpenRouterTTS

_FORMAT_BY_SUFFIX = {".mp3": "mp3", ".pcm": "pcm"}


def _output_format(out_path: Path, override: str | None) -> str:
    if override:
        return override
    return _FORMAT_BY_SUFFIX.get(out_path.suffix.lower(), "pcm")


async def _synthesize_once(
    *,
    api_key: str,
    model: str,
    voice: str,
    text: str,
    output_format: str,
    out_path: Path,
) -> int:
    """Make one request and write the artifact. Returns the byte count."""
    provider = OpenRouterTTS(
        api_key=api_key,
        model=model,
        voice_id=voice,
        output_format=output_format,
    )
    try:
        chunks = [chunk async for chunk in provider.synthesize(text)]
        audio = b"".join(chunks)
    finally:
        await provider.aclose()
    if not audio:
        raise TTSError("openrouter: no audio returned")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio)
    return len(audio)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="smoke_openrouter_tts",
        description="Make one OpenRouter TTS request and write the audio.",
    )
    parser.add_argument("--model", required=True, help="OpenRouter speech model id")
    parser.add_argument("--voice", default="", help="voice id supported by that model")
    parser.add_argument(
        "--language", default="", help="ISO-639-1 hint, metadata only (e.g. ru)"
    )
    parser.add_argument("--text", required=True, help="text to synthesize")
    parser.add_argument("--out", required=True, type=Path, help="output audio path")
    parser.add_argument(
        "--format",
        default=None,
        choices=["pcm", "mp3"],
        help="output format (default: inferred from --out, else pcm)",
    )
    args = parser.parse_args(argv)

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("error: OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2
    if not args.text.strip():
        print("error: --text must not be empty", file=sys.stderr)
        return 2

    output_format = _output_format(args.out, args.format)
    try:
        size = asyncio.run(
            _synthesize_once(
                api_key=api_key,
                model=args.model,
                voice=args.voice,
                text=args.text,
                output_format=output_format,
                out_path=args.out,
            )
        )
    except TTSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(
        f"wrote {args.out} ({size} bytes, {output_format}, "
        f"language={args.language or 'auto'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
