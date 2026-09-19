"""Google Gemini provider via google-generativeai."""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import google.generativeai as genai

from .base import LLMError, LLMProvider


class GeminiProvider(LLMProvider):
    def __init__(self, *, model: str, api_key: str, supports_vision: bool = True) -> None:
        if not api_key:
            raise LLMError("gemini: missing API key")
        self.name = "gemini"
        self.model = model
        self.supports_vision = supports_vision
        genai.configure(api_key=api_key)
        self._model_name = model

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.85,
        top_p: float = 0.95,
        max_tokens: int = 500,
        presence_penalty: float = 0.0,  # accepted for signature parity
        frequency_penalty: float = 0.0,
    ) -> AsyncIterator[str]:
        system_prompt, rest = self._split_system(messages)
        # Gemini needs at least ~40 tokens to avoid safety filter triggers.
        effective_tokens = max(40, max_tokens)
        model = genai.GenerativeModel(
            self._model_name,
            system_instruction=system_prompt or None,
            generation_config={
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": effective_tokens,
            },
        )
        contents = [self._encode_message(m) for m in rest]
        try:
            response = await model.generate_content_async(contents, stream=True)
        except Exception as e:
            raise LLMError(f"gemini request failed: {e}") from e

        async for chunk in response:
            try:
                text = getattr(chunk, "text", None)
            except ValueError:
                # A streamed chunk can carry only a finish reason or an internal
                # "thought" part, so its ``.text`` accessor raises. That is not
                # the end of the answer: skip the chunk and keep reading, or the
                # visible text would be cut off mid-sentence.
                continue
            if text:
                yield text

    async def aclose(self) -> None:
        await asyncio.sleep(0)

    # ----- helpers -----
    @staticmethod
    def _split_system(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        system = ""
        rest: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"] if isinstance(m["content"], str) else ""
            else:
                rest.append(m)
        return system, rest

    def _encode_message(self, m: dict[str, Any]) -> dict[str, Any]:
        role = "user" if m["role"] == "user" else "model"
        content = m.get("content")
        parts: list[Any] = []
        if isinstance(content, list):
            for b in content:
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif b.get("type") == "image":
                    parts.append(
                        {"mime_type": b.get("mime", "image/jpeg"), "data": b["data"]}
                    )
        else:
            parts.append(content or "")
        return {"role": role, "parts": parts}
