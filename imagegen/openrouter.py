"""Hosted image generation through OpenRouter's dedicated Image API.

Discovery first, then generation: the model capabilities come from
``GET /api/v1/images/models``, and an unknown model id is never invented or
substituted — the caller must choose from what the catalogue actually offers.
``POST /api/v1/images`` returns base64 image data plus a usage object with the
exact cost, which is recorded so a pack's real spend is known.

The API key is read from the environment by the caller and is never logged.

Tests inject a fake client/transport; this module itself makes no calls until
:meth:`OpenRouterImageClient.generate` is invoked.
"""
from __future__ import annotations

import base64
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import httpx

IMAGES_ENDPOINT = "https://openrouter.ai/api/v1/images"
IMAGE_MODELS_ENDPOINT = "https://openrouter.ai/api/v1/images/models"

# The requested family, in preference order. These are *looked up*, never
# assumed: an id missing from the catalogue is skipped and reported.
PREFERRED_HOST_MODELS: tuple[str, ...] = (
    "openai/gpt-image-2",
    "openai/gpt-5.4-image-2",
    "openai/gpt-image-2.5-sunburst",
    "google/gemini-3-pro-image",
    "google/gemini-3.1-flash-image",
)


class ImageModelError(RuntimeError):
    """No usable image model could be resolved from the live catalogue."""


class ImageGenerationError(RuntimeError):
    """The image provider failed or returned nothing usable."""


@dataclass(frozen=True)
class ImageModel:
    """One image model as the catalogue describes it."""

    id: str
    name: str = ""
    input_modalities: tuple[str, ...] = ()
    supported_parameters: Mapping[str, Any] = field(default_factory=dict)
    supports_streaming: bool = False

    @property
    def vendor(self) -> str:
        return self.id.split("/", 1)[0] if "/" in self.id else ""

    def supports_parameter(self, name: str) -> bool:
        return name in (self.supported_parameters or {})

    def parameter_values(self, name: str) -> tuple[str, ...]:
        spec = (self.supported_parameters or {}).get(name)
        if isinstance(spec, Mapping):
            values = spec.get("values")
            if isinstance(values, (list, tuple)):
                return tuple(str(value) for value in values)
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "vendor": self.vendor,
            "input_modalities": list(self.input_modalities),
            "supported_parameters": dict(self.supported_parameters),
            "supports_streaming": self.supports_streaming,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImageModel":
        architecture = data.get("architecture") or {}
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            input_modalities=tuple(
                str(item) for item in (architecture.get("input_modalities") or ())
            ),
            supported_parameters=dict(data.get("supported_parameters") or {}),
            supports_streaming=bool(data.get("supports_streaming", False)),
        )


def fetch_image_models(
    api_key: str = "",
    *,
    client: httpx.Client | None = None,
    timeout: float = 30.0,
) -> list[ImageModel]:
    """Return the live image-model catalogue (a free GET)."""
    owned = client is None
    active = client or httpx.Client(timeout=httpx.Timeout(timeout, connect=5.0))
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = active.get(IMAGE_MODELS_ENDPOINT, headers=headers)
        if response.status_code >= 400:
            raise ImageModelError(
                f"openrouter images/models {response.status_code}: "
                f"{response.text[:200]!r}"
            )
        payload = response.json()
    except httpx.HTTPError as exc:
        raise ImageModelError(f"openrouter images/models network error: {exc}") from exc
    finally:
        if owned:
            active.close()
    return [
        ImageModel.from_dict(item) for item in (payload.get("data") or [])
    ]


def find_model(models: Iterable[ImageModel], model_id: str) -> ImageModel | None:
    wanted = (model_id or "").strip()
    return next((model for model in models if model.id == wanted), None)


def resolve_host_model(
    models: Iterable[ImageModel],
    *,
    preferred: Sequence[str] = PREFERRED_HOST_MODELS,
) -> ImageModel:
    """Pick the first *available* preferred host model or raise a clear blocker."""
    catalogue = list(models)
    for candidate in preferred:
        found = find_model(catalogue, candidate)
        if found is not None:
            return found
    available = ", ".join(model.id for model in catalogue) or "none"
    raise ImageModelError(
        "none of the preferred host image models are offered; choose one of: "
        f"{available}"
    )


@dataclass
class GeneratedImage:
    """One generated image plus its measured cost."""

    data: bytes
    media_type: str = "image/png"
    model: str = ""
    cost_usd: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.data)
        return target


def _data_url(reference: str | Path | bytes, mime: str = "image/png") -> str:
    if isinstance(reference, bytes):
        raw = reference
    else:
        raw = Path(reference).read_bytes()
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


class OpenRouterImageClient:
    """Thin synchronous client for ``POST /api/v1/images``."""

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 420.0,
    ) -> None:
        if not api_key:
            raise ImageGenerationError("openrouter images: missing OPENROUTER_API_KEY")
        self._api_key = api_key
        self._owned = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout, connect=10.0)
        )

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        aspect_ratio: str = "",
        output_format: str = "png",
        quality: str = "",
        background: str = "",
        input_references: Sequence[str | Path | bytes] = (),
        seed: int | None = None,
        n: int = 1,
    ) -> list[GeneratedImage]:
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": max(1, int(n)),
            "output_format": output_format,
        }
        if aspect_ratio:
            body["aspect_ratio"] = aspect_ratio
        if quality:
            body["quality"] = quality
        if background:
            body["background"] = background
        if seed is not None:
            body["seed"] = int(seed)
        if input_references:
            body["input_references"] = [
                {"type": "image_url", "image_url": {"url": _data_url(reference)}}
                for reference in input_references
            ]
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = self._client.post(IMAGES_ENDPOINT, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise ImageGenerationError(f"openrouter images network error: {exc}") from exc
        if response.status_code >= 400:
            raise ImageGenerationError(
                f"openrouter images {response.status_code}: {response.text[:300]!r}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ImageGenerationError("openrouter images: response was not JSON") from exc
        data = payload.get("data") or []
        if not data:
            raise ImageGenerationError("openrouter images: response contained no image")
        usage = payload.get("usage") or {}
        cost = usage.get("cost")
        images: list[GeneratedImage] = []
        for item in data:
            encoded = item.get("b64_json")
            if not encoded:
                continue
            images.append(
                GeneratedImage(
                    data=base64.b64decode(encoded),
                    media_type=str(item.get("media_type", "image/png")),
                    model=model,
                    cost_usd=float(cost) if cost is not None else None,
                    usage=dict(usage),
                )
            )
        if not images:
            raise ImageGenerationError("openrouter images: no decodable image data")
        return images

    def close(self) -> None:
        if self._owned:
            self._client.close()

    def __enter__(self) -> "OpenRouterImageClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def remove_chroma_background(
    source: str | Path,
    target: str | Path,
    *,
    color: str = "0x00B140",
    method: str = "spill",
    similarity: float = 0.30,
    blend: float = 0.10,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Turn a flat chroma background into real alpha.

    ``gpt-image-2`` cannot emit ``background=transparent`` (only ``auto`` or
    ``opaque``), so the master is generated on a flat chroma colour and removed
    locally. The default ``method="spill"`` uses the chroma-spill key in
    :mod:`production.host_pack`: it keys the flat screen without making dark
    joints and contour lines translucent (the FFmpeg ``colorkey`` path with
    ``similarity=.30/blend=.10`` was measured doing exactly that). The legacy
    FFmpeg path stays available as ``method="ffmpeg_colorkey"``. The result is
    a true RGBA PNG that :func:`verify_alpha_png` can measure; it is never
    assumed transparent.
    """
    if method == "spill":
        from production.host_pack import key_chroma_to_alpha

        return key_chroma_to_alpha(source, target)
    if method != "ffmpeg_colorkey":
        raise ImageGenerationError(f"unknown background removal method: {method!r}")
    executable = shutil.which(ffmpeg) or ffmpeg
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-vf",
        f"colorkey={color}:{similarity}:{blend},format=rgba",
        "-frames:v",
        "1",
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if completed.returncode != 0:
        tail = (completed.stderr or "").strip().splitlines()[-4:]
        raise ImageGenerationError(
            "background removal failed: " + " / ".join(tail)
        )
    if not target.is_file():
        raise ImageGenerationError("background removal produced no file")
    return target


@dataclass(frozen=True)
class HostPackPlan:
    """What a host pack generation would do, before any money is spent."""

    model: str
    canvas: tuple[int, int]
    aspect_ratio: str
    quality: str
    validation_states: tuple[str, ...]
    remaining_states: tuple[str, ...]
    estimated_cost_per_image_usd: float = 0.0
    estimated_total_usd: float = 0.0
    budget_usd: float = 0.0
    within_budget: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "canvas": list(self.canvas),
            "aspect_ratio": self.aspect_ratio,
            "quality": self.quality,
            "validation_states": list(self.validation_states),
            "remaining_states": list(self.remaining_states),
            "estimated_cost_per_image_usd": round(
                self.estimated_cost_per_image_usd, 6
            ),
            "estimated_total_usd": round(self.estimated_total_usd, 6),
            "budget_usd": self.budget_usd,
            "within_budget": self.within_budget,
            "note": (
                "estimate only; the response usage object records the exact cost"
            ),
        }


def build_host_pack_plan(
    model: ImageModel | str,
    *,
    states: Sequence[str],
    validation_count: int = 3,
    canvas: tuple[int, int] = (1024, 1536),
    aspect_ratio: str = "2:3",
    quality: str = "medium",
    cost_per_image_usd: float = 0.0,
    budget_usd: float = 0.0,
) -> HostPackPlan:
    """Plan the validation sample and the full pack with an honest estimate."""
    state_list = tuple(str(state) for state in states)
    validation = state_list[: max(1, min(validation_count, len(state_list)))]
    remaining = state_list[len(validation) :]
    model_id = model.id if isinstance(model, ImageModel) else str(model)
    estimated = cost_per_image_usd * len(state_list)
    return HostPackPlan(
        model=model_id,
        canvas=canvas,
        aspect_ratio=aspect_ratio,
        quality=quality,
        validation_states=validation,
        remaining_states=remaining,
        estimated_cost_per_image_usd=cost_per_image_usd,
        estimated_total_usd=estimated,
        budget_usd=budget_usd,
        within_budget=(budget_usd <= 0) or estimated <= budget_usd,
    )


__all__ = [
    "GeneratedImage",
    "HostPackPlan",
    "ImageGenerationError",
    "ImageModel",
    "ImageModelError",
    "OpenRouterImageClient",
    "PREFERRED_HOST_MODELS",
    "build_host_pack_plan",
    "fetch_image_models",
    "find_model",
    "remove_chroma_background",
    "resolve_host_model",
]
