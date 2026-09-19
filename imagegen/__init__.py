"""Hosted image generation for the reusable PNG host pack.

The OpenRouter Image API client lives here, outside ``production/``, for the
same reason the LLM and TTS clients do: the production pipeline never opens a
network connection itself. ``production`` composes injected providers; scripts
and the agent compose this client.
"""
from .openrouter import (
    IMAGE_MODELS_ENDPOINT,
    IMAGES_ENDPOINT,
    PREFERRED_HOST_MODELS,
    GeneratedImage,
    HostPackPlan,
    ImageGenerationError,
    ImageModel,
    ImageModelError,
    OpenRouterImageClient,
    build_host_pack_plan,
    fetch_image_models,
    find_model,
    remove_chroma_background,
    resolve_host_model,
)

__all__ = [
    "GeneratedImage",
    "HostPackPlan",
    "IMAGE_MODELS_ENDPOINT",
    "IMAGES_ENDPOINT",
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
