"""
sign_language_translator.models
===============================

This module contains the various models in the sign language translator system
and their associated components.
"""

from importlib import import_module

from sign_language_translator.models._utils import get_model
from sign_language_translator.models.language_models import (
    BeamSampling,
    LanguageModel,
    MixerLM,
    NgramLanguageModel,
)

__all__ = [
    "get_model",
    "language_models",
    "sign_to_text",
    "text_to_sign",
    "utils",
    "video_embedding",
    "text_embedding",
    "ConcatenativeSynthesis",
    "NgramLanguageModel",
    "TransformerLanguageModel",
    "BeamSampling",
    "MixerLM",
    "LanguageModel",
    "TextToSignModel",
    "MediaPipeLandmarksModel",
    "VideoEmbeddingModel",
    "TextEmbeddingModel",
    "VectorLookupModel",
]


def __getattr__(name):
    if name == "TransformerLanguageModel":
        module = import_module("sign_language_translator.models.language_models")
        return getattr(module, name)

    if name in {"language_models", "sign_to_text", "text_to_sign", "utils", "video_embedding", "text_embedding"}:
        return import_module(f"sign_language_translator.models.{name}")

    if name in {"ConcatenativeSynthesis"}:
        module = import_module("sign_language_translator.models.text_to_sign")
        return getattr(module, name)

    if name in {"TextToSignModel"}:
        module = import_module("sign_language_translator.models.text_to_sign.t2s_model")
        return getattr(module, name)

    if name in {"MediaPipeLandmarksModel", "VideoEmbeddingModel"}:
        module = import_module("sign_language_translator.models.video_embedding")
        return getattr(module, name)

    if name in {"TextEmbeddingModel", "VectorLookupModel"}:
        module = import_module("sign_language_translator.models.text_embedding")
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
