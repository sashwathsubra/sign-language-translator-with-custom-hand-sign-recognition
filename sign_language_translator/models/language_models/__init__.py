"""
sign_language_translator.models.language_models
==============================================

This module contains various language models used in the sign language translator system.
"""

from importlib import import_module

from sign_language_translator.models.language_models.abstract_language_model import (
    LanguageModel,
)
from sign_language_translator.models.language_models.beam_sampling import BeamSampling
from sign_language_translator.models.language_models.mixer import MixerLM
from sign_language_translator.models.language_models.ngram_language_model import (
    NgramLanguageModel,
)

__all__ = [
    "NgramLanguageModel",
    "MixerLM",
    "BeamSampling",
    "LanguageModel",
    "TransformerLanguageModel",
    "transformer_language_model",
]


def __getattr__(name):
    if name == "TransformerLanguageModel":
        module = import_module(
            "sign_language_translator.models.language_models.transformer_language_model.model"
        )
        return getattr(module, name)

    if name == "transformer_language_model":
        return import_module("sign_language_translator.models.language_models.transformer_language_model")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
