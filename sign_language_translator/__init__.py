"""
sign_language_translator
========================

Code: https://github.com/sign-language-translator/sign-language-translator
Help: https://slt.readthedocs.io
Demo: https://huggingface.co/sltAI

This project is an effort to bridge the communication gap between the hearing and the hearing-impaired community using Artificial Intelligence.
The goal is to provide a user friendly API to novel Sign Language Translation solutions that can easily adapt to any regional sign language.
"""

from importlib import import_module

from sign_language_translator import config
from sign_language_translator.config.assets import Assets
from sign_language_translator.config.enums import ModelCodeGroups, ModelCodes
from sign_language_translator.config.enums import SignFormats as SignFormatCodes
from sign_language_translator.config.enums import SignLanguages as SignLanguageCodes
from sign_language_translator.config.enums import TextLanguages as TextLanguageCodes
from sign_language_translator.config.settings import Settings

__version__ = config.utils.get_package_version()

__all__ = [
    # config
    "Assets",
    "Settings",
    "__version__",
    # modules
    "vision",
    "text",
    "models",
    "languages",
    "utils",
    "config",
    "enums",
    # classes (enum)
    "ModelCodes",
    "TextLanguageCodes",
    "SignLanguageCodes",
    "SignFormatCodes",
    "ModelCodeGroups",
    # classes (wrappers)
    "Landmarks",
    "Video",
    # object loaders / factory functions
    "get_sign_language",
    "get_text_language",
    "get_model",
    "get_sign_wrapper_class",
]


def __getattr__(name):
    if name in {"vision", "text", "models", "languages", "utils", "config", "enums"}:
        return import_module(f"sign_language_translator.{name}")

    if name in {"Landmarks", "Video"}:
        module_name = {
            "Landmarks": "sign_language_translator.vision.landmarks.landmarks",
            "Video": "sign_language_translator.vision.video.video",
        }[name]
        module = import_module(module_name)
        return getattr(module, name)

    if name in {"get_sign_language", "get_text_language"}:
        module = import_module("sign_language_translator.languages")
        return getattr(module, name)

    if name == "get_model":
        module = import_module("sign_language_translator.models")
        return getattr(module, name)

    if name == "get_sign_wrapper_class":
        module = import_module("sign_language_translator.vision._utils")
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
