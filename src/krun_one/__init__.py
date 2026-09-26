"""Krun One — KRUN-005 (experimental). ``from krun_one import KrunOne``."""

import os

# Same workaround as KRUN-004 (bakeoff/__init__.py): torch>=2.14 JIT-compiles a C helper for some
# ModernBERT rotary ops that needs Python.h, missing on the dev machine; ATen fallback is used.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

from .model import KrunOne, DecisionModel, build_sequence, option_texts  # noqa: E402,F401

__all__ = ["KrunOne", "DecisionModel", "build_sequence", "option_texts"]
