#!/usr/bin/env python3
"""Deprecated continuation entry point for the removed legacy architecture."""

from __future__ import annotations

raise SystemExit(
    "prepare_confidence_fixed_continuation.py is obsolete: legacy confidence-rescue "
    "checkpoints are incompatible with the unified 800x800 LocatePose model. "
    "Train a new Stage1 checkpoint and continue with Stage2/Stage3."
)
