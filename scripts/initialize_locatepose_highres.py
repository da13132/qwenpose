#!/usr/bin/env python3
"""Deprecated migration entry point for the removed dual-RGB architecture."""

from __future__ import annotations

raise SystemExit(
    "initialize_locatepose_highres.py is obsolete: LocatePose now uses one unified "
    "800x800 P2/P3/P4 pose pyramid. Train a new Stage1 checkpoint instead of "
    "migrating a legacy dual-RGB checkpoint."
)
