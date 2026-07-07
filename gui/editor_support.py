# -*- coding: utf-8 -*-
"""Support helpers for the GUI file explorer and detached editor."""
from __future__ import annotations

import os


def resolve_explorer_root(configured_cwd: str, fallback_cwd: str | None = None) -> str:
    """Return a valid directory for the GUI file explorer root."""
    candidate = os.path.abspath((configured_cwd or "").strip()) if configured_cwd else ""
    if candidate and os.path.isdir(candidate):
        return candidate

    fallback = fallback_cwd or os.getcwd()
    return os.path.abspath(fallback)


def read_utf8_text(path: str) -> str:
    """Read a UTF-8 text file without normalizing line endings."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return handle.read()


def write_utf8_text(path: str, content: str) -> None:
    """Write a UTF-8 text file without forcing platform line-ending conversion."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)


def detect_line_ending(content: str) -> str:
    """Infer the line-ending style to preserve on save."""
    if "\r\n" in content:
        return "\r\n"
    if "\r" in content:
        return "\r"
    return "\n"


def normalize_for_editor(content: str) -> str:
    """Convert on-disk newlines to the editor's internal \\n representation."""
    return content.replace("\r\n", "\n").replace("\r", "\n")


def serialize_for_disk(content: str, line_ending: str) -> str:
    """Apply the chosen line-ending style before saving."""
    if line_ending == "\n":
        return content
    return content.replace("\n", line_ending)
