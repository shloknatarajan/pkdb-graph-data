"""Benchmark adapter and compatibility imports for extraction configuration."""

from __future__ import annotations

from .benchmark import Entry
from .timecourse_extractor import (
    INSTRUCTIONS,
    OUTPUT_SCHEMA,
    SYSTEM,
    build_extraction_text,
    truncate_paper,
)


def build_task_text(
    entry: Entry, *, paper: str | None, substance_hint: bool = False
) -> str:
    """Translate a benchmark entry into a dataset-independent extraction request."""
    return build_extraction_text(
        paper=paper,
        figure_name=entry.raw.get("figure", entry.id),
        paper_title=entry.reference_title,
        substance_vocabulary=entry.substance_names if substance_hint else None,
    )


__all__ = [
    "INSTRUCTIONS",
    "OUTPUT_SCHEMA",
    "SYSTEM",
    "build_extraction_text",
    "build_task_text",
    "truncate_paper",
]
