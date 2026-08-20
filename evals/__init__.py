"""Evaluation harness for the PK-DB graph -> timecourse benchmark."""

from .backends import BACKENDS, Prediction, build_backend
from .benchmark import Curve, Entry, load_benchmark, parse_prediction
from .metrics import aggregate, compare_curves, match_curves, score_figure
from .timecourse_extractor import ExtractionResult, TimecourseExtractor

__all__ = [
    "BACKENDS",
    "Curve",
    "Entry",
    "ExtractionResult",
    "Prediction",
    "TimecourseExtractor",
    "aggregate",
    "build_backend",
    "compare_curves",
    "load_benchmark",
    "match_curves",
    "parse_prediction",
    "score_figure",
]
