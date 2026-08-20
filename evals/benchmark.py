"""Loading and normalizing the PK-DB graph benchmark and model predictions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from .units import time_conversion

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmark"

# Fields compared between a predicted curve and its matched gold curve. Weights
# are used both for metadata accuracy and for the matching cost; substance
# dominates because it is what actually distinguishes curves within a figure.
META_FIELDS: dict[str, float] = {
    "substance": 0.45,
    "intervention": 0.20,
    "tissue": 0.15,
    "measurement_type": 0.10,
    "group": 0.10,
}

# `substance`, `tissue` and `measurement_type` are drawn from a controlled
# vocabulary, so exact match is a fair test. `intervention` and `group` are
# free-text curator codes -- gold spells one arm "paracetamol,propanolol"
# (sic) where a model would reasonably write "propranolol" -- so they are
# compared as fuzzy token sets instead.
FUZZY_FIELDS = frozenset({"intervention", "group"})

# Below this per-token similarity, two tokens are different words, not typos.
TOKEN_MATCH_FLOOR = 0.75


def _norm_text(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    return s or None


def _tokens(value: str | None) -> list[str]:
    if value is None:
        return []
    return [t for t in re.split(r"[,;/+&\s]+", value.strip().lower()) if t]


def _best_token_ratio(token: str, others: list[str]) -> float:
    best = 0.0
    for other in others:
        r = SequenceMatcher(None, token, other).ratio()
        if r > best:
            best = r
    return best if best >= TOKEN_MATCH_FLOOR else 0.0


def field_similarity(field_name: str, pred: str | None, gold: str | None) -> float:
    """Agreement in [0, 1] between a predicted and a gold metadata value."""
    if gold is None:
        return 0.0
    if field_name not in FUZZY_FIELDS:
        return 1.0 if pred == gold else 0.0

    p_toks, g_toks = _tokens(pred), _tokens(gold)
    if not g_toks:
        return 0.0
    if not p_toks:
        return 0.0
    recall = float(np.mean([_best_token_ratio(t, p_toks) for t in g_toks]))
    precision = float(np.mean([_best_token_ratio(t, g_toks) for t in p_toks]))
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass
class Curve:
    """One timecourse: metadata plus the (time, value) series read off a figure."""

    substance: str | None = None
    intervention: str | None = None
    tissue: str | None = None
    group: str | None = None
    measurement_type: str | None = None
    time_unit: str | None = None
    value_unit: str | None = None
    times: np.ndarray = field(default_factory=lambda: np.empty(0))
    values: np.ndarray = field(default_factory=lambda: np.empty(0))
    label: str | None = None

    @property
    def n_points(self) -> int:
        return int(self.times.size)

    def metadata(self) -> dict[str, str | None]:
        return {f: _norm_text(getattr(self, f)) for f in META_FIELDS}


def _points_to_arrays(points: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract (time, y) pairs, preferring `mean` and falling back to `value`.

    PK-DB stores aggregate curves as `mean` (with sd/se) and individual-subject
    curves as `value`; a point missing both carries no y and is dropped.
    """
    ts: list[float] = []
    ys: list[float] = []
    for p in points or []:
        t = p.get("time")
        y = p.get("mean")
        if y is None:
            y = p.get("value")
        if t is None or y is None:
            continue
        try:
            ts.append(float(t))
            ys.append(float(y))
        except (TypeError, ValueError):
            continue
    if not ts:
        return np.empty(0), np.empty(0)
    order = np.argsort(np.asarray(ts, dtype=float))
    return np.asarray(ts, dtype=float)[order], np.asarray(ys, dtype=float)[order]


def curve_from_dict(d: dict) -> Curve:
    times, values = _points_to_arrays(d.get("points", []))
    return Curve(
        substance=d.get("substance"),
        intervention=d.get("intervention"),
        tissue=d.get("tissue"),
        group=d.get("group"),
        measurement_type=d.get("measurement_type"),
        time_unit=d.get("time_unit"),
        value_unit=d.get("value_unit"),
        times=times,
        values=values,
        label=d.get("label"),
    )


def to_gold_time_base(curve: Curve, gold_time_unit: str | None) -> Curve:
    """Rescale a predicted curve's times into the gold curve's time unit.

    An unrecognized or absent unit on either side leaves times untouched: time
    units in this benchmark are `hr` almost everywhere, so assuming agreement is
    far less damaging than silently zeroing the series.
    """
    factor = time_conversion(curve.time_unit, gold_time_unit)
    if factor is None or factor == 1.0:
        return curve
    scaled = Curve(**{**curve.__dict__})
    scaled.times = curve.times * factor
    return scaled


@dataclass
class Entry:
    """One benchmark row: a figure, its source paper, and the gold timecourses."""

    id: str
    image: Path
    study_sid: str
    paper_text: str
    reference_title: str
    substances: list[dict]
    gold: list[Curve]
    raw: dict

    @property
    def substance_names(self) -> list[str]:
        return [s["name"] for s in self.substances if s.get("name")]


SKIPPED_EMPTY_GOLD: list[str] = []


def load_benchmark(
    path: Path | str = BENCHMARK_DIR / "benchmark.jsonl",
    *,
    ids: list[str] | None = None,
    limit: int | None = None,
    skip_empty_gold: bool = True,
) -> list[Entry]:
    """Read benchmark.jsonl into Entry objects, dropping curves with no y values.

    A few PK-DB timecourses carry times but no `mean`/`value` (the curator recorded
    the sampling schedule, not the digitized curve). Those curves cannot be scored
    against, and a figure whose curves are all like that -- PKDB00063__Fig1 -- has
    no answer at all, so it is excluded and recorded in SKIPPED_EMPTY_GOLD.
    """
    path = Path(path)
    root = path.parent
    entries: list[Entry] = []
    SKIPPED_EMPTY_GOLD.clear()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if ids is not None and row["id"] not in ids:
                continue
            gold = [curve_from_dict(t) for t in row.get("timecourses", [])]
            gold = [c for c in gold if c.n_points > 0]
            if not gold and skip_empty_gold:
                SKIPPED_EMPTY_GOLD.append(row["id"])
                continue
            entries.append(
                Entry(
                    id=row["id"],
                    image=root / row["image"],
                    study_sid=row["study_sid"],
                    paper_text=row.get("paper_text") or "",
                    reference_title=row.get("reference_title") or "",
                    substances=row.get("substances") or [],
                    gold=gold,
                    raw=row,
                )
            )
            if limit is not None and len(entries) >= limit:
                break
    return entries


def parse_prediction(obj: dict) -> list[Curve]:
    """Build Curves from a model's JSON response, tolerating a missing wrapper."""
    tcs = obj.get("timecourses")
    if tcs is None and isinstance(obj, list):
        tcs = obj
    curves = [curve_from_dict(t) for t in (tcs or [])]
    return [c for c in curves if c.n_points > 0]
