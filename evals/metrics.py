"""Scoring predicted timecourses against the PK-DB gold digitizations.

The task is set-valued: a model returns an unordered list of curves, so before
any numeric comparison the predicted curves must be matched to the gold ones.
Matching is a linear-sum assignment over a cost that blends metadata agreement
with a scale-tolerant numeric distance, so a curve with the right shape but the
wrong unit still matches (and is then penalized during scoring rather than
silently paired with the wrong gold curve).

Numeric accuracy is reported two ways because concentration-time curves span
decades on a log axis:

    median_rel_err  -- median |pred-gold| / |gold| over gold's sample times
    median_log10_err -- median |log10(pred/gold)|, the scale-free view

A predicted curve only earns credit if its unit is convertible into the gold
unit; otherwise the values are not comparable and the curve is counted wrong.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .benchmark import META_FIELDS, Curve, field_similarity, to_gold_time_base
from .units import value_conversion

# Cost weights for the assignment problem.
W_NUMERIC = 0.6
W_META = 0.4

# A matched curve counts as recovered when its unit is right, it spans most of
# the gold time range, and its median relative error is within tolerance.
DEFAULT_REL_TOL = 0.20
MIN_COVERAGE = 0.5


def _match_tolerance(gold_times: np.ndarray) -> float:
    """How far a predicted time may sit from a gold time and still be the same point.

    Bounded by a fraction of the smallest gap between distinct gold times, so a
    predicted point can never be claimed by two different gold times.
    """
    uniq = np.unique(gold_times)
    span = float(uniq[-1] - uniq[0]) if uniq.size > 1 else 1.0
    tol = 0.02 * span
    if uniq.size > 1:
        tol = min(tol, 0.4 * float(np.diff(uniq).min()))
    return max(tol, 1e-9)


def align_curves(pred: Curve, gold: Curve) -> tuple[np.ndarray, np.ndarray, float]:
    """Pair predicted values with gold values, returning (gold_y, pred_y, coverage).

    Gold points are grouped by time. A group of size one is an ordinary sample of a
    function, so if the model reported no point there we interpolate its curve --
    a model that traced the same curve with fewer markers still deserves credit.
    A group of size > 1 is several subjects measured at one time; there is no
    function to interpolate, so the two value multisets are paired in sorted order.

    Extrapolation is refused: a model that only digitized the first two hours gets
    no credit for the 24-hour tail, it just loses coverage.
    """
    if pred.n_points == 0 or gold.n_points == 0:
        return np.empty(0), np.empty(0), 0.0

    tol = _match_tolerance(gold.times)
    # np.interp requires a strictly increasing grid; duplicate predicted times mean
    # the prediction is not a function either, so interpolation is not available.
    pred_is_function = pred.n_points >= 2 and bool(np.all(np.diff(pred.times) > 0))
    lo, hi = float(pred.times.min()), float(pred.times.max())

    used = np.zeros(pred.n_points, dtype=bool)
    gold_y: list[float] = []
    pred_y: list[float] = []

    for t in np.unique(gold.times):
        group = np.sort(gold.values[gold.times == t])
        near = np.where((np.abs(pred.times - t) <= tol) & ~used)[0]
        if near.size:
            order = near[np.argsort(pred.values[near])]
            k = min(order.size, group.size)
            used[order[:k]] = True
            gold_y.extend(group[:k].tolist())
            pred_y.extend(pred.values[order[:k]].tolist())
        elif group.size == 1 and pred_is_function and lo - tol <= t <= hi + tol:
            gold_y.append(float(group[0]))
            pred_y.append(float(np.interp(t, pred.times, pred.values)))

    coverage = len(gold_y) / gold.n_points
    return np.asarray(gold_y), np.asarray(pred_y), coverage


def _relative_errors(pred_y: np.ndarray, gold_y: np.ndarray) -> np.ndarray:
    denom = np.abs(gold_y)
    ok = denom > 0
    out = np.full(gold_y.shape, np.nan)
    out[ok] = np.abs(pred_y[ok] - gold_y[ok]) / denom[ok]
    return out


def _log10_errors(pred_y: np.ndarray, gold_y: np.ndarray) -> np.ndarray:
    ok = (pred_y > 0) & (gold_y > 0)
    out = np.full(gold_y.shape, np.nan)
    out[ok] = np.abs(np.log10(pred_y[ok] / gold_y[ok]))
    return out


def _nanmedian(a: np.ndarray) -> float | None:
    """Return the median of finite numeric values, tolerating missing values."""
    numeric = np.asarray(
        [value for value in np.asarray(a, dtype=object).flat if value is not None],
        dtype=float,
    )
    numeric = numeric[np.isfinite(numeric)]
    return float(np.median(numeric)) if numeric.size else None


def metadata_scores(pred: Curve, gold: Curve) -> dict[str, float]:
    """Per-field agreement, over only the fields gold actually annotates.

    PK-DB omits `group` and `measurement_type` on many curves; scoring a model
    against a null target would penalize it for the annotation's gaps.
    """
    p, g = pred.metadata(), gold.metadata()
    return {f: field_similarity(f, p[f], g[f]) for f in META_FIELDS if g[f] is not None}


def meta_similarity(pred: Curve, gold: Curve) -> float:
    """Weight the per-field scores into a single agreement in [0, 1]."""
    scores = metadata_scores(pred, gold)
    if not scores:
        return 0.5  # nothing to compare on; stay neutral
    total = sum(META_FIELDS[f] for f in scores)
    return sum(META_FIELDS[f] * s for f, s in scores.items()) / total


def _shape_distance(pred: Curve, gold: Curve) -> float:
    """Scale-free curve distance in [0, 1], used only for matching.

    Each curve is divided by its own median before comparison, so a prediction
    reported in ng/ml still matches a gold curve in microg/ml -- the unit error is
    then charged during scoring rather than corrupting the assignment.
    """
    g, p, coverage = align_curves(pred, gold)
    if g.size == 0:
        return 1.0
    finite = np.isfinite(p) & np.isfinite(g)
    if not finite.any():
        return 1.0
    p, g = p[finite], g[finite]
    pm, gm = np.median(np.abs(p)), np.median(np.abs(g))
    if pm <= 0 or gm <= 0:
        return 1.0
    med = _nanmedian(_log10_errors(np.abs(p) / pm, np.abs(g) / gm))
    if med is None:
        return 1.0
    # A curve covering little of the gold time range is a poor match even if the
    # overlapping part lines up.
    return float(np.clip(med, 0.0, 1.0)) * coverage + (1.0 - coverage)


@dataclass
class CurveScore:
    """Numeric and metadata agreement for one matched (pred, gold) pair."""

    pred_index: int
    gold_index: int
    gold_substance: str | None
    pred_substance: str | None
    units_comparable: bool
    pred_value_unit: str | None
    gold_value_unit: str | None
    coverage: float
    n_gold_points: int
    n_pred_points: int
    median_rel_err: float | None
    median_log10_err: float | None
    frac_within_10pct: float | None
    frac_within_20pct: float | None
    meta_similarity: float
    meta_fields: dict[str, float]
    recovered: bool


def compare_curves(
    pred: Curve, gold: Curve, *, rel_tol: float = DEFAULT_REL_TOL
) -> CurveScore:
    """Score one predicted curve against the gold curve it was matched to."""
    pred_t = to_gold_time_base(pred, gold.time_unit)
    g, pred_y, coverage = align_curves(pred_t, gold)

    factor = value_conversion(pred.value_unit, gold.value_unit)
    units_ok = factor is not None

    median_rel = median_log = frac10 = frac20 = None
    if units_ok and g.size:
        p = pred_y * factor
        rel = _relative_errors(p, g)
        median_rel = _nanmedian(rel)
        median_log = _nanmedian(_log10_errors(p, g))
        valid = rel[np.isfinite(rel)]
        if valid.size:
            frac10 = float((valid <= 0.10).mean())
            frac20 = float((valid <= 0.20).mean())

    meta_fields = metadata_scores(pred, gold)

    recovered = bool(
        units_ok
        and coverage >= MIN_COVERAGE
        and median_rel is not None
        and median_rel <= rel_tol
    )
    return CurveScore(
        pred_index=-1,
        gold_index=-1,
        gold_substance=gold.substance,
        pred_substance=pred.substance,
        units_comparable=units_ok,
        pred_value_unit=pred.value_unit,
        gold_value_unit=gold.value_unit,
        coverage=coverage,
        n_gold_points=gold.n_points,
        n_pred_points=pred.n_points,
        median_rel_err=median_rel,
        median_log10_err=median_log,
        frac_within_10pct=frac10,
        frac_within_20pct=frac20,
        meta_similarity=meta_similarity(pred, gold),
        meta_fields=meta_fields,
        recovered=recovered,
    )


def match_curves(preds: list[Curve], golds: list[Curve]) -> list[tuple[int, int]]:
    """Assign predicted curves to gold curves, minimizing total matching cost."""
    if not preds or not golds:
        return []
    cost = np.zeros((len(preds), len(golds)))
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            p_t = to_gold_time_base(p, g.time_unit)
            numeric = _shape_distance(p_t, g)
            cost[i, j] = W_NUMERIC * numeric + W_META * (1.0 - meta_similarity(p, g))
    rows, cols = linear_sum_assignment(cost)
    return list(zip(rows.tolist(), cols.tolist()))


def score_figure(
    preds: list[Curve], golds: list[Curve], *, rel_tol: float = DEFAULT_REL_TOL
) -> dict[str, Any]:
    """Score all predicted curves for one figure against its gold timecourses."""
    pairs = match_curves(preds, golds)
    scores: list[CurveScore] = []
    for i, j in pairs:
        s = compare_curves(preds[i], golds[j], rel_tol=rel_tol)
        s.pred_index, s.gold_index = i, j
        scores.append(s)

    n_recovered = sum(s.recovered for s in scores)
    precision = n_recovered / len(preds) if preds else 0.0
    recall = n_recovered / len(golds) if golds else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    matched = [s for s in scores if s.units_comparable and s.median_rel_err is not None]
    meta_hits = [v for s in scores for v in s.meta_fields.values()]

    return {
        "n_gold": len(golds),
        "n_pred": len(preds),
        "n_recovered": n_recovered,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "count_error": len(preds) - len(golds),
        "unit_error_rate": (
            sum(not s.units_comparable for s in scores) / len(scores)
            if scores
            else None
        ),
        "median_rel_err": _nanmedian(np.array([s.median_rel_err for s in matched]))
        if matched
        else None,
        "median_log10_err": _nanmedian(np.array([s.median_log10_err for s in matched]))
        if matched
        else None,
        "mean_coverage": float(np.mean([s.coverage for s in scores]))
        if scores
        else None,
        "meta_accuracy": float(np.mean(meta_hits)) if meta_hits else None,
        "curves": [asdict(s) for s in scores],
    }


def aggregate(figure_scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Roll per-figure results into micro (curve-weighted) and macro (figure-weighted) views."""
    figs = list(figure_scores.values())
    if not figs:
        return {}

    tot_gold = sum(f["n_gold"] for f in figs)
    tot_pred = sum(f["n_pred"] for f in figs)
    tot_rec = sum(f["n_recovered"] for f in figs)
    micro_p = tot_rec / tot_pred if tot_pred else 0.0
    micro_r = tot_rec / tot_gold if tot_gold else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    )

    all_curves = [c for f in figs for c in f["curves"]]
    comparable = [c for c in all_curves if c["median_rel_err"] is not None]

    def _mean(key: str, source: list[dict]) -> float | None:
        vals = [s[key] for s in source if s.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def _median(key: str, source: list[dict]) -> float | None:
        vals = [s[key] for s in source if s.get(key) is not None]
        return float(np.median(vals)) if vals else None

    meta_hits = [v for c in all_curves for v in c["meta_fields"].values()]
    per_field: dict[str, float] = {}
    for f in META_FIELDS:
        vals = [c["meta_fields"][f] for c in all_curves if f in c["meta_fields"]]
        if vals:
            per_field[f] = float(np.mean(vals))

    return {
        "n_figures": len(figs),
        "n_gold_curves": tot_gold,
        "n_pred_curves": tot_pred,
        "n_recovered_curves": tot_rec,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": float(np.mean([f["f1"] for f in figs])),
        "exact_curve_count": float(np.mean([f["count_error"] == 0 for f in figs])),
        "mean_abs_count_error": float(np.mean([abs(f["count_error"]) for f in figs])),
        "unit_error_rate": float(
            np.mean([not c["units_comparable"] for c in all_curves])
        )
        if all_curves
        else None,
        "median_rel_err": _median("median_rel_err", comparable),
        "median_log10_err": _median("median_log10_err", comparable),
        "mean_frac_within_10pct": _mean("frac_within_10pct", comparable),
        "mean_frac_within_20pct": _mean("frac_within_20pct", comparable),
        "mean_coverage": _mean("coverage", all_curves),
        "meta_accuracy": float(np.mean(meta_hits)) if meta_hits else None,
        "meta_accuracy_by_field": per_field,
    }
