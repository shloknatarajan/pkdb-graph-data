"""Command line entry point: run a model on the benchmark, then score it.

python -m evals.run predict --backend claude-cli --model sonnet --limit 3
python -m evals.run score   --predictions predictions/sonnet.jsonl
python -m evals.run run     --backend oracle
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .backends import Prediction, build_backend
from .benchmark import BENCHMARK_DIR, load_benchmark, parse_prediction
from .metrics import DEFAULT_REL_TOL, aggregate, score_figure

PREDICTIONS_DIR = Path(__file__).resolve().parent.parent / "predictions"


def _add_selection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--benchmark", type=Path, default=BENCHMARK_DIR / "benchmark.jsonl")
    p.add_argument("--limit", type=int, default=None, help="only the first N figures")
    p.add_argument("--ids", nargs="*", default=None, help="specific figure ids")


def cmd_predict(args: argparse.Namespace) -> Path:
    entries = load_benchmark(args.benchmark, ids=args.ids, limit=args.limit)
    backend = build_backend(
        args.backend,
        args.model,
        paper_chars=None if args.paper_chars < 0 else args.paper_chars,
        substance_hint=args.substance_hint,
    )
    out = args.out or PREDICTIONS_DIR / f"{args.backend}_{backend.model}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[predict] {len(entries)} figures -> {backend.name}:{backend.model}",
        file=sys.stderr,
    )
    preds: list[Prediction] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, pred in enumerate(pool.map(backend.predict, entries), 1):
            status = pred.error or f"{len(pred.timecourses)} curves"
            print(
                f"  [{i}/{len(entries)}] {pred.id}: {status} ({pred.elapsed_s:.0f}s)",
                file=sys.stderr,
            )
            preds.append(pred)

    with out.open("w") as fh:
        for p in preds:
            fh.write(json.dumps(p.to_json()) + "\n")
    failed = sum(p.error is not None for p in preds)
    cost = sum(p.usage.get("cost_usd") or 0.0 for p in preds)
    wall = sum(p.elapsed_s for p in preds)
    note = f", ${cost:.2f}" if cost else ""
    print(
        f"[predict] wrote {out} ({failed} failures, {wall / 60:.1f} model-minutes{note})",
        file=sys.stderr,
    )
    return out


def _load_predictions(path: Path) -> dict[str, dict]:
    with path.open() as fh:
        return {
            row["id"]: row for row in (json.loads(line) for line in fh if line.strip())
        }


def cmd_score(args: argparse.Namespace) -> dict:
    entries = load_benchmark(args.benchmark, ids=args.ids, limit=args.limit)
    preds = _load_predictions(args.predictions)

    per_figure: dict[str, dict] = {}
    for entry in entries:
        row = preds.get(entry.id)
        if row is None:
            continue
        curves = [] if row.get("error") else parse_prediction(row)
        per_figure[entry.id] = score_figure(curves, entry.gold, rel_tol=args.rel_tol)
        per_figure[entry.id]["error"] = row.get("error")

    summary = aggregate(per_figure)
    report = {
        "predictions": str(args.predictions),
        "rel_tol": args.rel_tol,
        "n_failed_figures": sum(1 for f in per_figure.values() if f["error"]),
        "summary": summary,
        "per_figure": per_figure,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
    print_report(report)
    return report


def print_report(report: dict) -> None:
    s = report["summary"]
    if not s:
        print("no figures scored")
        return

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{100 * x:5.1f}%"

    def num(x: float | None, digits: int = 3) -> str:
        return "n/a" if x is None else f"{x:.{digits}f}"

    print(f"\n{report['predictions']}")
    print(f"  figures {s['n_figures']}  (failed: {report['n_failed_figures']})")
    print(f"  gold curves {s['n_gold_curves']}   predicted {s['n_pred_curves']}")
    print("\n  Curve recovery (a curve counts only if its unit converts, it covers")
    print(
        f"  >=50% of the gold time range, and median relative error <= {report['rel_tol']:.0%})"
    )
    print(
        f"    micro F1          {num(s['micro_f1'])}   (P {num(s['micro_precision'])} / R {num(s['micro_recall'])})"
    )
    print(f"    macro F1          {num(s['macro_f1'])}")
    print(f"    recovered curves  {s['n_recovered_curves']}/{s['n_gold_curves']}")
    print("\n  Curve counting")
    print(f"    exact count       {pct(s['exact_curve_count'])} of figures")
    print(f"    mean |count err|  {num(s['mean_abs_count_error'], 2)}")
    print("\n  Numeric accuracy (over matched, unit-comparable curves)")
    print(f"    median rel err    {pct(s['median_rel_err'])}")
    print(f"    median log10 err  {num(s['median_log10_err'])}")
    print(f"    points within 10% {pct(s['mean_frac_within_10pct'])}")
    print(f"    points within 20% {pct(s['mean_frac_within_20pct'])}")
    print(f"    time coverage     {pct(s['mean_coverage'])}")
    print("\n  Metadata")
    print(f"    unit error rate   {pct(s['unit_error_rate'])}")
    print(f"    field accuracy    {pct(s['meta_accuracy'])}")
    for fname, acc in sorted(
        s["meta_accuracy_by_field"].items(), key=lambda kv: -kv[1]
    ):
        print(f"      {fname:<18}{pct(acc)}")
    print()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="evals.run", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="run a model over the benchmark")
    _add_selection_args(p)
    p.add_argument("--backend", default="claude-cli")
    p.add_argument("--model", default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--paper-chars", type=int, default=40000, help="-1 for the full paper"
    )
    p.add_argument(
        "--substance-hint",
        action="store_true",
        help="list study substances in the prompt",
    )

    p = sub.add_parser("score", help="score an existing predictions file")
    _add_selection_args(p)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL)
    p.add_argument("--out", type=Path, default=None)

    p = sub.add_parser("run", help="predict then score")
    _add_selection_args(p)
    p.add_argument("--backend", default="claude-cli")
    p.add_argument("--model", default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--paper-chars", type=int, default=40000)
    p.add_argument("--substance-hint", action="store_true")
    p.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL)

    args = ap.parse_args(argv)
    if args.cmd == "predict":
        cmd_predict(args)
    elif args.cmd == "score":
        cmd_score(args)
    else:
        pred_path = cmd_predict(args)
        args.predictions = pred_path
        args.out = None
        cmd_score(args)


if __name__ == "__main__":
    main()
