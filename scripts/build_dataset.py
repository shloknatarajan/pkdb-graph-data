"""Assemble the VLM training dataset: figure image <-> digitized time-series.

Ground truth is the set of digitized ``.*_Fig*.tsv`` files PK-DB curators produced
by tracing each concentration-time figure. We therefore iterate over TSVs (not
PNGs) and pair each to its figure image:

    .<Study>_Fig<N>.tsv   ->  <Study>_Fig<N>.png          (exact stem)
    .<Study>_Fig<N>A.tsv  ->  <Study>_Fig<N>.png          (sub-panel -> parent)

Multiple sub-panel TSVs that map to the same PNG are merged into one record, so
each record is (one figure image  <->  all series digitized from it). Only
figures whose data is a genuine concentration-time course (numeric ``time``
column with >= MIN_POINTS distinct times) are kept; PK-parameter scatter figures
and pure table images are excluded.

Outputs (under dataset/):
    images/<sid>__<figure>.png   figure image (VLM input)
    annotations.jsonl            rich structured record per figure
    vlm_sft.jsonl                chat-format SFT records (image + instruction -> JSON)
    splits.json                  study-disjoint train/val split
    README.md                    dataset card
"""

from __future__ import annotations

import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MEDIA_DIR = ROOT / "data" / "media"
MANIFEST = ROOT / "data" / "manifest.json"
OUT = ROOT / "dataset"
IMG_OUT = OUT / "images"

MIN_POINTS = 3  # a "timecourse" needs at least this many distinct time points

# columns that identify *which* series a row belongs to (vs. the data point)
GROUPING_COLS = [
    "group",
    "individual",
    "count",
    "label",
    "measurement_type",
    "tissue",
    "intervention",
    "substance",
]
VALUE_COLS = ["mean", "median", "value", "sd", "se", "min", "max", "cv", "n", "count"]
POINT_VALUE_COLS = ["mean", "median", "value", "sd", "se", "min", "max", "cv"]


def clean(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def target_png(tsv_name: str, files: set[str]) -> str | None:
    """Map a .*_Fig*.tsv filename to its figure PNG (exact stem, else parent panel)."""
    stem = tsv_name[1:] if tsv_name.startswith(".") else tsv_name
    stem = stem[:-4] if stem.lower().endswith(".tsv") else stem
    if f"{stem}.png" in files:
        return f"{stem}.png"
    m = re.match(r"(.*_Fig\d+)", stem)
    if m and f"{m.group(1)}.png" in files:
        return f"{m.group(1)}.png"
    return None


def figure_id(png_name: str, study_name: str) -> str:
    stem = png_name.rsplit(".", 1)[0]
    prefix = f"{study_name}_"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    return stem


def parse_tsv_series(tsv_path: Path, source_file: str) -> list[dict]:
    """Parse one digitized TSV into timecourse series. Empty if not a timecourse."""
    try:
        df = pd.read_csv(tsv_path, sep="\t")
    except Exception:
        return []
    cols = list(df.columns)
    if "time" not in cols:
        return []
    df = df[pd.to_numeric(df["time"], errors="coerce").notna()].copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    if df["time"].nunique() < MIN_POINTS:
        return []

    group_cols = [c for c in GROUPING_COLS if c in cols]
    ycol = next((c for c in ["mean", "median", "value"] if c in cols), None)
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]

    series = []
    for key, g in grouped:
        g = g.sort_values("time")
        key = key if isinstance(key, tuple) else (key,)
        meta = {c: clean(v) for c, v in zip(group_cols, key)}
        if g["time"].nunique() < MIN_POINTS:
            continue
        points = []
        for _, row in g.iterrows():
            pt = {"time": clean(row.get("time"))}
            for vc in POINT_VALUE_COLS:
                if vc in cols and clean(row.get(vc)) is not None:
                    pt[vc] = clean(row.get(vc))
            points.append(pt)
        series.append(
            {
                "series_key": meta,
                "time_unit": clean(g["time_unit"].iloc[0]) if "time_unit" in cols else None,
                "value_unit": clean(g["unit"].iloc[0]) if "unit" in cols else None,
                "y_column": ycol,
                "n_points": len(points),
                "source_data_file": source_file,
                "points": points,
            }
        )
    return series


def series_caption(series: list[dict]) -> str:
    def uniq(field):
        return sorted({str(s["series_key"].get(field)) for s in series if s["series_key"].get(field)})

    parts = []
    for field, lbl in [("substance", "substance"), ("intervention", "intervention"), ("tissue", "tissue")]:
        vals = uniq(field)
        if vals:
            parts.append(f"{lbl}(s): {', '.join(vals)}")
    return "; ".join(parts)


INSTRUCTION = (
    "This is a pharmacokinetic figure from a scientific paper showing "
    "concentration-time curves. Extract every plotted time-series as structured "
    "data. For each curve, report the substance, intervention, tissue, subject "
    "group, the time unit and concentration unit, and the list of (time, value) "
    "points. Return the result as JSON."
)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    if IMG_OUT.exists():
        shutil.rmtree(IMG_OUT)
    IMG_OUT.mkdir(parents=True, exist_ok=True)

    records = []
    stats = defaultdict(int)
    for sid, m in manifest.items():
        if "files" not in m:
            continue
        name = m["name"]
        study_dir = MEDIA_DIR / name
        if not study_dir.exists():
            continue
        on_disk = {p.name for p in study_dir.iterdir()}
        fig_tsvs = sorted(
            f for f in on_disk if f.lower().endswith(".tsv") and re.search(r"_Fig", f, re.I)
        )
        # group tsvs by their target figure png
        png_to_tsvs: dict[str, list[str]] = defaultdict(list)
        for t in fig_tsvs:
            stats["fig_tsv_total"] += 1
            png = target_png(t, on_disk)
            if png is None:
                stats["tsv_no_png"] += 1
                continue
            png_to_tsvs[png].append(t)

        for png, tsvs in sorted(png_to_tsvs.items()):
            merged = []
            for t in tsvs:
                merged.extend(parse_tsv_series(study_dir / t, t))
            if not merged:
                stats["figure_not_timecourse"] += 1
                continue
            stats["timecourse_figures"] += 1
            stats["series_total"] += len(merged)

            fig_id = figure_id(png, name)
            out_name = f"{sid}__{fig_id}.png"
            shutil.copyfile(study_dir / png, IMG_OUT / out_name)
            records.append(
                {
                    "id": f"{sid}__{fig_id}",
                    "image": f"images/{out_name}",
                    "study_sid": sid,
                    "study_name": name,
                    "dump_name": m.get("dump_name", name),
                    "pmid": m.get("pmid"),
                    "reference_title": m.get("reference_title"),
                    "figure": fig_id,
                    "source_figure_file": png,
                    "source_data_files": tsvs,
                    "substances": m.get("substances"),
                    "caption": series_caption(merged),
                    "n_series": len(merged),
                    "series": merged,
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "annotations.jsonl").open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    with (OUT / "vlm_sft.jsonl").open("w") as f:
        for rec in records:
            target = {
                "figure": rec["figure"],
                "series": [
                    {
                        **s["series_key"],
                        "time_unit": s["time_unit"],
                        "value_unit": s["value_unit"],
                        "points": s["points"],
                    }
                    for s in rec["series"]
                ],
            }
            f.write(
                json.dumps(
                    {
                        "id": rec["id"],
                        "image": rec["image"],
                        "messages": [
                            {"role": "user", "content": "<image>\n" + INSTRUCTION},
                            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
                        ],
                        "provenance": {
                            "study_sid": rec["study_sid"],
                            "study_name": rec["study_name"],
                            "pmid": rec["pmid"],
                            "figure": rec["figure"],
                            "source": "PK-DB (pk-db.com), open-licence",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    sids = sorted({r["study_sid"] for r in records})
    val = set(sids[::5])
    split = {
        "train": [r["id"] for r in records if r["study_sid"] not in val],
        "val": [r["id"] for r in records if r["study_sid"] in val],
        "val_studies": sorted(val),
    }
    (OUT / "splits.json").write_text(json.dumps(split, indent=2))

    print("=== dataset stats ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    n_series = sum(r["n_series"] for r in records)
    n_points = sum(s["n_points"] for r in records for s in r["series"])
    print(f"  studies_with_data: {len(sids)}")
    print(f"  figure_records: {len(records)}")
    print(f"  total_series: {n_series}")
    print(f"  total_points: {n_points}")
    print(f"  train/val: {len(split['train'])}/{len(split['val'])}")

    write_readme(records, dict(stats), sids, split, n_series, n_points)


def write_readme(records, stats, sids, split, n_series, n_points) -> None:
    md = f"""# PK-DB Figure → Timecourse VLM Dataset

Paired **concentration-time figure images** and the **digitized time-series data**
each figure was traced from, for training a vision-language model to read
pharmacokinetic plots.

## Source & provenance

All data comes from [PK-DB](https://pk-db.com) (Grzegorzewski et al., *Nucleic
Acids Res.* 2021, [doi:10.1093/nar/gkaa990](https://doi.org/10.1093/nar/gkaa990)),
restricted to **open-licence** studies. PK-DB curators digitized each published
concentration-time figure; PK-DB stores both the cropped figure image
(`<Study>_Fig<N>.png`) and the digitized data behind it (`.<Study>_Fig<N>.tsv`).
The TSV is the ground-truth annotation for its sibling PNG. **Attribution required.**

## Contents

| file | description |
|---|---|
| `images/<sid>__<figure>.png` | figure image (VLM input) |
| `annotations.jsonl` | one rich record per figure: full structured series + provenance |
| `vlm_sft.jsonl` | chat-format supervised-fine-tuning records (`<image>` + instruction → JSON target) |
| `splits.json` | study-disjoint train/val split |

## Statistics

- Figure → timecourse pairs (images): **{len(records)}**
- Distinct studies: **{len(sids)}**
- Digitized series (individual curves): **{n_series}**
- Total (time, value) points: **{n_points}**
- Train / val records: **{len(split['train'])} / {len(split['val'])}** (val is study-disjoint)

## Record schema (`annotations.jsonl`)

```json
{{
  "id": "PKDB00024__Fig1",
  "image": "images/PKDB00024__Fig1.png",
  "study_sid": "PKDB00024", "study_name": "Baraka1990", "pmid": "2306420",
  "figure": "Fig1",
  "source_figure_file": "Baraka1990_Fig1.png",
  "source_data_files": [".Baraka1990_Fig1.tsv"],
  "n_series": 3,
  "series": [
    {{
      "series_key": {{"intervention": "paracetamol", "substance": "paracetamol",
                     "tissue": "plasma", "count": 10}},
      "time_unit": "hr", "value_unit": "microg/ml", "y_column": "mean",
      "n_points": 11,
      "points": [{{"time": 0.5, "mean": 18.05, "sd": 4.49, "se": 1.42}}, ...]
    }}
  ]
}}
```

## Method notes

- Records are built by inverting from the digitized TSVs (ground truth) to their
  figure images. Sub-panel TSVs (`_Fig2A`, `_Fig2B`) are merged onto their parent
  figure image (`_Fig2.png`), so one record = one image with all its curves.
- Only figures whose TSV encodes a concentration-time course (numeric `time`,
  ≥{MIN_POINTS} distinct times) are included. Table images (`_Tab*`) and
  PK-parameter/scatter figures are excluded.
- Pipeline: `scripts/build_manifest.py` → `scripts/recover_and_download.py` →
  `scripts/recover_by_pmid.py` → `scripts/build_dataset.py`.

## Coverage limits

The 2021 CSV dump lists 56 open-licence studies; ~20 (mostly glucose/insulin
metabolism papers) have since been removed from the live PK-DB and their figure
images are no longer served, so they are absent here. Pipeline diagnostics:
`{json.dumps(stats)}`.
"""
    (OUT / "README.md").write_text(md)


if __name__ == "__main__":
    main()
