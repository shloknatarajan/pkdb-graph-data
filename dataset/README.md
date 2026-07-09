# PK-DB Figure → Timecourse VLM Dataset

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

- Figure → timecourse pairs (images): **39**
- Distinct studies: **23**
- Digitized series (individual curves): **216**
- Total (time, value) points: **2830**
- Train / val records: **33 / 6** (val is study-disjoint)

## Record schema (`annotations.jsonl`)

```json
{
  "id": "PKDB00024__Fig1",
  "image": "images/PKDB00024__Fig1.png",
  "study_sid": "PKDB00024", "study_name": "Baraka1990", "pmid": "2306420",
  "figure": "Fig1",
  "source_figure_file": "Baraka1990_Fig1.png",
  "source_data_files": [".Baraka1990_Fig1.tsv"],
  "n_series": 3,
  "series": [
    {
      "series_key": {"intervention": "paracetamol", "substance": "paracetamol",
                     "tissue": "plasma", "count": 10},
      "time_unit": "hr", "value_unit": "microg/ml", "y_column": "mean",
      "n_points": 11,
      "points": [{"time": 0.5, "mean": 18.05, "sd": 4.49, "se": 1.42}, ...]
    }
  ]
}
```

## Method notes

- Records are built by inverting from the digitized TSVs (ground truth) to their
  figure images. Sub-panel TSVs (`_Fig2A`, `_Fig2B`) are merged onto their parent
  figure image (`_Fig2.png`), so one record = one image with all its curves.
- Only figures whose TSV encodes a concentration-time course (numeric `time`,
  ≥3 distinct times) are included. Table images (`_Tab*`) and
  PK-parameter/scatter figures are excluded.
- Pipeline: `scripts/build_manifest.py` → `scripts/recover_and_download.py` →
  `scripts/recover_by_pmid.py` → `scripts/build_dataset.py`.

## Coverage limits

The 2021 CSV dump lists 56 open-licence studies; ~20 (mostly glucose/insulin
metabolism papers) have since been removed from the live PK-DB and their figure
images are no longer served, so they are absent here. Pipeline diagnostics:
`{"fig_tsv_total": 66, "timecourse_figures": 39, "series_total": 216, "figure_not_timecourse": 14, "tsv_no_png": 1}`.
