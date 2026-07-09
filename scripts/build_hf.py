"""Package dataset/ into a HuggingFace-ready `imagefolder` dataset.

Reads dataset/annotations.jsonl + dataset/splits.json and emits dataset/hf/:

    hf/
      train/         image PNGs + metadata.jsonl   (HF imagefolder convention)
      validation/    image PNGs + metadata.jsonl
      README.md      dataset card WITH yaml frontmatter
      LICENSE        PK-DB attribution terms

`metadata.jsonl` uses the required `file_name` column plus flat scalar fields and
JSON-string columns (`target_json`, `series_json`) so Arrow never has to infer a
schema for the variable-key nested annotation. Load with:

    from datasets import load_dataset
    ds = load_dataset("imagefolder", data_dir="dataset/hf")   # train + validation

or push the hf/ folder to the Hub (auto-detected as imagefolder).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "dataset"
HF = DS / "hf"

INSTRUCTION = (
    "This is a pharmacokinetic figure from a scientific paper showing "
    "concentration-time curves. Extract every plotted time-series as structured "
    "data. For each curve, report the substance, intervention, tissue, subject "
    "group, the time unit and concentration unit, and the list of (time, value) "
    "points. Return the result as JSON."
)


def sft_target(rec: dict) -> dict:
    return {
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


def norm_substances(subs) -> list[str]:
    out = []
    for s in subs or []:
        if isinstance(s, dict):
            out.append(s.get("name") or s.get("label") or s.get("sid"))
        else:
            out.append(str(s))
    return [x for x in out if x]


def metadata_row(rec: dict) -> dict:
    return {
        "file_name": Path(rec["image"]).name,  # relative to this split folder
        "id": rec["id"],
        "study_sid": rec["study_sid"],
        "study_name": rec["study_name"],
        "pmid": str(rec["pmid"]) if rec.get("pmid") is not None else None,
        "reference_title": rec.get("reference_title"),
        "figure": rec["figure"],
        "substances": norm_substances(rec.get("substances")),
        "caption": rec.get("caption"),
        "n_series": rec["n_series"],
        "prompt": INSTRUCTION,
        # JSON-string columns -> stable Arrow schema regardless of nested keys
        "target_json": json.dumps(sft_target(rec), ensure_ascii=False),
        "series_json": json.dumps(rec["series"], ensure_ascii=False),
        "source": "PK-DB (pk-db.com), open-licence; attribution required",
    }


def main() -> None:
    recs = [json.loads(l) for l in (DS / "annotations.jsonl").open()]
    by_id = {r["id"]: r for r in recs}
    split = json.loads((DS / "splits.json").read_text())

    if HF.exists():
        shutil.rmtree(HF)
    counts = {}
    for hf_split, key in [("train", "train"), ("validation", "val")]:
        out = HF / hf_split
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for rid in split[key]:
            rec = by_id[rid]
            src = DS / rec["image"]
            shutil.copyfile(src, out / Path(rec["image"]).name)
            rows.append(metadata_row(rec))
        with (out / "metadata.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[hf_split] = len(rows)

    n_series = sum(r["n_series"] for r in recs)
    n_points = sum(s["n_points"] for r in recs for s in r["series"])
    n_studies = len({r["study_sid"] for r in recs})
    write_card(counts, len(recs), n_series, n_points, n_studies)
    write_license()

    print(f"HF dataset written to {HF}")
    print(f"  train={counts['train']}  validation={counts['validation']}")
    print("  load: datasets.load_dataset('imagefolder', data_dir='dataset/hf')")


def write_card(counts, n_rec, n_series, n_points, n_studies) -> None:
    yaml = f"""---
license: other
license_name: pk-db-terms
license_link: https://pk-db.com
pretty_name: PK-DB Figure to Timecourse (VLM)
language:
- en
task_categories:
- image-to-text
- visual-question-answering
tags:
- pharmacokinetics
- chart-understanding
- figure-extraction
- scientific-figures
- time-series
size_categories:
- n<1K
---
"""
    body = f"""# PK-DB Figure → Timecourse (VLM)

Vision-language dataset pairing **concentration-time figure images** from
pharmacokinetic papers with the **structured time-series data** each figure was
digitized from. Intended for training/evaluating a VLM to read PK plots.

## Load

```python
from datasets import load_dataset
ds = load_dataset("imagefolder", data_dir=".")   # 'image' + metadata columns
print(ds["train"][0]["prompt"])
print(ds["train"][0]["target_json"])              # JSON string: the extraction target
```

## Fields

| column | type | description |
|---|---|---|
| `image` | image | the figure (VLM input) |
| `id` | string | `<study_sid>__<figure>` |
| `study_sid`, `study_name`, `pmid`, `reference_title` | string | provenance |
| `figure` | string | e.g. `Fig1` |
| `substances` | list[string] | substances curated for the study |
| `caption` | string | short auto summary of the series in the figure |
| `n_series` | int | number of curves in the figure |
| `prompt` | string | the instruction (same for all rows) |
| `target_json` | string | JSON: `{{figure, series:[{{substance, intervention, tissue, ..., time_unit, value_unit, points:[{{time, mean, sd, se}}]}}]}}` — the supervised target |
| `series_json` | string | JSON: full per-series records incl. `y_column`, `n_points`, `source_data_file` |
| `source` | string | attribution string |

## Statistics

- Figure→timecourse image pairs: **{n_rec}**  (train **{counts['train']}** / validation **{counts['validation']}**, study-disjoint)
- Distinct studies: **{n_studies}**
- Digitized curves: **{n_series}** · (time, value) points: **{n_points}**

## Source, license & attribution

All data derives from [PK-DB](https://pk-db.com) (Grzegorzewski et al., *Nucleic
Acids Res.* 2021, [doi:10.1093/nar/gkaa990](https://doi.org/10.1093/nar/gkaa990)),
restricted to **open-licence** studies. PK-DB curators digitized each published
concentration-time figure; PK-DB serves both the cropped figure image and the
digitized data behind it, which forms the ground-truth pairing here.
**Attribution is required.** See `LICENSE`.

```bibtex
@article{{grzegorzewski2021pkdb,
  title={{PK-DB: pharmacokinetics database for individualized and stratified computational modeling}},
  author={{Grzegorzewski, Jan and Brandhorst, Janosch and Green, Kathleen and others}},
  journal={{Nucleic Acids Research}}, volume={{49}}, number={{D1}}, pages={{D1358--D1364}}, year={{2021}},
  doi={{10.1093/nar/gkaa990}}
}}
```

## Caveats

- Small (**{n_rec}** figures): curated open-licence subset, not a benchmark scale.
- Value units are as entered by curators (e.g. `microg/ml`, `µg/ml`, `mg/L`);
  normalize if you need canonical units.
- Includes mostly concentration curves plus some secretion-rate curves; a few
  figures are discrete-time strip/scatter plots (≥3 time points).
- ~20 open-licence studies from the 2021 dump were removed from live PK-DB and
  could not be included (figure images no longer served).
"""
    (HF / "README.md").write_text(yaml + body)


def write_license() -> None:
    txt = """PK-DB Figure → Timecourse (VLM) — license & attribution
========================================================

Data derived from PK-DB (https://pk-db.com), open-licence studies only.

PK-DB's TERMS_OF_USE state no restriction beyond the original data owners, but
ATTRIBUTION IS REQUIRED. When using this dataset you must cite:

  Grzegorzewski J, et al. PK-DB: pharmacokinetics database for individualized and
  stratified computational modeling. Nucleic Acids Research. 2021;49(D1):D1358-D1364.
  doi:10.1093/nar/gkaa990

and acknowledge PK-DB (https://pk-db.com) as the source of the figures and
digitized data. Underlying figures originate from the cited primary publications
(see each record's `pmid` / `reference_title`); respect those publishers' terms.
"""
    (HF / "LICENSE").write_text(txt)


if __name__ == "__main__":
    main()
