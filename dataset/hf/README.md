---
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
# PK-DB Figure → Timecourse (VLM)

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
| `target_json` | string | JSON: `{figure, series:[{substance, intervention, tissue, ..., time_unit, value_unit, points:[{time, mean, sd, se}]}]}` — the supervised target |
| `series_json` | string | JSON: full per-series records incl. `y_column`, `n_points`, `source_data_file` |
| `source` | string | attribution string |

## Statistics

- Figure→timecourse image pairs: **39**  (train **33** / validation **6**, study-disjoint)
- Distinct studies: **23**
- Digitized curves: **216** · (time, value) points: **2830**

## Source, license & attribution

All data derives from [PK-DB](https://pk-db.com) (Grzegorzewski et al., *Nucleic
Acids Res.* 2021, [doi:10.1093/nar/gkaa990](https://doi.org/10.1093/nar/gkaa990)),
restricted to **open-licence** studies. PK-DB curators digitized each published
concentration-time figure; PK-DB serves both the cropped figure image and the
digitized data behind it, which forms the ground-truth pairing here.
**Attribution is required.** See `LICENSE`.

```bibtex
@article{grzegorzewski2021pkdb,
  title={PK-DB: pharmacokinetics database for individualized and stratified computational modeling},
  author={Grzegorzewski, Jan and Brandhorst, Janosch and Green, Kathleen and others},
  journal={Nucleic Acids Research}, volume={49}, number={D1}, pages={D1358--D1364}, year={2021},
  doi={10.1093/nar/gkaa990}
}
```

## Caveats

- Small (**39** figures): curated open-licence subset, not a benchmark scale.
- Value units are as entered by curators (e.g. `microg/ml`, `µg/ml`, `mg/L`);
  normalize if you need canonical units.
- Includes mostly concentration curves plus some secretion-rate curves; a few
  figures are discrete-time strip/scatter plots (≥3 time points).
- ~20 open-licence studies from the 2021 dump were removed from live PK-DB and
  could not be included (figure images no longer served).
