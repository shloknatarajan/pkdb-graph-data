---
license: other
license_name: pkdb-and-publisher-terms
license_link: https://pk-db.com
pretty_name: PK-DB Graph to Timecourse Benchmark
language:
- en
task_categories:
- image-to-text
- visual-question-answering
tags:
- pharmacokinetics
- chart-understanding
- figure-extraction
- time-series
- scientific-figures
size_categories:
- n<1K
---
# PK-DB Graph → Timecourse Benchmark

Each entry is **one pharmacokinetic figure (graph) + the full text of its source
paper**, paired with the **timecourses digitized from that figure**.

    input   =  figure image  +  paper_text (full article)
    target  =  timecourses[]  (per curve: substance, intervention, tissue, group,
               n, time_unit, value_unit, and the (time, value) points)

## Load

```python
from datasets import load_dataset
ds = load_dataset("imagefolder", data_dir=".")   # split: 'test'
ex = ds["test"][0]
ex["image"]                    # PIL figure
ex["paper_text"]               # full paper text
import json; json.loads(ex["timecourses_json"])   # target series
```

## Columns

| column | description |
|---|---|
| `image` | figure image (input) |
| `paper_text` | full paper text (input); `paper_content_type` = `fulltext_pdf`/`abstract` |
| `paper_pdf` | filename under `papers/` of the original PDF |
| `timecourses_json` | JSON string: list of curves, each with substance/intervention/tissue/group/count/units/points (target) |
| `id`, `study_sid`, `study_name`, `pmid`, `reference_title`, `substances`, `figure`, `n_timecourses` | provenance |

`papers/` holds the original PDFs and extracted `.txt`/`.md` for each study.

## Statistics

- Entries (figures): **39** across **23** studies (single `test` split)
- Paper content: **38** full-text, **1** abstract-only
- Timecourses: **216** · (time, value) points: **2830**

## Why the paper is included

Timecourse *values* come from PK-DB curators' figure digitization (original paper
units). Fields like subject **n**, the **SD-vs-SE** distinction, the intervention
**dose/route**, and sometimes the **unit** are not visible in the graph and must
be read from the paper — so the full text is bundled as input.

## Source, license & attribution

Figures + digitized data: [PK-DB](https://pk-db.com), open-licence studies
(Grzegorzewski et al., *Nucleic Acids Res.* 2021, doi:10.1093/nar/gkaa990).
Paper text/PDFs are the property of their original publishers (see each `pmid`)
and are included here for research use; respect the publishers' terms. Attribution
to PK-DB and the primary publications is required. See `LICENSE`.
