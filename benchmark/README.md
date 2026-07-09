# PK-DB Graph → Timecourse Benchmark

Each entry is **one pharmacokinetic figure (graph) + the full content of its
source paper**, paired with the **timecourses digitized from that figure**. The
task: given the graph and the paper, recover the underlying concentration-time
series.

    input   =  figure image  +  paper_text (full paper or abstract)
    target  =  timecourses[]  (per curve: substance, intervention, tissue,
               group, n, units, and the (time, value) points)

## Files

| path | description |
|---|---|
| `benchmark.jsonl` | one self-contained row per figure (paper text inlined) |
| `images/<id>.png` | the figure image |
| `papers/<sid>.txt` | extracted full paper text (from PK-DB PDF) |
| `papers/<sid>.pdf` | original paper PDF |
| `papers/<sid>.md` | PubMed abstract markdown |

## Row schema

```json
{
  "id": "PKDB00024__Fig1",
  "image": "images/PKDB00024__Fig1.png",
  "study_sid": "PKDB00024", "study_name": "Baraka1990", "pmid": "2306420",
  "reference_title": "...", "substances": [...], "figure": "Fig1",
  "paper_content_type": "fulltext_pdf",
  "paper_text": "<entire paper text>",
  "paper_pdf": "papers/PKDB00024.pdf", "paper_md": "papers/PKDB00024.md",
  "n_timecourses": 4,
  "timecourses": [
    {"substance": "paracetamol", "intervention": "paracetamol", "tissue": "plasma",
      "group": "all", "count": 10, "measurement_type": "concentration",
      "time_unit": "hr", "value_unit": "microg/ml",
      "points": [{"time": 0.5, "mean": 18.05, "sd": 4.49, "se": 1.42}, ...]}
  ]
}
```

## Statistics

- Entries (figures): **39**  across **23** studies
- Paper content: **38** full-text (from PDF), **1** abstract-only
- Timecourses (curves): **216** · (time, value) points: **2830**

## Notes on provenance & why paper content is included

Timecourse *values* come from PK-DB curators' digitization of each figure
(original paper units, e.g. `microg/ml`). Several fields needed to fully specify
a timecourse are **not visible in the graph** — subject count `n`, the SD-vs-SE
distinction, the intervention dose/route, and sometimes the concentration unit —
and must be read from the paper. Including the full paper text makes every target
field legitimately derivable from the input.

## Source & license

Derived from [PK-DB](https://pk-db.com) (open-licence studies) and PubMed. Cite
Grzegorzewski et al., *Nucleic Acids Res.* 2021, doi:10.1093/nar/gkaa990.
Paper PDFs/text belong to their original publishers (see each `pmid`); respect
their terms. Attribution required.

Pipeline: `build_manifest.py` → `recover_and_download.py` → `fetch_papers.py`
→ `build_dataset.py` → `build_benchmark.py`.
