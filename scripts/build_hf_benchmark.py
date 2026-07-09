"""Package benchmark/ into a HuggingFace imagefolder (single `test` split).

Row = figure image + full paper text (input) -> timecourses (target). Nested
timecourses are stored as a JSON string for a stable Arrow schema. PDFs/txt/md
are copied to hf/papers/ for upload as plain repo files (not part of the parquet).

Outputs benchmark/hf/: test/ (images + metadata.jsonl), papers/, README.md, LICENSE.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BM = ROOT / "benchmark"
HF = BM / "hf"
TEST = HF / "test"
PAPERS = HF / "papers"


def norm_substances(subs) -> list[str]:
    out = []
    for s in subs or []:
        out.append(s.get("name") or s.get("label") or s.get("sid") if isinstance(s, dict) else str(s))
    return [x for x in out if x]


def main() -> None:
    rows = [json.loads(l) for l in (BM / "benchmark.jsonl").open()]
    for d in (TEST, PAPERS):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    meta = []
    copied_papers = set()
    for r in rows:
        img = Path(r["image"]).name
        shutil.copyfile(BM / r["image"], TEST / img)
        # copy paper assets once per study
        for rel in (r.get("paper_pdf"), r.get("paper_md")):
            if rel and rel not in copied_papers:
                src = BM / rel
                if src.exists():
                    shutil.copyfile(src, PAPERS / Path(rel).name)
                    copied_papers.add(rel)
        sid = r["study_sid"]
        txt_src = BM / "papers" / f"{sid}.txt"
        if txt_src.exists() and f"{sid}.txt" not in copied_papers:
            shutil.copyfile(txt_src, PAPERS / f"{sid}.txt")
            copied_papers.add(f"{sid}.txt")

        meta.append(
            {
                "file_name": img,
                "id": r["id"],
                "study_sid": sid,
                "study_name": r["study_name"],
                "pmid": str(r["pmid"]) if r.get("pmid") is not None else None,
                "reference_title": r.get("reference_title"),
                "substances": norm_substances(r.get("substances")),
                "figure": r["figure"],
                "paper_content_type": r["paper_content_type"],
                "paper_text": r["paper_text"],
                "paper_pdf": Path(r["paper_pdf"]).name if r.get("paper_pdf") else None,
                "n_timecourses": r["n_timecourses"],
                "timecourses_json": json.dumps(r["timecourses"], ensure_ascii=False),
            }
        )

    with (TEST / "metadata.jsonl").open("w") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    write_card(rows)
    write_license()
    print(f"HF benchmark at {HF}: {len(meta)} rows, {len(copied_papers)} paper files")


def write_card(rows) -> None:
    n = len(rows)
    n_studies = len({r["study_sid"] for r in rows})
    n_full = sum(1 for r in rows if r["paper_content_type"] == "fulltext_pdf")
    n_tc = sum(r["n_timecourses"] for r in rows)
    n_pts = sum(len(t["points"]) for r in rows for t in r["timecourses"])
    yaml = """---
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
"""
    body = f"""# PK-DB Graph → Timecourse Benchmark

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

- Entries (figures): **{n}** across **{n_studies}** studies (single `test` split)
- Paper content: **{n_full}** full-text, **{n - n_full}** abstract-only
- Timecourses: **{n_tc}** · (time, value) points: **{n_pts}**

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
"""
    (HF / "README.md").write_text(yaml + body)


def write_license() -> None:
    (HF / "LICENSE").write_text(
        "PK-DB Graph -> Timecourse Benchmark — license & attribution\n"
        "===========================================================\n\n"
        "Figures and digitized timecourse data derive from PK-DB (https://pk-db.com),\n"
        "open-licence studies only. Cite:\n\n"
        "  Grzegorzewski J, et al. PK-DB: pharmacokinetics database for individualized\n"
        "  and stratified computational modeling. Nucleic Acids Research. 2021;49(D1):\n"
        "  D1358-D1364. doi:10.1093/nar/gkaa990\n\n"
        "Paper full text and PDFs are the property of their respective publishers\n"
        "(identified by each record's pmid) and are provided for research use only.\n"
        "Users must respect the original publishers' copyright and terms of use.\n"
    )


if __name__ == "__main__":
    main()
