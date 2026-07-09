"""Build the figure-segmented benchmark: (graph + all paper content) -> timecourses.

Each entry is ONE figure image plus the full text of its source paper (input),
paired with the timecourses digitized from that figure (target). This is the unit
a model is asked to solve: "here is a concentration-time figure and the paper it
came from; produce the underlying time-series."

Inputs:
    dataset/annotations.jsonl        per-figure digitized series (from PK-DB figure TSVs)
    study_dataset/paper_index.json   paper markdown/pdf per study
    study_dataset/papers/<sid>.{md,pdf}

Paper content per entry, best available:
    fulltext_pdf  - text extracted from the PK-DB-hosted PDF (preferred)
    abstract      - abstract markdown from PubMed (fallback)

Outputs (benchmark/):
    images/<id>.png          the figure (graph)
    papers/<sid>.{pdf,txt,md}  paper assets
    benchmark.jsonl          one self-contained row per figure
    README.md
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
STUDY = ROOT / "study_dataset"
PAPERS_IN = STUDY / "papers"
OUT = ROOT / "benchmark"
IMG_OUT = OUT / "images"
PAP_OUT = OUT / "papers"

TC_KEYS = [
    "substance",
    "intervention",
    "tissue",
    "group",
    "individual",
    "count",
    "label",
    "measurement_type",
]


def pdf_text(pdf: Path) -> str | None:
    try:
        from pypdf import PdfReader

        r = PdfReader(str(pdf))
        txt = "\n".join((p.extract_text() or "") for p in r.pages)
        return txt if len(txt) > 800 else None
    except Exception:
        return None


def flatten_series(series: list[dict]) -> list[dict]:
    out = []
    for s in series:
        k = s.get("series_key", {})
        tc = {kk: k.get(kk) for kk in TC_KEYS if k.get(kk) is not None}
        tc["time_unit"] = s.get("time_unit")
        tc["value_unit"] = s.get("value_unit")
        tc["points"] = s.get("points", [])
        out.append(tc)
    return out


def main() -> None:
    recs = [json.loads(l) for l in (DATASET / "annotations.jsonl").open()]
    paper_index = json.loads((STUDY / "paper_index.json").read_text())

    for d in (IMG_OUT, PAP_OUT):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # cache extracted pdf text per study
    text_cache: dict[str, tuple[str, str]] = {}  # sid -> (content_type, text)

    def paper_for(sid: str) -> tuple[str, str | None, str | None, str | None]:
        """Return (content_type, text, pdf_rel, md_rel)."""
        pinfo = paper_index.get(sid, {})
        pdf_rel = md_rel = None
        # copy assets
        src_pdf = PAPERS_IN / f"{sid}.pdf"
        src_md = PAPERS_IN / f"{sid}.md"
        if src_pdf.exists():
            shutil.copyfile(src_pdf, PAP_OUT / f"{sid}.pdf")
            pdf_rel = f"papers/{sid}.pdf"
        if src_md.exists():
            shutil.copyfile(src_md, PAP_OUT / f"{sid}.md")
            md_rel = f"papers/{sid}.md"

        if sid in text_cache:
            ctype, text = text_cache[sid]
        else:
            text = pdf_text(src_pdf) if src_pdf.exists() else None
            if text:
                ctype = "fulltext_pdf"
                (PAP_OUT / f"{sid}.txt").write_text(text)
            elif src_md.exists():
                ctype, text = "abstract", src_md.read_text()
            else:
                ctype, text = "none", None
            text_cache[sid] = (ctype, text)
        return ctype, text, pdf_rel, md_rel

    rows = []
    for rec in recs:
        sid = rec["study_sid"]
        ctype, text, pdf_rel, md_rel = paper_for(sid)
        img_name = f"{rec['id']}.png"
        shutil.copyfile(DATASET / rec["image"], IMG_OUT / img_name)

        tcs = flatten_series(rec["series"])
        rows.append(
            {
                "id": rec["id"],
                "image": f"images/{img_name}",
                "study_sid": sid,
                "study_name": rec["study_name"],
                "pmid": rec.get("pmid"),
                "reference_title": rec.get("reference_title"),
                "substances": rec.get("substances"),
                "figure": rec["figure"],
                # ---- INPUT: all paper content ----
                "paper_content_type": ctype,
                "paper_text": text,
                "paper_pdf": pdf_rel,
                "paper_md": md_rel,
                # ---- TARGET: timecourses digitized from this figure ----
                "n_timecourses": len(tcs),
                "timecourses": tcs,
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "benchmark.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # stats
    n_full = sum(1 for r in rows if r["paper_content_type"] == "fulltext_pdf")
    n_abs = sum(1 for r in rows if r["paper_content_type"] == "abstract")
    n_tc = sum(r["n_timecourses"] for r in rows)
    n_pts = sum(len(t["points"]) for r in rows for t in r["timecourses"])
    n_studies = len({r["study_sid"] for r in rows})
    print(f"benchmark entries (figures): {len(rows)}")
    print(f"  distinct studies: {n_studies}")
    print(f"  paper content: fulltext_pdf={n_full}  abstract={n_abs}")
    print(f"  timecourses: {n_tc}  points: {n_pts}")
    write_readme(len(rows), n_studies, n_full, n_abs, n_tc, n_pts)


def write_readme(n, n_studies, n_full, n_abs, n_tc, n_pts) -> None:
    md = f"""# PK-DB Graph → Timecourse Benchmark

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
{{
  "id": "PKDB00024__Fig1",
  "image": "images/PKDB00024__Fig1.png",
  "study_sid": "PKDB00024", "study_name": "Baraka1990", "pmid": "2306420",
  "reference_title": "...", "substances": [...], "figure": "Fig1",
  "paper_content_type": "fulltext_pdf",
  "paper_text": "<entire paper text>",
  "paper_pdf": "papers/PKDB00024.pdf", "paper_md": "papers/PKDB00024.md",
  "n_timecourses": 4,
  "timecourses": [
    {{"substance": "paracetamol", "intervention": "paracetamol", "tissue": "plasma",
      "group": "all", "count": 10, "measurement_type": "concentration",
      "time_unit": "hr", "value_unit": "microg/ml",
      "points": [{{"time": 0.5, "mean": 18.05, "sd": 4.49, "se": 1.42}}, ...]}}
  ]
}}
```

## Statistics

- Entries (figures): **{n}**  across **{n_studies}** studies
- Paper content: **{n_full}** full-text (from PDF), **{n_abs}** abstract-only
- Timecourses (curves): **{n_tc}** · (time, value) points: **{n_pts}**

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
"""
    (OUT / "README.md").write_text(md)


if __name__ == "__main__":
    main()
