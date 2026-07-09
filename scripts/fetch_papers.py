"""Fetch paper content for every open study that has timecourses.

For each study PMID we save markdown via the `pubmed-markdown` package: PMC
open-access full text when available, otherwise the abstract (which still carries
title / authors / journal / DOI and usually the dose, n, and design). We also
pull the PK-DB-hosted PDF where the study manifest lists one.

Outputs:
    study_dataset/papers/<sid>.md        paper markdown (fulltext or abstract)
    study_dataset/papers/<sid>.pdf       original PDF (if PK-DB serves it)
    study_dataset/paper_index.json       {sid: {pmid, has_fulltext, md, pdf}}
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.json"
OUT = ROOT / "study_dataset"
PAPERS = OUT / "papers"
UA = "pkdb-graph-data/1.0 (+shlok@gxl.ai)"


def load_env_email() -> None:
    if os.environ.get("NCBI_EMAIL"):
        return
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            m = re.match(r"\s*NCBI_EMAIL\s*=\s*\"?([^\"\n]+)\"?", line)
            if m:
                os.environ["NCBI_EMAIL"] = m.group(1).strip()


def get_paper_markdown(pmid: str) -> tuple[str | None, bool]:
    """Return (markdown, has_fulltext). Falls back to abstract."""
    from pubmed_markdown import (
        get_abstract_markdown_from_pmid,
        get_html_from_pmcid,
        get_pmcid_from_pmid,
        markdown_from_html,
    )

    # try PMC open-access full text
    try:
        res = get_pmcid_from_pmid(pmid)
        pmcid = res.get(pmid) if isinstance(res, dict) else res
        if pmcid:
            html = get_html_from_pmcid(pmcid)
            if html:
                md = markdown_from_html(html)
                if md and len(md) > 500:
                    return md, True
    except Exception:
        pass
    # fall back to abstract
    try:
        return get_abstract_markdown_from_pmid(pmid), False
    except Exception:
        return None, False


def studies_with_timecourses() -> pd.DataFrame:
    """Open studies that have timecourses in the dump OR figures with digitized data.

    Some studies (e.g. Prescott1980) carry zero dump timecourses but still have
    figure-digitized series, so we union in every study referenced by the
    figure-level dataset.
    """
    st = pd.read_csv(RAW / "studies.csv", low_memory=False)
    tc = pd.read_csv(RAW / "timecourses.csv", low_memory=False)
    open_sids = set(st[st.licence == "open"].sid)
    sids = set(tc[tc.study_sid.isin(open_sids)].study_sid.unique())

    ann = ROOT / "dataset" / "annotations.jsonl"
    if ann.exists():
        for line in ann.read_text().splitlines():
            if line.strip():
                sids.add(json.loads(line)["study_sid"])
    return st[st.sid.isin(sids)][["sid", "name", "reference_pmid"]].reset_index(
        drop=True
    )


def main() -> None:
    load_env_email()
    PAPERS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    df = studies_with_timecourses()
    print(f"{len(df)} open studies with timecourses")

    index = {}
    for i, row in df.iterrows():
        sid, name = row["sid"], row["name"]
        pmid = (
            str(int(row["reference_pmid"])) if pd.notna(row["reference_pmid"]) else None
        )
        rec = {
            "sid": sid,
            "name": name,
            "pmid": pmid,
            "has_fulltext": False,
            "md": None,
            "pdf": None,
        }

        if pmid:
            md, full = get_paper_markdown(pmid)
            if md:
                (PAPERS / f"{sid}.md").write_text(md)
                rec["md"] = f"papers/{sid}.md"
                rec["has_fulltext"] = full

        # PK-DB PDF if listed
        m = manifest.get(sid, {})
        pdf = next((f for f in m.get("files", []) if f.get("kind") == "pdf"), None)
        if pdf:
            try:
                req = urllib.request.Request(pdf["url"], headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=120) as r:
                    (PAPERS / f"{sid}.pdf").write_bytes(r.read())
                rec["pdf"] = f"papers/{sid}.pdf"
            except Exception as e:  # noqa: BLE001
                print(f"  pdf ERR {sid}: {e}")

        index[sid] = rec
        tag = "FULL" if rec["has_fulltext"] else ("abs" if rec["md"] else "NONE")
        print(
            f"[{i + 1}/{len(df)}] {sid} {name:<20} pmid={pmid} paper={tag} pdf={'y' if rec['pdf'] else '-'}"
        )
        time.sleep(0.34)  # NCBI politeness (<=3 req/s)

    (OUT / "paper_index.json").write_text(json.dumps(index, indent=2))
    full = sum(1 for r in index.values() if r["has_fulltext"])
    absr = sum(1 for r in index.values() if r["md"] and not r["has_fulltext"])
    pdfs = sum(1 for r in index.values() if r["pdf"])
    print(f"\nDONE: {len(index)} studies | fulltext={full} abstract={absr} pdf={pdfs}")


if __name__ == "__main__":
    main()
