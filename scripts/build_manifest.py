"""Build a file manifest for PK-DB open-licence studies.

For each open study we fetch its detail record from the public PK-DB API and
record every associated /media file. Of particular interest:

  - ``*_Fig*.png`` / ``*_Tab*.png``  : the cropped figure/table images
  - ``.*_Fig*.tsv`` / ``.*_Tab*.tsv`` : the digitized data behind each image
  - ``*.pdf`` / ``*.xlsx``            : full paper + curation workbook

The digitized TSV is the ground-truth annotation for its sibling PNG, which is
exactly the (figure image <-> structured time-series) pairing we want for VLM
training.

Output: data/manifest.json
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "manifest.json"

API = "https://pk-db.com/api/v1/"
MEDIA = "https://pk-db.com"
UA = "pkdb-graph-data/1.0 (+shlok@gxl.ai)"


def api(path: str, **params) -> dict:
    params.setdefault("format", "json")
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def open_study_sids() -> list[str]:
    import pandas as pd

    st = pd.read_csv(RAW / "studies.csv", low_memory=False)
    return sorted(st.loc[st.licence == "open", "sid"].tolist())


def classify(name: str) -> str:
    base = name.rsplit("/", 1)[-1]
    stem = base[1:] if base.startswith(".") else base
    if re.search(r"_Fig", stem, re.I):
        kind = "figure"
    elif re.search(r"_Tab", stem, re.I):
        kind = "table"
    elif stem.lower().endswith(".pdf"):
        return "pdf"
    elif stem.lower().endswith(".xlsx"):
        return "xlsx"
    else:
        return "other"
    return kind


def main() -> None:
    sids = open_study_sids()
    print(f"{len(sids)} open studies", file=sys.stderr)
    manifest = {}
    for i, sid in enumerate(sids, 1):
        try:
            d = api(f"studies/{sid}/")
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(sids)}] {sid} ERROR {e}", file=sys.stderr)
            manifest[sid] = {"error": str(e)}
            continue
        ref = d.get("reference") or {}
        files = d.get("files") or []
        recs = []
        for f in files:
            name = f.get("name", "")
            path = f.get("file", "")
            recs.append(
                {
                    "pk": f.get("pk"),
                    "name": name,
                    "url": MEDIA + path if path.startswith("/") else path,
                    "kind": classify(name),
                }
            )
        manifest[sid] = {
            "sid": sid,
            "name": d.get("name"),
            "licence": d.get("licence"),
            "pmid": (ref.get("pmid") if isinstance(ref, dict) else None),
            "reference_title": (ref.get("title") if isinstance(ref, dict) else None),
            "timecourse_count": d.get("timecourse_count"),
            "scatter_count": d.get("scatter_count"),
            "output_count": d.get("output_count"),
            "substances": d.get("substances"),
            "files": recs,
        }
        nfig = sum(1 for r in recs if r["kind"] == "figure")
        print(
            f"[{i}/{len(sids)}] {sid} {d.get('name'):<18} figs={nfig} files={len(recs)}",
            file=sys.stderr,
        )
        time.sleep(0.3)  # be polite

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2))

    # summary
    tot_fig = sum(
        sum(1 for r in m.get("files", []) if r["kind"] == "figure")
        for m in manifest.values()
    )
    tot_tab = sum(
        sum(1 for r in m.get("files", []) if r["kind"] == "table")
        for m in manifest.values()
    )
    print(
        f"\nDONE: {len(manifest)} studies, {tot_fig} figure images, {tot_tab} table images",
        file=sys.stderr,
    )
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
