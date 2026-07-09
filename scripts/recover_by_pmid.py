"""Second-pass recovery: match the still-missing open studies to live PK-DB by PMID.

The 2021 dump and the live site disagree on ~20 study *names* (e.g. dump
``Healy1989`` vs live ``Healy1991``), so name lookup fails. The PMID is stable, so
we match on it, then download that live study's figure/table PNGs + digitized
``.tsv`` files (using their real manifest URLs).

Updates data/manifest.json in place and appends to data/media/<live_name>/.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MEDIA_DIR = ROOT / "data" / "media"
MANIFEST = ROOT / "data" / "manifest.json"
API = "https://pk-db.com/api/v1/"
HOST = "https://pk-db.com"
UA = "pkdb-graph-data/1.0 (+shlok@gxl.ai)"


def api(path: str, **params) -> dict:
    params.setdefault("format", "json")
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def classify(name: str) -> str:
    base = name.rsplit("/", 1)[-1]
    stem = base[1:] if base.startswith(".") else base
    if re.search(r"_Fig", stem, re.I):
        return "figure"
    if re.search(r"_Tab", stem, re.I):
        return "table"
    if stem.lower().endswith(".pdf"):
        return "pdf"
    if stem.lower().endswith(".xlsx"):
        return "xlsx"
    return "other"


def file_records(files: list[dict]) -> list[dict]:
    recs = []
    for f in files or []:
        name = f.get("name", "")
        path = f.get("file", "")
        recs.append(
            {
                "pk": f.get("pk"),
                "name": name,
                "url": HOST + path if path.startswith("/") else path,
                "kind": classify(name),
            }
        )
    return recs


def main() -> None:
    st = pd.read_csv(RAW / "studies.csv", low_memory=False)
    manifest = json.loads(MANIFEST.read_text())

    # live pmid -> record (records carry files + reference)
    d = api("studies/", page_size=2000)
    data = d.get("data", d)
    rows = data.get("data") if isinstance(data, dict) else data
    pmid2rec = {}
    for r in rows:
        ref = r.get("reference") or {}
        pmid = str(ref.get("pmid")) if ref.get("pmid") else None
        if pmid:
            pmid2rec[pmid] = r
    print(f"live studies indexed by pmid: {len(pmid2rec)}")

    missing = [sid for sid, m in manifest.items() if "files" not in m]
    print(f"still-missing studies: {len(missing)}")
    recovered = 0
    for sid in missing:
        row = st[st.sid == sid].iloc[0]
        pmid = row.get("reference_pmid")
        pmid = str(int(pmid)) if pd.notna(pmid) else None
        rec = pmid2rec.get(pmid) if pmid else None
        if not rec:
            print(f"  no-pmid-match {sid} (pmid={pmid}, dumpname={row['name']})")
            continue
        ref = rec.get("reference") or {}
        files = rec.get("files") or []
        manifest[sid] = {
            "sid": sid,
            "current_sid": rec.get("sid"),
            "name": rec.get("name"),  # live name -> matches media filenames
            "dump_name": row["name"],
            "licence": "open",
            "pmid": ref.get("pmid"),
            "reference_title": ref.get("title"),
            "timecourse_count": rec.get("timecourse_count"),
            "scatter_count": rec.get("scatter_count"),
            "output_count": rec.get("output_count"),
            "substances": rec.get("substances"),
            "files": file_records(files),
        }
        nfig = sum(1 for r in manifest[sid]["files"] if r["kind"] == "figure")
        print(
            f"  RECOVERED {sid} -> {rec.get('sid')} live={rec.get('name')} "
            f"(dump={row['name']}) figs~={nfig}"
        )
        recovered += 1

    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"\nrecovered {recovered}/{len(missing)} by pmid")

    # download media for the newly recovered studies
    dl_ok = dl_err = 0
    for sid in missing:
        m = manifest.get(sid, {})
        if "files" not in m:
            continue
        study_dir = MEDIA_DIR / m["name"]
        study_dir.mkdir(parents=True, exist_ok=True)
        for r in m["files"]:
            if r["kind"] not in ("figure", "table"):
                continue
            base = r["name"].rsplit("/", 1)[-1]
            dest = study_dir / base
            if dest.exists() and dest.stat().st_size > 0:
                continue
            try:
                dest.write_bytes(fetch_bytes(r["url"]))
                dl_ok += 1
            except Exception as e:  # noqa: BLE001
                dl_err += 1
                print(f"    ERR {base}: {e}")
            time.sleep(0.15)
    print(f"download: {dl_ok} ok, {dl_err} err")


if __name__ == "__main__":
    main()
