"""Recover 404 studies by name and download all figure/table images + digitized TSVs.

The 2021 CSV dump uses SIDs that the live PK-DB has partly re-numbered, so
``studies/{sid}/`` 404s for some open studies. Media files, however, are named by
study *name* (e.g. ``Baraka1990_Fig1.png``), which is stable. We therefore:

  1. Pull the live study list once -> {name: current_sid}.
  2. For every open study whose original manifest entry errored, re-fetch its
     detail via the current sid and splice the files in.
  3. Download every figure/table PNG and its sibling digitized ``.tsv`` into
     data/media/<name>/.

Writes an updated data/manifest.json and data/download_log.json.
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


def file_records(d: dict) -> list[dict]:
    recs = []
    for f in d.get("files") or []:
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


def entry_from_detail(d: dict) -> dict:
    ref = d.get("reference") or {}
    return {
        "sid": d.get("sid"),
        "name": d.get("name"),
        "licence": d.get("licence"),
        "pmid": ref.get("pmid") if isinstance(ref, dict) else None,
        "reference_title": ref.get("title") if isinstance(ref, dict) else None,
        "timecourse_count": d.get("timecourse_count"),
        "scatter_count": d.get("scatter_count"),
        "output_count": d.get("output_count"),
        "substances": d.get("substances"),
        "files": file_records(d),
    }


def main() -> None:
    st = pd.read_csv(RAW / "studies.csv", low_memory=False)
    manifest = json.loads(MANIFEST.read_text())

    # 1. live name -> sid map
    d = api("studies/", page_size=2000)
    data = d.get("data", d)
    rows = data.get("data") if isinstance(data, dict) else data
    name2sid = {r.get("name"): r.get("sid") for r in rows}
    print(f"live studies: {len(rows)}")

    # 2. recover errored entries by name
    missing = [sid for sid, m in manifest.items() if "error" in m]
    for sid in missing:
        nm = st.loc[st.sid == sid, "name"].iloc[0]
        cur = name2sid.get(nm)
        if not cur:
            print(f"  UNRECOVERABLE {sid} ({nm})")
            continue
        try:
            detail = api(f"studies/{cur}/")
            manifest[sid] = entry_from_detail(detail)
            manifest[sid]["original_sid"] = sid
            manifest[sid]["current_sid"] = cur
            nfig = sum(1 for r in manifest[sid]["files"] if r["kind"] == "figure")
            print(f"  RECOVERED {sid} -> {cur} ({nm}) figs={nfig}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {sid} -> {cur} ({nm}): {e}")
        time.sleep(0.3)

    MANIFEST.write_text(json.dumps(manifest, indent=2))

    # 3. download media
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    log = []
    for sid, m in manifest.items():
        if "files" not in m:
            continue
        name = m["name"]
        study_dir = MEDIA_DIR / name
        study_dir.mkdir(parents=True, exist_ok=True)
        for r in m["files"]:
            if r["kind"] not in ("figure", "table"):
                continue  # skip pdf/xlsx/other for the image dataset
            base = r["name"].rsplit("/", 1)[-1]
            dest = study_dir / base
            if dest.exists() and dest.stat().st_size > 0:
                log.append({"sid": sid, "file": base, "status": "cached"})
                continue
            try:
                blob = fetch_bytes(r["url"])
                dest.write_bytes(blob)
                log.append(
                    {"sid": sid, "file": base, "status": "ok", "bytes": len(blob)}
                )
            except Exception as e:  # noqa: BLE001
                log.append({"sid": sid, "file": base, "status": f"ERR {e}"})
            time.sleep(0.15)

        # also grab the hidden digitized tsv for each figure/table
        for r in m["files"]:
            if r["kind"] not in ("figure", "table"):
                continue
            base = r["name"].rsplit("/", 1)[-1]
            stem = base.rsplit(".", 1)[0]
            tsv_url = f"{HOST}/media/data/.{stem}.tsv"
            dest = study_dir / f".{stem}.tsv"
            if dest.exists() and dest.stat().st_size > 0:
                continue
            try:
                blob = fetch_bytes(tsv_url)
                dest.write_bytes(blob)
                log.append(
                    {"sid": sid, "file": f".{stem}.tsv", "status": "ok", "bytes": len(blob)}
                )
            except Exception as e:  # noqa: BLE001
                log.append({"sid": sid, "file": f".{stem}.tsv", "status": f"ERR {e}"})
            time.sleep(0.15)

    (ROOT / "data" / "download_log.json").write_text(json.dumps(log, indent=2))
    ok = sum(1 for x in log if x["status"] in ("ok", "cached"))
    err = sum(1 for x in log if str(x["status"]).startswith("ERR"))
    print(f"\ndownload: {ok} ok/cached, {err} errors, {len(log)} total attempts")


if __name__ == "__main__":
    main()
