# PK-DB Data Sources

How the PK-DB Viewer gets its data, why we download a GitHub artifact instead of
calling the live API, and how to reproduce the check that the API is still
missing the data we need.

> **TL;DR** — The public PK-DB API returns the *relational* entities (studies,
> groups, individuals, interventions) but returns the *measurement* data
> (**outputs, timecourses, scatters**) **empty** to anonymous callers. Those are
> the whole point of the viewer, so we ingest instead from a curated CSV dump the
> PK-DB maintainers publish on GitHub. Verified still-missing on **2026-07-08**
> (see [Verification](#verification)).

---

## The two ingest paths

There are two ingesters in `ingest/`. **`ingest_dump.py` is the primary one** and
produces the data the viewer actually ships.

| | `ingest/ingest_dump.py` (PRIMARY) | `ingest/ingest.py` (legacy) |
|---|---|---|
| Source of study design | GitHub CSV dump | live PK-DB API |
| Source of outputs / timecourses / scatters | **GitHub CSV dump** | ❌ not available |
| Source of reference + paper text | NCBI (reused from `ingest.py`) | NCBI |
| Status | current | kept only for its NCBI paper/reference pipeline, which `ingest_dump.py` imports |

Both write the same on-disk layout the viewer reads directly:

```
public/data/index.json          # study list for the picker
public/data/<sid>/study.json     # all extracted data for one study
public/data/<sid>/paper.md       # the original paper as markdown
```

---

## Method 1 — GitHub CSV dump (what we use)

The measurement data is downloaded as a single zip from the PK-DB maintainers'
analysis repository:

```
https://raw.githubusercontent.com/matthiaskoenig/pkdb_analysis/develop/tests/data/testdata_concise_false.zip
```

Human-browsable page:

```
https://github.com/matthiaskoenig/pkdb_analysis/blob/develop/tests/data/testdata_concise_false.zip
```

Facts about the artifact:

- **Repo / path:** `matthiaskoenig/pkdb_analysis` @ `develop` → `tests/data/testdata_concise_false.zip`
- **What it is:** a `pkdb_analysis` export — **8 CSVs**: `studies`, `groups`,
  `individuals`, `interventions`, `outputs`, `timecourses`, `scatters`,
  `info_nodes`
- **Coverage:** **661 studies** total; the viewer uses the **56 `licence=open`** ones
- **Snapshot date:** **2021-12-03** (`SNAPSHOT` constant in `ingest_dump.py`)
- **Size:** ~6.7 MB
- **Local cache:** `ingest/.cache_testdata_concise_false.zip` — **gitignored**
  (`.gitignore` line: `ingest/.cache_*.zip`). `load_zip()` reuses the cache if it
  exists and is > 1 MB, otherwise re-downloads and rewrites it.

Where this lives in code: `ingest/ingest_dump.py`, `ZIP_URL` (lines 57–60),
`SNAPSHOT` (line 61), `load_zip()` (lines 83–99).

Run it:

```bash
python ingest/ingest_dump.py               # all 56 open studies
python ingest/ingest_dump.py --limit 3     # smoke test
python ingest/ingest_dump.py --sid PKDB00024
python ingest/ingest_dump.py --no-paper    # skip NCBI paper fetch; data only
```

### ⚠️ The URL is not pinned

`ZIP_URL` points at the `develop` **branch**, not a commit SHA or tag. If that
repo rewrites history, moves, or deletes the file, a fresh checkout with no local
cache cannot reproduce the data. For durability, either pin the URL to a specific
commit SHA or commit the zip into this repo (drop the `.gitignore` line).

---

## Method 2 — Live PK-DB API (why it is NOT enough)

- **Base URL:** `https://pk-db.com/api/v1/`
- **Docs / browsable:** `https://pk-db.com/api/v1/` (DRF browsable API)

The anonymous API **does** serve:

- `studies/`, `studies/{sid}/` — metadata + counts
- `groups/`, `individuals/`, `interventions/` — the relational entities
  (fetched via a `filter/?studies__sid=<sid>&concise=false` → `uuid`, then paged)

The anonymous API **does NOT** serve the measurement data:

- `outputs/` returns **`count: 0`** and **zero rows** — even for studies whose own
  `studies/{sid}/` metadata advertises hundreds of outputs.
- `timecourses/` and `scatters/` return **HTTP 404** for anonymous callers.

Root cause: PK-DB serves outputs/timecourses/scatters from an Elasticsearch
backend that returns nothing without a curator login. The relational entities
come from the relational DB and respond fine. This is why the legacy
`ingest/ingest.py` can build a study's *design* but never its *findings*.

---

## Verification

Re-run this any time to confirm the API is still missing the measurement data.
It queries a study whose metadata advertises outputs/timecourses/scatters and
shows what the entity endpoints actually return.

```bash
python3 - <<'PY'
import json, urllib.parse, urllib.request
API="https://pk-db.com/api/v1/"
UA="pkdb-viewer-verify/1.0 (+you@example.com)"
def api(path,**p):
    p.setdefault("format","json")
    req=urllib.request.Request(f"{API}{path}?{urllib.parse.urlencode(p)}",headers={"User-Agent":UA})
    with urllib.request.urlopen(req,timeout=60) as r:
        return json.loads(r.read().decode("utf-8","replace"))

sid="PKDB00954"  # metadata claims: 669 outputs, 10 timecourses, 18 scatters
uuid=api("filter/", **{"studies__sid":sid,"concise":"false"}).get("uuid")
for ep,extra in [("groups",{}),("individuals",{}),("interventions",{"normed":"true"}),
                 ("outputs",{"normed":"true"}),("timecourses",{}),("scatters",{})]:
    try:
        d=api(ep+"/", uuid=uuid, page_size=5, **extra)["data"]
        print(f"{ep:14s} count={d.get('count')} rows={len(d.get('data') or [])}")
    except Exception as e:
        print(f"{ep:14s} ERROR {e}")
PY
```

**Result on 2026-07-08** (study `PKDB00954`, metadata advertises 669 outputs /
10 timecourses / 18 scatters):

```
groups         count=9   rows=5      ✅ relational entities returned
individuals    count=248 rows=5      ✅
interventions  count=6   rows=5      ✅
outputs        count=0   rows=0      ❌ empty despite claimed 669
timecourses    ERROR HTTP Error 404  ❌ endpoint unavailable to anonymous
scatters       ERROR HTTP Error 404  ❌ endpoint unavailable to anonymous
```

Conclusion: **the measurement data is still missing from the anonymous API**, so
the GitHub dump remains necessary.

---

## Data provenance notes (from investigating the dump)

Useful when reasoning about what the numbers mean:

- **Outputs** carry a `calculated` boolean in the raw dump (dropped by our
  `build_outputs()` today):
  - `calculated=False` (~87%) — the value **as reported by the paper's authors**,
    transcribed by a human curator.
  - `calculated=True` (~13%) — **derived by PK-DB's automated pharmacokinetics
    pipeline** (`pkdb_analysis`, non-compartmental analysis) from the
    concentration–time curves. Exactly 9 parameter types: `auc-end`, `auc-inf`,
    `cmax`, `tmax`, `kel`, `thalf`, `clearance`, `vd`, `vd-ss`. Each links back to
    the `intervention_pk` + `group_pk`/`individual_pk` it was computed for.
- **Timecourses** are, in this corpus, **digitized from figures** (line plots),
  not transcribed from tables. Of 16 open studies with readable full text, 15
  verified as figure-digitized and 1 was undeterminable (abstract-only scan); in
  every paper the concentration–time series lived only in figures while the
  tables held summary PK parameters. (Round sampling times like 0.5/1/2/4 h do
  NOT imply a table — the curves were simply sampled at nominal blood-draw times.)

---

## Attribution & terms

PK-DB: Grzegorzewski et al., *Nucleic Acids Res.* 2021,
[doi:10.1093/nar/gkaa990](https://doi.org/10.1093/nar/gkaa990).
The dump's `TERMS_OF_USE.md` states no restriction beyond the original data
owners; **attribution is required**.

