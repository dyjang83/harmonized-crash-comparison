"""Fetch NHTSA Crash Report Sampling System (CRSS) annual CSV releases.

CRSS is a nationally representative probability sample of POLICE-REPORTED crashes.
It is the human-driver comparator for the SGO corpus: same regulator, same crash
concepts, coded rather than narrated (CRSS carries no crash narrative, so the
extraction pipeline cannot be run on it -- the comparison is between our predicted
AV severity distribution and CRSS's coded one).

The two populations are NOT the same by construction. SGO's mandatory-filing
threshold sits far below the bar at which a human crash generates a police report,
so the raw corpora are not comparable and no amount of reweighting makes them so.
See models/av_vs_human.py for the severity-threshold harmonization that defines a
common population before anything is compared.

CRSS is a COMPLEX SURVEY SAMPLE. Every estimate must use the case weight
(WEIGHT), and variance must come from resampling PSUs within strata (PSU,
PSUSTRAT) -- an unweighted count of CRSS rows is not an estimate of anything.

URLs are static and permanent (verified August 2026). Usage:
    python -m fetch.fetch_crss                 # 2021-2024
    python -m fetch.fetch_crss --years 2023
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile

import requests

HEADERS = {"User-Agent": "av-crash-nlp research fetcher (contact: dyjang83@github)"}
OUT_DIR = os.path.join("data", "raw", "crss")

DEFAULT_YEARS = [2021, 2022, 2023, 2024]

# CRSS annual CSV bundle. ~50 MB each; contains accident.csv, vehicle.csv,
# person.csv and a set of supplementary files.
CSV_URL = "https://static.nhtsa.gov/nhtsa/downloads/CRSS/{year}/CRSS{year}CSV.zip"

# The Analytical User's Manual. The variable crosswalk in schema/crss_map.py must
# be checked against this rather than against recollection -- CRSS renames and
# recodes variables between releases, exactly as SGO did in its third amendment.
MANUAL_URL = ("https://static.nhtsa.gov/nhtsa/downloads/CRSS/"
              "Links%20for%20CRSS%20Manuals.pdf")


def download(url: str, dest: str, skip_existing: bool = True) -> str | None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if skip_existing and os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[crss] skip (exists): {dest}")
        return dest
    print(f"[crss] downloading {os.path.basename(dest)} ...")
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
    except requests.HTTPError as e:
        print(f"[crss] ERROR {url}: {e}", file=sys.stderr)
        # A partial file is worse than none: it would be skipped as "exists" on
        # the next run and then fail deep inside the crosswalk.
        if os.path.exists(dest):
            os.remove(dest)
        return None
    print(f"[crss] -> {dest} ({os.path.getsize(dest):,} bytes)")
    return dest


def extract(zip_path: str, year: int, force: bool = False) -> str:
    """Unzip one annual bundle into data/raw/crss/<year>/."""
    out = os.path.join(OUT_DIR, str(year))
    if os.path.isdir(out) and os.listdir(out) and not force:
        print(f"[crss] skip (extracted): {out}")
        return out
    os.makedirs(out, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        # Flatten: some releases nest the CSVs under a directory, some do not.
        for member in z.namelist():
            if member.endswith("/"):
                continue
            target = os.path.join(out, os.path.basename(member))
            with z.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
    names = sorted(os.listdir(out))
    print(f"[crss] extracted {len(names)} files to {out}/")
    print(f"[crss]   {', '.join(names[:8])}{' ...' if len(names) > 8 else ''}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Fetch NHTSA CRSS annual CSV releases.")
    ap.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    ap.add_argument("--no-manual", action="store_true",
                    help="Skip the Analytical User's Manual PDF.")
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-download and re-extract even if files exist.")
    a = ap.parse_args()

    got = []
    for year in a.years:
        dest = os.path.join(OUT_DIR, f"CRSS{year}CSV.zip")
        p = download(CSV_URL.format(year=year), dest, skip_existing=not a.force)
        if p and not a.no_extract:
            extract(p, year, force=a.force)
        if p:
            got.append(year)

    if not a.no_manual:
        download(MANUAL_URL, os.path.join(OUT_DIR, "CRSS_Manuals.pdf"),
                 skip_existing=not a.force)

    print(f"\n[crss] done. {len(got)}/{len(a.years)} year(s) available in {OUT_DIR}/")
    if not got:
        print("[crss] Nothing downloaded. Check your network connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
