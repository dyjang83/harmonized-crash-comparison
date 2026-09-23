"""AV (SGO) vs. human (CRSS) crash structure, on a harmonized population.

THE PROBLEM THIS MODULE EXISTS TO SOLVE. SGO and CRSS do not sample the same
crashes. SGO's Standing General Order compels a filing for qualifying ADS/ADAS
crashes at a threshold far below the bar at which a human crash generates a
police report, and CRSS samples police-reported crashes. Comparing the raw
corpora measures the difference in reporting rules at least as much as any
difference in crashes. No reweighting fixes this, because the populations
differ in what they CONTAIN, not merely in how they are sampled.

The approach taken here is SEVERITY-THRESHOLD HARMONIZATION: restrict both
sides to crashes at or above a common OUTCOME bar, so membership is defined by
what happened rather than by who was obliged to file. Two thresholds are
reported side by side --

    any-injury    CRSS MAXSEV_IM >= Possible Injury (C); SGO severity != none
    tow-away      CRSS TOWED on any vehicle;             SGO struct_towed

-- along with the retained N on each side, which IS the reportability argument
rather than a footnote to it.

PERIOD. Both sides are also restricted to the same incident years (--years).
The SGO corpus runs from 2020 to 2026 while CRSS ends at its latest annual
release, so an unrestricted AV side puts roughly half its crashes -- the 2025
and 2026 filings, which are also the ones from the most changed AV fleet --
against human years that do not exist in the comparison.

WHAT THIS DELIBERATELY DOES NOT DO. It does not compute crash RATES. A rate
needs an exposure denominator (vehicle miles travelled) that exists for neither
corpus here, and the underreporting corrections that rate comparisons require
(the IIHS/Waymo benchmarks apply a 32% adjustment for any-injury-reported) are
exactly the assumptions this design avoids needing. Everything below is
conditional on membership in the harmonized population: given that a crash
occurred and cleared the bar, how is it distributed?

SURVEY DESIGN. CRSS is a complex probability sample. Every human-side estimate
is WEIGHT-ed, and variance comes from resampling PSUs within strata, not from
an iid bootstrap -- an iid resample ignores clustering and reports intervals
that are far too narrow. The SGO side is a census of filings, not a sample, so
its uncertainty is an ordinary bootstrap over records.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Callable, Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema.crss_map import (FIELDS, SEVERITY_COMMON, COARSENED,  # noqa: E402
                             crss_severity_common, sgo_severity_common,
                             WEIGHT_COL, PSU_COL, STRATUM_COL)
from utils.config import canonical_model  # noqa: E402

RESULTS = os.path.join("data", "processed")
TABLES = os.path.join("paper", "tables")
CRSS_DIR = os.path.join("data", "raw", "crss")
SGO_DIR = os.path.join("data", "raw", "sgo")
NARRATIVES = os.path.join("data", "interim", "narratives.jsonl")

THRESHOLDS = ["all", "any_injury", "tow_away"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_crss_years(years, severity_var: str = "MAXSEV_IM") -> pd.DataFrame:
    """Pool several CRSS annual releases into one frame.

    WHY POOL. Comparing the SGO corpus against a single human year introduces a
    temporal mismatch that no weighting fixes, and leaves the fatal cell resting
    on too few crashes to separate from the human rate. Pooling is sound here
    because the comparison is DISTRIBUTIONAL: a pooled share is the crash-
    volume-weighted average of the annual shares, so no denominator
    reconciliation is needed. It would NOT be sound for a rate.

    These years also define the comparison window on BOTH sides: load_sgo drops
    AV crashes outside them. The SGO corpus runs to 2026 and CRSS ends at its
    latest release, so leaving the AV side unrestricted -- as this module used
    to -- silently compared ~half the AV crashes against human years that do not
    exist.

    DESIGN. Each annual release is an independent sample, and a PSU identifier
    means a different thing in a different year. Both the PSU and the stratum
    are therefore namespaced by year, so the bootstrap resamples PSUs within
    (year, stratum) cells rather than pooling a 2021 PSU with its 2024 namesake.
    """
    frames = []
    for y in years:
        f = load_crss(y, severity_var=severity_var)
        f["_year"] = y
        # Namespace the design identifiers: same id, different year, different unit.
        f["_psu"] = f"{y}_" + f["_psu"].astype(str)
        f["_stratum"] = f"{y}_" + f["_stratum"].astype(str)
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    print(f"[av_vs_human] pooled CRSS {list(years)}: {len(out):,} crashes, "
          f"weighted {out['_w'].sum():,.0f} ({out['_w'].sum()/len(years):,.0f}/yr avg)")
    return out


def load_crss(year: int = 2023, severity_var: str = "MAXSEV_IM") -> pd.DataFrame:
    """Crash-level CRSS frame with severity, thresholds, design vars and codes.

    severity_var defaults to the IMPUTED MAXSEV_IM rather than MAX_SEV: NHTSA
    imputes the ~2% of crashes coded 'Unknown/Not Reported', and dropping those
    instead would silently condition the comparison on severity being known,
    which is not independent of severity. Pass MAX_SEV for a sensitivity run.
    """
    d = os.path.join(CRSS_DIR, str(year))
    acc = pd.read_csv(os.path.join(d, "accident.csv"), low_memory=False)
    acc["sev"] = acc[f"{severity_var}NAME"].map(crss_severity_common)

    # Tow-away rolls up from the vehicle file: a crash is tow-away if ANY vehicle
    # was towed, matching SGO's 'Was Any Vehicle Towed?'. Note TOWED is the
    # disposition of THIS vehicle; TOW_VEH is about trailers and is not it.
    veh = pd.read_csv(os.path.join(d, "vehicle.csv"), low_memory=False,
                      usecols=["CASENUM", "TOWEDNAME"])
    towed = (veh.assign(t=veh["TOWEDNAME"].astype(str).str.strip().eq("Towed"))
                .groupby("CASENUM")["t"].any())
    acc["tow_away"] = acc["CASENUM"].map(towed).fillna(False)

    acc["any_injury"] = acc["sev"].isin(["minor", "serious", "fatal"])
    acc["all"] = True
    acc["_w"] = acc[WEIGHT_COL].astype(float)
    acc["_psu"] = acc[PSU_COL]
    acc["_stratum"] = acc[STRATUM_COL]
    return acc


def sgo_incident_years(sgo_dir: str = SGO_DIR) -> pd.Series:
    """Report ID -> incident year, read from the raw SGO CSVs.

    narratives.jsonl carries no date field, so the year has to come back from
    the raw CSVs. It is read through parse_ol316._load_sgo_frame rather than a
    fresh join so that it uses the same collapse to one row per Report ID
    (highest Report Version) that produced the narratives -- otherwise the year
    could be taken from a superseded version of the report whose narrative is
    not the one in the corpus.

    NHTSA publishes 'Incident Date' as MON-YYYY; the day is coarsened away.
    """
    from fetch.parse_ol316 import _load_sgo_frame
    paths = sorted(glob.glob(os.path.join(sgo_dir, "*Incident_Reports*.csv")))
    df = _load_sgo_frame(paths)
    cols = {c.lower().strip(): c for c in df.columns}
    c_id, c_date = cols.get("report id"), cols.get("incident date")
    if c_id is None or c_date is None:
        raise RuntimeError(f"SGO CSVs in {sgo_dir} lack Report ID / Incident Date")
    year = pd.to_datetime(df[c_date], format="%b-%Y", errors="coerce").dt.year
    return pd.Series(year.to_numpy(), index=df[c_id].astype(str).to_numpy())


def load_sgo(narratives: str, years) -> pd.DataFrame:
    """AV-side frame, restricted to incident years `years`.

    `years` is required, not optional: it is the same list that selects the CRSS
    releases, so the two sides cannot drift out of alignment. Crashes whose
    incident date is missing or unparseable are dropped -- they cannot be placed
    in the window, and keeping them would reopen the mismatch on a smaller scale.
    """
    df = pd.read_json(narratives, lines=True)
    df = df[df["source"] == "sgo"].copy()

    yr = sgo_incident_years()
    df["_year"] = df["report_id"].astype(str).map(yr)
    n_all = len(df)
    n_undated = int(df["_year"].isna().sum())
    df = df[df["_year"].isin(list(years))].copy()
    print(f"[av_vs_human] SGO restricted to {list(years)}: {len(df):,} of "
          f"{n_all:,} crashes ({n_all - len(df):,} outside the window, of which "
          f"{n_undated:,} undated)")

    df["sev"] = df["struct_severity"].map(sgo_severity_common)
    df["any_injury"] = df["sev"].isin(["minor", "serious", "fatal"])
    tow = df.get("struct_towed")
    df["tow_away"] = (tow.astype(str).str.strip().str.lower().isin({"y", "yes", "true"})
                      if tow is not None else False)
    df["all"] = True
    df["_w"] = 1.0            # a census of filings, not a sample
    return df


# ---------------------------------------------------------------------------
# Weighted estimates and design-aware resampling
# ---------------------------------------------------------------------------
def weighted_share(df: pd.DataFrame, col: str, levels: list) -> np.ndarray:
    w = df["_w"].to_numpy(float)
    v = df[col].to_numpy()
    tot = w.sum()
    if tot <= 0:
        return np.full(len(levels), np.nan)
    return np.array([w[v == lv].sum() / tot for lv in levels])


def psu_bootstrap(df: pd.DataFrame, stat: Callable[[pd.DataFrame], np.ndarray],
                  n_boot: int = 1000, seed: int = 11) -> tuple:
    """Resample PSUs WITH replacement within each stratum.

    This is the design-consistent bootstrap for a stratified cluster sample.
    Resampling rows instead would treat clustered observations as independent
    and understate the variance -- typically by a large factor for crash data,
    where PSUs are geographic and highly homogeneous internally.
    """
    rng = np.random.default_rng(seed)
    if "_psu" not in df.columns:      # SGO: census, ordinary bootstrap over rows
        idx = np.arange(len(df))
        vals = [stat(df.iloc[rng.choice(idx, len(idx), replace=True)])
                for _ in range(n_boot)]
    else:
        groups = {s: g["_psu"].unique() for s, g in df.groupby("_stratum")}
        by_psu = {(s, p): g for (s, p), g in df.groupby(["_stratum", "_psu"])}
        vals = []
        for _ in range(n_boot):
            parts = []
            for s, psus in groups.items():
                for p in rng.choice(psus, len(psus), replace=True):
                    parts.append(by_psu[(s, p)])
            vals.append(stat(pd.concat(parts, ignore_index=True)))
    a = np.vstack(vals)
    return (np.nanpercentile(a, 2.5, axis=0), np.nanpercentile(a, 97.5, axis=0))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare_severity(crss: pd.DataFrame, sgo: pd.DataFrame, threshold: str,
                     n_boot: int = 500) -> dict:
    c = crss[crss[threshold] & crss["sev"].notna()]
    s = sgo[sgo[threshold] & sgo["sev"].notna()]
    lv = SEVERITY_COMMON
    hc, hs = weighted_share(c, "sev", lv), weighted_share(s, "sev", lv)
    c_lo, c_hi = psu_bootstrap(c, lambda d: weighted_share(d, "sev", lv), n_boot)
    s_lo, s_hi = psu_bootstrap(s, lambda d: weighted_share(d, "sev", lv), n_boot)
    return {
        "threshold": threshold, "levels": lv,
        "n_crss": int(len(c)), "n_sgo": int(len(s)),
        "crss_weighted_n": float(c["_w"].sum()),
        "human_share": hc.tolist(), "human_ci": [c_lo.tolist(), c_hi.tolist()],
        "av_share": hs.tolist(), "av_ci": [s_lo.tolist(), s_hi.tolist()],
        "ratio": (hs / np.where(hc > 0, hc, np.nan)).tolist(),
    }


def compare_structure(crss: pd.DataFrame, sgo: pd.DataFrame, threshold: str
                      ) -> list[dict]:
    """Marginal distribution of each structural field, both sides, harmonized.

    Each side is projected independently into the shared space (crss_map.FIELDS);
    there is no pairing, because these are different crashes.
    """
    out = []
    c_all = crss[crss[threshold]]
    s_all = sgo[sgo[threshold]]
    for name, (crss_fn, av_fn, levels, schema_field) in FIELDS.items():
        cl = c_all.apply(lambda r: crss_fn(r.to_dict()), axis=1)
        cm = cl.notna()
        cdf = pd.DataFrame({"v": cl[cm], "_w": c_all.loc[cm, "_w"]})

        if schema_field not in s_all.columns:
            continue
        sl = s_all[schema_field].map(av_fn)
        sm = sl.notna()
        sdf = pd.DataFrame({"v": sl[sm], "_w": 1.0})
        if cdf.empty or sdf.empty:
            continue

        hc = weighted_share(cdf, "v", levels)
        hs = weighted_share(sdf, "v", levels)

        # AV-side abstention decides whether this field is comparable AT ALL.
        # The extractor answers `unknown` when the narrative does not state the
        # condition, and narratives state conditions SELECTIVELY -- weather gets
        # written down when it is remarkable. So the surviving AV rows for a
        # high-abstention field are not a random subsample of AV crashes, they
        # are the subsample whose narrative mentioned the thing, and their
        # distribution is shifted toward exactly the values that prompt a
        # mention. Comparing that against a police-coded human distribution,
        # which is coded for every crash regardless, measures narration habits
        # rather than crash structure.
        #
        # This is the same ceiling the extraction evaluation already reports
        # (weather abstention 82.0% at corpus scale); it is restated here
        # because it invalidates a COMPARISON, not just an accuracy figure.
        n_av, n_drop = int(sm.sum()), int((~sm).sum())
        abst = n_drop / max(n_av + n_drop, 1)
        out.append({
            "field": name, "levels": levels,
            "n_crss": int(cm.sum()), "n_sgo": n_av,
            "crss_dropped": int((~cm).sum()), "sgo_dropped": n_drop,
            "av_abstention_rate": float(abst),
            "interpretable": bool(abst < 0.20),
            "interpretability_note": (
                "" if abst < 0.20 else
                f"AV side abstains on {abst:.0%} of records; the surviving rows "
                "are conditioned on the narrative mentioning this field, which "
                "is not a random subsample. Not a structural comparison."),
            "human_share": hc.tolist(), "av_share": hs.tolist(),
            "ratio": (hs / np.where(hc > 0, hc, np.nan)).tolist(),
            "coarsened": COARSENED.get(name, ""),
        })
    return out


def attach_extractions(sgo: pd.DataFrame, extractions: str,
                       model: Optional[str] = None) -> pd.DataFrame:
    """Join the canonical model's extracted schema fields onto the SGO frame.

    Filtered to one model for the same reason build_features.load() is: an
    unfiltered dedup resolves by JSONL append order, which would mix extractions
    from different models into one distribution.
    """
    model = model or canonical_model()
    ext = pd.read_json(extractions, lines=True)
    ext = ext[(ext["ok"]) & (ext["model"] == model)].copy()
    if ext.empty:
        raise SystemExit(f"[av_vs_human] no extractions for model={model!r}")
    flat = pd.json_normalize(ext["extraction"])
    flat["report_id"] = ext["report_id"].values
    flat = flat.drop_duplicates("report_id")
    return sgo.merge(flat, on="report_id", how="left", suffixes=("", "_ext"))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="AV (SGO) vs. human (CRSS) structure.")
    ap.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024],
                    help="Comparison window, applied to BOTH sides: the CRSS "
                    "releases to pool and the SGO incident years to keep. "
                    "Defaults to 2021-2024, the CRSS releases that exist.")
    ap.add_argument("--per-year", action="store_true",
                    help="Also report each year separately, as a stability check "
                    "on the pooled estimate.")
    ap.add_argument("--narratives", default=NARRATIVES)
    ap.add_argument("--extractions",
                    default=os.path.join(RESULTS, "extractions.jsonl"))
    ap.add_argument("--extraction-model", default=None)
    ap.add_argument("--severity-var", default="MAXSEV_IM",
                    choices=["MAXSEV_IM", "MAX_SEV"],
                    help="MAXSEV_IM is NHTSA's imputed variable and is the "
                    "default; MAX_SEV drops ~2%% of crashes whose severity was "
                    "not reported, which is not independent of severity.")
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()

    crss = load_crss_years(a.years, severity_var=a.severity_var)
    sgo = attach_extractions(load_sgo(a.narratives, a.years), a.extractions,
                             a.extraction_model)

    report = {"years": a.years, "severity_var": a.severity_var,
              "n_boot": a.n_boot, "coarsened": COARSENED,
              "severity": {}, "structure": {}, "retained": {}}

    for th in THRESHOLDS:
        c, s = crss[crss[th]], sgo[sgo[th]]
        report["retained"][th] = {
            "crss_n": int(len(c)), "crss_weighted_n": float(c["_w"].sum()),
            "sgo_n": int(len(s)),
            "crss_share_of_all": float(len(c) / len(crss)),
            "sgo_share_of_all": float(len(s) / len(sgo)),
        }
        report["severity"][th] = compare_severity(crss, sgo, th, a.n_boot)
        report["structure"][th] = compare_structure(crss, sgo, th)
        print(f"[av_vs_human] {th}: CRSS n={len(c):,} "
              f"(wtd {c['_w'].sum():,.0f})  SGO n={len(s):,}", flush=True)

    if a.per_year and len(a.years) > 1:
        # Does the pooled severity distribution hide year-to-year drift? Reported
        # rather than assumed away: 2021 in particular still carries pandemic
        # effects on traffic volume and speeding.
        report["per_year"] = {}
        for y in a.years:
            sub = crss[crss["_year"] == y]
            report["per_year"][y] = compare_severity(sub, sgo, "any_injury",
                                                     n_boot=max(a.n_boot // 4, 100))
            sh = report["per_year"][y]["human_share"]
            print(f"[av_vs_human] {y} any-injury human shares: "
                  f"{[round(x, 4) for x in sh]}")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "av_vs_human.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[av_vs_human] wrote {out}")

    for th in THRESHOLDS:
        st = report["structure"][th]
        bad = [f["field"] for f in st if not f["interpretable"]]
        if bad:
            print(f"[av_vs_human] {th}: NOT interpretable due to AV abstention: "
                  f"{', '.join(bad)}")

    for th in THRESHOLDS:
        sv = report["severity"][th]
        print(f"\n=== severity, threshold={th}  (CRSS n={sv['n_crss']:,}, "
              f"SGO n={sv['n_sgo']:,})")
        print(f"{'level':10s} {'human':>8s} {'AV':>8s} {'AV/human':>9s}")
        for i, lv in enumerate(sv["levels"]):
            h, v = sv["human_share"][i], sv["av_share"][i]
            r = sv["ratio"][i]
            print(f"{lv:10s} {h:8.4f} {v:8.4f} {r:9.2f}x")


if __name__ == "__main__":
    main()
