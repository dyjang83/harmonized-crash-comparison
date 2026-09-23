"""Crosswalk between CRSS (police-reported human crashes) and the AV schema.

Structure deliberately mirrors schema/distant_map.py: a REDUCERS registry of
`row -> Optional[(human_label, av_label_space)]` functions that project BOTH
sides into the coarsest space the two sources can jointly express, and return
None to DROP a row whose mapping would require inventing a convention. Dropping
rather than guessing is the same discipline the distant-supervision key already
uses, and for the same reason: a fabricated mapping shows up as a real finding.

Two granularity gaps are structural and cannot be closed:

  SEVERITY. CRSS codes KABCO; SGO codes a five-level 'highest injury severity
  alleged' that includes a *Moderate* grade KABCO has no counterpart for. The
  common space is therefore FOUR levels, with SGO Minor+Moderate merged and
  KABCO C+B merged. This is the load-bearing mapping for the severity comparison.

  LIGHTING. CRSS 'Dark - Unknown Lighting' cannot separate lit from unlit dark,
  exactly as SGO's identical category cannot (see distant_map.reduce_lighting).

A construct caveat that no mapping fixes, and which belongs in the write-up:
SGO severity is ALLEGED by the reporting entity; CRSS severity is a police
officer's KABCO assessment. Different observers, different incentives. They are
the closest available pair, not the same measurement.

Variables are matched on the decoded *NAME columns CRSS ships alongside every
coded variable, not on the integer codes. NHTSA renumbers codes between
releases -- the SGO third amendment did exactly this -- and a silently shifted
integer would corrupt every downstream estimate without raising anything.
"""
from __future__ import annotations

from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
_MISSING = {"", "not reported", "reported as unknown", "unknown", "unknown/not reported",
            "no person involved", "died prior to crash", "died prior to crash*"}


def _norm(v) -> Optional[str]:
    """Lowercase, strip punctuation-ish noise, collapse whitespace."""
    if v is None:
        return None
    s = str(v).strip().lower()
    for ch in "-/,.()*":
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s or None


def is_missing(v) -> bool:
    n = _norm(v)
    return n is None or n in {_norm(m) for m in _MISSING}


# ---------------------------------------------------------------------------
# Severity: the common four-level ordinal
# ---------------------------------------------------------------------------
# Ordered. Index is the ordinal value used by the distribution comparison.
SEVERITY_COMMON = ["none", "minor", "serious", "fatal"]

# CRSS MAXSEV_IM / MAX_SEV (KABCO), matched on MAXSEV_IMNAME.
#   O -> none, C+B -> minor, A -> serious, K -> fatal
# 'Injured, Severity Unknown', 'Died Prior to Crash' and 'No person involved'
# have no defensible placement on the ordinal and are dropped.
_KABCO = {
    "no apparent injury o": "none",
    "possible injury c": "minor",
    "suspected minor injury b": "minor",
    "suspected serious injury a": "serious",
    "fatal injury k": "fatal",
}

# SGO 'Highest Injury Severity Alleged'. The raw field is free-ish text with
# hospitalization qualifiers ('Minor W/O Hospitalization') and a property-damage
# phrasing that means no injury ('Property Damage. No Injured Reported'), so it
# is matched by substring in severity order -- the same approach and ordering as
# features.build_features._norm_severity, kept consistent with it on purpose.
def sgo_severity_common(s) -> Optional[str]:
    if not isinstance(s, str):
        return None
    t = s.strip().lower()
    if "fatal" in t:
        return "fatal"
    if "serious" in t:
        return "serious"
    # Moderate merges DOWN into minor: KABCO has no moderate grade, and the
    # alternative (merging up into serious) would inflate the AV serious share
    # against a human key that cannot reciprocate.
    if "moder" in t or "minor" in t:
        return "minor"
    if "no inj" in t or t in {"none", "no"}:
        return "none"
    return None


def crss_severity_common(name) -> Optional[str]:
    if is_missing(name):
        return None
    return _KABCO.get(_norm(name))


# ---------------------------------------------------------------------------
# Structural fields: one-sided projections into a shared space
# ---------------------------------------------------------------------------
# NOTE ON STRUCTURE. distant_map.py pairs two codings OF THE SAME CRASH (the
# reporting entity's code vs. the model's extraction), so its reducers take both
# sides at once. Nothing of the kind applies here: an SGO crash and a CRSS crash
# are different crashes, and what is compared is two MARGINAL DISTRIBUTIONS. So
# each field gets two independent projections -- one from CRSS codes, one from
# schema values -- into a shared label space, and the comparison is between the
# resulting distributions.
#
# Each shared space is the COARSEST space both sides can express TOTALLY, so
# that projection drops only genuinely missing values. That matters more here
# than in the paired setting: dropping a category present on one side only
# (CRSS's 'Cloudy', 13% of its crashes) would not merely shrink n, it would
# renormalize the remaining shares and manufacture a difference.

# --- lighting: {daylight, dark, dawn_dusk} ---------------------------------
# CRSS 'Dark - Unknown Lighting' cannot separate lit from unlit dark, and the
# schema can. Comparing in the finer space would force those CRSS rows to be
# dropped, biasing the human dark share; comparing in {daylight, dark, dawn_dusk}
# keeps every row on both sides.
LIGHTING_LEVELS = ["daylight", "dark", "dawn_dusk"]
_CRSS_LIGHTING = {
    "daylight": "daylight", "dark not lighted": "dark", "dark lighted": "dark",
    "dark unknown lighting": "dark", "dawn": "dawn_dusk", "dusk": "dawn_dusk",
}
_AV_LIGHTING = {
    "daylight": "daylight", "dark_lighted": "dark", "dark_unlighted": "dark",
    "dawn_dusk": "dawn_dusk",
}


def crss_lighting(row: dict) -> Optional[str]:
    return _CRSS_LIGHTING.get(_norm(row.get("LGT_CONDNAME")))


def av_lighting(v) -> Optional[str]:
    return _AV_LIGHTING.get(str(v)) if v is not None else None


# --- weather: {clear_or_cloudy, rain, snow, fog, other} --------------------
# The schema has no 'cloudy' member, so CRSS's Cloudy cannot map to a schema
# value one-for-one. Rather than drop 13% of the human sample, both sides
# collapse to the distinction both can actually express: precipitation type,
# with dry-but-overcast pooled with clear.
WEATHER_LEVELS = ["clear_or_cloudy", "rain", "snow", "fog", "other"]
_CRSS_WEATHER = {
    "clear": "clear_or_cloudy", "cloudy": "clear_or_cloudy",
    "rain": "rain", "freezing rain or drizzle": "rain",
    "snow": "snow", "blowing snow": "snow", "sleet or hail": "snow",
    "fog smog smoke": "fog",
    "severe crosswinds": "other", "blowing sand soil dirt": "other",
    "other": "other",
}
_AV_WEATHER = {"clear": "clear_or_cloudy", "rain": "rain", "snow": "snow",
               "fog": "fog", "other": "other"}


def crss_weather(row: dict) -> Optional[str]:
    return _CRSS_WEATHER.get(_norm(row.get("WEATHERNAME")))


def av_weather(v) -> Optional[str]:
    return _AV_WEATHER.get(str(v)) if v is not None else None


# --- locality: {intersection, non_intersection} ----------------------------
LOCALITY_LEVELS = ["intersection", "non_intersection"]
_CRSS_INTERSECTION = {"intersection", "intersection related"}
_AV_INTERSECTION = {"intersection", "intersection_related"}
_AV_LOCALITY_KNOWN = {"intersection", "intersection_related", "segment",
                      "driveway", "other"}


def crss_locality(row: dict) -> Optional[str]:
    n = _norm(row.get("RELJCT2NAME"))
    if n is None or is_missing(n):
        return None
    return "intersection" if n in _CRSS_INTERSECTION else "non_intersection"


def av_locality(v) -> Optional[str]:
    sv = str(v) if v is not None else ""
    if sv not in _AV_LOCALITY_KNOWN:
        return None
    return "intersection" if sv in _AV_INTERSECTION else "non_intersection"


# --- collision type: manner, with the no-motor-vehicle-partner bucket ------
# CRSS MAN_COLL describes manner BETWEEN motor vehicles and reserves one code
# for 'first harmful event was not a collision with a motor vehicle
# in-transport', which pools single-vehicle and VRU crashes. The schema
# separates those two, so the shared space keeps them pooled.
COLLISION_LEVELS = ["rear_end", "head_on", "cross_path", "sideswipe",
                    "no_mv_partner", "other"]
_CRSS_NO_MV = ("the first harmful event was not a collision with a motor "
               "vehicle in transport")
_CRSS_MANCOLL = {
    "front to rear": "rear_end", "front to front": "head_on", "angle": "cross_path",
    "sideswipe same direction": "sideswipe",
    "sideswipe opposite direction": "sideswipe",
    "rear to side": "other", "rear to rear": "other", "other": "other",
}
_AV_COLLISION = {
    "rear_end": "rear_end", "head_on": "head_on", "cross_path": "cross_path",
    "sideswipe": "sideswipe", "single_vehicle": "no_mv_partner",
    "vru": "no_mv_partner", "backing": "other", "other": "other",
}


def crss_collision(row: dict) -> Optional[str]:
    n = _norm(row.get("MAN_COLLNAME"))
    if n is None or is_missing(n):
        return None
    if n == _CRSS_NO_MV:
        return "no_mv_partner"
    return _CRSS_MANCOLL.get(n)


def av_collision(v) -> Optional[str]:
    return _AV_COLLISION.get(str(v)) if v is not None else None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# field -> (crss_projection, av_projection, shared_levels, schema_field)
FIELDS: dict[str, tuple[Callable, Callable, list, str]] = {
    "lighting": (crss_lighting, av_lighting, LIGHTING_LEVELS, "lighting"),
    "weather": (crss_weather, av_weather, WEATHER_LEVELS, "weather"),
    "locality": (crss_locality, av_locality, LOCALITY_LEVELS, "locality"),
    "collision_type": (crss_collision, av_collision, COLLISION_LEVELS,
                       "collision_type"),
}

COMPARED_FIELDS = list(FIELDS)

# Coarsening applied, surfaced so no reader mistakes a coarsened comparison for
# a full-granularity one. Mirrors distant_map.COARSENED.
COARSENED = {
    "lighting": "dark_lighted/dark_unlighted pooled: CRSS 'Dark - Unknown Lighting'",
    "weather": "clear and cloudy pooled: the schema has no cloudy member",
    "locality": "intersection vs. not-intersection",
    "collision_type": "single_vehicle and vru pooled: CRSS codes them as one",
    "severity": "four-level: SGO Minor+Moderate merged, KABCO C+B merged",
}

# CRSS survey design. Every estimate must be WEIGHT-ed; variance comes from
# resampling PSU_VAR within PSUSTRAT. PSU_VAR (not PSU) is the variance-estimation
# unit -- CRSS splits and collapses PSUs for variance purposes, which is why the
# two columns have different cardinality (67 vs 60 in 2023). Verify against the
# Analytical User's Manual before publishing.
WEIGHT_COL = "WEIGHT"
PSU_COL = "PSU_VAR"
STRATUM_COL = "PSUSTRAT"
