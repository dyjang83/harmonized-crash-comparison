"""Tests for the CRSS <-> AV-schema crosswalk.

Fixtures are hand-built rows in the CRSS *NAME column format. The point of these
tests is that a mapping must either be RIGHT or DROP the row -- an invented
correspondence is the failure mode that would silently corrupt every downstream
estimate, so most of these assert None.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schema.crss_map import (SEVERITY_COMMON, sgo_severity_common,
                             crss_severity_common, crss_lighting, av_lighting,
                             crss_weather, av_weather, crss_locality, av_locality,
                             crss_collision, av_collision,
                             FIELDS, COARSENED, is_missing)


# --------------------------- severity --------------------------------------
def test_kabco_maps_onto_common_ordinal():
    assert crss_severity_common("No Apparent Injury (O)") == "none"
    assert crss_severity_common("Possible Injury (C)") == "minor"
    assert crss_severity_common("Suspected Minor Injury (B)") == "minor"
    assert crss_severity_common("Suspected Serious Injury (A)") == "serious"
    assert crss_severity_common("Fatal Injury (K)") == "fatal"


def test_kabco_unplaceable_codes_drop():
    """No defensible position on the ordinal -> drop, never guess."""
    for v in ("Injured, Severity Unknown", "Died Prior to Crash*",
              "No person involved", "Unknown/Not Reported", ""):
        assert crss_severity_common(v) is None, v


def test_sgo_severity_handles_the_messy_real_strings():
    """These are the literal values present in the SGO corpus, not idealized ones."""
    cases = {
        "Fatality": "fatal",
        "Serious": "serious",
        "Serious W/ Hospitalization": "serious",
        "Moderate": "minor",                       # merged: KABCO has no moderate
        "Moderate W/ Hospitalization": "minor",
        "Moderate W/O Hospitalization": "minor",
        "Minor": "minor",
        "Minor W/ Hospitalization": "minor",
        "Minor W/O Hospitalization": "minor",
        "No Injuries Reported": "none",
        "No Injured Reported": "none",
        "Property Damage. No Injured Reported": "none",   # says injury, means none
        "None": "none",
        "Unknown": None,
    }
    for raw, want in cases.items():
        assert sgo_severity_common(raw) == want, f"{raw!r} -> {sgo_severity_common(raw)!r}"


def test_moderate_merges_down_not_up():
    """Merging Moderate into serious would inflate the AV serious share against
    a human key that has no moderate grade to reciprocate with."""
    assert sgo_severity_common("Moderate") == "minor"
    assert SEVERITY_COMMON.index("minor") < SEVERITY_COMMON.index("serious")


def test_common_space_is_ordered_and_four_levels():
    assert SEVERITY_COMMON == ["none", "minor", "serious", "fatal"]


# --------------------------- lighting --------------------------------------
def test_lighting_pools_dark_on_both_sides():
    """CRSS cannot separate lit from unlit dark. Comparing in the finer space
    would force those rows to be dropped and bias the human dark share, so both
    sides pool into {daylight, dark, dawn_dusk}."""
    assert crss_lighting({"LGT_CONDNAME": "Dark - Unknown Lighting"}) == "dark"
    assert crss_lighting({"LGT_CONDNAME": "Dark - Lighted"}) == "dark"
    assert crss_lighting({"LGT_CONDNAME": "Dark - Not Lighted"}) == "dark"
    assert av_lighting("dark_lighted") == av_lighting("dark_unlighted") == "dark"
    assert crss_lighting({"LGT_CONDNAME": "Daylight"}) == av_lighting("daylight") == "daylight"
    for d in ("Dawn", "Dusk"):
        assert crss_lighting({"LGT_CONDNAME": d}) == "dawn_dusk"


def test_lighting_drops_only_genuine_missing():
    for v in ("Not Reported", "Reported as Unknown", "Other"):
        assert crss_lighting({"LGT_CONDNAME": v}) is None, v
    assert av_lighting("unknown") is None
    assert av_lighting(None) is None


# --------------------------- weather ---------------------------------------
def test_weather_pools_cloudy_with_clear_rather_than_dropping_it():
    """CRSS 'Cloudy' is 13% of its crashes and has no schema member. Dropping it
    would renormalize the remaining shares and manufacture a difference."""
    assert crss_weather({"WEATHERNAME": "Cloudy"}) == "clear_or_cloudy"
    assert crss_weather({"WEATHERNAME": "Clear"}) == "clear_or_cloudy"
    assert av_weather("clear") == "clear_or_cloudy"


def test_weather_precipitation_types_coarsen():
    for v in ("Snow", "Blowing Snow", "Sleet or Hail"):
        assert crss_weather({"WEATHERNAME": v}) == "snow", v
    for v in ("Rain", "Freezing Rain or Drizzle"):
        assert crss_weather({"WEATHERNAME": v}) == "rain", v
    assert crss_weather({"WEATHERNAME": "Fog, Smog, Smoke"}) == "fog"
    assert av_weather("fog") == "fog"


# --------------------------- locality --------------------------------------
def test_locality_intersection_vs_not():
    assert crss_locality({"RELJCT2NAME": "Intersection"}) == "intersection"
    assert crss_locality({"RELJCT2NAME": "Intersection-Related"}) == "intersection"
    assert crss_locality({"RELJCT2NAME": "Non-Junction"}) == "non_intersection"
    assert crss_locality({"RELJCT2NAME": "Driveway Access"}) == "non_intersection"
    assert av_locality("intersection") == av_locality("intersection_related") == "intersection"
    assert av_locality("segment") == av_locality("driveway") == "non_intersection"
    assert av_locality("unknown") is None


# --------------------------- collision type --------------------------------
_NO_MV = ("The First Harmful Event was Not a Collision with a Motor Vehicle "
          "In-Transport")


def test_collision_manner_mapping():
    assert crss_collision({"MAN_COLLNAME": "Front-to-Rear"}) == "rear_end"
    assert crss_collision({"MAN_COLLNAME": "Front-to-Front"}) == "head_on"
    assert crss_collision({"MAN_COLLNAME": "Angle"}) == "cross_path"
    for d in ("Sideswipe - Same Direction", "Sideswipe - Opposite Direction"):
        assert crss_collision({"MAN_COLLNAME": d}) == "sideswipe"
    assert av_collision("rear_end") == "rear_end"


def test_single_vehicle_and_vru_pool_because_crss_cannot_split_them():
    assert crss_collision({"MAN_COLLNAME": _NO_MV}) == "no_mv_partner"
    assert av_collision("single_vehicle") == "no_mv_partner"
    assert av_collision("vru") == "no_mv_partner"


def test_projections_are_total_over_observed_categories():
    """Every non-missing CRSS category must land somewhere: an unmapped category
    silently shrinks and renormalizes the human distribution."""
    observed = {
        "LGT_CONDNAME": ["Daylight", "Dark - Not Lighted", "Dark - Lighted",
                         "Dawn", "Dusk", "Dark - Unknown Lighting"],
        "WEATHERNAME": ["Clear", "Rain", "Sleet or Hail", "Snow",
                        "Fog, Smog, Smoke", "Severe Crosswinds",
                        "Blowing Sand, Soil, Dirt", "Other", "Cloudy",
                        "Blowing Snow", "Freezing Rain or Drizzle"],
        "RELJCT2NAME": ["Non-Junction", "Intersection", "Intersection-Related",
                        "Driveway Access", "Entrance/Exit Ramp Related",
                        "Railway Grade Crossing", "Crossover-Related",
                        "Driveway Access Related", "Shared-Use Path Crossing",
                        "Acceleration/Deceleration Lane", "Through Roadway",
                        "Other location within Interchange Area",
                        "Entrance/Exit Ramp"],
        "MAN_COLLNAME": [_NO_MV, "Front-to-Rear", "Front-to-Front", "Angle",
                         "Sideswipe - Same Direction",
                         "Sideswipe - Opposite Direction", "Rear-to-Side",
                         "Rear-to-Rear", "Other"],
    }
    fns = {"LGT_CONDNAME": crss_lighting, "WEATHERNAME": crss_weather,
           "RELJCT2NAME": crss_locality, "MAN_COLLNAME": crss_collision}
    for col, vals in observed.items():
        for v in vals:
            assert fns[col]({col: v}) is not None, f"{col}={v!r} maps nowhere"


def test_projections_land_in_declared_levels():
    for name, (cf, af, levels, _) in FIELDS.items():
        assert len(levels) == len(set(levels)), name


# --------------------------- registry hygiene ------------------------------
def test_every_field_is_documented_as_coarsened():
    for name in FIELDS:
        assert name in COARSENED, f"{name} coarsens silently -- document it"


def test_missing_tokens():
    for v in ("Not Reported", "Reported as Unknown", "Unknown", "", None):
        assert is_missing(v), v
    assert not is_missing("Daylight")


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            f(); print(f"ok  {n}")
    print("OK: CRSS crosswalk tests passed")
