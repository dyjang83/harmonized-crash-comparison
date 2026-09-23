"""Parse raw sources into a unified narratives table.

Outputs data/interim/narratives.jsonl with one row per report:
    {report_id, source, manufacturer, narrative, struct_severity, ...}

Two readers:
  - parse_ol316_pdf: pull the free-text accident-description narrative out of an
    OL 316 PDF using pdfplumber. OL 316 places the description after the
    "DESCRIBE ACCIDENT" / "Additional information attached" region; we extract
    text and isolate the narrative block heuristically, then keep the raw page
    text as a fallback for manual review.
  - load_sgo_csv: read the SGO incident CSVs, keep the Narrative field (Field
    115) and the structured outcome fields used downstream, dropping rows whose
    narrative is fully redacted as CBI.

The structured severity field from SGO ('Highest Injury Severity Alleged') is
preserved verbatim as `struct_severity`; it is the lift-test target and must NOT
be derived from text. OL 316 has no equivalent multi-grade severity code, but
its Section 4 injured/deceased checkboxes do resolve a coarser, still
non-narrative `struct_severity` signal (see _ol316_injury_flag) -- the lift
test itself still runs on SGO rows only (see features/build_features.py),
since it needs the full ordinal scale OL 316 cannot support.
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Iterator, Optional

import logging

import numpy as np
import pandas as pd
import pdfplumber

# Suppress noisy pdfplumber/pdfminer font-descriptor warnings from some OL 316 PDFs.
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# SGO CSVs come from a Windows pipeline and are usually Windows-1252, not UTF-8.
_CSV_ENCODINGS = ["utf-8-sig", "cp1252", "latin-1"]

INTERIM = os.path.join("data", "interim")
NARR_OUT = os.path.join(INTERIM, "narratives.jsonl")

# A "soft gap" for regexes that span two anchor phrases with unrelated text in
# between. Stops at a genuine sentence boundary (period, whitespace, capital
# letter -- "...the General Order. The vehicle was...") but crosses abbreviation
# periods that aren't sentence ends ("Request No. 1,", "rev.202110153149").
# Blocking ALL periods (an earlier version of this) is too strict: legal
# boilerplate frequently embeds "No. <n>" mid-sentence, which then hides the
# second anchor phrase and makes the whole regex silently fail to match.
_SOFT_GAP = r"(?:[^.]|\.(?!\s+[A-Z]))"

# Page-text fallback markers (only used for non-fillable / scanned OL 316 PDFs).
# The real Section 5 header is "SECTION 5 — ACCIDENT DETAILS - DESCRIPTION".
_DESC_START = re.compile(
    r"(ACCIDENT\s+DETAILS\s*[-–—]\s*DESCRIPTION|"
    r"DESCRIBE\s+(?:HOW\s+THE\s+)?ACCIDENT|DESCRIPTION\s+OF\s+(?:ACCIDENT|COLLISION))",
    re.I)
_DESC_END = re.compile(
    r"(ITEMS\s+MARKED\s+BELOW|SECTION\s+6|OL\s*316\s*\(REV|"
    r"Additional\s+information\s+attached)", re.I)
# "Autonomous Mode / Conventional Mode" is a checkbox label, not narrative.
_MODE_LINE = re.compile(r"\b(Autonomous|Conventional)\s+Mode\b", re.I)

# Matches an ENTIRE bracketed redaction placeholder, whatever descriptive text it
# contains, e.g. "[REDACTED]", "[XXX]", or "[REDACTED, MAY CONTAIN CONFIDENTIAL
# BUSINESS INFORMATION]". A narrower regex that only strips the trigger word
# (REDACTED/XXX) leaves the placeholder's own explanatory text behind (e.g. "MAY
# CONTAIN CONFIDENTIAL BUSINESS INFORMATION") looking like narrative content and
# padding the word count -- that text describes the redaction, it is not a
# description of the crash, so the whole bracket span must be removed as one unit.
_REDACTED_BRACKET = re.compile(
    r"\[[^\[\]]{0,300}?(?:REDACTED|XXX|CONFIDENTIAL\s+BUSINESS\s+INFORMATION)"
    r"[^\[\]]{0,300}?\]",
    re.I,
)
# Bare (unbracketed) redaction tokens, for narratives using REDACTED/XXX with no
# surrounding brackets at all.
_REDACTED_BARE = re.compile(r"\bREDACTED\b|\bXXX\b", re.I)

# Some reporting entities preface every SGO filing with a fixed legal-preservation
# disclaimer -- standard boilerplate a legal team writes once to preserve a
# jurisdictional/authority objection while still nominally complying, e.g.
# "Without agreeing with NHTSA's interpretations or conceding that [Company] is
# obligated to respond to the General Order, [Company] provides this
# information:". This is not narrative content. Left unstripped, this ~20-word
# span pads the word count enough to hide a report whose actual content is
# nothing but a bare reference (e.g. "Video:[XXX]") with no crash description at
# all -- exactly like the reported case. Matches the company name generically
# rather than hardcoding one, since any manufacturer may use this template.
_LEGAL_DISCLAIMER = re.compile(
    r"without\s+(?:agreeing|conceding)\b"
    + _SOFT_GAP + r"{0,300}?"
    r"obligat\w*\s+to\s+respond\s+to\s+the\s+general\s+order"
    + _SOFT_GAP + r"{0,150}?"
    r"provides?\s+this\s+information\s*:?",
    re.I | re.S,
)

# Minimum words of genuine narrative that must remain after stripping every
# redaction placeholder for an SGO row to be considered usable at all. Below
# this, the row is boilerplate about the redaction, not a narrative, and is
# excluded (not annotated as unknown/null) the same way a CBI refusal is.
_MIN_SGO_NARRATIVE_WORDS = 8

# Some rows contain no crash description at all, only a cross-reference to
# ANOTHER report's narrative, e.g. "See Waymo report WaymoLLC-202109XXX-0
# (rev.202110XXX153149) for Narrative details.", "Please refer to the
# description provided by Beep in incident report number 30490-11294-1.", or
# "No update to narrative. Please refer to Waymo LLC Report - 202107XXX-0
# ... submitted on July XXX, 2021." These are long enough in raw word count to
# survive the redaction/disclaimer stripping above, but they describe zero
# collision content -- the actual narrative lives in a different report
# entirely. Detected structurally (cue + "report" + a report-ID-like token)
# rather than by hardcoding exact phrasings, since manufacturers word this
# differently; requiring all three conditions on the SAME sentence keeps this
# from stripping a genuine narrative that merely mentions "the police report"
# in passing without pointing to another filing's narrative.
_POINTER_CUE = re.compile(
    r"\b(?:see|refer(?:s|red|ring)?\s+to|please\s+refer)\b|"
    r"no\s+update\s+to\s+narrative\b",
    re.I,
)
_REPORT_NOUN = re.compile(r"\breport\b", re.I)
_REPORT_ID_TOKEN = re.compile(r"\b[A-Za-z0-9]*\d[A-Za-z0-9\-]*\d[A-Za-z0-9\-]*\b")


def _split_sentences(text: str) -> list:
    """Naive sentence splitter, adequate for this narrow purpose."""
    return re.split(r"(?<=[.!?])\s+", text)


def _strip_report_pointer_sentences(text: str) -> str:
    """Remove any sentence that is merely a cross-reference to another
    report's narrative rather than a description of this crash."""
    kept = [s for s in _split_sentences(text)
            if not (_POINTER_CUE.search(s) and _REPORT_NOUN.search(s)
                    and _REPORT_ID_TOKEN.search(s))]
    return " ".join(kept).strip()


def _narrative_after_redaction(narr: str) -> str:
    """Return narr with every known non-narrative element removed: bracketed
    redaction placeholders, bare REDACTED/XXX tokens, the legal-disclaimer
    preamble some entities prepend to every filing, and any sentence that is
    merely a cross-reference to another report's narrative. What remains, if
    anything, is the actual attempt at a narrative (which may still be short
    or absent)."""
    stripped = _LEGAL_DISCLAIMER.sub(" ", narr)
    stripped = _REDACTED_BRACKET.sub(" ", stripped)
    stripped = _REDACTED_BARE.sub(" ", stripped)
    stripped = _strip_report_pointer_sentences(stripped)
    return re.sub(r"\s+", " ", stripped).strip()


# Field VALUES that are checkboxes (start with '/') or empty are not narrative.
_MIN_NARRATIVE_WORDS = 12

# Detects a CBI-refusal statement standing in for the narrative, e.g. "Waymo is
# seeking CBI protection for the narrative and I am unable to provide one." This
# is NOT a narrative -- it is a non-response -- and must not be harvested as one
# just because it clears the word-count threshold. Records matching this are
# excluded from the corpus entirely (same treatment as fully-redacted SGO rows),
# not annotated as null/unknown, and counted separately as a corpus statistic.
_CBI_REFUSAL = re.compile(
    r"(confidential\s+business\s+information|\bCBI\b)"
    + _SOFT_GAP + r"{0,150}?"
    r"(unable\s+to\s+provide|will\s+not\s+provide|cannot\s+provide|"
    r"declin\w*\s+to\s+provide|withheld|withholding|not\s+(?:being\s+)?provided|"
    r"redact(?:ed|ing)?\s+in\s+full)",
    re.I | re.S,
)


def _is_cbi_refusal(text: str) -> bool:
    return bool(text) and bool(_CBI_REFUSAL.search(text))


# Some SGO rows contain substantial, well-formed prose that nonetheless
# describes NO crash at all, because the report itself states there is nothing
# to report. Two sub-cases seen in practice:
#   (a) a "no new/updated incident" administrative filing -- a routine
#       compliance filing confirming zero incidents this period, e.g. "...comma
#       confirms that it has no reportable incident information."; NHTSA's own
#       documentation calls this report type "No New or Updated Incident
#       Reports".
#   (b) a retraction -- the reporting entity states that further investigation
#       (often by law enforcement) determined the incident did NOT actually
#       involve their ADS/ADAS vehicle, and formally asks that the report be
#       removed/voided, e.g. "Tesla requests removal of this report and its
#       count from the SGO reporting."
# Unlike redaction/disclaimer/pointer content, these can run 30-60+ words of
# genuine prose, so a word-count threshold cannot catch them -- the row must be
# classified by content, the same way _is_cbi_refusal works, not by how much
# text remains after stripping known boilerplate.
_NON_INCIDENT = re.compile(
    r"no\s+reportable\s+incident(?:s)?\s+(?:information|to\s+report)\b|"
    r"requests?\s+removal\s+of\s+(?:this|the)\s+report\b|"
    r"request(?:ing|ed)?\s+(?:that\s+)?(?:this|the)\s+report\s+be\s+removed\b|"
    r"did\s+not\s+involve\s+the\s+(?:original|subject)\s+vehicle\b|"
    r"not\s+a\s+reportable\s+(?:incident|event|crash)\b",
    re.I,
)


def _is_non_incident(text: str) -> bool:
    """True if the narrative reports zero qualifying AV incident: either a
    routine 'nothing to report' compliance filing, or a retraction stating the
    incident didn't actually involve the reporting entity's AV. Unlike the
    fixed corporate boilerplate templates (_LEGAL_DISCLAIMER, _CBI_REFUSAL),
    retraction language in particular is free-form investigative prose, so this
    detector has lower recall -- it catches known phrasings, not the concept in
    general. Additional real-world phrasings should be added here as found."""
    return bool(text) and bool(_NON_INCIDENT.search(text))


# Some manufacturers file a report as TWO PDFs: a "-a"/"-redacted" main OL 316
# form (metadata + checkboxes, narrative section says "Additional information
# attached") and a separate "-b"/"-narrative" PDF that is nothing but the prose
# description, with no form scaffold at all. Such a document has no "SECTION 5"
# header to slice after, so the Section-5 slicer correctly finds nothing. We
# detect this case by checking whether the page text looks like the standard
# multi-section OL 316 form at all; if it doesn't, the page text itself likely
# *is* the narrative attachment.
_FORM_MARKERS = [re.compile(p, re.I) for p in (
    r"SECTION\s+1\b", r"SECTION\s+2\b", r"SECTION\s+3\b", r"SECTION\s+4\b",
    r"MANUFACTURER.?S\s+INFORMATION", r"CERTIFICATION",
)]

# Boilerplate lines to strip from a standalone attachment page before treating
# the remainder as narrative (letterhead / footer / barcode artifacts).
_BOILERPLATE_LINE = re.compile(
    r"^\s*(OL\s*316\s*\(REV[^\n]*|DMV\s+USE\s+ONLY|A\s+Public\s+Service\s+Agency|"
    r"\*OL316\*|Page\s+\d+(\s+of\s+\d+)?|AVT\s+NUMBER\s*:?\s*)\s*$",
    re.I,
)


def _looks_like_standard_form(raw_text: str) -> bool:
    """True if raw_text is (a copy of) the standard multi-section OL 316 form.

    Requires >=2 section markers so a narrative attachment that happens to
    mention one incidental term (e.g. "certification") isn't misclassified.
    """
    return sum(1 for pat in _FORM_MARKERS if pat.search(raw_text)) >= 2


def _clean_attachment_text(raw_text: str) -> str:
    """Strip known OL 316 letterhead/footer lines from a standalone attachment."""
    lines = [ln for ln in raw_text.splitlines() if not _BOILERPLATE_LINE.match(ln)]
    text = " ".join(ln.strip() for ln in lines if ln.strip())
    return re.sub(r"\s+", " ", text).strip()


def _require_pypdf():
    """Import pypdf, failing loudly and once if it is unavailable.

    The narrative on an OL 316 lives in an AcroForm text-field widget, not in
    the page content stream, so pypdf is not optional for this corpus -- it is
    the only thing that reads the narrative at all. Swallowing the ImportError
    and returning {} makes a missing dependency look exactly like a scanned,
    fieldless PDF: every file falls through to the page-text path, finds only
    the blank form template, and is reported as "no narrative found". That is a
    silent total-corpus failure that reads as a data problem, so it is raised
    rather than caught.
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError(
            "pypdf is required to read OL 316 narratives (they live in AcroForm "
            "fields, not page text). Install it with `pip install pypdf`. Note "
            "that the legacy `PyPDF2` package is NOT a substitute: its "
            "get_fields() has a different return shape."
        ) from e
    return PdfReader


def _form_field_values(path: str) -> dict:
    """Return {field_name: text_value} for AcroForm TEXT fields with content.

    OL 316 is a fillable form; the typed narrative lives in a form-field widget
    that pdfplumber's text extraction does not read. We pull values via pypdf.
    Checkbox/radio states are excluded here and collected separately by
    _form_checkbox_states -- they are not narrative, but they are not noise
    either: they carry the reporter's own coding of weather, lighting, roadway
    and pre-crash movement, which is the OL 316 side of the distant-supervision
    key. Discarding them (as this function previously did for all '/'-prefixed
    values) threw that key away.
    """
    PdfReader = _require_pypdf()
    try:
        fields = PdfReader(path).get_fields() or {}
    except Exception as e:  # noqa: BLE001 - genuinely fieldless or damaged PDF
        print(f"[parse] no readable AcroForm in {os.path.basename(path)}: {e}")
        return {}
    out = {}
    for name, f in fields.items():
        val = f.get("/V")
        if val is None:
            continue
        s = str(val)
        if s.startswith("/"):      # checkbox / radio state, e.g. '/Yes', '/Off'
            continue
        s = re.sub(r"\s+", " ", s.replace("\r", " ").replace("\n", " ")).strip()
        if s:
            out[str(name)] = s
    return out


# Checkbox "off" states. Anything else in /V means the box is ticked; the export
# value itself varies by form generation ('/Yes', '/On', '/1', '/Choice1'), so
# the tick is detected by exclusion rather than by matching a fixed on-value.
_CHECKBOX_OFF = {"/off", "/no", "", "/0"}


def _form_checkbox_states(path: str) -> dict:
    """Return {field_name: state} for AcroForm checkbox/radio fields that are ON."""
    PdfReader = _require_pypdf()
    try:
        fields = PdfReader(path).get_fields() or {}
    except Exception:  # noqa: BLE001 - already reported by _form_field_values
        return {}
    on = {}
    for name, f in fields.items():
        val = f.get("/V")
        if val is None:
            continue
        s = str(val).strip()
        if not s.startswith("/"):
            continue
        if s.lower() in _CHECKBOX_OFF:
            continue
        on[str(name)] = s
    return on


# OL 316 checkbox groups, keyed by the letter codes printed on the form.
#
# VERIFIED against the 868 filings in data/raw/ol316 via --audit-checkboxes.
# The AcroForm names follow the pattern "<GROUP> <LETTER> <VEHICLE>", e.g.
# "WEATHER A 1", "MOVEMENT  B 2" (note the double space in the MOVEMENT group,
# which is present in the form itself). The letters index the printed option
# list; the trailing digit is the VEHICLE NUMBER, not a second choice.
#
# The vehicle suffix matters. On the OL 316, vehicle 1 is the autonomous
# vehicle and vehicle 2 is the other party (Section 3 is headed "OTHER PARTY'S
# INFORMATION/VEHICLE 2"). Environmental groups are ticked redundantly for both
# vehicles with the same value, but MOVEMENT and TYPE are per-vehicle: reading
# the "2" boxes for pre-crash maneuver would code the other party's maneuver as
# the subject AV's. Only "1" is read for those groups.
#
# Note also that the form's ROADWAY group is ROADWAY SURFACE (dry/wet/icy), not
# a road classification. OL 316 has no road-class or locality group at all, so
# it supplies no supervision for those two fields.
_OL316_WEATHER = {"a": "clear", "b": "cloudy", "c": "rain", "d": "snow",
                  "e": "fog", "f": "other", "g": "wind"}
_OL316_LIGHTING = {"a": "daylight", "b": "dawn_dusk", "c": "dark_lighted",
                   "d": "dark_unlighted", "e": "dark_unlighted"}
_OL316_MOVEMENT = {
    "a": "stopped", "b": "proceeding straight", "d": "making right turn",
    "e": "making left turn", "g": "backing", "j": "changing lanes",
    "o": "parked",
    # c (ran off road), f (u-turn), h (slowing/stopping), i (passing),
    # k (parking maneuver), l (entering traffic), m (other unsafe turning),
    # n (crossing into opposing lane), p (merging), q (wrong way), r (other)
    # have no clean schema counterpart and are deliberately omitted so those
    # rows drop rather than being forced to `other`.
}
_OL316_TYPE = {"a": "head_on", "b": "sideswipe", "c": "rear_end",
               "d": "cross_path", "e": "single_vehicle", "g": "vru"}
# f (overturned) and h (other) omitted deliberately.

_OL316_GROUPS = {
    "WEATHER": ("weather", _OL316_WEATHER, False),
    "LIGHTING": ("lighting", _OL316_LIGHTING, False),
    "MOVEMENT": ("movement", _OL316_MOVEMENT, True),
    "TYPE": ("collision", _OL316_TYPE, True),
}

# Letter-coded groups that are real, verified form fields (same "<GROUP>
# <LETTER> <VEHICLE>" naming convention as _OL316_GROUPS) but have no
# counterpart in the extraction schema: ROADWAY is road SURFACE
# (dry/wet/snowy-icy/slippery) and ROAD CONDITIONS is road-defect codes
# (potholes/construction/flooded/none unusual) -- neither is a road
# classification or locality concept, so they supply no distant supervision
# for `road_class`/`locality` (see the comment above _OL316_GROUPS). Listed
# here, rather than left to fall through to "unmapped", so the audit reports
# a genuinely unrecognized field name, not routine form furniture.
_OL316_NO_SCHEMA_LETTER_GROUPS = {"ROADWAY", "ROAD CONDITIONS"}

_OL316_BOX_RE = re.compile(
    r"^\s*(WEATHER|LIGHTING|MOVEMENT|TYPE|ROADWAY|ROAD\s+CONDITIONS)\s+([A-R])\s*(\d)?\s*$",
    re.IGNORECASE)

# Non-lettered checkbox fields verified (via --audit-checkboxes across the
# full corpus) to be real OL 316 form fields with no extraction-schema
# counterpart: the other-party type ("Vehicle was: Moving/Stopped in
# Traffic", "Involved in the Accident: Pedestrian/Bicyclist/undefined" --
# "undefined" is the literal AcroForm field name on the printed form, not a
# parsing artifact), the vehicle-DAMAGE severity scale (UNK/NONE/MINOR/MOD/
# MAJOR -- property damage grade, not injury severity, so this is NOT
# narrative_injury_severity/struct_severity), the damage-location diagram,
# the "OTHER ASSOCIATED FACTOR(S)" contributing-factor codes, and AM/PM.
# Section 4's per-person "Injured/Deceased/Driver/Passenger/Bicyclist/
# Proper ty" checkboxes are the one exception with schema signal: they are
# both recognized here (so the raw tick names don't pollute the audit) AND
# read again, by name, in _ol316_injury_flag below to derive struct_severity
# (the embedded space in "Proper ty" is a literal artifact of the form, not
# a typo introduced here).
_OL316_PARTY_TYPE_RE = re.compile(
    r"^(Moving|Stopped in Traffic|Pedestrian|Bicyclist|undefined)(_\d+)?$", re.I)
_OL316_PERSON_RE = re.compile(
    r"^(Injured|Deceased|Driver|Passenger|Proper\s?ty)(_\d+)?$", re.I)
_OL316_DAMAGE_SEVERITY = {"unknown", "none", "minor", "moderate", "major"}
_OL316_DAMAGE_LOCATIONS = (
    "Rear Bumper", "Front Bumper",
    "Left Rear Passenger", "Right Rear Passenger",
    "Left Rear", "Right Rear",
    "Front Driver Side", "Front Passenger Side",
    "Left Front Corner", "Right Front Corner",
)
_OL316_DAMAGE_DIAGRAM_RE = re.compile(
    r"^(?:" + "|".join(re.escape(s) for s in _OL316_DAMAGE_LOCATIONS) + r")"
    r"(?:\s*\d+)?$", re.IGNORECASE)
_OL316_OTHER_FACTOR_RE = re.compile(r"^OTHER\s+[A-L](\s+(YES|NO))?$", re.I)
_OL316_TIME_RE = re.compile(r"^(AM|PM)$", re.I)


def _is_known_nonschema_field(name: str) -> bool:
    s = name.strip()
    if s.lower() in _OL316_DAMAGE_SEVERITY:
        return True
    return bool(
        _OL316_PARTY_TYPE_RE.match(s) or _OL316_PERSON_RE.match(s) or
        _OL316_DAMAGE_DIAGRAM_RE.match(s) or _OL316_OTHER_FACTOR_RE.match(s) or
        _OL316_TIME_RE.match(s)
    )


def _ol316_injury_flag(states: dict) -> Optional[str]:
    """Derive a coarse, non-narrative severity signal from OL 316 Section 4
    ("INJURY/DEATH, PROPERTY DAMAGE"): per-person "CHECK ALL THAT APPLY
    Injured/Deceased/Driver/Passenger/Bicyclist/Proper ty" checkboxes, up to
    2 named people on this form revision (suffix "_2" for the second).

    Unlike SGO's "Highest Injury Severity Alleged", OL 316 has no minor/
    moderate/serious grade -- only a binary injured/not per person -- so this
    can resolve at most three coarse buckets (handled in distant_map.py's
    reduce_narrative_severity the same way "Dark - Unknown Lighting" is
    projected onto dark_any: both sides of the comparison collapse onto the
    coarsest space the key can actually support):
      "fatal"       -- any listed person's Deceased box is ticked.
      "some injury" -- any listed person's Injured box is ticked (and no
                       Deceased tick); which of minor/moderate/serious is
                       unresolvable from this form.
      "none"        -- at least one person-row was actually filled in (not
                       left blank) and every such row has Proper ty ticked
                       with neither Injured nor Deceased -- i.e. the section
                       was used to record property-only damage.
    Returns None (no supervision, not "none") when the section carries no
    tick at all -- that is indistinguishable from the section being left
    blank rather than affirmatively completed.
    """
    rows = []
    for suffix in ("", "_2"):
        injured = f"Injured{suffix}" in states
        deceased = f"Deceased{suffix}" in states
        prop = f"Proper ty{suffix}" in states
        if injured or deceased or prop:
            rows.append((injured, deceased, prop))
    if not rows:
        return None
    if any(deceased for _, deceased, _ in rows):
        return "fatal"
    if any(injured for injured, _, _ in rows):
        return "some injury"
    if all(prop and not injured and not deceased for injured, deceased, prop in rows):
        return "none"
    return None


def _ol316_structured(states: dict) -> dict:
    """Reduce ticked OL 316 checkboxes to the same struct_* keys the SGO loader
    emits, so both corpora feed one distant-supervision path.

    Only groups resolving to exactly one value are emitted. A group with two
    conflicting ticks is an ambiguous filing, not a label, and yields nothing --
    except weather, which the form allows to be marked twice and which is
    therefore carried through as the same flag dict the SGO indicators produce.
    """
    hits: dict[str, set] = {g: set() for g in ("weather", "lighting",
                                               "movement", "collision")}
    # "Autonomous Mode" / "Conventional Mode" is the OL 316 side of
    # engagement_state: OL 316 is filed only for SAE L3-5 systems
    # (struct_report_type is always "ads"), so this pair is exactly what
    # reduce_engagement() in distant_map.py needs to resolve
    # ads_engaged/not_engaged -- previously not read at all, so OL 316 rows
    # contributed zero distant supervision for that field.
    engagement_ticks: set = set()
    unmapped = []
    for name in states:
        m = _OL316_BOX_RE.match(str(name))
        if m:
            group = re.sub(r"\s+", " ", m.group(1).upper())
            if group in _OL316_NO_SCHEMA_LETTER_GROUPS:
                continue
            letter, vehicle = m.group(2).lower(), m.group(3)
            field, table, subject_only = _OL316_GROUPS[group]
            # Vehicle 1 is the AV. For per-vehicle groups, ignore the other party.
            if subject_only and vehicle not in (None, "1"):
                continue
            if letter in table:
                hits[field].add(table[letter])
            continue
        if name in ("Autonomous Mode", "Conventional Mode"):
            engagement_ticks.add(name)
            continue
        if _is_known_nonschema_field(str(name)):
            continue
        unmapped.append(name)

    out: dict = {}
    if hits["weather"]:
        out["struct_weather_flags"] = {w: "Y" for w in hits["weather"]}
    for field, key in (("lighting", "struct_lighting"),
                       ("movement", "struct_movement"),
                       ("collision", "struct_collision_type")):
        if len(hits[field]) == 1:
            out[key] = next(iter(hits[field]))
    if len(engagement_ticks) == 1:
        out["struct_engaged"] = "yes" if "Autonomous Mode" in engagement_ticks else "no"
    severity = _ol316_injury_flag(states)
    if severity is not None:
        out["struct_severity"] = severity
    if unmapped:
        out["_unmapped_checkboxes"] = unmapped
    return out


def _narrative_from_fields(values: dict) -> str:
    """Pick the prose field(s). The narrative is far longer than any other field.

    Field names in OL 316 are unreliable (the narrative is often mislabeled, e.g.
    'ADDRESS_2...'), so we select by content: any value with >= _MIN_NARRATIVE_WORDS
    words, joined in descending length order; otherwise the single longest value if
    it is at least a short sentence. CBI-refusal statements are excluded from
    consideration here -- they are long enough to clear the word threshold, but
    they are a non-response, not a narrative, and must not be selected as one.
    """
    prose = [(len(v.split()), v) for v in values.values()
             if len(v.split()) >= _MIN_NARRATIVE_WORDS and not _is_cbi_refusal(v)]
    if prose:
        prose.sort(reverse=True)
        # Join all genuine prose fragments (handles narratives split across fields).
        return " ".join(v for _, v in prose)
    # Fallback: longest non-refusal value if it is at least a short clause.
    candidates = [v for v in values.values() if not _is_cbi_refusal(v)]
    longest = max(candidates, key=lambda v: len(v), default="")
    return longest if len(longest.split()) >= 6 else ""


def _manufacturer_from_fields(values: dict) -> Optional[str]:
    for key in values:
        k = key.lower()
        if "manufacturer" in k and "name" in k:
            return values[key]
    for key in values:
        if "business name" in key.lower():
            return values[key]
    return None


def parse_ol316_pdf(path: str) -> tuple[Optional[dict], str]:
    """Return (record_or_None, status).

    status is one of:
      "ok"           - usable narrative extracted.
      "cbi_refusal"  - the only candidate text was a CBI-refusal statement
                       ("Waymo is seeking CBI protection ... unable to provide
                       one"). This is a non-response, not a narrative. Excluded
                       from the corpus and from annotation; counted separately.
      "no_narrative" - no narrative text found by any method. Commonly the main
                       "-a"/"-redacted" half of a report filed as two PDFs, whose
                       narrative lives entirely in a sibling "-b"/"-narrative"
                       file; that sibling is its own record (see the standalone-
                       attachment fallback below), so this is expected, not data
                       loss. Can also be a genuinely unreadable scan.
      "failed"       - the PDF could not be opened/parsed at all.
    """
    values = _form_field_values(path)
    checkboxes = _form_checkbox_states(path)
    saw_cbi_in_fields = any(_is_cbi_refusal(v) for v in values.values())

    narrative = _narrative_from_fields(values) if values else ""
    mfr_text = _manufacturer_from_fields(values) if values else None

    raw = ""
    saw_cbi_in_pagetext = False
    if not narrative or len(narrative.split()) < _MIN_NARRATIVE_WORDS:
        try:
            with pdfplumber.open(path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            raw = "\n".join(pages)

            sliced = _slice_narrative(raw)
            if _is_cbi_refusal(sliced):
                saw_cbi_in_pagetext = True
            elif len(sliced.split()) >= _MIN_NARRATIVE_WORDS:
                narrative = sliced

            # Standalone-attachment fallback: some manufacturers file the
            # narrative as its own separate PDF with no OL 316 form scaffold at
            # all (often named with "narrative" in the filename). Such a
            # document has no "SECTION 5" header to slice after -- the relevant
            # content usually *is* the page text. Only engage this when the
            # page text does NOT look like the standard multi-section OL 316
            # form, to avoid swallowing a normal form page some other way.
            if (not narrative or len(narrative.split()) < _MIN_NARRATIVE_WORDS) \
                    and raw and not _looks_like_standard_form(raw):
                candidate = _clean_attachment_text(raw)
                if _is_cbi_refusal(candidate):
                    saw_cbi_in_pagetext = True
                elif len(candidate.split()) >= _MIN_NARRATIVE_WORDS:
                    narrative = candidate
        except Exception as e:  # noqa: BLE001
            print(f"[parse] page-text fallback failed {path}: {e}")
            return None, "failed"

    manufacturer = _guess_manufacturer(os.path.basename(path),
                                       (mfr_text or "") + " " + raw[:400])

    if not narrative or len(narrative.split()) < _MIN_NARRATIVE_WORDS:
        if saw_cbi_in_fields or saw_cbi_in_pagetext:
            print(f"[parse] CBI refusal (no narrative provided): "
                  f"{os.path.basename(path)} -- excluded, not annotated as null")
            return None, "cbi_refusal"
        print(f"[parse] no narrative found in {os.path.basename(path)} "
              f"(may be on a separate attached sheet, or unreadable scan)")
        return None, "no_narrative"

    rec = {
        "report_id": os.path.splitext(os.path.basename(path))[0],
        "source": "ol316",
        "manufacturer": manufacturer,
        "narrative": narrative.strip(),
        "struct_severity": None,  # default; overwritten below if Section 4 resolves it
    }
    # Reporter-filed checkbox codes: the OL 316 side of the distant-supervision
    # key. Absent for scanned/flattened filings, which simply contribute no
    # supervision rather than being coded as unknown.
    struct = _ol316_structured(checkboxes)
    struct.pop("_unmapped_checkboxes", None)
    rec.update(struct)
    # OL 316 is filed only by manufacturers testing SAE L3-5 vehicles, so an
    # engaged system on this form is always ADS, never L2 ADAS.
    rec["struct_report_type"] = "ads"
    return rec, "ok"


def _slice_narrative(text: str) -> str:
    m = _DESC_START.search(text)
    if not m:
        return ""
    tail = text[m.end():]
    e = _DESC_END.search(tail)
    block = tail[: e.start()] if e else tail
    block = _MODE_LINE.sub(" ", block)          # strip the mode-checkbox label
    return re.sub(r"\s+", " ", block).strip()


def _longest_text_block(page_text: str) -> str:
    blocks = re.split(r"\n{2,}", page_text)
    blocks = [re.sub(r"\s+", " ", b).strip() for b in blocks]
    return max(blocks, key=len) if blocks else ""


def _guess_manufacturer(filename: str, text: str) -> str:
    known = ["waymo", "cruise", "zoox", "nuro", "apple", "pony", "aurora",
             "mercedes", "tesla", "gatik", "motional", "wayve", "didi"]
    hay = (filename + " " + text[:400]).lower()
    for k in known:
        if k in hay:
            return k
    return "unknown"


def _read_sgo_csv(path: str) -> Optional[pd.DataFrame]:
    for enc in _CSV_ENCODINGS:
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False,
                               low_memory=False, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    print(f"[parse] could not decode {path} with any known encoding; skipping")
    return None


def _load_sgo_frame(paths: list[str]) -> pd.DataFrame:
    """Read every SGO CSV (both the "current" and "Archive" file sets) and
    collapse to one row per Report ID, keeping the highest Report Version.

    NHTSA publishes SGO incident reports as a "current" file set (third
    amendment, June 2025 onward) plus an "Archive" set (2021-2025) that cover
    overlapping Report IDs: a report can appear in the archive at Report
    Version 1-2 and again in the current file at Version 3 once amended, with
    different narrative text each time. `glob` matches both file sets (both
    contain "Incident_Reports" in the name), so simply concatenating them --
    as this used to do -- puts the same incident in the corpus multiple times.
    That is not just redundant: because the extraction step and the gold
    hand-annotation sample each independently pick a row per Report ID, they
    can end up disagreeing on which version, silently scoring the model
    against narrative text the human annotators never saw. Collapsing to the
    max Report Version here, before any row reaches load_sgo_csv, makes every
    downstream step key off a single canonical narrative per Report ID.
    """
    frames = []
    for p in paths:
        df = _read_sgo_csv(p)
        if df is None:
            continue
        # The ADS and ADAS incident files share a schema but describe different
        # automation levels, and the "Automation Engaged?" column is Yes/No in
        # both. Which file a row came from is the ONLY thing that resolves an
        # engaged system to ads_engaged vs adas_engaged, so it is preserved.
        df = df.assign(_source_file=os.path.basename(p))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True)
    cols = {c.lower().strip(): c for c in all_df.columns}
    c_id = cols.get("report id") or cols.get("report_id")
    if c_id is None:
        return all_df  # no id column at all; nothing to key the dedup on
    c_ver = cols.get("report version")
    version = (pd.to_numeric(all_df[c_ver], errors="coerce")
              if c_ver is not None else pd.Series(np.nan, index=all_df.index))
    all_df = (all_df.assign(_version=version.fillna(-1))
                    .sort_values("_version", ascending=False)
                    .drop_duplicates(subset=c_id, keep="first")
                    .drop(columns="_version")
                    .reset_index(drop=True))
    return all_df


def load_sgo_csv(df: pd.DataFrame) -> Iterator[dict]:
    """Yield usable SGO rows, skipping redacted, disclaimer-only,
    pointer-only, and CBI-refusal narratives.

    `df` must already be deduplicated to one row per Report ID (see
    _load_sgo_frame) -- this function does not dedupe.

    Two distinct non-narrative patterns are excluded here, both counted by the
    caller (see build()):
      - insufficient narrative content: after stripping every known non-
        narrative element -- bracketed redaction placeholders (whole-span, not
        just the trigger word: "[REDACTED, MAY CONTAIN CONFIDENTIAL BUSINESS
        INFORMATION]" strips entirely, not just "REDACTED"), bare REDACTED/XXX
        tokens, and a fixed legal-preservation disclaimer some entities prepend
        to every filing (e.g. "Without agreeing ... that [Company] is
        obligated to respond to the General Order, [Company] provides this
        information: Video:[XXX]") -- what remains must clear
        _MIN_SGO_NARRATIVE_WORDS or the row is excluded (_narrative_after_redaction).
        This also catches short boilerplate lead-ins like "As stated by the
        vehicle custodian" with nothing narrative following.
      - a natural-language CBI-refusal sentence in the narrative field itself,
        e.g. "Waymo is seeking CBI protection for the narrative and I am unable
        to provide one." (_is_cbi_refusal). This is a non-response, not a
        narrative, and must not be passed through to annotation/extraction just
        because it reads as a well-formed sentence.
    """
    cols = {c.lower().strip(): c for c in df.columns}

    def col(*cands):
        """Resolve to the FIRST matching column name (single-column fields)."""
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    def cols_all(*cands):
        """Resolve to EVERY matching column name, in candidate order.

        The ADS and ADAS incident files do not use identical column names for
        the same concept. Concatenating them produces a frame in which both
        spellings exist as separate columns and each is NaN for the rows that
        came from the other file. Resolving such a field to a single column
        therefore silently drops every row from the other file -- which is why
        `lighting` and `automation engaged` were populated on only 2270 and 2259
        of 3503 rows respectively. Reading all matching columns and coalescing
        row-wise (see `g`) recovers them.
        """
        return [cols[c] for c in cands if c in cols]

    c_narr = col("narrative")
    c_id = col("report id", "report_id", "report id version", "reportid")
    c_make = col("make", "reporting entity", "manufacturer")
    # These share the cross-file column-name divergence described in cols_all,
    # so they are coalesced too rather than resolved to a single spelling.
    c_sev = cols_all("highest injury severity alleged",
                     "highest severity alleged", "highest injury severity")
    # The SGO third amendment (June 2025) renamed several of these and dropped
    # others outright, so the current and archive files must both be covered:
    #   SV Was Vehicle Towed?      -> Was Any Vehicle Towed?
    #   SV Any Air Bags Deployed?  -> Any Air Bags Deployed?
    #   ADS Equipped?              -> Engagement Status
    #   Lighting                   -> dropped, no replacement
    c_tow = cols_all("sv was vehicle towed?", "was any vehicle towed?",
                     "vehicle towed", "towed")
    c_air = cols_all("sv any air bags deployed?", "any air bags deployed?",
                     "air bags deployed?", "airbag deployed")
    c_eng = cols_all("engagement status", "ads/adas - automation engaged?",
                     "automation engaged", "ads equipped?")
    # Distant-supervision key: the reporting entity's own structured coding of
    # fields the extraction schema also targets. These are labels we did not
    # produce, available on every filing, and they are what allows extraction
    # accuracy to be measured at corpus scale rather than at hand-annotation
    # scale. See src/schema/distant_map.py for the value-level mapping.
    c_crashwith = cols_all("crash with", "sv contact area - crash partner",
                           "crash partner", "other vehicle - vehicle type")
    c_move = cols_all("sv precrash movement", "sv pre-crash movement",
                      "precrash movement")
    c_road = cols_all("roadway type", "roadway", "road type")
    # Lighting exists only in the archive files. The third amendment removed the
    # column with no replacement, so roughly a third of the corpus supplies no
    # lighting supervision at all -- a limitation of the source, not a parse
    # failure. Coverage is reported per field for this reason.
    c_light = cols_all("lighting", "light condition", "lighting conditions")
    c_speed = cols_all("sv precrash speed (mph)", "sv precrash speed",
                       "precrash speed (mph)")

    # SGO does NOT store weather as a categorical. It stores a row of
    # independent indicator columns ("Weather - Clear", "Weather - Rain",
    # "Weather - Snow", ...), each 'Y' or blank, because a filing may assert
    # several conditions at once. Resolving `weather` to the first matching
    # column read the CLEAR INDICATOR as though it were the weather value,
    # producing a field whose only observed values were 'Y' and ' ' -- i.e. no
    # weather information at all. All indicator columns are collected here and
    # reduced jointly downstream.
    c_weather_flags = {c[len("weather - "):].strip(): cols[c]
                       for c in cols if c.startswith("weather - ")}

    def g(row, c):
        """Coalesce the first non-missing value across candidate columns.

        NaN is truthy in Python, so the previous `value or None` idiom did NOT
        filter the NaNs introduced by concatenating files with differing
        columns; they propagated into narratives.jsonl and then into every
        downstream consumer as float('nan') labels.
        """
        if not c:
            return None
        for name in ([c] if isinstance(c, str) else c):
            v = row.get(name)
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            if isinstance(v, str) and not v.strip():
                continue
            return v
        return None

    def weather_flags(row):
        on = {k: str(row.get(v)).strip()
              for k, v in c_weather_flags.items()
              if str(row.get(v, "")).strip().lower() in {"y", "yes", "1", "true"}}
        return on or None

    def report_type(row):
        """ADS vs ADAS, recovered from the originating filename."""
        src = str(row.get("_source_file", "")).lower()
        if "adas" in src:
            return "adas"
        if "ads" in src:
            return "ads"
        return None

    for i, row in df.iterrows():
        narr = (row.get(c_narr, "") if c_narr else "").strip()
        rid = (row.get(c_id) if c_id else f"sgo_{i}") or f"sgo_{i}"
        if not narr:
            continue
        remaining = _narrative_after_redaction(narr)
        if len(remaining.split()) < _MIN_SGO_NARRATIVE_WORDS:
            print(f"[parse] redacted (insufficient narrative remains): "
                  f"sgo report {rid} -- excluded, not annotated as null")
            yield {"report_id": rid, "_status": "redacted"}
            continue
        if _is_cbi_refusal(narr):
            print(f"[parse] CBI refusal (no narrative provided): "
                  f"sgo report {rid} -- excluded, not annotated as null")
            yield {"report_id": rid, "_status": "cbi_refusal"}
            continue
        if _is_non_incident(narr):
            print(f"[parse] non-qualifying incident / retraction: "
                  f"sgo report {rid} -- excluded, not annotated as null")
            yield {"report_id": rid, "_status": "non_incident"}
            continue
        yield {
            "report_id": rid,
            "source": "sgo",
            "manufacturer": (row.get(c_make, "unknown") if c_make else "unknown"),
            "narrative": narr,
            "struct_severity": g(row, c_sev),
            "struct_towed": g(row, c_tow),
            "struct_airbag": g(row, c_air),
            "struct_engaged": g(row, c_eng),
            "struct_crashwith": g(row, c_crashwith),
            "struct_movement": g(row, c_move),
            "struct_roadway": g(row, c_road),
            "struct_lighting": g(row, c_light),
            "struct_weather_flags": weather_flags(row),
            "struct_speed": g(row, c_speed),
            "struct_report_type": report_type(row),
            "_status": "ok",
        }


def build(narr_out: str = NARR_OUT) -> dict:
    os.makedirs(INTERIM, exist_ok=True)
    n_ol316 = n_sgo = 0
    n_cbi_refusal = n_no_narrative = n_failed = 0
    n_sgo_cbi_refusal = n_sgo_redacted = n_sgo_non_incident = 0
    with open(narr_out, "w") as out:
        for pdf in sorted(glob.glob("data/raw/ol316/*.pdf")):
            rec, status = parse_ol316_pdf(pdf)
            if status == "ok":
                out.write(json.dumps(rec) + "\n")
                n_ol316 += 1
            elif status == "cbi_refusal":
                n_cbi_refusal += 1
            elif status == "no_narrative":
                n_no_narrative += 1
            else:  # "failed"
                n_failed += 1
        sgo_paths = sorted(glob.glob("data/raw/sgo/*Incident_Reports*.csv"))
        sgo_df = _load_sgo_frame(sgo_paths)
        for rec in load_sgo_csv(sgo_df):
            status = rec.pop("_status", "ok")
            if status == "cbi_refusal":
                n_sgo_cbi_refusal += 1
                continue
            if status == "redacted":
                n_sgo_redacted += 1
                continue
            if status == "non_incident":
                n_sgo_non_incident += 1
                continue
            out.write(json.dumps(rec) + "\n")
            n_sgo += 1

    n_ol316_total = n_ol316 + n_cbi_refusal + n_no_narrative + n_failed
    summary = {
        "ol316_usable": n_ol316,
        "ol316_cbi_refusal": n_cbi_refusal,     # report this rate in the paper
        "ol316_no_narrative": n_no_narrative,
        "ol316_failed_to_parse": n_failed,
        "ol316_total_pdfs": n_ol316_total,
        "sgo_usable": n_sgo,
        "sgo_cbi_refusal": n_sgo_cbi_refusal,   # report this rate in the paper too
        "sgo_redacted": n_sgo_redacted,         # bracket-redacted, insufficient text
        "sgo_non_incident": n_sgo_non_incident, # no-incident filings + retractions
        "total_usable": n_ol316 + n_sgo,
    }
    print(f"[parse] {summary}")
    stats_path = os.path.join(INTERIM, "corpus_stats.json")
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[parse] wrote {stats_path}")
    return summary


def audit_checkboxes(pattern: str = "data/raw/ol316/*.pdf", limit: int = 0) -> dict:
    """Report which OL 316 AcroForm checkbox names are ticked across the corpus
    and which of them _OL316_CHECKBOX_MAP fails to recognize.

    The internal field names on this form are generated from the printed labels
    but are not stable across form revisions, so the name map cannot be assumed
    correct without checking it against the actual filings. Run this before
    reporting any OL 316 distant-supervision numbers: an unrecognized revision
    should surface here as a large `unmapped` list, not as quiet missing
    coverage in the results table.
    """
    from collections import Counter
    seen, unmapped, n_with_boxes = Counter(), Counter(), 0
    paths = sorted(glob.glob(pattern))
    if limit:
        paths = paths[:limit]
    for path in paths:
        states = _form_checkbox_states(path)
        if not states:
            continue
        n_with_boxes += 1
        seen.update(states.keys())
        struct = _ol316_structured(states)
        unmapped.update(struct.get("_unmapped_checkboxes", []))
    report = {
        "pdfs_scanned": len(paths),
        "pdfs_with_ticked_checkboxes": n_with_boxes,
        "most_common_ticked": seen.most_common(40),
        "unmapped_ticked": unmapped.most_common(40),
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import sys
    if "--audit-checkboxes" in sys.argv:
        audit_checkboxes()
    else:
        build()