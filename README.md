# Harmonized crash comparison: AV (SGO) vs. human (CRSS)

Extracted from `av-crash-nlp`. Compares NHTSA SGO autonomous-vehicle crashes
against NHTSA CRSS police-reported human crashes. Both sides are restricted to a
common outcome bar (any-injury, and separately tow-away) and to the same incident
years.

## Layout

```
src/fetch/fetch_crss.py       download + unzip CRSS releases -> data/raw/crss/
src/fetch/parse_ol316.py      dependency only: _load_sgo_frame (SGO incident years)
src/schema/crss_map.py        CRSS codes <-> shared schema enums
src/models/av_vs_human.py     SGO vs CRSS, threshold-harmonized, PSU bootstrap
src/models/severity_dist.py   severity distributions + stochastic dominance tests
src/utils/config.py           reads config.yaml
tests/                        test_crss_map.py, test_severity_dist.py (synthetic)
data/raw/{crss,sgo}           raw inputs
data/interim/narratives.jsonl, data/processed/extractions.jsonl   AV-side inputs
data/processed/lift_test.json                                     severity_dist input
data/processed/{av_vs_human,severity_dist}.json                   current outputs
```

## Run (from this folder)

```bash
export PYTHONPATH=src
python3 -m fetch.fetch_crss
python3 -m models.av_vs_human --n-boot 500
python3 -m models.severity_dist --threshold any_injury
python3 -m pytest tests -q
```

Table and figure outputs are written to `paper/tables/` and `paper/figures/`.
