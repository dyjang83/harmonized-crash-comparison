"""Severity DISTRIBUTION comparison: predicted AV vs. observed AV vs. human.

The lift test asks whether narrative text improves POINT predictions of severity.
This asks a different question: what does the pipeline say the whole severity
distribution looks like, and how does its shape differ from the human one?

Three distributions, all in the four-level common space (schema/crss_map.py):

  predicted AV   column means of the out-of-fold probability matrix produced by
                 lift_test.py. Summing predicted PROBABILITIES gives an expected
                 share at each level; a histogram of argmax labels would not --
                 with 14 fatalities in 3,272 records, argmax can assign the
                 fatal class zero mass while the model still places real
                 probability there. Reported as a DIAGNOSTIC, not as the basis
                 for the human comparison -- see the calibration caveat below.
  observed AV    the structured SGO severity marginal. A sanity check: a model
                 that cannot reproduce its own training marginal is not going to
                 say anything trustworthy about a different population.
  human          CRSS, survey-weighted, restricted to the same harmonized
                 population (models/av_vs_human.py).

WHY SHAPE, NOT DIRECTION. The early read on these data is that AV severity is
neither uniformly higher nor uniformly lower -- it sits below the human
distribution in some regions and above it in others. A single directional
summary ("AVs are safer") would be false in one tail whichever way it is
stated. So the comparison tests for CROSSING rather than for a shift:

  * first-order stochastic dominance is checked in BOTH directions. If both
    fail, neither distribution dominates and no directional claim is supported.
  * the CDF difference and its crossing points are located explicitly.
  * per-level ratios carry intervals, so a crossing driven by a cell of a dozen
    crashes is visibly distinguishable from an established one.
  * a Cramer-von Mises statistic on the ordered categories gives one overall
    shape test.

A point-estimate crossing is NOT a finding on its own. With the fatal cell
resting on ~14 AV crashes, the interval is what decides whether a crossing is
real, and this module reports both so they cannot be conflated.

WHY THE VERDICT RESTS ON THE OBSERVED DISTRIBUTION, NOT THE PREDICTED ONE.
Both lift-test model families are trained with balanced class handling, which is
correct for a ranking task and wrong for a distributional one: predict_proba then
estimates p(y|x) under a uniform prior, and summing it describes a world with as
many fatal crashes as no-injury ones. Uncorrected, gradient boosting puts 11.7%
of harmonized AV crashes at 'fatal' against 3.1% observed, which would report AV
crashes as dramatically more severe purely as a training artifact.

prior_correct() undoes this under a uniform-training-prior assumption, and that
assumption does not hold equally for the two families: it moves gradient
boosting's total-variation distance from the observed marginal from 0.241 to
0.051, but moves logistic regression's from 0.036 to 0.084. Fitting the
correction empirically instead is circular -- matching a five-level marginal with
five free parameters reproduces it on held-out folds almost by construction, and
proves nothing about the model's distributional accuracy.

The decisive point is simpler: the AV severity marginal does not need a model.
`struct_severity` is a directly observed structured SGO field, so estimating it
from predictions adds calibration error for no information gain. The predicted
distribution earns its place where counting cannot reach -- the SGO records whose
structured severity is 'Unknown' -- and that is imputation, a narrower claim than
the marginal. So: the human comparison uses the OBSERVED AV distribution, and the
predicted one is reported beside it with its calibration distance, as a check on
the pipeline rather than as evidence about the population.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema.crss_map import SEVERITY_COMMON  # noqa: E402

# Total-variation distance beyond which the predicted distribution is treated as
# uninformative about any population. A model that cannot reproduce the marginal
# it was fit on is not evidence about a different one.
CALIBRATION_TV_MAX = 0.05

RESULTS = os.path.join("data", "processed")
FIGURES = os.path.join("paper", "figures")
TABLES = os.path.join("paper", "tables")

# lift_test uses the five-level SGO ordinal; the comparison space is four-level
# (Minor and Moderate merged, because KABCO has no moderate grade).
SGO_5 = ["No Injuries Reported", "Minor", "Moderate", "Serious", "Fatality"]
_TO_COMMON = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}   # -> none, minor, serious, fatal


def to_common(p5: np.ndarray) -> np.ndarray:
    """Collapse a 5-level distribution (or probability matrix) to 4 levels."""
    p5 = np.atleast_2d(np.asarray(p5, dtype=float))
    out = np.zeros((p5.shape[0], 4))
    for src, dst in _TO_COMMON.items():
        out[:, dst] += p5[:, src]
    return out


def prior_correct(proba: np.ndarray, prior: np.ndarray,
                  train_prior: Optional[np.ndarray] = None) -> np.ndarray:
    """Put class-balanced probabilities back on the population scale.

    THE PROBLEM. Both model families in the lift test are given balanced class
    handling -- logistic regression via class_weight="balanced", gradient
    boosting via balanced sample weights -- because without it the majority
    class swamps the structured-only baseline and the lift becomes an
    imbalance artifact rather than a representation effect. That is the right
    call FOR THE LIFT TEST, which reads argmax labels and only cares about
    ranking. It is the wrong scale for a DISTRIBUTION: balanced training fits
    p(y|x) as though every class were equally common, so summing predict_proba
    gives the severity distribution of a world with equal numbers of fatal and
    no-injury crashes.

    Measured here: uncorrected, the model puts 11.7% of harmonized AV crashes at
    'fatal' against 3.1% observed, and the distribution comparison then reports
    AV crashes as dramatically more severe -- entirely as an artifact of the
    training weights.

    THE CORRECTION. With a uniform effective training prior, Bayes gives
        p_pop(c|x) proportional to p_model(c|x) * pi_c
    where pi_c is the true class prior. Applied PER ROW and renormalized, since
    renormalization is nonlinear and correcting the aggregated means instead
    would give a different (wrong) answer.

    This does not rescue a badly ranked model -- it only fixes the scale. The
    calibration_to_observed check in compare() is what verifies it worked.
    """
    proba = np.asarray(proba, dtype=float)
    prior = np.asarray(prior, dtype=float)
    k = proba.shape[1]
    train_prior = (np.full(k, 1.0 / k) if train_prior is None
                   else np.asarray(train_prior, dtype=float))
    w = np.divide(prior, train_prior, out=np.zeros(k), where=train_prior > 0)
    p = proba * w
    tot = p.sum(axis=1, keepdims=True)
    # A row whose entire mass sat on classes with zero prior has nothing to
    # rescale onto; leave it uniform rather than emit NaN.
    return np.where(tot > 0, p / np.where(tot > 0, tot, 1.0), 1.0 / k)


def cdf(p: np.ndarray) -> np.ndarray:
    return np.cumsum(np.asarray(p, dtype=float))


def dominance(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> dict:
    """First-order stochastic dominance in both directions, plus crossings.

    On an ordinal severity scale with 0 = least severe, distribution A is
    *less severe* than B in the first-order sense when CDF_A >= CDF_B at every
    level. Checking only one direction and reporting its failure would leave a
    reader to assume the other holds; both are checked, and both failing is the
    informative outcome.
    """
    ca, cb = cdf(a), cdf(b)
    d = ca - cb
    a_less = bool(np.all(d >= -tol))
    b_less = bool(np.all(d <= tol))
    signs = np.sign(np.where(np.abs(d) < tol, 0.0, d))
    nz = signs[signs != 0]
    crossings = int(np.sum(nz[1:] * nz[:-1] < 0)) if len(nz) > 1 else 0
    return {"cdf_a": ca.tolist(), "cdf_b": cb.tolist(),
            "cdf_diff": d.tolist(), "a_first_order_less_severe": a_less,
            "b_first_order_less_severe": b_less,
            "neither_dominates": (not a_less and not b_less),
            "n_crossings": crossings}


def cramer_von_mises(a: np.ndarray, b: np.ndarray, n_a: int, n_b: int) -> float:
    """Two-sample Cramer-von Mises statistic for ordered categories.

    Weighted by the pooled distribution so levels contribute in proportion to
    how often they occur, rather than a rare tail dominating the statistic.
    """
    ca, cb = cdf(a), cdf(b)
    pooled = (n_a * np.asarray(a) + n_b * np.asarray(b)) / (n_a + n_b)
    return float((n_a * n_b) / (n_a + n_b) * np.sum((ca - cb) ** 2 * pooled))


def compare(pred: np.ndarray, observed: np.ndarray, human: np.ndarray,
            n_av: int, n_human: int) -> dict:
    pred, observed, human = (np.asarray(x, dtype=float)
                             for x in (pred, observed, human))
    return {
        "levels": SEVERITY_COMMON,
        "predicted_av": pred.tolist(),
        "observed_av": observed.tolist(),
        "human": human.tolist(),
        # Does the model reproduce the marginal it was fit on? If not, its
        # predicted distribution is not evidence about anything else.
        "calibration_to_observed": {
            "max_abs_diff": float(np.max(np.abs(pred - observed))),
            "total_variation": float(0.5 * np.sum(np.abs(pred - observed))),
        },
        "predicted_vs_human": {
            **dominance(pred, human),
            "cvm": cramer_von_mises(pred, human, n_av, n_human),
            "ratio": (pred / np.where(human > 0, human, np.nan)).tolist(),
        },
        "observed_vs_human": {
            **dominance(observed, human),
            "cvm": cramer_von_mises(observed, human, n_av, n_human),
            "ratio": (observed / np.where(human > 0, human, np.nan)).tolist(),
        },
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Severity distribution: AV vs human.")
    ap.add_argument("--lift-json", default=os.path.join(RESULTS, "lift_test.json"))
    ap.add_argument("--av-human-json",
                    default=os.path.join(RESULTS, "av_vs_human.json"))
    ap.add_argument("--threshold", default="any_injury",
                    help="Harmonized population to compare within.")
    ap.add_argument("--rep", default="S+T", choices=["S", "T", "S+T"])
    ap.add_argument("--family", default=None,
                    help="Model family; defaults to the lift test's headline.")
    a = ap.parse_args()

    lift = json.load(open(a.lift_json))
    pd_block = lift.get("predicted_distribution")
    if not pd_block:
        raise SystemExit(
            f"[severity_dist] {a.lift_json} has no predicted_distribution. "
            "Re-run models.lift_test -- probabilities are only collected by the "
            "current version.")
    family = a.family or lift["headline"]["model"]

    avh = json.load(open(a.av_human_json))
    sev = avh["severity"][a.threshold]
    human = np.array(sev["human_share"])
    n_av, n_human = sev["n_sgo"], sev["n_crss"]

    # The OBSERVED AV distribution must come from the same harmonized population
    # as the human one, so it is read from av_vs_human's threshold-filtered
    # shares -- NOT from the lift test's marginal over the whole SGO pool.
    #
    # Those two agree for the any-injury bar only by coincidence: "any injury" is
    # exactly the complement of "none", so renormalizing the full marginal off
    # `none` reproduces the injury subset. Tow-away is an orthogonal filter and no
    # such equivalence holds -- reading the unfiltered marginal there compared a
    # whole-corpus AV distribution against a tow-away-filtered human one and
    # overstated the leading CDF difference as 0.299 instead of 0.193.
    observed = np.array(sev["av_share"])

    pred5 = np.array(pd_block["by_model"][family][a.rep])
    pred = to_common(pred5)[0]

    # The predicted distribution is over the FULL SGO corpus, while the human
    # distribution is restricted to the harmonized population. Renormalizing the
    # AV side onto the same support is what makes them comparable; without it,
    # the comparison would silently reintroduce the reporting-threshold gap that
    # av_vs_human.py exists to remove.
    # The PREDICTED distribution still comes from the lift test, which fits the
    # whole SGO pool, so it alone needs projecting onto the harmonized support.
    # `observed` is already on it.
    if a.threshold == "any_injury":
        keep = np.array([0.0, 1.0, 1.0, 1.0])
        pred = (pred * keep) / max((pred * keep).sum(), 1e-12)

    rep = compare(pred, observed, human, n_av, n_human)
    rep["threshold"], rep["family"], rep["representation"] = a.threshold, family, a.rep

    out = os.path.join(RESULTS, "severity_dist.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[severity_dist] wrote {out}\n")

    print(f"threshold={a.threshold}  family={family}  rep={a.rep}  "
          f"(AV n={n_av:,}, human n={n_human:,})")
    print(f"{'level':9s} {'pred AV':>9s} {'obs AV':>9s} {'human':>9s} {'pred/human':>11s}")
    for i, lv in enumerate(SEVERITY_COMMON):
        print(f"{lv:9s} {rep['predicted_av'][i]:9.4f} {rep['observed_av'][i]:9.4f} "
              f"{rep['human'][i]:9.4f} {rep['predicted_vs_human']['ratio'][i]:10.2f}x")

    def _verdict(d):
        if d["neither_dominates"]:
            return (f"NEITHER dominates ({d['n_crossings']} CDF crossing(s)) "
                    "-- no directional claim is supported")
        return ("AV distribution is first-order LESS severe"
                if d["a_first_order_less_severe"]
                else "human distribution is first-order LESS severe")

    c = rep["calibration_to_observed"]
    ok = c["total_variation"] <= CALIBRATION_TV_MAX
    rep["predicted_distribution_usable"] = bool(ok)

    d = rep["observed_vs_human"]
    print(f"\n>>> VERDICT (observed AV vs human): {_verdict(d)}")
    print(f"    CvM={d['cvm']:.4f}")

    d = rep["predicted_vs_human"]
    print(f"\n    predicted AV vs human: {_verdict(d)}   CvM={d['cvm']:.4f}")
    print(f"    calibration to observed AV marginal: TV={c['total_variation']:.4f} "
          f"({'within' if ok else 'EXCEEDS'} the {CALIBRATION_TV_MAX:.2f} threshold)")
    if not ok:
        print("    -> the predicted distribution does NOT reproduce the marginal it")
        print("       was fit on, so it is reported as a pipeline diagnostic only and")
        print("       carries no weight in the verdict above. Balanced class training")
        print("       puts predict_proba on a uniform-prior scale; see prior_correct().")


if __name__ == "__main__":
    main()
