"""Tests for the severity-distribution comparison.

Hand-built distributions with known shapes. The cases that matter are the ones
where a directional summary would be WRONG -- crossing CDFs -- because that is
the shape these data actually have.
"""
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.severity_dist import (to_common, cdf, dominance, cramer_von_mises,
                                  compare, SEVERITY_COMMON)


def test_to_common_merges_minor_and_moderate():
    """SGO Minor+Moderate collapse to one level; KABCO has no moderate grade."""
    p5 = np.array([0.70, 0.15, 0.10, 0.03, 0.02])
    got = to_common(p5)[0]
    assert np.allclose(got, [0.70, 0.25, 0.03, 0.02])
    assert abs(got.sum() - 1.0) < 1e-12


def test_to_common_handles_a_matrix():
    m = np.array([[0.6, 0.2, 0.1, 0.05, 0.05], [0.9, 0.05, 0.05, 0.0, 0.0]])
    got = to_common(m)
    assert got.shape == (2, 4)
    assert np.allclose(got.sum(axis=1), 1.0)


def test_dominance_detects_a_clear_shift():
    safer = np.array([0.90, 0.08, 0.015, 0.005])
    riskier = np.array([0.70, 0.20, 0.070, 0.030])
    d = dominance(safer, riskier)
    assert d["a_first_order_less_severe"] is True
    assert d["neither_dominates"] is False
    assert d["n_crossings"] == 0


def test_dominance_detects_crossing_and_refuses_a_direction():
    """The real shape: AV lower at 'serious' but higher at 'fatal'.

    Neither distribution dominates, so no single directional claim holds -- which
    is the finding, not a failure to find one.
    """
    av = np.array([0.0, 0.9152, 0.0536, 0.0312])
    human = np.array([0.0, 0.8885, 0.0897, 0.0218])
    d = dominance(av, human)
    assert d["neither_dominates"] is True
    assert d["a_first_order_less_severe"] is False
    assert d["b_first_order_less_severe"] is False
    assert d["n_crossings"] >= 1


def test_dominance_is_symmetric_in_its_two_checks():
    a = np.array([0.9, 0.08, 0.015, 0.005])
    b = np.array([0.7, 0.20, 0.070, 0.030])
    assert dominance(a, b)["a_first_order_less_severe"] == \
        dominance(b, a)["b_first_order_less_severe"]


def test_identical_distributions_dominate_both_ways():
    p = np.array([0.7, 0.2, 0.07, 0.03])
    d = dominance(p, p)
    assert d["a_first_order_less_severe"] and d["b_first_order_less_severe"]
    assert d["neither_dominates"] is False
    assert cramer_von_mises(p, p, 100, 100) == 0.0


def test_cdf_is_monotone_and_ends_at_one():
    p = np.array([0.7, 0.2, 0.07, 0.03])
    c = cdf(p)
    assert np.all(np.diff(c) >= 0)
    assert abs(c[-1] - 1.0) < 1e-12


def test_cvm_grows_with_separation():
    base = np.array([0.7, 0.2, 0.07, 0.03])
    near = np.array([0.68, 0.22, 0.07, 0.03])
    far = np.array([0.3, 0.4, 0.2, 0.1])
    assert cramer_von_mises(base, far, 500, 500) > \
        cramer_von_mises(base, near, 500, 500)


def test_compare_reports_both_directions_and_self_calibration():
    pred = np.array([0.0, 0.92, 0.05, 0.03])
    obs = np.array([0.0, 0.90, 0.07, 0.03])
    human = np.array([0.0, 0.8885, 0.0897, 0.0218])
    r = compare(pred, obs, human, 448, 25790)
    assert r["levels"] == SEVERITY_COMMON
    assert "predicted_vs_human" in r and "observed_vs_human" in r
    assert r["calibration_to_observed"]["total_variation"] >= 0
    assert len(r["predicted_vs_human"]["ratio"]) == 4


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            f(); print(f"ok  {n}")
    print("OK: severity-distribution tests passed")


# --------------------------- prior correction ------------------------------
def test_prior_correct_rows_stay_distributions():
    from models.severity_dist import prior_correct
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(5), size=200)
    prior = np.array([0.86, 0.10, 0.025, 0.0073, 0.0043])
    prior = prior / prior.sum()
    q = prior_correct(P, prior)
    assert q.shape == P.shape
    assert np.allclose(q.sum(axis=1), 1.0)
    assert (q >= 0).all()


def test_prior_correct_shifts_mass_toward_the_common_class():
    """Balanced training over-weights rare classes; the correction undoes that."""
    from models.severity_dist import prior_correct
    P = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
    prior = np.array([0.86, 0.10, 0.025, 0.0073, 0.0043])
    prior = prior / prior.sum()
    q = prior_correct(P, prior)[0]
    assert q[0] > P[0, 0]          # no-injury share rises
    assert q[4] < P[0, 4]          # fatal share falls
    assert np.allclose(q, prior)   # uniform input recovers the prior exactly


def test_prior_correct_is_identity_under_a_uniform_prior():
    from models.severity_dist import prior_correct
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(4), size=50)
    assert np.allclose(prior_correct(P, np.full(4, 0.25)), P)


def test_prior_correct_handles_zero_prior_classes():
    """A class absent from the corpus must not produce NaN."""
    from models.severity_dist import prior_correct
    P = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    q = prior_correct(P, np.array([0.7, 0.3, 0.0]))
    assert np.isfinite(q).all()
    assert np.allclose(q.sum(axis=1), 1.0)


def test_calibration_threshold_is_declared():
    from models.severity_dist import CALIBRATION_TV_MAX
    assert 0.0 < CALIBRATION_TV_MAX < 1.0
