"""Smoke tests on synthetic data so the skeleton imports and core math runs."""
import numpy as np

from caz import __version__
from caz.config import DEFAULT
from caz.model.pca import fit_pca
from caz.model.regression import fit_regression, identity_link, link_log_pr
from caz.predict import project_pca, back_transform
from caz.uncertainty import monte_carlo_pi
from caz.validate import mb, mae, kge, skill_score


def test_version():
    assert __version__


def test_config_defaults():
    assert DEFAULT.region.lon_min < DEFAULT.region.lon_max
    assert DEFAULT.calibration.cal_years > 0


def test_pca_roundtrip():
    rng = np.random.default_rng(0)
    Nt, Ng, k = 200, 50, 4
    E_true = rng.standard_normal((Ng, k))
    S_true = rng.standard_normal((Nt, k))
    M = S_true @ E_true.T
    M -= M.mean(0)
    res = fit_pca(M, n_components=k)
    rec = res.S @ res.E.T
    assert np.allclose(M, rec, atol=1e-8)


def test_regression_recovers_beta():
    rng = np.random.default_rng(1)
    Nt, Ne = 300, 5
    S = rng.standard_normal((Nt, Ne))
    beta = rng.standard_normal(Ne)
    y = S @ beta + 0.1 * rng.standard_normal(Nt)
    fit = fit_regression(S, y)
    assert np.allclose(fit.beta, beta, atol=0.05)
    assert fit.mse < 0.05


def test_links():
    O = np.array([0.0, 1.0, 5.0])
    g, off = identity_link(O)
    assert np.array_equal(g, O) and off == 0.0
    g2, off2 = link_log_pr(O)
    assert off2 == 1.0 + 0.0


def test_pi_shape():
    Sbeta = np.zeros(50)
    pi = monte_carlo_pi(Sbeta, mse=1.0, n_sim=500)
    assert pi.shape == (50, 2)
    assert (pi[:, 1] > pi[:, 0]).all()


def test_metrics():
    o = np.arange(10.0)
    p = o + 1.0
    assert mb(p, o) == 1.0
    assert mae(p, o) == 1.0
    assert -1 < kge(p, o) <= 1
    assert skill_score(0.5, 1.0, 0.0) == 50.0


def test_back_transform():
    g = np.array([0.0, 1.0])
    assert np.array_equal(back_transform(g, "identity"), g)
    out = back_transform(np.log(np.array([2.0, 3.0])), "log_pr", offset=1.0)
    assert np.allclose(out, [1.0, 2.0])


def test_project_pca():
    rng = np.random.default_rng(2)
    M = rng.standard_normal((20, 8))
    E = rng.standard_normal((8, 3))
    S = project_pca(M, E)
    assert S.shape == (20, 3)
