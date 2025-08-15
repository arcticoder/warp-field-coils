import numpy as np
from reactor import (
    Reactor,
    bennett_profile,
    vorticity_evolution,
    drift_poisson_step,
    microwave_maxwell,
    lg_mode,
    kinetics_update,
    adiabatic_mu,
)


def test_bennett_profile_basic():
    r = np.linspace(0, 1.0, 16)
    n = bennett_profile(1.0, 2.0, r)
    assert n.shape == r.shape
    assert n[0] >= n[-1] > 0


def test_kinetics_nonnegativity():
    N = np.array([1.0, 0.5])
    out = kinetics_update(N, dt=1e-3, rates={"k": 0.1, "S": 0.0})
    assert np.all(out >= 0)


def test_vorticity_and_poisson_step():
    omega = np.zeros((32, 32))
    omega[16, 16] = 1.0
    psi = drift_poisson_step(omega, max_iter=5)
    omega2 = vorticity_evolution(omega, psi, nu=1e-3, dt=1e-3, forcing=None)
    assert omega2.shape == omega.shape


def test_microwave_maxwell_dims():
    E = np.ones((8, 8, 3))
    out = microwave_maxwell(E, sigma=1.0, eps_r=2.0, mu_r=1.0, k0=1.0)
    assert out.shape == E.shape
    assert np.all(out <= E + 1e-12)


def test_lg_mode_norm():
    rho = np.linspace(0, 3, 64)
    R = lg_mode(1, 2, rho, w=1.0)
    assert R.shape == rho.shape
    assert R.max() <= 1.0 + 1e-9


def test_adiabatic_mu_scaling():
    mu1 = adiabatic_mu(1.0, 1.0, 1.0)
    mu2 = adiabatic_mu(1.0, 1.0, 2.0)
    assert np.isclose(mu2, mu1 / 2.0)


def test_reactor_step_smoke():
    R = Reactor(grid=(32, 32))
    s0 = R.state.copy()
    s1 = R.step(dt=1e-3)
    assert s1.shape == s0.shape
    assert not np.allclose(s1, s0)
