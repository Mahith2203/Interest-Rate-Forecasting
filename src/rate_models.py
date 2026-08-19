"""
Stochastic short-rate models: Vasicek and Cox-Ingersoll-Ross (CIR).

Both are one-factor mean-reverting models of the form:
    dr_t = a(b - r_t) dt + sigma * f(r_t) dW_t

Vasicek uses f(r_t) = 1        -> allows negative rates, closed-form tractable.
CIR     uses f(r_t) = sqrt(r_t) -> keeps rates non-negative (given Feller condition).
"""

import numpy as np


def vasicek_sim(r0: float, a: float, b: float, sigma: float,
                 T: float, dt: float, seed: int | None = None) -> np.ndarray:
    """
    Simulate a short-rate path under the Vasicek model using Euler-Maruyama.

    Parameters
    ----------
    r0    : initial short rate
    a     : speed of mean reversion
    b     : long-run mean rate
    sigma : volatility
    T     : total time horizon (years)
    dt    : time step (years)
    seed  : optional RNG seed for reproducibility

    Returns
    -------
    np.ndarray of shape (N+1,) with the simulated rate path.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    rates = np.empty(n_steps + 1)
    rates[0] = r0
    for t in range(n_steps):
        dr = a * (b - rates[t]) * dt + sigma * np.sqrt(dt) * rng.standard_normal()
        rates[t + 1] = rates[t] + dr
    return rates


def cir_sim(r0: float, a: float, b: float, sigma: float,
            T: float, dt: float, seed: int | None = None) -> np.ndarray:
    """
    Simulate a short-rate path under the Cox-Ingersoll-Ross (CIR) model
    using Euler-Maruyama with full truncation to keep rates non-negative.

    Feller condition (ensures r_t stays strictly positive in the exact
    process): 2*a*b >= sigma**2

    Parameters are the same as `vasicek_sim`.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    rates = np.empty(n_steps + 1)
    rates[0] = r0
    for t in range(n_steps):
        sqrt_r = np.sqrt(max(rates[t], 0.0))
        dr = a * (b - rates[t]) * dt + sigma * sqrt_r * np.sqrt(dt) * rng.standard_normal()
        rates[t + 1] = max(rates[t] + dr, 0.0)
    return rates


def feller_condition(a: float, b: float, sigma: float) -> bool:
    """Return True if the CIR Feller condition (2ab >= sigma^2) holds."""
    return 2 * a * b >= sigma ** 2
