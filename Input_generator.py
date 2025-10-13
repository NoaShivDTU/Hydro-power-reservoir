import numpy as np
import pandas as pd

# -------------------------
# Helpers
# -------------------------

def convolve_poly(a, b):
    """Convolve two 1D polynomials a(B) and b(B) given as arrays of coefficients in increasing lag order."""
    out = np.zeros(len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        out[i:i+len(b)] += ai * b
    return out

def simulate_ar_from_poly(mu, ar_poly, eps, T_total, y0=None):
    """
    Simulate y_t with (ar_poly)(y_t - mu) = eps_t.
    ar_poly is array a with a[0]=1 and y_t - mu satisfies sum_k a[k]*(y_{t-k}-mu) = eps_t.
    """
    assert np.isclose(ar_poly[0], 1.0), "AR poly must have leading 1"
    y = np.zeros(T_total)
    if y0 is None:
        # start at mean
        y[:len(ar_poly)] = mu
    else:
        y[:len(y0)] = y0[:len(y)]
        if len(y0) < len(y):
            y[len(y0):] = mu
    for t in range(T_total):
        s = eps[t]
        # sum a[k]*(y_{t-k}-mu) for k>=1
        for k in range(1, min(len(ar_poly), t+1)):
            s -= ar_poly[k] * (y[t-k] - mu)
        y[t] = mu + s
    return y

def simulate_arma_inflow(ps1, ph1, ph2, ph41, xi, T_total, nu0=None):
    """
    (1 - ps1 B)(1 - B) nu_t = (1 - ph1 B - ph2 B^2)(1 - ph41 B^41) xi_t
    Expand to get AR: a(B) = 1 - (1+ps1)B + ps1 B^2
    MA: c(B) = 1 - ph1 B - ph2 B^2 - ph41 B^41 + ph1*ph41 B^42 + ph2*ph41 B^43
    """
    # AR coefficients (a0..a2)
    a = np.zeros(3)
    a[0] = 1.0
    a[1] = -(1.0 + ps1)
    a[2] = ps1

    # MA coefficients length 44 (up to lag 43)
    c = np.zeros(44)
    c[0] = 1.0
    if len(c) > 1: c[1] -= ph1
    if len(c) > 2: c[2] -= ph2
    if len(c) > 41: c[41] -= ph41
    if len(c) > 42: c[42] += ph1 * ph41
    if len(c) > 43: c[43] += ph2 * ph41

    nu = np.zeros(T_total)
    if nu0 is not None and len(nu0) >= 2:
        nu[0] = nu0[0]
        nu[1] = nu0[1]
    for t in range(T_total):
        rhs = 0.0
        # MA side
        for k in range(min(len(c), t+1)):
            rhs += c[k] * xi[t-k]
        # AR side: a0*nu_t + a1*nu_{t-1} + a2*nu_{t-2} = rhs
        # => nu_t = rhs - a1*nu_{t-1} - a2*nu_{t-2}
        if t == 0:
            nu[t] = rhs
        elif t == 1:
            nu[t] = rhs - a[1]*nu[t-1]
        else:
            nu[t] = rhs - a[1]*nu[t-1] - a[2]*nu[t-2]
    return nu

# -------------------------
# Main generator (paper-aligned)
# -------------------------
# --------------------- NORDIC SCENARIO GENERATOR (drop-in) ---------------------
import numpy as np

def _nordic_diurnal_profile():
    """
    24h shape roughly matching Nord Pool: low overnight, morning ramp, evening peak.
    Returns length-24 array with mean 0 and unit std (we scale it later).
    """
    p = np.array([
        -0.55, -0.60, -0.62, -0.60, -0.50, -0.35, -0.10,  0.10,
         0.25,  0.35,  0.30,  0.20,  0.10,  0.05,  0.10,  0.25,
         0.45,  0.70,  0.85,  0.55,  0.25,  0.00, -0.25, -0.45
    ], dtype=float)
    p = p - p.mean()
    p = p / (np.std(p) + 1e-9)
    return p

def _ar1_series(T, phi, sigma, rng, x0=0.0):
    x = np.empty(T, dtype=float)
    x_prev = x0
    std_eps = sigma
    for t in range(T):
        eps = rng.normal(0.0, std_eps)
        x_t = phi * x_prev + eps
        x[t] = x_t
        x_prev = x_t
    return x

def _spike_process(T, rng, p_pos=0.02, p_neg=0.015, pos_scale=22.0, neg_scale=15.0):
    """
    Mixture spikes: rare, heavy-tailed. magnitudes ~ lognormal.
    Positive & negative spikes; negative spikes allow negative prices.
    """
    s = np.zeros(T, dtype=float)
    if p_pos > 0:
        pos_mask = rng.random(T) < p_pos
        s[pos_mask] += rng.lognormal(mean=np.log(pos_scale), sigma=0.5, size=pos_mask.sum())
    if p_neg > 0:
        neg_mask = rng.random(T) < p_neg
        s[neg_mask] -= rng.lognormal(mean=np.log(neg_scale), sigma=0.5, size=neg_mask.sum())
    return s

def generate_nordic_prices(N, T, seed=42, burn_in=400,
                           base=20.0,
                           diurnal_amp=7.0,     # €/MWh amplitude of daily shape
                           weekend_drop=2.0,    # €/MWh on Sat/Sun
                           phi=0.85, sigma=4.0, # AR(1) residuals
                           p_pos=0.02, p_neg=0.015,
                           spike_pos_scale=22.0, spike_neg_scale=15.0,
                           p_min=-80.0, p_max=250.0):
    """
    Returns price_samples with shape [N, T-1].
    """
    rng = np.random.default_rng(seed)
    prof24 = _nordic_diurnal_profile()  # mean 0, unit std
    H = burn_in + (T - 1)  # price length per scenario

    prices = np.empty((N, T - 1), dtype=float)
    for n in range(N):
        # Deterministic components
        hours = np.arange(H)
        diurnal = diurnal_amp * prof24[hours % 24]
        dow = (hours // 24) % 7
        weekend_adj = np.where(dow >= 5, -weekend_drop, 0.0)  # Sat=5, Sun=6

        # Stochastic components
        resid = _ar1_series(H, phi=phi, sigma=sigma, rng=rng, x0=0.0)
        spikes = _spike_process(H, rng, p_pos=p_pos, p_neg=p_neg,
                                pos_scale=spike_pos_scale, neg_scale=spike_neg_scale)

        path = base + diurnal + weekend_adj + resid + spikes
        path = np.clip(path, p_min, p_max)

        # drop burn-in
        prices[n, :] = path[burn_in:]
    return prices

def generate_correlated_inflows(N, T, J, seed=42, burn_in=0,
                                means=None,       # per-reservoir mean inflow
                                phi=0.985,        # high persistence
                                sigmas=None,      # per-reservoir innovation std
                                rho=0.85,         # cross-correlation of noise
                                storm_rate_per_day=0.35,
                                storm_dur_mean_h=12,
                                storm_amp_scale=1.6,
                                min_inflow=0.0):
    """
    Returns inflow_samples with shape [N, T, J].
    High-persistence AR(1) with correlated noise + shared storm pulses.
    """
    rng = np.random.default_rng(seed + 10)
    if means is None:
        means = np.full(J, 50.0, dtype=float)
    if sigmas is None:
        sigmas = np.full(J, 0.30, dtype=float)

    # Correlated noise via Cholesky (same rho for off-diagonals)
    C = np.full((J, J), rho, dtype=float)
    np.fill_diagonal(C, 1.0)
    L = np.linalg.cholesky(C)
    D = np.diag(sigmas)

    inflow = np.empty((N, T, J), dtype=float)
    for n in range(N):
        # AR(1) baseline
        x = np.zeros((burn_in + T, J), dtype=float)
        x[0, :] = means  # start at mean
        for t in range(1, burn_in + T):
            z = L @ rng.standard_normal(J)  # correlated N(0, I)
            eps = (D @ z)                   # scale per reservoir
            x[t, :] = means * (1 - phi) + phi * x[t-1, :] + eps

        series = x[burn_in:, :]  # drop burn-in

        # Storm pulses (compound Poisson), shared across reservoirs
        lam = storm_rate_per_day * (T / 24.0)
        k = rng.poisson(lam=lam)
        if k > 0:
            pulses = np.zeros((T, J), dtype=float)
            for _ in range(k):
                start = rng.integers(0, T)
                dur = int(np.clip(rng.normal(storm_dur_mean_h, 4.0), 4, 30))
                amp = rng.lognormal(mean=np.log(storm_amp_scale), sigma=0.5)
                # shared shape (triangular), mild spatial heterogeneity (5%)
                shape = np.linspace(0.2, 1.0, num=dur)
                shape = np.concatenate([shape, shape[::-1]])  # up & down
                end = min(T, start + shape.size)
                seg = shape[:end - start][:, None] * amp
                hetero = 1.0 + 0.05 * rng.standard_normal(J)  # ±5%
                pulses[start:end, :] += seg * hetero[None, :]
            series += pulses

        inflow[n, :, :] = np.clip(series, min_inflow, None)

    return inflow
# ---------------------------------------------------------------------------


def generate_input(N, T=48, burn_in=200, seed=None):
    rng = np.random.default_rng(seed)

    # --- System setup (Table 2) ---
    J = 2
    pi_max = np.array([57.96, 121.36])      # 10^3 m^3/h
    pi_min = np.zeros_like(pi_max)
    l_max  = np.array([1130.0, 1000.0])     # 10^3 m^3
    l_min  = np.array([ 113.0,  100.0])     # 10^3 m^3
    l0     = np.array([124.3, 110.0])       # 10^3 m^3
    g      = np.array([0.1101, 0.5051])     # MWh per 10^3 m^3

    # Cascade topology: Upper discharges into Lower; Lower to sink
    # Level update: l_{t+1} = l_t + inflow + R * discharge
    R = np.array([[-1.0,  0.0],
                  [ 1.0, -1.0]])
    
        # --- Simulation lengths ---
    T_total = burn_in + T
    

    # --- Price process (Eq. 13a) in LEVELS (no log) ---
    mu_price = 30.0    # $/MWh (level)
    theta1   = 0.6874
    eta1     = 0.9234
    eta24    = 0.8502
    eta168   = 0.9665
    sigma_eps = 0.2369  # innovation std in $/MWh (level)

    # AR polynomial: (1 - θ1 B)(1 - η1 B)(1 - η24 B^24)(1 - η168 B^168)
    p1   = np.array([1.0, -theta1])
    p1b  = np.array([1.0, -eta1])
    p24  = np.zeros(25);  p24[0] = 1.0;  p24[24] = -eta24
    p168 = np.zeros(169); p168[0] = 1.0; p168[168] = -eta168
    ar_price_poly = convolve_poly(convolve_poly(convolve_poly(p1, p1b), p24), p168)

    price_samples = np.zeros((N, T))
    for n in range(N):
        eps = rng.normal(0.0, sigma_eps, size=T_total)  # LEVEL innovations
        y = simulate_ar_from_poly(mu_price, ar_price_poly, eps, T_total)
        # optional, for safety only: keep prices within a realistic band
        y = np.clip(y, 0.0, 200.0)
        price_samples[n, :] = y[-T:]

        
    # --- Inflow process (Eq. 13b) params by reservoir ---
    psi1  = np.array([0.9899, 0.9775])
    phi1  = np.array([1.3156, 1.4442])
    phi2  = np.array([-0.3504, -0.5509])
    phi41 = np.array([0.8424, 0.8304])

    # Noise stds and correlation
    sigma_xi   = np.array([0.6549, 0.1646])
    corr_xi12  = 0.0417
    cov12      = corr_xi12 * sigma_xi[0] * sigma_xi[1]
    cov        = np.array([[sigma_xi[0]**2, cov12],
                        [cov12,          sigma_xi[1]**2]])

    mu_inflow = np.array([50.0, 50.0])   # mean inflow (10^3 m^3/h)

    # --- Simulate inflow paths (correlated noises), on anomalies then add μ back ---
    inflow_samples = np.zeros((N, T, J))
    for n in range(N):
        Xi = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=T_total)
        for j in range(J):
            # simulate anomaly process ~ Eq.13b on (nu - mu)
            nu_anom = simulate_arma_inflow(
                ps1=psi1[j], ph1=phi1[j], ph2=phi2[j], ph41=phi41[j],
                xi=Xi[:, j], T_total=T_total,
                nu0=np.array([0.0, 0.0])   # anomalies start at 0
            )
            nu_j = mu_inflow[j] + nu_anom
            inflow_samples[n, :, j] = np.clip(nu_j[-T:], 0.0, None)  # keep nonnegative
    # --- Nordic-style synthetic data (keeps shapes expected by ADP/SDDP) ---
    # price_samples: [N, T-1]
    price_samples = generate_nordic_prices(
        N=N, T=T, seed=seed, burn_in=burn_in,
        base=30.0,          # typical Nord Pool level (adjust freely)
        diurnal_amp=7.0,    # daily swing amplitude in €/MWh
        weekend_drop=2.0,   # lower prices on Sat/Sun
        phi=0.85, sigma=4.0,
        p_pos=0.02, p_neg=0.015,
        spike_pos_scale=22.0, spike_neg_scale=15.0,
        p_min=-80.0, p_max=250.0
    )

    # inflow_samples: [N, T, J]
    inflow_samples = generate_correlated_inflows(
        N=N, T=T, J=J, seed=seed, burn_in=0,
        means=np.full(J, 50.0),
        phi=0.985, sigmas=np.full(J, 0.30),
        rho=0.85,                      # strong spatial correlation
        storm_rate_per_day=0.35,       # ~one storm every ~3 days
        storm_dur_mean_h=12,           # 6–24h typical
        storm_amp_scale=1.6,           # average storm bump (units)
        min_inflow=0.0
    )
# ----------------------------------------------------------------------


    """
    # --- Price process (Eq. 13a style) ---
    mu_price = 20.0            # $/MWh
    theta1   = 0.6874
    eta1     = 0.9234
    eta24    = 0.8502
    eta168   = 0.9665
    sigma_eps = 0.2369         # std of ε_t

    # Build AR polynomial: (1 - θ1 B)(1 - η1 B)(1 - η24 B^24)(1 - η168 B^168)
    p1   = np.array([1.0, -theta1])
    p1b  = np.array([1.0, -eta1])
    p24  = np.zeros(25); p24[0] = 1.0; p24[24] = -eta24
    p168 = np.zeros(169); p168[0] = 1.0; p168[168] = -eta168
    ar_price_poly = convolve_poly(convolve_poly(convolve_poly(p1, p1b), p24), p168)

    # --- Inflow process (Eq. 13b) params by reservoir ---
    psi1 = np.array([0.9899, 0.9775])
    phi1 = np.array([1.3156, 1.4442])
    phi2 = np.array([-0.3504, -0.5509])
    phi41 = np.array([0.8424, 0.8304])

    # Noise stds and correlation
    sigma_xi = np.array([0.6549, 0.1646])
    corr_xi12 = 0.0417
    cov12 = corr_xi12 * sigma_xi[0] * sigma_xi[1]
    cov = np.array([[sigma_xi[0]**2, cov12],
                    [cov12,          sigma_xi[1]**2]])

    # --- Simulation lengths ---
    T_total = burn_in + T

    # --- Simulate price paths ---
    price_samples = np.zeros((N, T))
    for n in range(N):
        eps = rng.normal(0.0, sigma_eps, size=T_total)
        # simulate y_t = price_t with AR poly on (y - mu)
        y = simulate_ar_from_poly(mu_price, ar_price_poly, eps, T_total)
        price_samples[n, :] = y[-T:]

    # --- Simulate inflow paths (correlated noises) ---
    inflow_samples = np.zeros((N, T, J))
    for n in range(N):
        # draw correlated xi_t for all t
        Xi = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=T_total)
        # Initial two inflow values from paper (50) to anchor recursion
        nu0 = np.array([50.0, 50.0])
        for j in range(J):
            nu_j = simulate_arma_inflow(
                ps1=psi1[j], ph1=phi1[j], ph2=phi2[j], ph41=phi41[j],
                xi=Xi[:, j], T_total=T_total,
                nu0=np.array([nu0[j], nu0[j]])
            )
            # keep last T and clip to nonnegative
            inflow_samples[n, :, j] = np.clip(nu_j[-T:], 0.0, None)
    """
    # --- Initial values per paper ---
    nu0_vec = np.array([50.0, 50.0])   # 10^3 m^3/h
    rho0 = 20.0                        # $/MWh

    # --- ADP init scaffolding (unchanged) ---
    a_t = [np.zeros((J, 1)) for _ in range(T)]
    b_t = [np.zeros(J) for _ in range(T)]
    l_bar = np.tile(l0, (N, 1))

    return (J, T, l_max, l_min, l0, pi_max, pi_min,
            price_samples, inflow_samples, nu0_vec, rho0, R, a_t, b_t, l_bar, g)
