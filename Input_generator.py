import numpy as np
import pandas as pd

def generate_input(N):
    # -------------------------
    # System Setup Parameters
    # -------------------------
    J = 2    # two reservoirs: Upper, Lower
    T = 48   # time steps (hours)

    # Reservoir characteristics from table (in 10^3 m^3 or 10^3 m^3/h)
    pi_max = np.array([57.96, 121.36])  # Max discharge (Upper, Lower)
    pi_min = np.zeros_like(pi_max)     # Min discharge

    l_max = np.array([1130.0, 1000.0])  # Max capacity
    l_min = np.array([113.0, 100.0])    # Min capacity
    l0    = np.array([124.3, 110.0])    # Initial level

    # Conversion rate to energy (MWh per 10^3 m^3)
    alpha_energy = np.array([0.1101, 0.5051])  # g_j values

    # Simple discharge-only topology (each reservoir discharges to sink)
    R = -np.eye(J)

    # -------------------------
    # ARMA Model Parameters
    # -------------------------

    # Price model AR(1)
    price_mu = 20.0
    price_phi = 0.7
    price_sigma = 10.0

    # Inflow model AR(1) for each reservoir
    scarcity = 0.35
    volatility = 0.25
    phi = 0.9

    inflow_mu_j = scarcity * pi_max
    inflow_sigma_j = np.minimum(volatility * pi_max, 0.8 * np.maximum(inflow_mu_j, 1e-6))



    # Generate price samples
    price_samples = generate_arma_samples(price_mu, price_phi, price_sigma, N, T)
    df_price = pd.DataFrame(price_samples, columns=[f"Hour {t+1}" for t in range(T)])
    #df_price.to_csv("prices.csv", index=False)

    # Generate inflow samples
    inflow_samples = np.zeros((N, T, J))
    for j in range(J):
        inflow_samples[:, :, j] = generate_arma_samples(
            mu=float(inflow_mu_j[j]),
            phi=phi,
            sigma=float(inflow_sigma_j[j]),
            N=N,
            T=T
        )
    inflow_samples = np.clip(inflow_samples, 0.0, None)

    df_inflow = pd.DataFrame(
        inflow_samples.reshape(N * T, J),
        columns=["Upper", "Lower"]
    )
    #df_inflow.to_csv("inflows.csv", index=False)

    # -------------------------
    # Initial Values
    # -------------------------
    nu0 = inflow_samples[:, 0, :].mean(axis=0)  # Initial inflow
    rho0 = price_mu                             # Initial price

    # -------------------------
    # ADP Initialization
    # -------------------------
    a_t = [np.zeros((J, 1)) for _ in range(T)]
    b_t = [np.zeros(J) for _ in range(T)]
    l_bar = np.tile(l0, (N, 1))  # Post-decision state matrix

    # -------------------------
    # Exportable Variables
    # -------------------------
    return J, T, l_max, l_min, l0, pi_max, pi_min,price_samples, inflow_samples, nu0, rho0, R, a_t, b_t, l_bar, alpha_energy
    
# -------------------------
# Sample Generation with ARMA
# -------------------------

def generate_arma_samples(mu, phi, sigma, N, T):
    samples = np.zeros((N, T))
    for n in range(N):
        x = np.zeros(T)
        eps = np.random.normal(0, sigma, size=T)
        x[0] = mu
        for t in range(1, T):
            x[t] = mu + phi * (x[t-1] - mu) + eps[t]
        samples[n] = x
    return samples