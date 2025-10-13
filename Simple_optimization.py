import os
import numpy as np
from gurobipy import Model, GRB, quicksum
from Input_generator import generate_input


# ---------- user knobs (non-overlapping with ADP) ----------
OUTPUT_DIR = "results_eev_ws"
N_list          =  [100, 200, 500, 1000]
OUT_OF_SAMPLE_M = 200
SEED_TRAIN      = 42
SEED_OOS        = 42 + 1               # ADP uses eval_seed = SEED + 1
T_USE           = 48                  # HEAD ONLY
BURN_IN =200
# -----------------------------------------------------------


def _align_time_axes(prices_t, inflows_tj, T_cap=None):
    """
    Make prices_t (T,), inflows_tj (T,J) share the same T and optionally cap to T_cap.
    Returns (prices_t_aligned, inflows_tj_aligned, T_used).
    """
    prices_t = np.asarray(prices_t, dtype=float)
    inflows_tj = np.asarray(inflows_tj, dtype=float)

    Tp = int(prices_t.shape[0])
    Ti = int(inflows_tj.shape[0])
    T = min(Tp, Ti)
    if T_cap is not None:
        T = min(T, int(T_cap))
    if T <= 0:
        raise ValueError(f"Inferred non-positive T (prices={Tp}, inflows={Ti}, cap={T_cap}).")

    if Tp != T:
        prices_t = prices_t[:T]
    if Ti != T:
        inflows_tj = inflows_tj[:T, :]

    return prices_t, inflows_tj, T


def _salvage_price_from_series(prices_t):
    """
    Proxy for E[price_{T+1} | price_{1:T}]. Uses the mean of the last 24 prices
    if available, else mean of all. Keeps the rule consistent across EV/WS/eval.
    """
    prices_t = np.asarray(prices_t, dtype=float)
    k = min(24, prices_t.size)
    return float(np.mean(prices_t[-k:]))


def build_model(J, l_max, l_min, l0, pi_max, pi_min, R,
                prices_t, inflows_tj, g,
                T_cap=None, fix_schedule=None,
                penalty_per_unit=None,  # None => no slack; float => 2-sided slack with penalty
               ):
    """
    Single-scenario model with hard bounds and salvage value.
    If fix_schedule is not None, actions are fixed exactly (pi == schedule).
    If penalty_per_unit is not None, adds two-sided slack to the water balance
    with linear penalty: penalty_per_unit * sum(slack_plus + slack_minus).
    Returns (model, pi, l, T, slack_tot) where slack_tot is a Gurobi linexpr (for logging).
    """
    model = Model("hydro_revenue_max")
    model.Params.OutputFlag = 0

    prices_t, inflows_tj, T = _align_time_axes(prices_t, inflows_tj, T_cap=T_cap)
    assert inflows_tj.shape == (T, J), f"inflows_tj shape {inflows_tj.shape} != (T={T}, J={J})"

    Jset, Tset = range(J), range(T)

    # decision vars
    pi = model.addVars(((j, t) for j in Jset for t in Tset),
                       lb={(j, t): float(pi_min[j]) for j in Jset for t in Tset},
                       ub={(j, t): float(pi_max[j]) for j in Jset for t in Tset},
                       name="pi")

    l  = model.addVars(((j, t) for j in Jset for t in range(T+1)),
                       lb={(j, t): float(l_min[j]) for j in Jset for t in range(T+1)},
                       ub={(j, t): float(l_max[j]) for j in Jset for t in range(T+1)},
                       name="l")

    use_slack = (penalty_per_unit is not None and penalty_per_unit > 0.0)
    if use_slack:
        # two-sided slack: slack_plus (extra inflow), slack_minus (extra outflow)
        sp = model.addVars(((j, t) for j in Jset for t in Tset), lb=0.0, name="slack_plus")
        sm = model.addVars(((j, t) for j in Jset for t in Tset), lb=0.0, name="slack_minus")
    else:
        sp = sm = None

    # initial levels
    for j in Jset:
        model.addConstr(l[j, 0] == float(l0[j]))

    # water balance
    for t in Tset:
        for j in Jset:
            if use_slack:
                model.addConstr(
                    l[j, t+1] == l[j, t]
                    + float(inflows_tj[t, j])
                    + quicksum(float(R[j, k]) * pi[k, t] for k in Jset)
                    + sp[j, t] - sm[j, t]
                )
            else:
                model.addConstr(
                    l[j, t+1] == l[j, t]
                    + float(inflows_tj[t, j])
                    + quicksum(float(R[j, k]) * pi[k, t] for k in Jset)
                )

    # fix actions exactly if requested
    if fix_schedule is not None:
        fix_schedule = np.asarray(fix_schedule, dtype=float)
        assert fix_schedule.shape == (J, T), f"fix_schedule shape {fix_schedule.shape} != (J={J}, T={T})"
        for j in Jset:
            for t in Tset:
                model.addConstr(pi[j, t] == float(fix_schedule[j, t]))

    # objective: revenue + salvage - penalty*slack
    revenue = quicksum(
        float(prices_t[t]) * quicksum(float(g[j]) * pi[j, t] for j in Jset)
        for t in Tset
    )
    salvage_price = _salvage_price_from_series(prices_t)
    salvage = salvage_price * quicksum(float(g[j]) * l[j, T] for j in Jset)

    slack_penalty = 0.0
    slack_tot = 0.0
    if use_slack:
        slack_tot = quicksum(sp[j, t] + sm[j, t] for j in Jset for t in Tset)  # for logging
        slack_penalty = penalty_per_unit * slack_tot

    model.setObjective(revenue + salvage - slack_penalty, GRB.MAXIMIZE)
    return model, pi, l, T, slack_tot




def solve_ws_for_scenario(params, prices_t, inflows_tj):
    J, l_max, l_min, l0, pi_max, pi_min, R, g = params
    model, pi, l, T, _ = build_model(J, l_max, l_min, l0, pi_max, pi_min, R,
                                     prices_t, inflows_tj, g, T_cap=T_USE,
                                     fix_schedule=None, penalty_per_unit=None)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        return np.nan, None
    schedule = np.array([[pi[j, t].X for t in range(T)] for j in range(J)])
    return model.ObjVal, schedule



def evaluate_fixed_schedule(params, prices_t, inflows_tj, schedule):
    """
    Evaluate a fixed π schedule with pi == schedule.
    To avoid infeasibilities under OOS, add two-sided balance slack with a
    moderate economic penalty (≈ 5 × max_price × max(g)).
    """
    J, l_max, l_min, l0, pi_max, pi_min, R, g = params
    T_cap = min(schedule.shape[1], T_USE)
    schedule = np.asarray(schedule, dtype=float)[:, :T_cap]

    # economic upper bound for marginal value of water
    max_price = float(np.nanmax(np.asarray(prices_t, dtype=float)))
    max_g = float(np.nanmax(np.asarray(g, dtype=float)))
    penalty_per_unit = 5.0 * max_price * max_g  # deterrent but not astronomically large

    model, pi, l, T, slack_tot = build_model(J, l_max, l_min, l0, pi_max, pi_min, R,
                                             prices_t, inflows_tj, g,
                                             T_cap=T_cap, fix_schedule=schedule,
                                             penalty_per_unit=penalty_per_unit)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        return np.nan
    # (Optional) you can inspect slack_tot.getValue() here if you want
    return model.ObjVal



def _avg_clean(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    return float(np.nanmean(x)) if np.any(~np.isnan(x)) else np.nan


def run_for_N(N):
    # ---- Generate HEAD-only scenarios (no tail), training seed ----
    (J, T_gen, l_max, l_min, l0, pi_max, pi_min,
     price_samples, inflow_samples,
     _nu0, _rho0, R, _a_t, _b_t, _l_bar,
     g) = generate_input(N=N, T=T_USE, seed=SEED_TRAIN, burn_in=BURN_IN)

    # Pack parameters
    params = (J, np.array(l_max), np.array(l_min), np.array(l0),
              np.array(pi_max), np.array(pi_min), np.array(R), np.array(g))

    # ---- EV/EEV schedule from means over ALL N scenarios (no fixed split) ----
    prices_mean  = np.mean(price_samples,  axis=0)    # (T_USE,)
    inflows_mean = np.mean(inflow_samples, axis=0)    # (T_USE, J)
    _, eev_schedule = solve_ws_for_scenario(params, prices_mean, inflows_mean)

    # ---- In-sample evaluation (across all N) ----
    eev_is, ws_is = [], []
    for s in range(N):
        ws_obj, _ = solve_ws_for_scenario(params, price_samples[s], inflow_samples[s])
        ws_is.append(ws_obj)
        eev_obj = evaluate_fixed_schedule(params, price_samples[s], inflow_samples[s], eev_schedule)
        eev_is.append(eev_obj)

    # ---- Out-of-sample (HEAD only, eval seed) ----
    (_J_o, _T_o, _l_max_o, _l_min_o, _l0_o, _pi_max_o, _pi_min_o,
    price_oos, inflow_oos,
    _nu0_o, _rho0_o, _R_o, _a_t_o, _b_t_o, _l_bar_o,
    _g_o) = generate_input(N=OUT_OF_SAMPLE_M, T=T_USE, seed=SEED_OOS, burn_in=BURN_IN)

    params_o = params  # <--- reuse training structure to ensure feasibility of fixed schedule

    eev_oos, ws_oos = [], []
    for s in range(OUT_OF_SAMPLE_M):
        # WS: perfect info under the same (training) structure
        ws_obj, _ = solve_ws_for_scenario(params_o, price_oos[s], inflow_oos[s])
        ws_oos.append(ws_obj)
        # EEV: evaluate the fixed training schedule under OOS prices/inflows
        eev_obj = evaluate_fixed_schedule(params_o, price_oos[s], inflow_oos[s], eev_schedule)
        eev_oos.append(eev_obj)

    # ---- Write per-scenario CSVs ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, f"in_sample_N{N}.csv"), "w") as f:
        f.write("scenario,WS,EEV,regret\n")
        for idx, (w, e) in enumerate(zip(ws_is, eev_is), 1):
            wv = "" if np.isnan(w) else f"{w:.8f}"
            ev = "" if np.isnan(e) else f"{e:.8f}"
            rv = "" if (np.isnan(w) or np.isnan(e)) else f"{(w - e):.8f}"
            f.write(f"{idx},{wv},{ev},{rv}\n")

    with open(os.path.join(OUTPUT_DIR, f"out_of_sample_N{N}.csv"), "w") as f:
        f.write("scenario,WS,EEV,regret\n")
        for idx, (w, e) in enumerate(zip(ws_oos, eev_oos), 1):
            wv = "" if np.isnan(w) else f"{w:.8f}"
            ev = "" if np.isnan(e) else f"{e:.8f}"
            rv = "" if (np.isnan(w) or np.isnan(e)) else f"{(w - e):.8f}"
            f.write(f"{idx},{wv},{ev},{rv}\n")

    # EEV schedule CSV (rows=time, cols=reservoirs) — use actual schedule width
    T_used = eev_schedule.shape[1]
    with open(os.path.join(OUTPUT_DIR, f"eev_schedule_N{N}.csv"), "w") as f:
        header = ",".join([f"Reservoir{j+1}" for j in range(J)])
        f.write("time," + header + "\n")
        for t in range(T_used):
            row = ",".join(f"{eev_schedule[j, t]:.8f}" for j in range(J))
            f.write(f"{t+1},{row}\n")

    return {
        "N": N,
        "EEV_in_sample": _avg_clean(eev_is),
        "WS_in_sample":  _avg_clean(ws_is),
        "EEV_out_of_sample": _avg_clean(eev_oos),
        "WS_out_of_sample":  _avg_clean(ws_oos),
    }


def main():
    rows = []
    for N in N_list:
        rows.append(run_for_N(N))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(summary_path, "w") as f:
        f.write("N,EEV_in_sample,WS_in_sample,EEV_out_of_sample,WS_out_of_sample\n")
        for r in rows:
            def fmt(x):
                return "" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.8f}"
            f.write(f"{r['N']},{fmt(r['EEV_in_sample'])},{fmt(r['WS_in_sample'])},"
                    f"{fmt(r['EEV_out_of_sample'])},{fmt(r['WS_out_of_sample'])}\n")

if __name__ == "__main__":
    main()
