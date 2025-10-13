import os
import csv
import time
import argparse
import numpy as np
from gurobipy import Model, GRB, quicksum
from Input_generator import generate_input

# -------------------------- small I/O helpers --------------------------

def ensure_dir(d):
    """Create a directory if it doesn't exist and return its path."""
    os.makedirs(d, exist_ok=True); return d

def save_csv(path, rows, header):
    """Write a CSV with a header and list-of-lists rows."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

# -------------------------- adapter (align shapes) ---------------------

def get_problem_inputs(seed, T, burnin, M):
    """
    Wrap the project-specific input generator and trim arrays to a common T.
    Returns a compact dict with the arrays used downstream.
    """
    (Jg, Tg, l_max, l_min, l0, pi_max, pi_min,
     price, inflow, nu0, rho0, R, a0, b0, lbar0, g) = generate_input(
        N=M, T=T, burn_in=burnin, seed=seed
    )

    price  = np.asarray(price, dtype=float)           # prices:  (M, Tp)
    inflow = np.asarray(inflow, dtype=float)          # inflows: (M, Ti, J?)
    Tp, Ti = price.shape[1], inflow.shape[1]
    T_use  = min(Tp, Ti)                              # use overlap across series

    price  = price[:, :T_use]
    inflow = inflow[:, :T_use, :]
    J = inflow.shape[2]                               # number of reservoirs/units

    def trim1(x): x = np.asarray(x, dtype=float); return x[:J]
    R = np.asarray(R, dtype=float)[:J, :J]            # production matrix (JxJ)
    g = trim1(g)                                      # turbine gains per unit (J,)
    lmin = trim1(l_min); lmax = trim1(l_max)          # storage bounds (J,)
    pimin = trim1(pi_min); pimax = trim1(pi_max)      # flow bounds (J,)
    l1 = trim1(l0)                                    # initial post-decision level (J,)

    return dict(T=T_use, J=J, R=R, g=g, lmin=lmin, lmax=lmax,
                pimin=pimin, pimax=pimax, l1=l1,
                prices=price, inflows=inflow)

# -------------------------- stage LP (value + duals) -------------------

# Small two-sided slack to guarantee feasibility; penalized in the objective
PENALTY_MULT = 5.0

def stage_lp_value_and_dual(lbar_t, nu_t, nu_next, price_now,
                            R, g, lmin, lmax, pimin, pimax, a_next):
    """
    One-hour stage LP (relaxed physics) used to:
      • pick flow π_t and next post-decision level l̄_{t+1}
      • get the duals ω_t of the balance constraints (subgradient wrt l̄_{t+1})
      • compute the stage value (immediate revenue + continuation proxy a_{t+1}^T l̄_{t+1})

    max   price_now * g^T π + a_next^T lbar_next - penalty * 1^T(s+ + s-)
    s.t.  lbar_next = lbar_t + nu_t + R π + s+ - s-
          lmin - nu_next ≤ lbar_next ≤ lmax - nu_next
          pimin ≤ π ≤ pimax
    """
    J = len(g)
    m = Model()
    m.Params.OutputFlag = 0
    m.Params.Method = 1       # dual simplex for fast duals
    m.Params.Presolve = 2
    m.Params.Crossover = 0
    m.Params.Threads = 1

    # decision vars: generation, next post-decision level, and feasibility slacks
    pi = m.addVars(J, name="pi")
    for j in range(J):
        pi[j].LB = float(pimin[j]); pi[j].UB = float(pimax[j])

    lbar_next = m.addVars(J, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="lbar_next")
    splus  = m.addVars(J, lb=0.0, name="splus")     # upward slack on balance
    sminus = m.addVars(J, lb=0.0, name="sminus")    # downward slack on balance

    # scale penalty with price magnitude to keep units comparable
    penalty = PENALTY_MULT * max(1.0, abs(float(price_now)) * float(max(g) if np.size(g) else 1.0))

    # balance + storage bounds
    bal = []
    for j in range(J):
        c = m.addConstr(
            lbar_next[j] == float(lbar_t[j]) + float(nu_t[j]) +
                           quicksum(float(R[j, k]) * pi[k] for k in range(J)) +
                           splus[j] - sminus[j]
        )
        bal.append(c)
        m.addConstr(lbar_next[j] >= float(lmin[j] - nu_next[j]))
        m.addConstr(lbar_next[j] <= float(lmax[j] - nu_next[j]))

    # objective = immediate revenue + continuation proxy - slack penalty
    obj = (float(price_now) * quicksum(float(g[j]) * pi[j] for j in range(J)) +
           quicksum(float(a_next[j]) * lbar_next[j] for j in range(J)) -
           penalty * quicksum(splus[j] + sminus[j] for j in range(J)))

    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Stage LP not optimal (status={m.Status}).")

    # stage metrics + subgradient
    val = float(m.ObjVal)
    pi_opt = np.array([pi[j].X for j in range(J)], dtype=float)
    lbar_next_val = np.array([lbar_next[j].X for j in range(J)], dtype=float)
    omega = np.array([bal[j].Pi for j in range(J)], dtype=float)  # duals of balance (∂V/∂l̄)

    try: m.dispose()
    except Exception: pass
    return val, pi_opt, lbar_next_val, omega

# -------------------------- smoothing (Option B) -----------------------

def backward_update(path_vals, path_lbar, path_omegas, a_prev, b_prev, alpha):
    """
    Batch the subgradients across sampled paths and update (a,b) with an EMA:
      a_bar = mean_m ω_t^m
      b_bar = mean_m ( Vhat_t^m - (ω_t^m)^T lbar_t^m )
      a_new = (1-α)a_prev + α a_bar
      b_new = (1-α)b_prev + α b_bar
    """
    a_bar = np.mean(np.stack(path_omegas, axis=0), axis=0)
    b_bar = float(np.mean([path_vals[m] - float(np.dot(path_omegas[m], path_lbar[m]))
                           for m in range(len(path_vals))]))
    a_new = (1.0 - alpha) * a_prev + alpha * a_bar
    b_new = (1.0 - alpha) * b_prev + alpha * b_bar
    return a_new, b_new

# -------------------------- optional salvage --------------------------

def compute_salvage_a(prices, g, k_last=24):
    """
    Terminal slope a_T: average price over the last k hours times g.
    Set use_salvage=False to revert to a_T = 0.
    """
    T = prices.shape[1]
    k = max(1, min(k_last, T))
    salvage_price = float(np.mean(prices[:, -k:]))
    return salvage_price * g

# -------------------------- trainer --------------------------

def adp_train_and_evaluate(outdir, Ns, seed, T, burnin,
                           train_M, test_M, batch=4, alpha=0.5,
                           use_salvage=False, rng_seed=None):
    """
    Main ADP loop:
      • iterate N times; each iteration uses a mini-batch of paths
      • forward simulate with current (a,b) to get Vhat and duals
      • backward update (a,b) using averaged subgradients
      • export lower bounds, last-5 traces, std over time, and realized profits
    """
    rng = np.random.default_rng(rng_seed if rng_seed is not None else seed)

    std_rows   = []   # rows for std_last5_over_time.csv
    profits    = []   # rows for profits.csv

    for N in Ns:
        # generate training and test pools with the same parameters but different seeds
        train = get_problem_inputs(seed, T, burnin, train_M)
        test  = get_problem_inputs(seed+777, T, burnin, test_M)

        T_use = train['T']; J = train['J']
        R, g = train['R'], train['g']
        lmin, lmax = train['lmin'], train['lmax']
        pimin, pimax = train['pimin'], train['pimax']
        l1 = train['l1']
        prices_tr, inflows_tr = train['prices'], train['inflows']
        prices_te, inflows_te = test['prices'],  test['inflows']

        # value function approximation parameters (post-decision): V̂_t(l̄)=a_t^T l̄ + b_t
        a_hat = [np.zeros(J) for _ in range(T_use+1)]
        b_hat = [0.0        for _ in range(T_use+1)]
        a_hat[T_use] = compute_salvage_a(prices_tr, g, k_last=24) if use_salvage else np.zeros(J)

        # arrays for figure exports
        last5_traces = np.zeros((5, T_use))     # store Vhat along last 5 training paths
        lower_bounds_trace = []                 # per-iteration batch averages

        t0 = time.time()

        # --------------------- training iterations ---------------------
        for n in range(1, N+1):
            B = max(1, min(train_M, batch))                 # mini-batch size
            idxs = rng.integers(0, train_M, size=B)         # sampled path indices

            # per-time-step collectors for this mini-batch
            vals_t = [ [] for _ in range(T_use) ]           # stage Vhat values
            lbar_t = [ [] for _ in range(T_use) ]           # states used for subgradients
            omg_t  = [ [] for _ in range(T_use) ]           # duals ω

            batch_totals = []                                # total over T for each path

            # --------------- forward pass on sampled paths ---------------
            for m in idxs:
                lbar = l1.copy()                             # reset state at t=1
                total_val = 0.0
                for t in range(T_use):
                    price_now = prices_tr[m, t]
                    nu_t = inflows_tr[m, t, :]
                    nu_next = inflows_tr[m, t+1, :] if (t < T_use-1) else np.zeros(J)
                    a_next = a_hat[t+1]

                    # solve stage LP, log value & subgradient, advance state
                    val, _, lbar_next, omega = stage_lp_value_and_dual(
                        lbar, nu_t, nu_next, price_now, R, g, lmin, lmax, pimin, pimax, a_next
                    )
                    vals_t[t].append(val)
                    lbar_t[t].append(lbar.copy())
                    omg_t[t].append(omega.copy())
                    lbar = lbar_next
                    total_val += val

                batch_totals.append(total_val)

            # store a lower-bound-like statistic: avg total value over this mini-batch
            lower_bounds_trace.append([n, float(np.mean(batch_totals))])

            # --------------- backward pass: update (a,b) ----------------
            for t in reversed(range(T_use)):
                a_hat[t], b_hat[t] = backward_update(
                    vals_t[t], lbar_t[t], omg_t[t], a_hat[t], b_hat[t], alpha
                )

        train_time_s = time.time() - t0

        # export per-iteration lower bounds (same format as SDDP)
        save_csv(os.path.join(outdir, f"lower_bounds_N{N}.csv"),
                 lower_bounds_trace, header=["iter", "lower_bound_batch_avg"])

        # ---------------- last five training paths for figures ----------------
        last5_idx = list(range(max(0, train_M-5), train_M))
        for i, m in enumerate(last5_idx):
            lbar = l1.copy()
            for t in range(T_use):
                price_now = prices_tr[m, t]
                nu_t = inflows_tr[m, t, :]
                nu_next = inflows_tr[m, t+1, :] if (t < T_use-1) else np.zeros(J)
                a_next = a_hat[t+1]
                val, _, lbar, _ = stage_lp_value_and_dual(
                    lbar, nu_t, nu_next, price_now, R, g, lmin, lmax, pimin, pimax, a_next
                )
                last5_traces[i, t] = val

        rows_last5 = [[t+1, s+1, last5_traces[s, t]]
                      for s in range(5) for t in range(T_use)]
        save_csv(os.path.join(outdir, f"values_last5_N{N}.csv"),
                 rows_last5, header=["t", "last5_index", "Vhat"])

        # std across the last-5 traces per t
        std_vec = np.std(last5_traces, axis=0, ddof=1)
        for t in range(T_use):
            std_rows.append([N, t+1, std_vec[t]])

        # ---------------- realized revenue (policy evaluation) ----------------
        # Important: only immediate revenue (no continuation term)
        def realized_profit_mean(prices, inflows):
            M = prices.shape[0]; totals = np.zeros(M)
            for m in range(M):
                lbar = l1.copy(); tot = 0.0
                for t in range(T_use):
                    price_now = prices[m, t]
                    nu_t = inflows[m, t, :]
                    nu_next = inflows[m, t+1, :] if (t < T_use-1) else np.zeros(J)
                    a_next = a_hat[t+1]
                    # take the LP's π decision driven by current a_{t+1}
                    _, pi, lbar, _ = stage_lp_value_and_dual(
                        lbar, nu_t, nu_next, price_now, R, g, lmin, lmax, pimin, pimax, a_next
                    )
                    tot += price_now * float(g @ pi)  # ρ_t * G(π_t)
                totals[m] = tot
            return float(np.mean(totals))

        ADP_in = realized_profit_mean(prices_tr, inflows_tr)

        t1 = time.time()
        ADP_out = realized_profit_mean(prices_te, inflows_te)
        eval_time_s = time.time() - t1
        online_ms_per_path = 1000.0 * eval_time_s / max(test_M, 1)

        # keep SDDP column names for seamless plotting
        profits.append([N, train_M, test_M, ADP_in, ADP_out,
                        round(train_time_s, 3), round(online_ms_per_path, 3)])

        print(f"[ADP N={N}] in={ADP_in:.3f}  out={ADP_out:.3f}  "
              f"train={train_time_s:.1f}s  online~{online_ms_per_path:.1f}ms")

    # aggregate exports
    save_csv(os.path.join(outdir, "std_last5_over_time.csv"),
             std_rows, header=["N", "t", "std_last5"])

    save_csv(os.path.join(outdir, "profits.csv"),
             profits,
             header=["N","train_M","test_M","SDDP_in","SDDP_out","train_time_s","online_ms_per_path"])

# ------------------------------- CLI ---------------------------------

def main():
    """Parse CLI args, run training/evaluation, write CSVs."""
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="results_adp")
    p.add_argument("--Ns", type=int, nargs="+",
                   default=list(range(100, 1001, 100)),
                   help="ADP iterations (number of mini-batch updates)")
    p.add_argument("--train_M", type=int, default=1000, help="training paths (pool)")
    p.add_argument("--test_M",  type=int, default=1000, help="test paths for OOS evaluation")
    p.add_argument("--batch",   type=int, default=8, help="mini-batch size per ADP iteration")
    p.add_argument("--alpha",   type=float, default=0.5, help="fixed stepsize for (a,b) smoothing")
    p.add_argument("--no_salvage", action="store_true", help="set terminal slope a_T = 0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--T", type=int, default=48)
    p.add_argument("--burnin", type=int, default=400)
    args = p.parse_args()

    ensure_dir(args.outdir)
    adp_train_and_evaluate(args.outdir, args.Ns, args.seed, args.T, args.burnin,
                           args.train_M, args.test_M,
                           batch=args.batch, alpha=args.alpha,
                           use_salvage=(not args.no_salvage),  # default False to match SDDP
                           rng_seed=args.seed)

if __name__ == "__main__":
    main()
