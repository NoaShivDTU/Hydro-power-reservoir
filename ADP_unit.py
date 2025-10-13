import os
import csv
import time
import argparse
import numpy as np
from gurobipy import Model, GRB, quicksum
from Input_generator import generate_input

# -------------------------- small I/O helpers --------------------------

def ensure_dir(d):
    os.makedirs(d, exist_ok=True); return d

def save_csv(path, rows, header):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

# -------------------------- adapter (align shapes) ---------------------

def get_problem_inputs(seed, T, burnin, M, lambda_start_scalar=0.0):
    (Jg, Tg, l_max, l_min, l0, pi_max, pi_min,
     price, inflow, nu0, rho0, R, a0, b0, lbar0, g) = generate_input(
        N=M, T=T, burn_in=burnin, seed=seed
    )

    price  = np.asarray(price, dtype=float)           # (M, Tp)
    inflow = np.asarray(inflow, dtype=float)          # (M, Ti, J?)
    Tp, Ti = price.shape[1], inflow.shape[1]
    T_use  = min(Tp, Ti)

    price  = price[:, :T_use]
    inflow = inflow[:, :T_use, :]
    J = inflow.shape[2]

    def trim1(x): x = np.asarray(x, dtype=float); return x[:J]
    R = np.asarray(R, dtype=float)[:J, :J]
    g = trim1(g); lmin = trim1(l_min); lmax = trim1(l_max)
    pimin = trim1(pi_min); pimax = trim1(pi_max); l1 = trim1(l0)

    # If your generator provides lambda_start as a vector, replace the next line accordingly.
    lambda_start = np.full(J, float(lambda_start_scalar), dtype=float)

    return dict(T=T_use, J=J, R=R, g=g, lmin=lmin, lmax=lmax,
                pimin=pimin, pimax=pimax, l1=l1,
                prices=price, inflows=inflow, lambda_start=lambda_start)

# -------------------------- stage models (MILP + LP) -------------------

# Small two-sided slack to guarantee feasibility; penalized in the objective
PENALTY_MULT = 5.0

def _penalty_scale(price_now, g):
    return PENALTY_MULT * max(1.0, abs(float(price_now)) * float(np.max(g) if g.size else 1.0))

def _build_common_vars(model, J):
    pi = model.addVars(J, name="pi")                                  # generation flow
    lbar_next = model.addVars(J, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="lbar_next")
    splus  = model.addVars(J, lb=0.0, name="splus")
    sminus = model.addVars(J, lb=0.0, name="sminus")
    u = model.addVars(J, lb=0.0, ub=1.0, name="u")                    # may be relaxed to [0,1]
    v = model.addVars(J, lb=0.0, ub=1.0, name="v")                    # may be relaxed to [0,1]
    return pi, lbar_next, splus, sminus, u, v

def _add_constraints(model, J, pi, lbar_next, splus, sminus, u, v,
                     lbar_t, u_prev, nu_t, nu_next, R, lmin, lmax, pimin, pimax):
    """Balance, bounds, commitment, and start-up logic."""
    bal = []
    for j in range(J):
        # inventory balance (post-decision lbar scheme)
        c = model.addConstr(
            lbar_next[j] == float(lbar_t[j]) + float(nu_t[j]) +
                           quicksum(float(R[j, k]) * pi[k] for k in range(J)) +
                           splus[j] - sminus[j]
        )
        bal.append(c)
        # storage bounds shift with next inflow
        model.addConstr(lbar_next[j] >= float(lmin[j] - nu_next[j]))
        model.addConstr(lbar_next[j] <= float(lmax[j] - nu_next[j]))
        # generation only when on
        model.addConstr(pi[j] >= float(pimin[j]) * u[j])
        model.addConstr(pi[j] <= float(pimax[j]) * u[j])
        # start-up indicator
        model.addConstr(-float(u_prev[j]) + u[j] - v[j] <= 0.0)
    return bal

def _set_objective(model, price_now, g, pi, lbar_next, splus, sminus, a_next, lambda_start, v):
    penalty = _penalty_scale(price_now, g)
    obj = (float(price_now) * quicksum(float(g[j]) * pi[j] for j in range(len(g))) +
           quicksum(float(a_next[j]) * lbar_next[j] for j in range(len(g))) -
           quicksum(float(lambda_start[j]) * v[j] for j in range(len(g))) -
           penalty * quicksum(splus[j] + sminus[j] for j in range(len(g))))
    model.setObjective(obj, GRB.MAXIMIZE)

def stage_commitment_solve_pair(lbar_t, u_prev, nu_t, nu_next, price_now,
                                R, g, lmin, lmax, pimin, pimax, a_next, lambda_start):
    """
    Two solves per stage:
      1) MILP for realized decisions (u,v,pi,lbar_next) and realized reward
      2) LP relaxation (u,v ∈ [0,1]) to extract duals of the balance equalities and
         a smooth stage value for training/diagnostics.

    Returns:
      vhat_relax, omega_relax,  (training value & subgradient)
      decisions: (pi_milp, lbar_next_milp, u_milp, v_milp, realized_reward)
    """
    J = len(g)

    # ---------- MILP (forward / realized) ----------
    m = Model()
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    pi, lbar_next, splus, sminus, u, v = _build_common_vars(m, J)
    # set integrality
    for j in range(J):
        u[j].VType = GRB.BINARY
        v[j].VType = GRB.BINARY
        pi[j].LB = float(pimin[j]); pi[j].UB = float(pimax[j])
    bal = _add_constraints(m, J, pi, lbar_next, splus, sminus, u, v,
                           lbar_t, u_prev, nu_t, nu_next, R, lmin, lmax, pimin, pimax)
    _set_objective(m, price_now, g, pi, lbar_next, splus, sminus, a_next, lambda_start, v)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Stage MILP not optimal (status={m.Status}).")

    pi_m = np.array([pi[j].X for j in range(J)], dtype=float)
    lbar_next_m = np.array([lbar_next[j].X for j in range(J)], dtype=float)
    u_m = np.array([u[j].X for j in range(J)], dtype=float)
    v_m = np.array([v[j].X for j in range(J)], dtype=float)
    realized_reward = float(price_now) * float(g @ pi_m) - float(lambda_start @ v_m)

    try: m.dispose()
    except Exception: pass

    # ---------- LP relaxation (backward / training) ----------
    r = Model()
    r.Params.OutputFlag = 0
    r.Params.Method = 1
    r.Params.Presolve = 2
    r.Params.Crossover = 0
    r.Params.Threads = 1
    pi, lbar_next, splus, sminus, u, v = _build_common_vars(r, J)
    for j in range(J):
        pi[j].LB = float(pimin[j]); pi[j].UB = float(pimax[j])
        # u, v already continuous in [0,1]
    bal = _add_constraints(r, J, pi, lbar_next, splus, sminus, u, v,
                           lbar_t, u_prev, nu_t, nu_next, R, lmin, lmax, pimin, pimax)
    _set_objective(r, price_now, g, pi, lbar_next, splus, sminus, a_next, lambda_start, v)
    r.optimize()
    if r.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Stage LP relaxation not optimal (status={r.Status}).")

    vhat_relax = float(r.ObjVal)
    omega_relax = np.array([bal[j].Pi for j in range(J)], dtype=float)

    try: r.dispose()
    except Exception: pass

    return (vhat_relax, omega_relax,
            (pi_m, lbar_next_m, u_m, v_m, realized_reward))

# -------------------------- smoothing (Option B) -----------------------

def backward_update(path_vals, path_lbar, path_omegas, a_prev, b_prev, alpha):
    """
    a_bar = mean_m ω_t^m
    b_bar = mean_m ( Vhat_t^m - (ω_t^m)^T lbar_t^m )
    EMA: a_new = (1-α)a_prev + α a_bar,   b_new = (1-α)b_prev + α b_bar
    """
    a_bar = np.mean(np.stack(path_omegas, axis=0), axis=0)
    b_bar = float(np.mean([path_vals[m] - float(np.dot(path_omegas[m], path_lbar[m]))
                           for m in range(len(path_vals))]))
    a_new = (1.0 - alpha) * a_prev + alpha * a_bar
    b_new = (1.0 - alpha) * b_prev + alpha * b_bar
    return a_new, b_new

# -------------------------- optional salvage --------------------------

def compute_salvage_a(prices, g, k_last=24):
    T = prices.shape[1]
    k = max(1, min(k_last, T))
    salvage_price = float(np.mean(prices[:, -k:]))
    return salvage_price * g

# -------------------------- trainer --------------------------

def adp_train_and_evaluate(outdir, Ns, seed, T, burnin,
                           train_M, test_M, batch=4, alpha=0.5,
                           use_salvage=False, rng_seed=None,
                           lambda_start_scalar=0.0):
    rng = np.random.default_rng(rng_seed if rng_seed is not None else seed)

    std_rows   = []   # (N, t, std_last5)
    profits    = []   # rows for profits.csv

    for N in Ns:
        # data
        train = get_problem_inputs(seed, T, burnin, train_M, lambda_start_scalar=lambda_start_scalar)
        test  = get_problem_inputs(seed+777, T, burnin, test_M, lambda_start_scalar=lambda_start_scalar)

        T_use = train['T']; J = train['J']
        R, g = train['R'], train['g']
        lmin, lmax = train['lmin'], train['lmax']
        pimin, pimax = train['pimin'], train['pimax']
        l1 = train['l1']
        prices_tr, inflows_tr = train['prices'], train['inflows']
        prices_te, inflows_te = test['prices'],  test['inflows']
        lambda_start = train['lambda_start']

        # value-function params a_t, b_t (post-decision); terminal slope a_T
        a_hat = [np.zeros(J) for _ in range(T_use+1)]
        b_hat = [0.0        for _ in range(T_use+1)]
        a_hat[T_use] = compute_salvage_a(prices_tr, g, k_last=24) if use_salvage else np.zeros(J)

        last5_traces = np.zeros((5, T_use))
        lower_bounds_trace = []  # (iter, lower_bound_batch_avg)

        t0 = time.time()

        # iterations
        for n in range(1, N+1):
            B = max(1, min(train_M, batch))
            idxs = rng.integers(0, train_M, size=B)

            vals_t = [ [] for _ in range(T_use) ]
            lbar_t = [ [] for _ in range(T_use) ]
            omg_t  = [ [] for _ in range(T_use) ]

            batch_totals = []

            for m in idxs:
                lbar = l1.copy()
                u_prev = np.zeros(J)  # all off initially
                total_val = 0.0
                for t in range(T_use):
                    price_now = prices_tr[m, t]
                    nu_t = inflows_tr[m, t, :]
                    nu_next = inflows_tr[m, t+1, :] if (t < T_use-1) else np.zeros(J)
                    a_next = a_hat[t+1]

                    # Solve (MILP for forward) + (LP relax for learning/logging)
                    vhat_relax, omega, (pi_m, lbar_next_m, u_m, v_m, realized_reward) = \
                        stage_commitment_solve_pair(
                            lbar, u_prev, nu_t, nu_next, price_now,
                            R, g, lmin, lmax, pimin, pimax, a_next, lambda_start
                        )

                    vals_t[t].append(vhat_relax)     # training/diagnostic value
                    lbar_t[t].append(lbar.copy())    # state used for gradient
                    omg_t[t].append(omega.copy())
                    lbar = lbar_next_m               # advance using MILP decision
                    u_prev = u_m                     # advance commitment
                    total_val += vhat_relax

                batch_totals.append(total_val)

            lower_bounds_trace.append([n, float(np.mean(batch_totals))])

            for t in reversed(range(T_use)):
                a_hat[t], b_hat[t] = backward_update(
                    vals_t[t], lbar_t[t], omg_t[t], a_hat[t], b_hat[t], alpha
                )

        train_time_s = time.time() - t0

        # export per-iteration lower bounds (same format)
        save_csv(os.path.join(outdir, f"lower_bounds_N{N}.csv"),
                 lower_bounds_trace, header=["iter", "lower_bound_batch_avg"])

        # last five training paths (for figure exports) – log LP-relaxation Vhat
        last5_idx = list(range(max(0, train_M-5), train_M))
        for i, m in enumerate(last5_idx):
            lbar = l1.copy()
            u_prev = np.zeros(J)
            for t in range(T_use):
                price_now = prices_tr[m, t]
                nu_t = inflows_tr[m, t, :]
                nu_next = inflows_tr[m, t+1, :] if (t < T_use-1) else np.zeros(J)
                a_next = a_hat[t+1]
                vhat_relax, _, (pi_m, lbar_next_m, u_m, v_m, _) = stage_commitment_solve_pair(
                    lbar, u_prev, nu_t, nu_next, price_now,
                    R, g, lmin, lmax, pimin, pimax, a_next, lambda_start
                )
                last5_traces[i, t] = vhat_relax
                lbar = lbar_next_m
                u_prev = u_m

        rows_last5 = [[t+1, s+1, last5_traces[s, t]]
                      for s in range(5) for t in range(T_use)]
        save_csv(os.path.join(outdir, f"values_last5_N{N}.csv"),
                 rows_last5, header=["t", "last5_index", "Vhat"])

        std_vec = np.std(last5_traces, axis=0, ddof=1)
        for t in range(T_use):
            std_rows.append([N, t+1, std_vec[t]])

        # realized revenue mean: ∑_t (ρ_t g^T π_t − λ^T v_t) under MILP policy
        def realized_profit_mean(prices, inflows):
            M = prices.shape[0]; totals = np.zeros(M)
            for m in range(M):
                lbar = l1.copy(); u_prev = np.zeros(J); tot = 0.0
                for t in range(T_use):
                    price_now = prices[m, t]
                    nu_t = inflows[m, t, :]
                    nu_next = inflows[m, t+1, :] if (t < T_use-1) else np.zeros(J)
                    a_next = a_hat[t+1]  # policy uses V approx in objective
                    _, _, (pi_m, lbar_next_m, u_m, v_m, realized_reward) = stage_commitment_solve_pair(
                        lbar, u_prev, nu_t, nu_next, price_now,
                        R, g, lmin, lmax, pimin, pimax, a_next, lambda_start
                    )
                    tot += realized_reward
                    lbar = lbar_next_m
                    u_prev = u_m
                totals[m] = tot
            return float(np.mean(totals))

        ADP_in = realized_profit_mean(prices_tr, inflows_tr)

        t1 = time.time()
        ADP_out = realized_profit_mean(prices_te, inflows_te)
        eval_time_s = time.time() - t1
        online_ms_per_path = 1000.0 * eval_time_s / max(test_M, 1)

        profits.append([N, train_M, test_M, ADP_in, ADP_out,
                        round(train_time_s, 3), round(online_ms_per_path, 3)])

        print(f"[ADP-UC N={N}] in={ADP_in:.3f}  out={ADP_out:.3f}  "
              f"train={train_time_s:.1f}s  online~{online_ms_per_path:.1f}ms")

    # aggregate exports
    save_csv(os.path.join(outdir, "std_last5_over_time.csv"),
             std_rows, header=["N", "t", "std_last5"])

    save_csv(os.path.join(outdir, "profits.csv"),
             profits,
             header=["N","train_M","test_M","SDDP_in","SDDP_out","train_time_s","online_ms_per_path"])

# ------------------------------- CLI ---------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="results_adp_unit")
    p.add_argument("--Ns", type=int, nargs="+",
                   default=list(range(100, 1001, 100)),
                   help="ADP iterations")
    p.add_argument("--train_M", type=int, default=1000, help="training paths (pool)")
    p.add_argument("--test_M",  type=int, default=1000, help="test paths for OOS evaluation")
    p.add_argument("--batch",   type=int, default=8, help="mini-batch size per ADP iteration")
    p.add_argument("--alpha",   type=float, default=0.5, help="fixed stepsize for (a,b) smoothing")
    p.add_argument("--no_salvage", action="store_true", help="set terminal slope a_T = 0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--T", type=int, default=48)
    p.add_argument("--burnin", type=int, default=400)
    p.add_argument("--lambda_start", type=float, default=0.0,
                   help="start-up cost λ (scalar applied to all units unless your Input_generator provides a vector)")
    args = p.parse_args()

    ensure_dir(args.outdir)
    adp_train_and_evaluate(args.outdir, args.Ns, args.seed, args.T, args.burnin,
                           args.train_M, args.test_M,
                           batch=args.batch, alpha=args.alpha,
                           use_salvage=(not args.no_salvage),
                           rng_seed=args.seed,
                           lambda_start_scalar=args.lambda_start)

if __name__ == "__main__":
    main()
