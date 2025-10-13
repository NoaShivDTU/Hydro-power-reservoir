import os
import csv
import time
import argparse
import numpy as np
from gurobipy import Model, GRB, quicksum
from Input_generator import generate_input


# ---------- small io helpers ----------

def ensure_dir(d):
    """Create folder d if it doesn’t exist."""
    os.makedirs(d, exist_ok=True)
    return d

def save_csv(path, rows, header):
    """Write rows to CSV at 'path' with a one-line header."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---------- adapt your generator output ----------
# (Leave this section as-is unless your Input_generator changes its return format.)

def get_problem_inputs(seed, T, burnin, M):
    """
    Wrap the input generator so all arrays line up:
    - trim prices/inflows to a common time horizon T_use
    - adopt J from the inflow tensor and trim vectors/matrices to that J
    """
    (J_gen, T_gen, l_max, l_min, l0, pi_max, pi_min,
     price_samples, inflow_samples, nu0_vec, rho0, R, a_t, b_t, l_bar, g) = \
        generate_input(N=M, T=T, burn_in=burnin, seed=seed)

    # Align time axes (some generators may return Tp != Ti)
    price_samples  = np.asarray(price_samples, dtype=float)      # (M, Tp)
    inflow_samples = np.asarray(inflow_samples, dtype=float)     # (M, Ti, J?)
    Tp = price_samples.shape[1]
    Ti = inflow_samples.shape[1]
    T_use = min(Tp, Ti)
    price_samples  = price_samples[:, :T_use]            # (M, T_use)
    inflow_samples = inflow_samples[:, :T_use, :]        # (M, T_use, J_inflow)

    # Use J coming from the inflow tensor
    J_inflow = int(inflow_samples.shape[2])

    def trim1(x):
        x = np.asarray(x, dtype=float)
        return x[:J_inflow]

    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ValueError(f"R must be 2D, got shape {R.shape}")
    R = R[:J_inflow, :J_inflow]   # square J×J

    g     = trim1(g)
    lmin  = trim1(l_min)
    lmax  = trim1(l_max)
    pimin = trim1(pi_min)
    pimax = trim1(pi_max)
    l1    = trim1(l0)

    return dict(
        T=T_use,
        J=J_inflow,
        R=R,
        g=g,
        lmin=lmin,
        lmax=lmax,
        pimin=pimin,
        pimax=pimax,
        l1=l1,
        prices=price_samples,   # (M, T_use)
        inflows=inflow_samples  # (M, T_use, J)
    )


# ---------- one-stage LP (value + subgradient wrt next state) ----------
# Gurobi parameters below are where to tweak speed/robustness if needed.

def stage_lp_value_and_dual(lbar_t, nu_t, nu_next, price_now,
                            R, g, lmin, lmax, pimin, pimax, cuts_next):
    """
    Stage LP with post-decision state lbar_{t+1} and value proxy θ:

      maximize   price_now * g^T * pi + θ
      s.t.       lbar_next = lbar_t + nu_t + R*pi
                 lmin - nu_next <= lbar_next <= lmax - nu_next
                 pimin <= pi <= pimax
                 θ <= a^T lbar_next + b,  for all (a,b) in cuts_next
                 (if no cuts, θ <= 0)

    Returns objective value, optimal (pi, lbar_next), duals on balance (ω), and θ.
    """
    J = len(g)
    m = Model()
    # Solver speed knobs (safe defaults):
    m.Params.OutputFlag = 0
    m.Params.Method = 1       # 0 auto, 1 dual simplex (good for small LPs), 2 barrier
    m.Params.Threads = 1      # increase if you have more CPU cores available
    m.Params.Presolve = 2
    m.Params.Crossover = 0

    # Variables
    pi = m.addVars(J, name="pi")
    for j in range(J):
        pi[j].LB = float(pimin[j])
        pi[j].UB = float(pimax[j])

    lbar_next = m.addVars(J, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="lbar_next")
    theta = m.addVar(lb=-GRB.INFINITY, name="theta")

    # Balance constraints and collect duals (ω) from them
    bal = []
    for j in range(J):
        c = m.addConstr(
            lbar_next[j] == lbar_t[j] + nu_t[j] + quicksum(R[j, k]*pi[k] for k in range(J)),
            name=f"bal_{j}"
        )
        bal.append(c)

    # Box on the next *pre-decision* level -> bounds on lbar_next
    for j in range(J):
        m.addConstr(lbar_next[j] >= float(lmin[j] - nu_next[j]))
        m.addConstr(lbar_next[j] <= float(lmax[j] - nu_next[j]))

    # Value cuts
    if cuts_next and len(cuts_next) > 0:
        for (a, b) in cuts_next:
            m.addConstr(theta <= quicksum(float(a[j])*lbar_next[j] for j in range(J)) + float(b))
    else:
        m.addConstr(theta <= 0.0)

    # Objective = immediate revenue + continuation θ
    m.setObjective(price_now * quicksum(float(g[j])*pi[j] for j in range(J)) + theta, GRB.MAXIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi status {m.Status} in stage LP")

    pi_opt = np.array([pi[j].X for j in range(J)], dtype=float)
    lbar_next_val = np.array([lbar_next[j].X for j in range(J)], dtype=float)
    val = float(m.ObjVal)
    omega = np.array([bal[j].Pi for j in range(J)], dtype=float)  # ω = duals on balance
    theta_val = float(theta.X)

    # Free native resources promptly
    try:
        m.dispose()
    except Exception:
        pass

    return val, pi_opt, lbar_next_val, omega, theta_val


# ---------- backward cut update with fixed alpha (Option B) ----------
# If learning is too jumpy/smooth, adjust alpha in CLI (below).

def backward_update(path_vals, path_lbar, path_omegas, a_prev, b_prev, alpha):
    """
    Smooth the sampled subgradient and intercept:
      a_bar = mean_m ω_t^m
      b_bar = mean_m (Vhat_t^m - (ω_t^m)^T lbar_t^m)
      a_new = (1-α) a_prev + α a_bar
      b_new = (1-α) b_prev + α b_bar
    """
    a_bar = np.mean(np.stack(path_omegas, axis=0), axis=0)
    b_bar = float(np.mean([path_vals[m] - np.dot(path_omegas[m], path_lbar[m])
                           for m in range(len(path_vals))]))
    a_new = (1.0 - alpha) * a_prev + alpha * a_bar
    b_new = (1.0 - alpha) * b_prev + alpha * b_bar
    return a_new, b_new


# ---------- policy evaluation helpers ----------
# Profit definition is “realized revenue only”. Change here if you need a different metric.

def solve_path_values_and_realized(cuts, l1, prices, inflows, R, g, lmin, lmax, pimin, pimax):
    """
    Roll the policy on one path with frozen cuts.
    Use cuts[t+1] as the continuation at stage t (cuts[T] is empty: no salvage).
    Returns:
      - sum of stage objectives (immediate + θ, diagnostic only)
      - list of stage objectives
      - realized revenue sum (used as “profit”)
    """
    T = prices.shape[0]
    J = len(g)
    lbar = l1.copy()
    Vhats = []
    realized = 0.0

    for t in range(T):
        price_now = prices[t]
        nu_t = inflows[t, :]
        nu_next = inflows[t+1, :] if (t < T-1) else np.zeros(J)
        cuts_next = cuts[t+1] if (t+1) < len(cuts) else []  # continuation from t+1

        val, pi, lbar_next, _, _ = stage_lp_value_and_dual(
            lbar, nu_t, nu_next, price_now, R, g, lmin, lmax, pimin, pimax, cuts_next
        )
        Vhats.append(val)
        realized += price_now * float(g @ pi)  # only immediate revenue counts as profit
        lbar = lbar_next

    return float(np.sum(Vhats)), Vhats, float(realized)


# ---------- training + evaluation loop ----------
# Most user-facing knobs appear in the CLI (bottom).
#    You can also hard-code them here if you prefer.

def sddp_train_and_evaluate(outdir, Ns, seed, T, burnin,
                            train_M, test_M, batch=8, alpha=0.5,
                            max_cuts_per_stage=None, rng_seed=None):
    """
    Train SDDP for each N in Ns, export diagnostics + realized profits.
    If max_cuts_per_stage is set (>0), keep only the K most recent cuts per stage (FIFO).
    """
    rng = np.random.default_rng(rng_seed if rng_seed is not None else seed)

    std_rows = []      # (N, t, std_last5)
    profit_rows = []   # (N, train_M, test_M, SDDP_in, SDDP_out, train_time_s, online_ms_per_path)

    for N in Ns:
        # Draw training/test pools (fixed seeds keep price/inflow consistent across runs)
        train = get_problem_inputs(seed, T, burnin, train_M)
        test  = get_problem_inputs(seed+777, T, burnin, test_M)

        T_train = train['T']
        J = train['J']
        R, g = train['R'], train['g']
        lmin, lmax = train['lmin'], train['lmax']
        pimin, pimax = train['pimin'], train['pimax']
        l1 = train['l1']
        prices_tr, inflows_tr = train['prices'], train['inflows']
        prices_te, inflows_te = test['prices'],  test['inflows']

        # Value approximation: cuts for V_{t+1}, t=0..T-1, plus an empty slot at T (no salvage)
        cuts = [[] for _ in range(T_train + 1)]
        a_hat = [np.zeros(J) for _ in range(T_train + 1)]
        b_hat = [0.0        for _ in range(T_train + 1)]

        lower_bounds_trace = []  # batch-average forward total per iteration

        t0 = time.time()

        # Main SDDP loop
        for n in range(1, N+1):
            # 👉 Batch size: trade off speed vs. stability (set via CLI --batch)
            B = max(1, min(train_M, batch))
            idxs = rng.integers(0, train_M, size=B)

            # Placeholders for backward update
            vals_t = [ [] for _ in range(T_train) ]
            lbar_t = [ [] for _ in range(T_train) ]
            omg_t  = [ [] for _ in range(T_train) ]

            batch_totals = []  # forward totals for the lower-bound trace

            # Forward pass (with current cuts)
            for m in idxs:
                lbar = l1.copy()
                total_val = 0.0
                for t in range(T_train):
                    price_now = prices_tr[m, t]
                    nu_t = inflows_tr[m, t, :]
                    nu_next = inflows_tr[m, t+1, :] if (t < T_train-1) else np.zeros(J)
                    cuts_next = cuts[t+1]          # continuation is V_{t+1}

                    val, _, lbar_next, omega, _ = stage_lp_value_and_dual(
                        lbar, nu_t, nu_next, price_now, R, g, lmin, lmax, pimin, pimax, cuts_next
                    )
                    vals_t[t].append(val)
                    lbar_t[t].append(lbar.copy())
                    omg_t[t].append(omega.copy())
                    lbar = lbar_next
                    total_val += val

                batch_totals.append(total_val)

            # Iteration-level “lower bound” (diagnostic)
            lower_bounds_trace.append([n, float(np.mean(batch_totals))])

            # Backward smoothing: one cut per stage, appended to the NEXT stage (t+1)
            for t in reversed(range(T_train)):
                a_hat[t+1], b_hat[t+1] = backward_update(
                    vals_t[t], lbar_t[t], omg_t[t], a_hat[t+1], b_hat[t+1], alpha
                )
                cuts[t+1].append((a_hat[t+1].copy(), b_hat[t+1]))
                # 👉 Memory/speed knob: cap cuts per stage with --max_cuts_per_stage (FIFO keep most recent K)
                if max_cuts_per_stage and max_cuts_per_stage > 0 and len(cuts[t+1]) > max_cuts_per_stage:
                    cuts[t+1] = cuts[t+1][-max_cuts_per_stage:]

        train_time_s = time.time() - t0

        # Save per-iteration lower bound trace
        save_csv(os.path.join(outdir, f"lower_bounds_N{N}.csv"),
                 lower_bounds_trace, header=["iter", "lower_bound_batch_avg"])

        # Last-5 trajectories: evaluate on the final 5 training paths
        last5_idx = list(range(max(0, train_M-5), train_M))
        last5_traces = np.zeros((5, T_train))
        for i, m in enumerate(last5_idx):
            _, Vhats, _ = solve_path_values_and_realized(
                cuts, l1,
                prices_tr[m, :],
                inflows_tr[m, :, :],
                R, g, lmin, lmax, pimin, pimax
            )
            last5_traces[i, :] = Vhats

        # Export last-5 for plotting
        rows_last5 = []
        for s in range(5):
            for t in range(T_train):
                rows_last5.append([t+1, s+1, last5_traces[s, t]])
        save_csv(os.path.join(outdir, f"values_last5_N{N}.csv"),
                 rows_last5, header=["t", "last5_index", "Vhat"])

        # Export std over time of the last-5 V̂
        std_vec = np.std(last5_traces, axis=0, ddof=1)
        for t in range(T_train):
            std_rows.append([N, t+1, std_vec[t]])

        # Realized profit (mean over scenarios) — used for “profit vs N”
        def realized_mean(prices, inflows):
            M = prices.shape[0]
            totals = np.zeros(M)
            for m in range(M):
                _, _, realized = solve_path_values_and_realized(
                    cuts, l1,
                    prices[m, :],
                    inflows[m, :, :],
                    R, g, lmin, lmax, pimin, pimax
                )
                totals[m] = realized
            return float(np.mean(totals))

        SDDP_in = realized_mean(prices_tr, inflows_tr)

        t_eval0 = time.time()
        SDDP_out = realized_mean(prices_te, inflows_te)
        eval_time_s = time.time() - t_eval0
        online_ms_per_path = 1000.0 * eval_time_s / max(test_M, 1)

        # Keep SDDP column names to match downstream figure scripts
        profit_rows.append([N, train_M, test_M, SDDP_in, SDDP_out,
                            round(train_time_s, 3), round(online_ms_per_path, 3)])
        print(f"[N={N}] SDDP_in={SDDP_in:.3f}, SDDP_out={SDDP_out:.3f}, "
              f"train={train_time_s:.1f}s, online~{online_ms_per_path:.1f}ms")

    # Save aggregated figure data + profits
    save_csv(os.path.join(outdir, "std_last5_over_time.csv"),
             std_rows, header=["N", "t", "std_last5"])

    save_csv(os.path.join(outdir, "profits.csv"),
             profit_rows,
             header=["N","train_M","test_M","SDDP_in","SDDP_out","train_time_s","online_ms_per_path"])


# ---------- CLI ----------
# Most users will tweak things here via command-line flags.

def main():
    p = argparse.ArgumentParser()
    # Output folder (use different names to avoid overwriting)
    p.add_argument("--outdir", type=str, default="results_sddp")

    # Iteration schedule: pick which N values to train
    p.add_argument("--Ns", type=int, nargs="+",
                   default=[100,200,500,1000],
                   help="SDDP iterations per run")

    # Scenario pool sizes: more paths = smoother learning but slower
    p.add_argument("--train_M", type=int, default=1000, help="training paths (for sampling/eval)")
    p.add_argument("--test_M",  type=int, default=1000, help="test paths for OOS evaluation")

    # Learning knobs
    p.add_argument("--batch",   type=int, default=8,   help="mini-batch size per SDDP iteration")
    p.add_argument("--alpha",   type=float, default=0.5, help="fixed stepsize for (a,b) smoothing")
    p.add_argument("--max_cuts_per_stage", type=int, default=0,
                   help="FIFO cap for cuts kept at each stage (0 means no cap)")

    # Randomness + horizon controls (match your generator)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--T", type=int, default=48)
    p.add_argument("--burnin", type=int, default=400)

    args = p.parse_args()

    ensure_dir(args.outdir)
    sddp_train_and_evaluate(args.outdir, args.Ns, args.seed, args.T, args.burnin,
                            args.train_M, args.test_M,
                            batch=args.batch, alpha=args.alpha,
                            max_cuts_per_stage=(args.max_cuts_per_stage or None),
                            rng_seed=args.seed)

if __name__ == "__main__":
    main()
