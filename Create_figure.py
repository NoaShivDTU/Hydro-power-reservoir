import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional: for the "mean price/inflow" figure
try:
    import Input_generator as IG
    HAS_IG = True
except Exception:
    HAS_IG = False


# ----------------------------- Helpers -----------------------------
def _pick_col(df_or_series, candidates, default=None):
    """Return the first column name (or index key) that exists (case sensitive)."""
    cols = df_or_series.columns if isinstance(df_or_series, pd.DataFrame) else df_or_series.index
    for c in candidates:
        if c in cols:
            return c
    return default

def _read_profits(indir: Path) -> pd.DataFrame:
    p = indir / "profits.csv"
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    if "N" in df.columns:
        df = df.sort_values("N")
    return df

def _read_std_last5(indir: Path) -> pd.DataFrame:
    p = indir / "std_last5_over_time.csv"
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    return df

def _read_values_last5(indir: Path, N: int) -> pd.DataFrame:
    p = indir / f"values_last5_N{int(N)}.csv"
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    # Expect columns: t, last5_index, Vhat
    return df

def _read_lower_bounds(indir: Path, N: int) -> pd.DataFrame | None:
    p = indir / f"lower_bounds_N{int(N)}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    return df


# ----------------------- EEV/WS Readers (robust) -----------------------
def _read_summary_eev_ws(indir: Path) -> pd.DataFrame | None:
    """Try reading results_eev_ws/summary.csv if present."""
    p = indir / "summary.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    # Normalize likely columns
    col_method = _pick_col(df, ["method", "approach", "algo", "Algorithm"])
    col_N      = _pick_col(df, ["N", "n", "samples"])
    col_in     = _pick_col(df, ["in_sample", "in", "in_sample_mean_total_profit"])
    col_out    = _pick_col(df, ["out_of_sample", "out", "out_of_sample_mean_total_profit"])
    if not all([col_method, col_N, col_in, col_out]):
        return None
    # Keep only EEV/WS
    df = df.rename(columns={col_method: "method", col_N: "N", col_in: "in_sample", col_out: "out_of_sample"})
    df["method"] = df["method"].astype(str).str.upper()
    df = df[df["method"].isin(["EEV", "WS"])]
    return df[["method", "N", "in_sample", "out_of_sample"]]

def _read_wide_or_long(file: Path) -> pd.DataFrame:
    """
    Read a small CSV that might be:
      - wide: one row with columns 'EEV','WS' (values)
      - long: columns like ('method'|'approach', 'value')
    Returns a long-form DataFrame with columns ['method','value'] (method uppercased).
    """
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    # wide?
    if "EEV" in df.columns or "WS" in df.columns:
        # assume first row is the values
        row0 = df.iloc[0]
        data = []
        for m in ["EEV", "WS"]:
            if m in row0.index:
                try:
                    val = float(row0[m])
                    data.append({"method": m, "value": val})
                except Exception:
                    pass
        return pd.DataFrame(data)
    # long?
    col_m = _pick_col(df, ["method", "approach", "algo", "Algorithm"])
    col_v = _pick_col(df, ["value", "profit", "mean", "estimate"])
    if col_m and col_v:
        out = df[[col_m, col_v]].copy()
        out.columns = ["method", "value"]
        out["method"] = out["method"].astype(str).str.upper()
        return out
    # fallback: empty
    return pd.DataFrame(columns=["method", "value"])

def _read_eev_ws_pair_from_split_files(indir: Path, N: int) -> dict[str, tuple[float, float]] | None:
    """
    When no summary exists, try:
      in_sample_N{N}.csv    -> contains EEV/WS in-sample
      out_of_sample_N{N}.csv-> contains EEV/WS out-of-sample
    Returns dict like {'EEV': (in,out), 'WS': (in,out)} or None.
    """
    pin  = indir / f"in_sample_N{int(N)}.csv"
    pout = indir / f"out_of_sample_N{int(N)}.csv"
    if not (pin.exists() and pout.exists()):
        return None
    din  = _read_wide_or_long(pin)
    dout = _read_wide_or_long(pout)
    if din.empty or dout.empty:
        return None
    result = {}
    for m in ["EEV", "WS"]:
        vin  = din.loc[din["method"] == m, "value"]
        vout = dout.loc[dout["method"] == m, "value"]
        if not vin.empty and not vout.empty:
            result[m] = (float(vin.iloc[0]), float(vout.iloc[0]))
    return result if result else None

def read_eev_ws(indir: Path, Ns: list[int]) -> dict[int, dict[str, tuple[float, float]]]:
    """
    Returns: mapping N -> {'EEV': (in,out), 'WS': (in,out)} for all Ns we can parse.
    Tries summary.csv first; falls back to in/out_of_sample_N*.csv files.
    """
    out: dict[int, dict[str, tuple[float, float]]] = {}
    df_sum = _read_summary_eev_ws(indir)
    if df_sum is not None and not df_sum.empty:
        for N in Ns:
            g = df_sum[df_sum["N"] == N]
            if g.empty:
                continue
            entry = {}
            for m in ["EEV", "WS"]:
                gm = g[g["method"] == m]
                if not gm.empty:
                    entry[m] = (float(gm.iloc[0]["in_sample"]), float(gm.iloc[0]["out_of_sample"]))
            if entry:
                out[N] = entry
    # fallback per-N files (also fills any missing Ns)
    for N in Ns:
        if N in out and all(k in out[N] for k in ["EEV", "WS"]):
            continue
        pair = _read_eev_ws_pair_from_split_files(indir, N)
        if pair:
            out.setdefault(N, {}).update(pair)
    return out


# ----------------------------- Figures -----------------------------
def fig_std_over_time_new(indir: Path, Ns: list[int], title_prefix: str) -> go.Figure:
    df = _read_std_last5(indir)
    fig = go.Figure()
    for N in Ns:
        g = df[df["N"] == N]
        if g.empty:
            continue
        fig.add_trace(go.Scatter(x=g["t"], y=g["std_last5"], mode="lines", name=f"N={N}"))
    fig.update_layout(
        title=f"{title_prefix} — standard deviation of last five V̂ over time",
        xaxis_title="Time (h)",
        yaxis_title="Std of last five V̂ ($)",
        legend_title_text="Training size",
        template="simple_white",
        hovermode="x unified"
    )
    return fig


def fig_vhat_last5_new(indir: Path, N_small: int, N_large: int,
                       title_prefix: str) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=(f"Last five out of N={N_small}",
                                        f"Last five out of N={N_large}"))
    for col, N in enumerate([N_small, N_large], start=1):
        df = _read_values_last5(indir, N)
        # reshape: each last5_index is a line over t
        for s_id, g in df.groupby("last5_index"):
            g = g.sort_values("t")
            fig.add_trace(
                go.Scatter(x=g["t"], y=g["Vhat"], mode="lines",
                           name=f"N={N} sample {int(s_id)}", showlegend=False),
                row=1, col=col
            )
        fig.update_xaxes(title_text="Time (h)", row=1, col=col)
    fig.update_yaxes(title_text="Estimate of post-decision value ($)", row=1, col=1)
    fig.update_layout(title=f"{title_prefix} — post-decision value V̂ (last five samples)",
                      template="simple_white", hovermode="x")
    return fig

def fig_runtime_vs_samples_new(indir_adp: Path, Ns_adp: list[int],
                               indir_sddp: Path, Ns_sddp: list[int]) -> go.Figure:
    def _times(indir, Ns):
        if not Ns: return []
        df = _read_profits(indir)
        col = _pick_col(df, ["train_time_s", "train_time_sec", "train_sec"])
        return [float(df.loc[df["N"] == N, col].values[0]) / 3600.0 for N in Ns]

    hours_adp  = _times(indir_adp, Ns_adp)  if Ns_adp  else []
    hours_sddp = _times(indir_sddp, Ns_sddp) if Ns_sddp else []

    fig = go.Figure()
    if Ns_adp:
        fig.add_trace(go.Scatter(x=Ns_adp, y=hours_adp, mode="lines+markers",
                                 name="ADP", marker_symbol="asterisk-open", marker_size=10))
    if Ns_sddp:
        fig.add_trace(go.Scatter(x=Ns_sddp, y=hours_sddp, mode="lines+markers",
                                 name="SDDP", marker_symbol="circle-open", marker_size=10))
    fig.update_layout(title="Running time vs number of samples",
                      xaxis_title="Number of samples",
                      yaxis_title="Running time (h)",
                      template="simple_white", hovermode="x unified")
    return fig

def fig_profit_vs_samples_new(indir_adp: Path, Ns_adp: list[int],
                              indir_sddp: Path, Ns_sddp: list[int],
                              style: str = "markers") -> go.Figure:
    def _profits(indir, Ns):
        if not Ns: return [], []
        df = _read_profits(indir)
        col_out = _pick_col(df, ["SDDP_out", "ADP_out", "out", "out_of_sample_mean_total_profit"])
        X, Y = [], []
        for N in Ns:
            row = df.loc[df["N"] == N]
            if row.empty: continue
            X.append(N); Y.append(float(row.iloc[0][col_out]))
        return X, Y

    X_adp, MU_adp     = _profits(indir_adp,  Ns_adp)
    X_sddp, MU_sddp   = _profits(indir_sddp, Ns_sddp)

    fig = go.Figure()
    if X_adp:
        if style.lower() == "bars":
            fig.add_trace(go.Bar(name="ADP", x=X_adp, y=MU_adp))
        else:
            fig.add_trace(go.Scatter(x=X_adp, y=MU_adp,
                                     mode="lines+markers", name="ADP",
                                     marker_symbol="asterisk-open", marker_size=10))
    if X_sddp:
        if style.lower() == "bars":
            fig.add_trace(go.Bar(name="SDDP", x=X_sddp, y=MU_sddp))
        else:
            fig.add_trace(go.Scatter(x=X_sddp, y=MU_sddp,
                                     mode="lines+markers", name="SDDP",
                                     marker_symbol="circle-open", marker_size=10))
    if style.lower() == "bars":
        fig.update_layout(barmode="group")

    fig.update_layout(
        title="Out-of-sample profit vs number of samples",
        xaxis_title="Number of samples (N)",
        yaxis_title="Out-of-sample mean total profit ($)",
        template="simple_white",
        hovermode="x unified",
        legend_title_text="Algorithm"
    )
    return fig



def fig_lower_bounds(indir: Path, N: int, title_prefix: str) -> go.Figure | None:
    df = _read_lower_bounds(indir, N)
    if df is None:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["iter"], y=df["lower_bound_batch_avg"], mode="lines",
                             name=f"N={N}"))
    fig.update_layout(
        title=f"{title_prefix} — per-iteration lower bound (N={N})",
        xaxis_title="Iteration",
        yaxis_title="Batch-average forward objective",
        template="simple_white",
        hovermode="x unified"
    )
    return fig

def fig_mean_price_inflow(mean_seed: int, mean_N: int, mean_T: int, mean_burnin: int) -> go.Figure:
    if not HAS_IG:
        raise RuntimeError("Input_generator.py not found. Place it next to this script.")
    (J, T_ret, l_max, l_min, l0, pi_max, pi_min,
     price_samples, inflow_samples, nu0_vec, rho0, R,
     _a, _b, _c, g) = IG.generate_input(N=mean_N, T=mean_T, seed=mean_seed, burn_in=mean_burnin)
    p_mean = np.mean(price_samples, axis=0)
    q_mean = np.mean(inflow_samples, axis=0)  # shape [T, J]
    fig = make_subplots(rows=1, cols=2, shared_xaxes=False,
                        subplot_titles=("Mean price over time","Mean inflow per reservoir"))
    x_price = np.arange(1, p_mean.shape[0] + 1)
    fig.add_trace(go.Scatter(x=x_price, y=p_mean, mode="lines", name="Price"), row=1, col=1)
    fig.update_xaxes(title_text="Time (h)", row=1, col=1)
    fig.update_yaxes(title_text="Mean price ($)", row=1, col=1)
    x_inflow = np.arange(1, q_mean.shape[0] + 1)
    for j in range(q_mean.shape[1]):
        fig.add_trace(go.Scatter(x=x_inflow, y=q_mean[:, j], mode="lines", name=f"Inflow R{j+1}"),
                      row=1, col=2)
    fig.update_xaxes(title_text="Time (h)", row=1, col=2)
    fig.update_yaxes(title_text="Mean inflow (units)", row=1, col=2)
    fig.update_layout(template="simple_white", hovermode="x unified",
                      title=f"Mean price and inflow (N={mean_N}, T={mean_T}, seed={mean_seed}, burn-in={mean_burnin})")
    return fig


# ----------------------------- Table -----------------------------
def build_table_profit(indir_adp: Path, Ns_adp: list[int],
                       indir_sddp: Path, Ns_sddp: list[int],
                       outdir: Path,
                       scale_to_1e4: bool = True,
                       Ns_table: list[int] = [500, 1000],
                       # NEW: EEV/WS
                       indir_eev_ws: Path | None = None,
                       Ns_eev_ws: list[int] | None = None):
    """
    Compact table for requested N values (default: [500, 1000]).
    Rows: ADP, SDDP, EEV, WS; Columns: In-sample / Out-of-sample.
    EEV/WS pulled from 'indir_eev_ws' via summary.csv or per-N files.
    """
    Ns_all = [N for N in Ns_table if (N in set(Ns_adp) | set(Ns_sddp) | set(Ns_eev_ws or []))]
    if not Ns_all:
        print("No matching Ns to tabulate — skipping table.")
        return

    def _pair(indir, N):
        df = _read_profits(indir)
        row = df.loc[df["N"] == N]
        if row.empty: return (np.nan, np.nan)
        col_in  = _pick_col(row, ["SDDP_in",  "ADP_in",  "in",  "in_sample_mean_total_profit"])
        col_out = _pick_col(row, ["SDDP_out", "ADP_out", "out", "out_of_sample_mean_total_profit"])
        return float(row.iloc[0][col_in]), float(row.iloc[0][col_out])

    # Read EEV/WS map if provided
    eev_ws_map: dict[int, dict[str, tuple[float, float]]] = {}
    if indir_eev_ws is not None:
        eev_ws_map = read_eev_ws(indir_eev_ws, Ns_eev_ws or [])

    # Build the MultiIndex structure
    cols = []
    for N in Ns_all:
        cols.append(("In-sample",  f"N={N}"))
        cols.append(("Out-of-sample", f"N={N}"))
    idx_rows = ["ADP", "SDDP", "EEV", "WS"]
    df_tab = pd.DataFrame(index=idx_rows,
                          columns=pd.MultiIndex.from_tuples(cols, names=["Set", "N"]),
                          dtype=float)

    # Fill ADP
    for N in Ns_all:
        if N in Ns_adp:
            mu_is, mu_oos = _pair(indir_adp, N)
            df_tab.loc["ADP", ("In-sample",  f"N={N}")] = mu_is
            df_tab.loc["ADP", ("Out-of-sample", f"N={N}")] = mu_oos

    # Fill SDDP
    for N in Ns_all:
        if N in Ns_sddp:
            mu_is, mu_oos = _pair(indir_sddp, N)
            df_tab.loc["SDDP", ("In-sample",  f"N={N}")] = mu_is
            df_tab.loc["SDDP", ("Out-of-sample", f"N={N}")] = mu_oos

    # Fill EEV/WS
    for N in Ns_all:
        pair = eev_ws_map.get(N, {})
        if "EEV" in pair:
            mu_is, mu_oos = pair["EEV"]
            df_tab.loc["EEV", ("In-sample",  f"N={N}")] = mu_is
            df_tab.loc["EEV", ("Out-of-sample", f"N={N}")] = mu_oos
        if "WS" in pair:
            mu_is, mu_oos = pair["WS"]
            df_tab.loc["WS", ("In-sample",  f"N={N}")] = mu_is
            df_tab.loc["WS", ("Out-of-sample", f"N={N}")] = mu_oos

    df_save = (df_tab / 1e4) if scale_to_1e4 else df_tab.copy()
    csv_path = outdir / "table_profit_estimates.csv"
    df_save.to_csv(csv_path, float_format="%.4f")

    # Build LaTeX manually
    def fmt(x): return "" if (pd.isna(x)) else f"{x:.4f}"
    rows = idx_rows
    data = {row: [] for row in rows}
    for N in Ns_all:
        for set_name in ["In-sample", "Out-of-sample"]:
            series = df_save.loc[:, (set_name, f"N={N}")]
            for r in rows:
                data[r].append(fmt(series.loc[r]))
    unit = r" (\(\times 10^{4}\)\$)" if scale_to_1e4 else ""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Profit estimates{unit} for ADP, SDDP, EEV, and WS.}}")
    lines.append(r"\label{tab:profit_estimates}")
    lines.append(r"\begin{tabular}{l" + "cc"*len(Ns_all) + r"}")
    lines.append(r"\toprule")
    lines.append("& " + " & ".join([rf"\multicolumn{{2}}{{c}}{{N={N}}}" for N in Ns_all]) + r" \\")
    # cmidrules
    start = 2
    lines.append("".join([rf"\cmidrule(lr){{{start+i*2}-{start+i*2+1}}}" for i in range(len(Ns_all))]))
    lines.append("Approach & " + " & ".join(["In-sample & Out-of-sample"]*len(Ns_all)) + r" \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(rf"{r} & " + " & ".join(data[r]) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    latex_text = "\n".join(lines)
    tex_path = outdir / "table_profit_estimates.tex"
    with open(tex_path, "w") as f:
        f.write(latex_text)
    print(f"Saved table to:\n  {csv_path}\n  {tex_path}\n")
    print("---- Paste-ready LaTeX table ----")
    print(latex_text)


# ------------------------------ Main ------------------------------
def main():
    p = argparse.ArgumentParser()
    # Inputs
    p.add_argument("--indir_adp", type=str, default="results_adp")
    p.add_argument("--indir_adp_uc", type=str, default="results_adp_unit")
    p.add_argument("--indir_sddp", type=str, default="results_sddp")
    p.add_argument("--indir_eev_ws", type=str, default="results_eev_ws")  # NEW
    p.add_argument("--Ns_adp", type=int, nargs="*", default=[100, 200, 500, 1000])
    p.add_argument("--Ns_adp_uc", type=int, nargs="*", default=[100, 200, 500, 1000])
    p.add_argument("--Ns_sddp", type=int, nargs="*", default=[100, 200, 500, 1000])
    p.add_argument("--Ns_eev_ws", type=int, nargs="*", default=None)  # NEW (defaults to Ns_adp if omitted)
    # Last-5 panels (defaults to smallest & largest in each list)
    p.add_argument("--N_small_adp", type=int, default=None)
    p.add_argument("--N_large_adp", type=int, default=None)
    p.add_argument("--N_small_adp_uc", type=int, default=None)
    p.add_argument("--N_large_adp_uc", type=int, default=None)
    p.add_argument("--N_small_sddp", type=int, default=None)
    p.add_argument("--N_large_sddp", type=int, default=None)
    # Profit plot style
    p.add_argument("--profit_style", type=str, default="markers",
                   choices=["markers", "bars"])
    # (Optional) mean price/inflow generation params
    p.add_argument("--mean_seed", type=int, default=42)
    p.add_argument("--mean_N", type=int, default=1000)
    p.add_argument("--mean_T", type=int, default=48)
    p.add_argument("--mean_burnin", type=int, default=400)
    # Output
    p.add_argument("--outdir", type=str, default="figures")
    # Table options
    p.add_argument("--scale_1e4", action="store_true", default=True)
    p.add_argument("--Ns_table", type=int, nargs="*", default=[500, 1000])
    args = p.parse_args()

    indir_adp     = Path(args.indir_adp)
    indir_adp_uc  = Path(args.indir_adp_uc)
    indir_sddp    = Path(args.indir_sddp)
    indir_eev_ws  = Path(args.indir_eev_ws)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    Ns_adp    = sorted(list(args.Ns_adp))     if args.Ns_adp    else []
    Ns_adp_uc = sorted(list(args.Ns_adp_uc))  if args.Ns_adp_uc else []
    Ns_sddp   = sorted(list(args.Ns_sddp))    if args.Ns_sddp   else []
    Ns_eev_ws = sorted(list(args.Ns_eev_ws))  if args.Ns_eev_ws else Ns_adp  # default to ADP Ns

    # Small/large picker
    def pick_small_large(Ns, given_small, given_large):
        if not Ns: return None, None
        n_small = given_small if given_small is not None else Ns[0]
        n_large = given_large if given_large is not None else Ns[-1]
        return n_small, n_large

    N_small_adp,    N_large_adp    = pick_small_large(Ns_adp,    args.N_small_adp,    args.N_large_adp)
    N_small_adp_uc, N_large_adp_uc = pick_small_large(Ns_adp_uc, args.N_small_adp_uc, args.N_large_adp_uc)
    N_small_sddp,   N_large_sddp   = pick_small_large(Ns_sddp,   args.N_small_sddp,   args.N_large_sddp)

    # ---------------- Figures ----------------
    # Optional (if Input_generator is present)
    if HAS_IG:
        fig5 = fig_mean_price_inflow(args.mean_seed, args.mean_N, args.mean_T, args.mean_burnin)
        fig5.write_image(str(outdir / "fig_mean_price_inflow.pdf"))
    else:
        print("Note: Input_generator.py not found — skipping fig_mean_price_inflow.pdf")

    # ADP (baseline)
    if Ns_adp:
        fig1a = fig_std_over_time_new(indir_adp, Ns_adp, "ADP")
        fig1a.write_image(str(outdir / "fig_std_over_time_adp.pdf"))

        if N_small_adp is not None and N_large_adp is not None:
            fig2a = fig_vhat_last5_new(indir_adp, N_small_adp, N_large_adp, "ADP")
            fig2a.write_image(str(outdir / "fig_vhat_last5_adp.pdf"))

    # ADP Unit Commitment
    if Ns_adp_uc:
        fig1uc = fig_std_over_time_new(indir_adp_uc, Ns_adp_uc, "ADP-UC")
        fig1uc.write_image(str(outdir / "fig_std_over_time_adp_uc.pdf"))

        if N_small_adp_uc is not None and N_large_adp_uc is not None:
            fig2uc = fig_vhat_last5_new(indir_adp_uc, N_small_adp_uc, N_large_adp_uc, "ADP-UC")
            fig2uc.write_image(str(outdir / "fig_vhat_last5_adp_uc.pdf"))

    # SDDP
    if Ns_sddp:
        fig1b = fig_std_over_time_new(indir_sddp, Ns_sddp, "SDDP")
        fig1b.write_image(str(outdir / "fig_std_over_time_sddp.pdf"))

        if N_small_sddp is not None and N_large_sddp is not None:
            fig2b = fig_vhat_last5_new(indir_sddp, N_small_sddp, N_large_sddp, "SDDP")
            fig2b.write_image(str(outdir / "fig_vhat_last5_sddp.pdf"))

    # Runtime comparison
    fig3 = fig_runtime_vs_samples_new(indir_adp, Ns_adp, indir_sddp, Ns_sddp)
    fig3.write_image(str(outdir / "fig_runtime_vs_samples.pdf"))

    # Profit vs samples (OOS)
    fig4 = fig_profit_vs_samples_new(indir_adp, Ns_adp, indir_sddp, Ns_sddp,
                                     style=args.profit_style)
    fig4.write_image(str(outdir / "fig_profit_vs_samples.pdf"))

    # ---------------- Table (ADP + SDDP + EEV + WS) ----------------
    build_table_profit(indir_adp=indir_adp, Ns_adp=Ns_adp,
                       indir_sddp=indir_sddp, Ns_sddp=Ns_sddp,
                       outdir=outdir, scale_to_1e4=args.scale_1e4,
                       Ns_table=args.Ns_table,
                       indir_eev_ws=indir_eev_ws, Ns_eev_ws=Ns_eev_ws)

    print(f"Figures and table saved in {outdir.resolve()}")

if __name__ == "__main__":
    main()
