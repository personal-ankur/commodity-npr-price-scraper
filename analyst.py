#!/usr/bin/env python3
import argparse, os, sys, math
import pandas as pd, numpy as np

NEPALI_MONTHS = [
    "Baisakh","Jestha","Ashadh","Shrawan","Bhadra","Ashwin",
    "Kartik","Mangsir","Poush","Magh","Falgun","Chaitra"
]
MONTH_TO_ORDER = {m: i for i, m in enumerate(NEPALI_MONTHS, start=1)}
REQUIRED_COLS = {"date","year","month","day","fine_gold","standard_gold","silver"}

def winsorize(s, lower=0.05, upper=0.95):
    if s.empty: return s
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=lo, upper=hi)

def trimmed_mean(s, p=0.10):
    if s.empty: return np.nan
    s = s.sort_values()
    n = len(s); k = int(math.floor(n*(p/2)))
    return s.iloc[k:n-k].mean() if 2*k < n else s.mean()

def load_and_validate(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    for col in ["year","day","fine_gold","standard_gold","silver"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["month"] = df["month"].astype(str).str.strip()
    unknown = sorted(set(df["month"]) - set(NEPALI_MONTHS))
    if unknown:
        print(f"[WARN] Ignoring unrecognized month names: {unknown}", file=sys.stderr)
    df = df[df["month"].isin(NEPALI_MONTHS)].copy()
    if df.empty:
        raise ValueError("No rows with recognized Nepali month names after filtering.")
    df["month_order"] = df["month"].map(MONTH_TO_ORDER)
    df = df.sort_values(["year","month_order","day"]).reset_index(drop=True)
    df["t"] = np.arange(1, len(df)+1, dtype=float)  # time index
    return df

def _fit_log_trend(series: pd.Series, t: pd.Series):
    y = np.log(series.values); x = t.values
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 2:
        return pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index)
    b, a = np.polyfit(x[mask], y[mask], 1)  # returns slope, intercept
    yhat_log = a + b*x
    trend_level = np.exp(yhat_log)
    resid_log = y - yhat_log
    return pd.Series(trend_level, index=series.index), pd.Series(resid_log, index=series.index)

def _recent_weights(df: pd.DataFrame, half_life_years: float):
    age = df["year"].astype(float).max() - df["year"].astype(float)
    lam = math.log(2) / max(half_life_years, 1e-6)
    w = np.exp(-lam * age)
    return w / w.sum()

def compute_robust_monthly(df: pd.DataFrame,
                           metals=("fine_gold","standard_gold","silver"),
                           trim_p=0.10,
                           winsor=None,
                           half_life_years=2.0):
    out_rows = []
    df = df.copy()
    if winsor is not None:
        l,u = winsor
        for m in metals:
            df[m] = winsorize(df[m], lower=l, upper=u)

    # Year averages for YoY normalization
    year_avgs = (df.groupby("year", as_index=False)[list(metals)].mean()
                   .rename(columns={m: f"{m}_year_avg" for m in metals}))
    df = df.merge(year_avgs, on="year", how="left")

    # Trend & residuals
    for m in metals:
        trend_level, resid_log = _fit_log_trend(df[m], df["t"])
        df[f"{m}_trend"] = trend_level
        df[f"{m}_resid_log"] = resid_log
        df[f"{m}_yoy_norm"] = df[m] / df[f"{m}_year_avg"] - 1.0

    df["_w"] = _recent_weights(df, half_life_years)

    for month in NEPALI_MONTHS:
        sub = df[df["month"] == month]
        if sub.empty: continue
        row = {"month": month, "samples": int(len(sub))}
        for m in metals:
            s = sub[m].dropna()
            row[f"{m}_median"] = s.median() if not s.empty else np.nan
            row[f"{m}_trimmed_mean_{int(trim_p*100)}p"] = trimmed_mean(s, trim_p)
            w_sub = sub.loc[s.index, "_w"]
            row[f"{m}_recent_weighted_mean_hl{half_life_years}y"] = float((s * (w_sub/w_sub.sum())).sum()) if w_sub.sum() > 0 else np.nan
            r = sub[f"{m}_resid_log"].dropna()
            row[f"{m}_seasonality_%"] = float(r.mean()*100.0) if not r.empty else np.nan
            yni = sub[f"{m}_yoy_norm"].dropna()
            row[f"{m}_yoy_month_index_%"] = float(yni.mean()*100.0) if not yni.empty else np.nan
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    out["month_order"] = out["month"].map(MONTH_TO_ORDER)
    out = out.sort_values("month_order").drop(columns=["month_order"]).reset_index(drop=True)
    for c in out.columns:
        if c not in ("month","samples") and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(3)
    return out, df

# ----------------- PLOTS -----------------
def _maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        print(f"[WARN] matplotlib not available: {e}", file=sys.stderr)
        return None

def save_plots(avg_df: pd.DataFrame, df_full: pd.DataFrame, out_dir: str):
    import os, sys
    os.makedirs(out_dir, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib not available: {e}", file=sys.stderr)
        return

    metals = ["fine_gold","standard_gold","silver"]
    nepali_months = ["Baisakh","Jestha","Ashadh","Shrawan","Bhadra","Ashwin",
                     "Kartik","Mangsir","Poush","Magh","Falgun","Chaitra"]

    def bar_months(values, title, fname, ylabel="%"):
        months = avg_df["month"].tolist()
        plt.figure()
        plt.bar(months, values)
        plt.title(title); plt.xlabel("Nepali Months"); plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=160); plt.close()
        print(f"[OK] Saved plot: {path}")

    def line_months(values, title, fname, ylabel="Price"):
        months = avg_df["month"].tolist()
        plt.figure()
        plt.plot(months, values, marker="o")
        plt.title(title); plt.xlabel("Nepali Months"); plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=160); plt.close()
        print(f"[OK] Saved plot: {path}")

    def line_trend_vs_actual(metal, title, fname):
        plt.figure()
        plt.plot(df_full["t"], df_full[metal], label="actual")
        plt.plot(df_full["t"], df_full[f"{metal}_trend"], label="trend")
        plt.title(title); plt.xlabel("Time Index"); plt.ylabel("Price")
        plt.legend(); plt.tight_layout()
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=160); plt.close()
        print(f"[OK] Saved plot: {path}")

    # === NEW HELPERS FOR ALL-YEARS MONTH-BY-MONTH CHARTS ===
    def month_year_pivot(metal: str):
        # average price per (year, month_order)
        g = (df_full
             .groupby(["year","month_order"], as_index=False)[metal]
             .mean())
        # pivot to years x 12 months
        piv = g.pivot(index="year", columns="month_order", values=metal)
        # ensure all 12 months in order
        piv = piv.reindex(columns=range(1, 13))
        return piv

    def heatmap_month_by_year(metal: str):
        piv = month_year_pivot(metal)
        plt.figure()
        im = plt.imshow(piv.values, aspect="auto", interpolation="nearest")
        plt.title(f"Month-by-Month Across Years — {metal.replace('_',' ').title()}")
        plt.xlabel("Nepali Months"); plt.ylabel("Year")
        plt.xticks(ticks=np.arange(12), labels=nepali_months, rotation=45, ha="right")
        plt.yticks(ticks=np.arange(len(piv.index)), labels=piv.index.astype(int))
        plt.colorbar(im, label="Average Price")
        plt.tight_layout()
        fname = f"monthly_by_year_heatmap_{metal}.png"
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=160); plt.close()
        print(f"[OK] Saved plot: {path}")

    def lines_month_by_year(metal: str):
        piv = month_year_pivot(metal)
        plt.figure()
        years_sorted = list(piv.index.astype(int))
        # Light lines for older years, slightly bolder for the most recent
        for y in years_sorted[:-1]:
            series = piv.loc[y].values
            plt.plot(nepali_months, series, marker="o", alpha=0.5)
        if years_sorted:
            y_last = years_sorted[-1]
            plt.plot(nepali_months, piv.loc[y_last].values, marker="o", linewidth=2.5, label=f"{y_last}")
            plt.legend(title="Most recent year", loc="best")
        plt.title(f"Month-by-Month Lines by Year — {metal.replace('_',' ').title()}")
        plt.xlabel("Nepali Months"); plt.ylabel("Average Price")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fname = f"monthly_by_year_lines_{metal}.png"
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=160); plt.close()
        print(f"[OK] Saved plot: {path}")

    # ====== EXISTING PLOTS (seasonality, YoY, recent-weighted, trend) ======
    for m in metals:
        bar_months(avg_df[f"{m}_seasonality_%"],
                   f"Detrended Seasonality (%): {m.replace('_',' ').title()}",
                   f"seasonality_{m}.png")
        bar_months(avg_df[f"{m}_yoy_month_index_%"],
                   f"YoY-Normalized Month Index (%): {m.replace('_',' ').title()}",
                   f"yoy_index_{m}.png")
        col = [c for c in avg_df.columns if c.startswith(f"{m}_recent_weighted_mean")][0]
        line_months(avg_df[col],
                    f"Recent-Weighted Mean by Month: {m.replace('_',' ').title()}",
                    f"recent_weighted_mean_{m}.png",
                    ylabel="Price")
        line_trend_vs_actual(m,
                             f"Actual vs Trend: {m.replace('_',' ').title()}",
                             f"trend_vs_actual_{m}.png")

        # ====== NEW CHARTS YOU ASKED FOR ======
        heatmap_month_by_year(m)
        lines_month_by_year(m)

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser(description="Robust gold/silver monthly trend analyst with plots")
    ap.add_argument("--csv", required=True, help="Path to prices CSV")
    ap.add_argument("--out", default="monthly_robust_averages.csv", help="Output CSV file")
    ap.add_argument("--trim", type=float, default=0.10, help="Trimmed-mean proportion (e.g., 0.10)")
    ap.add_argument("--winsor", type=float, nargs=2, metavar=("LOW","HIGH"),
                    help="Winsorize at quantiles LOW/HIGH (e.g., 0.05 0.95)")
    ap.add_argument("--half-life", type=float, default=2.0, help="Half-life in years for recency weights")
    ap.add_argument("--plots", action="store_true", help="Save diagnostic plots (PNGs)")
    ap.add_argument("--plot-dir", default="plots", help="Directory to save plots")
    args = ap.parse_args()

    df = load_and_validate(args.csv)
    winsor_q = tuple(args.winsor) if args.winsor else None

    avg, df_full = compute_robust_monthly(
        df,
        trim_p=args.trim,
        winsor=winsor_q,
        half_life_years=args.half_life
    )

    avg.to_csv(args.out, index=False)
    print(f"[OK] Wrote: {args.out}")
    print(avg.to_string(index=False))

    if args.plots:
        save_plots(avg, df_full, args.plot_dir)

if __name__ == "__main__":
    main()

