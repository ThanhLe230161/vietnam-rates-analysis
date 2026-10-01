"""Vietnam rates: time plot, ADF/KPSS, differencing, ACF/PACF, ARIMA-by-AIC forecast,
pseudo-out-of-sample check vs. random walk, and time-series momentum.

Usage
  python vgb_analysis.py --csv vn_2s10s_spread_bps.csv --unit bp  --name "VN 2s10s spread"
  python vgb_analysis.py --csv vn_10y_yield_volatility.csv --unit raw --log --name "VN 10Y yield volatility"
  python vgb_analysis.py --investing "Vietnam 10-Year Bond Yield Historical Data.csv" --unit pct --name "VN 10Y yield"
  python vgb_analysis.py --fred DGS10 --unit pct --name "US 10Y UST"

--unit tells the script how to convert a 1-unit move into basis points
(pct: x100, bp: x1, raw: no bp conversion, changes reported in native units).
"""
import argparse
import itertools
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")
BP = {"pct": 100.0, "bp": 1.0, "raw": 1.0}


# ---------------------------------------------------------------- data
def load_csv(path):
    """Two-column file with 'date' and 'value' (AsianBondsOnline export via VN_bonds.ipynb)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"].astype(float).sort_index()


def load_investing(path):
    """Investing.com 'Historical Data' export: Date, Price, Open, High, Low, Change %."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")["Price"].astype(float).sort_index()


def load_fred(series_id):
    df = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}")
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"]


def to_monthly(s):
    """Month average: smooths daily liquidity spikes (quarter-end, Tet)."""
    y = s.resample("ME").mean().dropna()
    return y.asfreq("ME").interpolate()


# ---------------------------------------------------------------- tests
def stationarity(x, label):
    adf_p = adfuller(x, autolag="AIC")[1]
    try:
        kpss_p = kpss(x, regression="c", nlags="auto")[1]
    except Exception:
        kpss_p = np.nan
    # ADF H0: unit root. KPSS H0: stationary. Statsmodels caps KPSS p in [0.01, 0.10].
    if adf_p < 0.05 and (np.isnan(kpss_p) or kpss_p > 0.05):
        v = "stationary"
    elif adf_p >= 0.05 and kpss_p <= 0.05:
        v = "non-stationary"
    else:
        v = "inconclusive"
    return f"{label:<16} ADF p={adf_p:6.3f} | KPSS p={kpss_p:6.3f} -> {v}", v


def choose_d(y, report):
    line, v0 = stationarity(y, "level"); report.append(line)
    line, v1 = stationarity(y.diff().dropna(), "1st difference"); report.append(line)
    if v0 == "stationary":
        return 0
    # Inconclusive level + stationary difference: difference (conservative for rates).
    return 1


def select_arima(y, d, max_p=3, max_q=3):
    best, rows = None, []
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        try:
            res = ARIMA(y, order=(p, d, q), trend="n" if d else "c").fit()
        except Exception:
            continue
        rows.append((p, d, q, res.aic))
        if best is None or res.aic < best[1].aic:
            best = ((p, d, q), res)
    table = pd.DataFrame(rows, columns=["p", "d", "q", "AIC"]).sort_values("AIC")
    return best, table


def oos_vs_random_walk(y, order, d, h=3, n_origins=24):
    """Rolling-origin h-step forecast errors: ARIMA vs. no-change (random walk)."""
    e_arima, e_rw = [], []
    for k in range(n_origins, 0, -1):
        end = len(y) - k - h + 1
        if end < 36:
            continue
        train, actual = y.iloc[:end], y.iloc[end + h - 1]
        try:
            f = ARIMA(train, order=order, trend="n" if d else "c").fit().forecast(h).iloc[-1]
        except Exception:
            continue
        e_arima.append(actual - f)
        e_rw.append(actual - train.iloc[-1])
    rmse = lambda e: float(np.sqrt(np.mean(np.square(e)))) if e else np.nan
    return rmse(e_arima), rmse(e_rw), len(e_arima)


def momentum(y, scale):
    """TSMOM read: sign of 12M change = trend; 3M change agreeing = trend intact."""
    ch3 = (y.iloc[-1] - y.iloc[-4]) * scale
    ch12 = (y.iloc[-1] - y.iloc[-13]) * scale
    sig = int(np.sign(ch12))
    label = {1: "UP-trend", -1: "DOWN-trend", 0: "flat"}[sig]
    agree = np.sign(ch3) == sig
    # z-score of the 12M change vs. its own history: is the move unusual?
    hist12 = (y.diff(12).dropna() * scale)
    z = (ch12 - hist12.mean()) / hist12.std()
    return ch3, ch12, label, agree, z


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv"); g.add_argument("--investing"); g.add_argument("--fred")
    ap.add_argument("--name", default="Rate")
    ap.add_argument("--unit", choices=BP, default="pct")
    ap.add_argument("--log", action="store_true", help="model log(series); use for volatility")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--out", default="output")
    a = ap.parse_args()

    raw = load_csv(a.csv) if a.csv else load_investing(a.investing) if a.investing else load_fred(a.fred)
    raw = raw[raw.index >= a.start]
    if a.log:
        raw = np.log(raw[raw > 0])
    y = to_monthly(raw)
    scale = 1.0 if a.log else BP[a.unit]
    unit = "log-pts" if a.log else ("bp" if a.unit != "raw" else "units")
    out = os.path.join(a.out, a.name.replace(" ", "_").replace("/", ""))
    os.makedirs(out, exist_ok=True)

    rep = [f"Series: {a.name}{' (log)' if a.log else ''}",
           f"Sample: {y.index[0]:%Y-%m} to {y.index[-1]:%Y-%m} ({len(y)} monthly obs)",
           f"Last daily obs: {raw.index[-1]:%Y-%m-%d} = {np.exp(raw.iloc[-1]) if a.log else raw.iloc[-1]:.4g}",
           f"Share of unchanged daily prints since 2025: {(raw['2025':].diff() == 0).mean():.0%}", ""]
    if len(y) < 48:
        rep.append("WARNING: < 48 monthly obs; ARIMA estimates and CIs are unreliable.\n")

    # 1. time plot
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(y.index, y.values, lw=1.6); ax.set_title(f"{a.name}: monthly average"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{out}/1_timeplot.png", dpi=160); plt.close(fig)

    # 2. stationarity
    rep.append("Stationarity (ADF H0 = unit root; KPSS H0 = stationary)")
    d = choose_d(y, rep)
    rep.append(f"-> differencing order d = {d}\n")

    # 3. ACF / PACF
    z = y.diff().dropna() if d else y
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    lags = min(24, len(z) // 2 - 1)
    plot_acf(z, lags=lags, ax=axes[0]); plot_pacf(z, lags=lags, ax=axes[1], method="ywm")
    axes[0].set_title(f"ACF (d={d})"); axes[1].set_title(f"PACF (d={d})")
    fig.tight_layout(); fig.savefig(f"{out}/2_acf_pacf.png", dpi=160); plt.close(fig)

    # 4. ARIMA by AIC
    (order, res), table = select_arima(y, d)
    rep += ["Top 5 ARIMA candidates by AIC:", table.head(5).to_string(index=False),
            f"-> selected ARIMA{order}"]
    lb_p = res.test_serial_correlation("ljungbox", lags=[12])[0][1][-1]
    rep.append(f"Residual Ljung-Box(12) p = {lb_p:.3f} "
               f"({'white noise' if lb_p > 0.05 else 'autocorrelation remains'})")
    r_a, r_rw, n = oos_vs_random_walk(y, order, d)
    rep.append(f"Out-of-sample 3M-ahead RMSE over last {n} origins: ARIMA {r_a * scale:.1f} {unit} "
               f"vs random walk {r_rw * scale:.1f} {unit} "
               f"-> {'ARIMA adds value' if r_a < 0.95 * r_rw else 'ARIMA does NOT beat no-change'}\n")

    # 5. forecast
    fc = res.get_forecast(a.horizon)
    mean, ci = fc.predicted_mean, fc.conf_int(alpha=0.05)
    tf = np.exp if a.log else (lambda v: v)
    rep.append(f"{a.horizon}-month forecast (95% CI):")
    for t in mean.index:
        rep.append(f"  {t:%Y-%m}: {tf(mean[t]):.4g}  [{tf(ci.loc[t].iloc[0]):.4g}, {tf(ci.loc[t].iloc[1]):.4g}]")
    move = (mean.iloc[-1] - y.iloc[-1]) * scale
    width = (ci.iloc[-1, 1] - ci.iloc[-1, 0]) * scale
    rep.append(f"Point move over horizon: {move:+.1f} {unit}; 95% CI width at horizon: {width:.1f} {unit}")
    if abs(move) < width / 4:
        rep.append("READ: move is small vs. uncertainty -> ARIMA alone does NOT support a directional call.")

    hist = y[y.index >= y.index[-1] - pd.DateOffset(years=4)]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(hist.index, tf(hist.values), lw=1.6, label="actual (monthly avg)")
    ax.plot(mean.index, tf(mean.values), "--", lw=1.6, label=f"ARIMA{order}")
    ax.fill_between(mean.index, tf(ci.iloc[:, 0]), tf(ci.iloc[:, 1]), alpha=.2, label="95% CI")
    ax.set_title(f"{a.name}: {a.horizon}-month ARIMA forecast"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{out}/3_forecast.png", dpi=160); plt.close(fig)

    # 6. momentum
    ch3, ch12, label, agree, zz = momentum(y, scale)
    rep.append(f"\nMomentum: 3M {ch3:+.1f} {unit} | 12M {ch12:+.1f} {unit} | 12M z-score {zz:+.2f}")
    rep.append(f"Signal: {label}; 3M {'confirms' if agree else 'diverges from'} 12M "
               f"({'trend intact' if agree else 'possible turning point'})")

    text = "\n".join(rep)
    open(f"{out}/report.txt", "w").write(text)
    print(text, "\n" + "=" * 70)


if __name__ == "__main__":
    main()
