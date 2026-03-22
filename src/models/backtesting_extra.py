"""
Backtesting extra: IC, subperíodos, turnover, Monte Carlo, métricas avanzadas.
Cálculos adicionales para la memoria del TFG — NO modifica archivos existentes.

Autor: Mateo Madrigal Arteaga, UFV
Uso:   python3 src/models/backtesting_extra.py
"""

import matplotlib
matplotlib.use("Agg")

import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, percentileofscore

# ── Constantes ──────────────────────────────────────────────────────

ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "LQD", "TIP", "GLD", "VNQ"]
EXTRA_DIR = "data/results/extra"
FIG_DIR = "docs/figures"

# Paleta Okabe-Ito para estrategias
C_XGB = "#D55E00"
C_LGB = "#E69F00"
C_6040 = "#0072B2"
C_EW = "#999999"
CLR_TEXT = "#333333"
CLR_ANNOT = "#666666"

plt.rcParams.update({
    "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF",
    "savefig.facecolor": "#FFFFFF", "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.5, "text.color": CLR_TEXT,
})

os.makedirs(EXTRA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ── Utilidades ──────────────────────────────────────────────────────

def load_predictions(path):
    """Carga predicciones (formato largo) y devuelve dict {etf: Series y_pred}."""
    df = pd.read_csv(path, parse_dates=["date"])
    result = {}
    for etf in ETFS:
        sub = df[df["etf"] == etf].set_index("date")
        sub.index = pd.to_datetime(sub.index)
        result[etf] = sub
    return result


def load_weights(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)


def portfolio_returns(weights, actuals):
    """Calcula retornos de cartera dados pesos y retornos reales."""
    common = weights.index.intersection(actuals.index)
    return (weights.loc[common].values * actuals.loc[common].values).sum(axis=1), common


def sharpe(rets, ann=52):
    if len(rets) < 2 or np.std(rets) == 0:
        return 0.0
    return np.mean(rets) / np.std(rets) * np.sqrt(ann)


def max_drawdown(rets):
    equity = np.exp(np.cumsum(rets))
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return dd.min()


def ann_return(rets, ann=52):
    return np.mean(rets) * ann


# ══════════════════════════════════════════════════════════════════
# CARGAR DATOS
# ══════════════════════════════════════════════════════════════════

print("=" * 62)
print("  BACKTESTING EXTRA — 5 CÁLCULOS ADICIONALES")
print("=" * 62)

try:
    master = pd.read_csv("data/processed/master_weekly_raw.csv",
                          parse_dates=["date"], index_col="date")
    target_cols = [f"target_{e}" for e in ETFS]
    actuals = master[target_cols].copy()
    actuals.columns = ETFS

    xgb_preds = load_predictions("data/results/xgb_tuned_predictions.csv")
    lgb_preds = load_predictions("data/results/lgb_tuned_predictions.csv")
    xgb_w = load_weights("data/results/xgb_tuned_weights.csv")
    lgb_w = load_weights("data/results/lgb_tuned_weights.csv")

    # Reconstruir retornos de carteras
    xgb_ret_arr, common = portfolio_returns(xgb_w, actuals)
    xgb_ret = pd.Series(xgb_ret_arr, index=common)
    lgb_ret_arr, common_l = portfolio_returns(lgb_w, actuals)
    lgb_ret = pd.Series(lgb_ret_arr, index=common_l)

    # Benchmarks
    act_aligned = actuals.loc[common]
    ret_6040 = 0.6 * act_aligned["SPY"] + 0.4 * act_aligned["AGG"]
    ret_ew = act_aligned.mean(axis=1)

    # Sanity check
    xgb_sharpe_check = sharpe(xgb_ret.values)
    print(f"\n  Sanity check: XGB Tuned Sharpe = {xgb_sharpe_check:.3f}", end="")
    if abs(xgb_sharpe_check - 1.397) > 0.05:
        print(f"  WARNING: esperado ~1.397")
    else:
        print("  OK")

    print(f"  Período: {common[0].date()} -> {common[-1].date()} ({len(common)} semanas)")
except Exception as e:
    print(f"ERROR cargando datos: {e}")
    sys.exit(1)

results = {}  # Almacena resultados para resumen final


# ══════════════════════════════════════════════════════════════════
# CÁLCULO 1: INFORMATION COEFFICIENT
# ══════════════════════════════════════════════════════════════════

print(f"\n{'=' * 62}")
print("  CÁLCULO 1: INFORMATION COEFFICIENT (IC)")
print(f"{'=' * 62}")

try:
    # Construir matrices de predicciones y actuals alineadas
    pred_matrix = pd.DataFrame(index=xgb_preds["SPY"].index)
    actual_matrix = pd.DataFrame(index=xgb_preds["SPY"].index)
    for etf in ETFS:
        pred_matrix[etf] = xgb_preds[etf]["y_pred"]
        actual_matrix[etf] = xgb_preds[etf]["y_true"]

    pred_matrix.index = pd.to_datetime(pred_matrix.index)
    actual_matrix.index = pd.to_datetime(actual_matrix.index)

    # IC cross-section semanal (Spearman)
    ic_weekly = []
    for date in pred_matrix.index:
        p = pred_matrix.loc[date].values
        a = actual_matrix.loc[date].values
        if np.any(np.isnan(p)) or np.any(np.isnan(a)):
            continue
        corr, _ = spearmanr(p, a)
        if not np.isnan(corr):
            ic_weekly.append(corr)

    ic_weekly = np.array(ic_weekly)
    ic_mean = ic_weekly.mean()
    ic_std = ic_weekly.std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
    ic_hit = (ic_weekly > 0).mean() * 100

    # IC por ETF (Pearson time-series)
    ic_by_etf = {}
    for etf in ETFS:
        p = pred_matrix[etf].values
        a = actual_matrix[etf].values
        mask = ~(np.isnan(p) | np.isnan(a))
        if mask.sum() > 10:
            ic_by_etf[etf] = np.corrcoef(p[mask], a[mask])[0, 1]
        else:
            ic_by_etf[etf] = np.nan

    print(f"\n  IC medio (cross-section Spearman): {ic_mean:.4f}")
    print(f"  IC std:                           {ic_std:.4f}")
    print(f"  IC IR (IC/std):                   {ic_ir:.4f}")
    print(f"  IC hit rate (>0):                 {ic_hit:.1f}%")
    print(f"\n  IC por ETF (Pearson time-series):")
    for etf in ETFS:
        print(f"    {etf}: {ic_by_etf[etf]:.4f}")

    # Guardar
    ic_df = pd.DataFrame({
        "Metric": ["IC_mean", "IC_std", "IC_IR", "IC_hit_rate"] + [f"IC_{e}" for e in ETFS],
        "Value": [ic_mean, ic_std, ic_ir, ic_hit] + [ic_by_etf[e] for e in ETFS],
    })
    ic_df.to_csv(os.path.join(EXTRA_DIR, "information_coefficient.csv"), index=False)

    results["ic"] = {"mean": ic_mean, "std": ic_std, "ir": ic_ir,
                      "hit": ic_hit, "by_etf": ic_by_etf}

except Exception as e:
    print(f"  ERROR: {e}")


# ══════════════════════════════════════════════════════════════════
# CÁLCULO 2: ANÁLISIS POR SUBPERÍODO
# ══════════════════════════════════════════════════════════════════

print(f"\n{'=' * 62}")
print("  CÁLCULO 2: ANÁLISIS POR SUBPERÍODO")
print(f"{'=' * 62}")

try:
    periods = [
        ("Post-crisis", "2011-04-01", "2015-12-31"),
        ("Pre-COVID",   "2016-01-01", "2019-12-31"),
        ("COVID",       "2020-01-01", "2020-12-31"),
        ("Inflación",   "2021-01-01", "2022-12-31"),
        ("Recuperación","2023-01-01", "2026-02-28"),
    ]

    strategies = {
        "XGB Tuned": xgb_ret,
        "LGB Tuned": lgb_ret,
        "60/40": ret_6040,
        "Equal Wt": ret_ew,
    }

    rows = []
    print(f"\n  {'Período':<14s} {'Estrategia':<12s} {'Sharpe':>7s} {'Ret.Ann':>8s} "
          f"{'MaxDD':>8s} {'Semanas':>8s} {'SE(Sh)':>7s}")
    print(f"  {'-'*66}")

    for pname, pstart, pend in periods:
        for sname, sret in strategies.items():
            mask = (sret.index >= pstart) & (sret.index <= pend)
            sub = sret[mask].values
            n = len(sub)
            if n < 4:
                continue
            sh = sharpe(sub)
            ar = ann_return(sub)
            md = max_drawdown(sub)
            se = 1 / np.sqrt(n)  # SE del Sharpe
            rows.append({
                "Period": pname, "Strategy": sname, "Sharpe": sh,
                "Ann_Return": ar, "MaxDD": md, "Weeks": n, "SE_Sharpe": se,
            })
            print(f"  {pname:<14s} {sname:<12s} {sh:>7.3f} {ar:>7.2%} "
                  f"{md:>7.2%} {n:>8d} {se:>7.3f}")

    sub_df = pd.DataFrame(rows)
    sub_df.to_csv(os.path.join(EXTRA_DIR, "subperiod_analysis.csv"), index=False)

    # Figura: barras agrupadas por período
    fig, ax = plt.subplots(figsize=(12, 6))
    period_names = [p[0] for p in periods]
    strat_names = list(strategies.keys())
    strat_colors = [C_XGB, C_LGB, C_6040, C_EW]
    x = np.arange(len(period_names))
    width = 0.2

    for k, (sname, color) in enumerate(zip(strat_names, strat_colors)):
        vals = []
        for pname in period_names:
            row = sub_df[(sub_df["Period"] == pname) & (sub_df["Strategy"] == sname)]
            vals.append(row["Sharpe"].values[0] if len(row) > 0 else 0)
        ax.bar(x + k * width, vals, width, label=sname, color=color)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(period_names, fontsize=10)
    ax.set_ylabel("Sharpe Ratio", fontsize=11)
    ax.set_title("Sharpe Ratio por subperíodo y estrategia", fontsize=14, weight="bold")
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="none")
    ax.axhline(y=0, color="#cccccc", linewidth=0.5)
    ax.yaxis.grid(True, alpha=0.3, color="#cccccc")
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "subperiod_sharpe.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figura: {FIG_DIR}/subperiod_sharpe.png")

    results["subperiod"] = sub_df

except Exception as e:
    print(f"  ERROR: {e}")


# ══════════════════════════════════════════════════════════════════
# CÁLCULO 3: TURNOVER DE PESOS
# ══════════════════════════════════════════════════════════════════

print(f"\n{'=' * 62}")
print("  CÁLCULO 3: TURNOVER DE PESOS")
print(f"{'=' * 62}")

try:
    turnover_results = {}
    for name, w, ret in [("XGB Tuned", xgb_w, xgb_ret), ("LGB Tuned", lgb_w, lgb_ret)]:
        # Turnover semanal = sum(|w_t - w_{t-1}|) / 2
        diff = w.diff().abs()
        turnover = diff.sum(axis=1) / 2
        turnover = turnover.iloc[1:]  # Descartar primera fila NaN

        to_mean = turnover.mean()
        to_median = turnover.median()
        to_p95 = turnover.quantile(0.95)
        to_max = turnover.max()
        to_annual = to_mean * 52

        # Coste estimado
        cost_5bps = to_mean * 0.0005 * 52  # anual
        cost_10bps = to_mean * 0.0010 * 52

        # Sharpe neto
        ann_ret_val = ann_return(ret.values)
        ann_vol = np.std(ret.values) * np.sqrt(52)
        sharpe_5 = (ann_ret_val - cost_5bps) / ann_vol if ann_vol > 0 else 0
        sharpe_10 = (ann_ret_val - cost_10bps) / ann_vol if ann_vol > 0 else 0

        turnover_results[name] = {
            "mean": to_mean, "median": to_median, "p95": to_p95, "max": to_max,
            "annual": to_annual, "cost_5bps": cost_5bps, "cost_10bps": cost_10bps,
            "sharpe_net_5": sharpe_5, "sharpe_net_10": sharpe_10,
        }

        print(f"\n  {name}:")
        print(f"    Turnover semanal: media={to_mean:.4f} ({to_mean*100:.2f}%), "
              f"mediana={to_median:.4f}, P95={to_p95:.4f}, max={to_max:.4f}")
        print(f"    Turnover anualizado: {to_annual:.2f} ({to_annual*100:.0f}%)")
        print(f"    Coste anual (5 bps):  {cost_5bps:.4f} ({cost_5bps*100:.2f}%)")
        print(f"    Coste anual (10 bps): {cost_10bps:.4f} ({cost_10bps*100:.2f}%)")
        print(f"    Sharpe neto (5 bps):  {sharpe_5:.3f}")
        print(f"    Sharpe neto (10 bps): {sharpe_10:.3f}")

    # Guardar
    to_rows = []
    for name, d in turnover_results.items():
        for k, v in d.items():
            to_rows.append({"Strategy": name, "Metric": k, "Value": v})
    pd.DataFrame(to_rows).to_csv(os.path.join(EXTRA_DIR, "turnover_analysis.csv"), index=False)

    # Figura: serie temporal de turnover XGB
    xgb_turnover = xgb_w.diff().abs().sum(axis=1).iloc[1:] / 2
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(xgb_turnover.index, xgb_turnover.values * 100,
            color=C_XGB, linewidth=0.7, alpha=0.8)
    ax.axhline(y=xgb_turnover.mean() * 100, color=C_XGB, linestyle="--",
               linewidth=1, label=f"Media: {xgb_turnover.mean()*100:.1f}%")
    ax.set_ylabel("Turnover semanal (%)", fontsize=11)
    ax.set_title("Turnover semanal de la cartera XGBoost Tuned", fontsize=14, weight="bold")
    ax.legend(fontsize=9, edgecolor="none")
    ax.yaxis.grid(True, alpha=0.3, color="#cccccc")
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "turnover_timeseries.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figura: {FIG_DIR}/turnover_timeseries.png")

    results["turnover"] = turnover_results

except Exception as e:
    print(f"  ERROR: {e}")


# ══════════════════════════════════════════════════════════════════
# CÁLCULO 4: MONTE CARLO DE PESOS ALEATORIOS
# ══════════════════════════════════════════════════════════════════

print(f"\n{'=' * 62}")
print("  CÁLCULO 4: MONTE CARLO (10,000 carteras aleatorias)")
print(f"{'=' * 62}")

try:
    np.random.seed(42)
    n_sim = 10000
    returns_matrix = act_aligned.values  # (778, 10)

    mc_sharpes = []
    for _ in range(n_sim):
        w = np.random.dirichlet(np.ones(10))
        # Cartera estática: mismos pesos todas las semanas
        port_ret = returns_matrix @ w
        mc_sharpes.append(sharpe(port_ret))

    mc_sharpes = np.array(mc_sharpes)
    mc_mean = mc_sharpes.mean()
    mc_std = mc_sharpes.std()
    mc_p5 = np.percentile(mc_sharpes, 5)
    mc_p95 = np.percentile(mc_sharpes, 95)

    xgb_percentile = percentileofscore(mc_sharpes, 1.397)
    p_value = 1 - xgb_percentile / 100

    print(f"\n  Sharpe aleatorio: {mc_mean:.3f} +/- {mc_std:.3f}")
    print(f"  Rango P5-P95: [{mc_p5:.3f}, {mc_p95:.3f}]")
    print(f"  XGB Tuned (1.397) -> percentil {xgb_percentile:.1f}%")
    print(f"  p-value: {p_value:.4f}")

    # Guardar
    pd.DataFrame({"sharpe": mc_sharpes}).to_csv(
        os.path.join(EXTRA_DIR, "montecarlo_results.csv"), index=False)

    # Figura: histograma
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(mc_sharpes, bins=80, color="#0072B2", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(x=1.397, color=C_XGB, linewidth=2.5, linestyle="-",
               label=f"XGB Tuned: 1.397 (percentil {xgb_percentile:.1f}%)")
    ax.axvline(x=0.847, color=C_6040, linewidth=1.5, linestyle="--",
               label="60/40: 0.847")

    ax.set_xlabel("Sharpe Ratio", fontsize=11)
    ax.set_ylabel("Frecuencia", fontsize=11)
    ax.set_title("Distribución de Sharpe de 10,000 carteras aleatorias vs XGBoost Tuned",
                 fontsize=13, weight="bold")
    ax.legend(fontsize=10, edgecolor="none", framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3, color="#cccccc")
    ax.set_axisbelow(True)

    # Anotación
    ax.annotate(f"p-value = {p_value:.4f}",
                xy=(1.397, ax.get_ylim()[1] * 0.85),
                fontsize=11, fontweight="bold", color=C_XGB, ha="center")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "montecarlo_histogram.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura: {FIG_DIR}/montecarlo_histogram.png")

    results["mc"] = {"mean": mc_mean, "std": mc_std, "percentile": xgb_percentile,
                     "p_value": p_value}

except Exception as e:
    print(f"  ERROR: {e}")


# ══════════════════════════════════════════════════════════════════
# CÁLCULO 5: BURKE RATIO + CONDITIONAL SHARPE
# ══════════════════════════════════════════════════════════════════

print(f"\n{'=' * 62}")
print("  CÁLCULO 5: MÉTRICAS DE RIESGO AVANZADAS")
print(f"{'=' * 62}")

try:
    adv_rows = []
    strategies_adv = {
        "XGB Tuned": xgb_ret, "LGB Tuned": lgb_ret,
        "60/40": ret_6040, "Equal Wt": ret_ew,
    }

    print(f"\n  {'Estrategia':<12s} {'Burke':>8s} {'CondSh':>8s} {'VaR95':>8s} {'CVaR95':>8s}")
    print(f"  {'-'*48}")

    for sname, sret in strategies_adv.items():
        r = sret.values

        # Burke ratio
        equity = np.exp(np.cumsum(r))
        peak = np.maximum.accumulate(equity)
        dd_series = (equity - peak) / peak
        dd_neg = dd_series[dd_series < 0]
        burke = ann_return(r) / np.sqrt(np.mean(dd_neg ** 2)) if len(dd_neg) > 0 else 0

        # Conditional Sharpe (peor 25%)
        sorted_r = np.sort(r)
        q25 = sorted_r[: len(sorted_r) // 4]
        cond_sharpe = np.mean(q25) / np.std(q25) * np.sqrt(52) if np.std(q25) > 0 else 0

        # VaR / CVaR
        var_95 = np.percentile(r, 5)
        cvar_95 = r[r <= var_95].mean() if (r <= var_95).sum() > 0 else var_95

        adv_rows.append({
            "Strategy": sname, "Burke": burke, "Cond_Sharpe_25": cond_sharpe,
            "VaR_95": var_95, "CVaR_95": cvar_95,
        })

        print(f"  {sname:<12s} {burke:>8.3f} {cond_sharpe:>8.3f} "
              f"{var_95:>7.4f} {cvar_95:>7.4f}")

    adv_df = pd.DataFrame(adv_rows)
    adv_df.to_csv(os.path.join(EXTRA_DIR, "advanced_risk_metrics.csv"), index=False)

    results["adv"] = adv_df

except Exception as e:
    print(f"  ERROR: {e}")


# ══════════════════════════════════════════════════════════════════
# RESUMEN EJECUTIVO
# ══════════════════════════════════════════════════════════════════

print(f"\n\n{'=' * 62}")
print(f"  RESUMEN EJECUTIVO — BACKTESTING EXTRA")
print(f"{'=' * 62}")

if "ic" in results:
    ic = results["ic"]
    print(f"\n  1. INFORMATION COEFFICIENT")
    print(f"     IC medio: {ic['mean']:.4f} | IC IR: {ic['ir']:.2f} | Hit rate: {ic['hit']:.1f}%")
    etf_str = " | ".join(f"{e}:{ic['by_etf'][e]:.3f}" for e in ETFS[:5])
    etf_str2 = " | ".join(f"{e}:{ic['by_etf'][e]:.3f}" for e in ETFS[5:])
    print(f"     {etf_str}")
    print(f"     {etf_str2}")

if "subperiod" in results:
    print(f"\n  2. ANÁLISIS POR SUBPERÍODO")
    df = results["subperiod"]
    for pname in df["Period"].unique():
        sub = df[df["Period"] == pname]
        xgb_sh = sub[sub["Strategy"] == "XGB Tuned"]["Sharpe"].values
        b60_sh = sub[sub["Strategy"] == "60/40"]["Sharpe"].values
        if len(xgb_sh) > 0 and len(b60_sh) > 0:
            diff = xgb_sh[0] - b60_sh[0]
            marker = "+" if diff > 0 else ""
            print(f"     {pname:<14s}: XGB={xgb_sh[0]:.3f} vs 60/40={b60_sh[0]:.3f} ({marker}{diff:.3f})")

if "turnover" in results:
    to = results["turnover"]
    print(f"\n  3. TURNOVER")
    for name, d in to.items():
        print(f"     {name}: media={d['mean']*100:.1f}% sem | "
              f"coste 5bps={d['cost_5bps']*100:.2f}% | Sharpe neto={d['sharpe_net_5']:.3f}")

if "mc" in results:
    mc = results["mc"]
    print(f"\n  4. MONTE CARLO (10,000 carteras)")
    print(f"     Sharpe aleatorio: {mc['mean']:.3f} +/- {mc['std']:.3f}")
    print(f"     XGB percentil: {mc['percentile']:.1f}% | p-value: {mc['p_value']:.4f}")

if "adv" in results:
    print(f"\n  5. MÉTRICAS AVANZADAS")
    df = results["adv"]
    for _, row in df.iterrows():
        print(f"     {row['Strategy']:<12s}: Burke={row['Burke']:.3f} | "
              f"CondSh={row['Cond_Sharpe_25']:.3f} | "
              f"VaR95={row['VaR_95']:.4f} | CVaR95={row['CVaR_95']:.4f}")

print(f"\n{'=' * 62}")
print(f"  Figuras: {FIG_DIR}/subperiod_sharpe.png, turnover_timeseries.png, montecarlo_histogram.png")
print(f"  Datos: {EXTRA_DIR}/")
print(f"{'=' * 62}")
