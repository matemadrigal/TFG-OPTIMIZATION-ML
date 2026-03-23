"""
Regenera las 3 figuras de backtesting extra con estilo académico Tufte + Okabe-Ito.
Lee datos ya calculados, NO recalcula nada.

Autor: Mateo Madrigal Arteaga, UFV
Uso:   python3 src/models/fix_figures.py
"""

import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import percentileofscore

# ── Estilo global ─────────────────────────────────────────────────

C_XGB = "#D55E00"
C_LGB = "#E69F00"
C_6040 = "#0072B2"
C_EW = "#999999"
CLR_TEXT = "#333333"
CLR_ANNOT = "#666666"
CLR_GRID = "#CCCCCC"
CLR_SPINE = "#999999"

plt.rcParams.update({
    "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF",
    "savefig.facecolor": "#FFFFFF",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": True, "axes.spines.bottom": True,
    "axes.linewidth": 0.5,
    "axes.edgecolor": CLR_SPINE,
    "axes.labelcolor": CLR_TEXT,
    "axes.titlecolor": CLR_TEXT,
    "xtick.color": CLR_TEXT, "ytick.color": CLR_TEXT,
    "text.color": CLR_TEXT,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
})

FIG_DIR = "docs/figures"
EXTRA_DIR = "data/results/extra"


# ══════════════════════════════════════════════════════════════════
# FIGURA 1: MONTE CARLO HISTOGRAM
# ══════════════════════════════════════════════════════════════════

def fig_montecarlo():
    mc = pd.read_csv(os.path.join(EXTRA_DIR, "montecarlo_results.csv"))
    sharpes = mc["sharpe"].values

    xgb_sharpe = 1.397
    b60_sharpe = 0.847
    pctl = percentileofscore(sharpes, xgb_sharpe)
    pval = 1 - pctl / 100

    fig, ax = plt.subplots(figsize=(11, 6))

    # Histograma en gris neutro
    ax.hist(sharpes, bins=80, color="#BBBBBB", edgecolor="#999999",
            linewidth=0.3, zorder=2)

    # Líneas verticales
    ax.axvline(x=xgb_sharpe, color=C_XGB, linewidth=2.5, linestyle="-",
               label=f"XGBoost Tuned (Sharpe {xgb_sharpe})", zorder=4)
    ax.axvline(x=b60_sharpe, color=C_6040, linewidth=1.5, linestyle="--",
               label=f"60/40 (Sharpe {b60_sharpe})", zorder=3)

    # Formato
    ax.set_xlabel("Sharpe Ratio", fontsize=11)
    ax.set_ylabel("Frecuencia", fontsize=11)
    ax.set_title("Distribución de Sharpe de 10.000 carteras aleatorias vs XGBoost Tuned",
                 fontsize=14, weight="bold", pad=12)

    # Grid horizontal suave
    ax.yaxis.grid(True, alpha=0.2, color=CLR_GRID)
    ax.set_axisbelow(True)

    # Leyenda compacta
    ax.legend(fontsize=10, loc="upper right", framealpha=0.8,
              edgecolor=CLR_GRID, facecolor="white")

    # Anotación p-value en recuadro
    ax.annotate(
        f"p-value = {pval:.4f}\nPercentil {pctl:.1f}%",
        xy=(0.97, 0.72), xycoords="axes fraction",
        ha="right", va="top", fontsize=12, fontweight="bold", color=C_XGB,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=C_XGB, alpha=0.85),
    )

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "montecarlo_histogram.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════
# FIGURA 2: TURNOVER TIMESERIES
# ══════════════════════════════════════════════════════════════════

def fig_turnover():
    # Recalcular turnover desde pesos (más fiable que CSV)
    xgb_w = pd.read_csv("data/results/xgb_tuned_weights.csv",
                          index_col=0, parse_dates=True)
    turnover = xgb_w.diff().abs().sum(axis=1).iloc[1:] / 2
    turnover_pct = turnover * 100

    # Media móvil 12 semanas
    ma12 = turnover_pct.rolling(12, min_periods=1).mean()
    mean_val = turnover_pct.mean()

    fig, ax = plt.subplots(figsize=(13, 5.5))

    # Sombreados de crisis
    crisis = [
        ("2020-02-21", "2020-06-30", "COVID-19"),
        ("2022-01-01", "2022-12-31", "Subidas tipos"),
    ]
    for cs, ce, label in crisis:
        cs_dt, ce_dt = pd.Timestamp(cs), pd.Timestamp(ce)
        ax.axvspan(cs_dt, ce_dt, alpha=0.15, color="#CCCCCC", zorder=0)
        mid = cs_dt + (ce_dt - cs_dt) / 2
        ax.text(mid, 92, label, ha="center", fontsize=8, color=CLR_ANNOT,
                style="italic", zorder=1)

    # Línea raw (tenue)
    ax.plot(turnover_pct.index, turnover_pct.values,
            color=C_LGB, alpha=0.25, linewidth=0.5, label="Turnover semanal", zorder=2)

    # Media móvil 12 sem (protagonista)
    ax.plot(ma12.index, ma12.values,
            color=C_XGB, linewidth=2, alpha=1.0, label="Media móvil 12 sem.", zorder=3)

    # Media global
    ax.axhline(y=mean_val, color=CLR_ANNOT, linestyle="--", linewidth=1,
               alpha=0.7, label=f"Media: {mean_val:.1f}%", zorder=2)

    # Formato
    ax.set_ylabel("Turnover semanal (%)", fontsize=11)
    ax.set_title("Turnover semanal de la cartera XGBoost Tuned (media móvil 12 semanas)",
                 fontsize=14, weight="bold", pad=12)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, alpha=0.2, color=CLR_GRID)
    ax.set_axisbelow(True)

    # Ticks cada 2 años
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Leyenda
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85,
              edgecolor=CLR_GRID, facecolor="white")

    # Anotación de coste
    ax.annotate("Coste estimado: 0.97% anual (5 bps)",
                xy=(0.98, 0.05), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=9, color=CLR_ANNOT)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "turnover_timeseries.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════
# FIGURA 3: SUBPERIOD SHARPE
# ══════════════════════════════════════════════════════════════════

def fig_subperiod():
    sub_df = pd.read_csv(os.path.join(EXTRA_DIR, "subperiod_analysis.csv"))

    periods = sub_df["Period"].unique()
    strats = ["XGB Tuned", "LGB Tuned", "60/40", "Equal Wt"]
    colors = [C_XGB, C_LGB, C_6040, C_EW]

    # Etiquetas con fechas
    period_labels = {
        "Post-crisis": "Post-crisis\n2011-2015",
        "Pre-COVID": "Pre-COVID\n2016-2019",
        "COVID": "COVID\n2020",
        "Inflación": "Inflación\n2021-2022",
        "Recuperación": "Recuperación\n2023-2026",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(periods))
    width = 0.18

    for k, (sname, color) in enumerate(zip(strats, colors)):
        vals = []
        for pname in periods:
            row = sub_df[(sub_df["Period"] == pname) & (sub_df["Strategy"] == sname)]
            vals.append(row["Sharpe"].values[0] if len(row) > 0 else 0)
        bars = ax.bar(x + k * width, vals, width, label=sname, color=color,
                      edgecolor="white", linewidth=0.5, zorder=3)

        # Valores numéricos encima de las barras de XGB
        if sname == "XGB Tuned":
            for xi, v in zip(x + k * width, vals):
                offset = 0.05 if v >= 0 else -0.15
                ax.text(xi, v + offset, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color=C_XGB)

    # Línea en Sharpe=0
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.5, zorder=2)

    # Formato
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([period_labels.get(p, p) for p in periods], fontsize=10)
    ax.set_ylabel("Sharpe Ratio", fontsize=11)
    ax.set_title("Sharpe Ratio por subperíodo y estrategia (walk-forward OOS)",
                 fontsize=14, weight="bold", pad=12)

    # Grid horizontal suave
    ax.yaxis.grid(True, alpha=0.2, color=CLR_GRID)
    ax.set_axisbelow(True)

    # Leyenda abajo del gráfico
    ax.legend(fontsize=9, loc="lower center", ncol=4,
              bbox_to_anchor=(0.5, -0.18), framealpha=0.85,
              edgecolor=CLR_GRID, facecolor="white")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "subperiod_sharpe.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from PIL import Image

    paths = []
    for name, func in [("montecarlo_histogram", fig_montecarlo),
                        ("turnover_timeseries", fig_turnover),
                        ("subperiod_sharpe", fig_subperiod)]:
        path = func()
        img = Image.open(path)
        w, h = img.size
        size_kb = os.path.getsize(path) / 1024
        print(f"  {path:<45s} {w}x{h} px, 300 DPI, {size_kb:.0f} KB")
        paths.append(path)

    print(f"\nFiguras regeneradas correctamente.")
