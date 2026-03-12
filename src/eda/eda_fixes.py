"""
Correcciones de 4 gráficos del EDA.
Sobreescribe los PNGs anteriores en docs/figures/.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Configuración visual global ───────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DIR_FIGURES = os.path.join("docs", "figures")
DIR_PROCESSED = os.path.join("data", "processed")
DIR_INTERIM = os.path.join("data", "interim")

COLORES = {
    "SPY": "#1f77b4", "QQQ": "#2ca02c", "IWM": "#9467bd",
    "EFA": "#17becf", "EEM": "#ff7f0e",
    "AGG": "#7f7f7f", "LQD": "#8c564b", "TIP": "#bcbd22",
    "GLD": "#d4af37", "VNQ": "#d62728",
}

NOMBRES = {
    "SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq 100 (QQQ)", "IWM": "Small Cap (IWM)",
    "EFA": "Desarrollados ex-US (EFA)", "EEM": "Emergentes (EEM)",
    "AGG": "Renta Fija US (AGG)", "LQD": "Corp. Inv. Grade (LQD)", "TIP": "TIPS Inflación (TIP)",
    "GLD": "Oro (GLD)", "VNQ": "Inmobiliario (VNQ)",
}

ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "LQD", "TIP", "GLD", "VNQ"]

CRISIS = [
    ("2008-09-01", "2009-03-31", "GFC 2008"),
    ("2020-02-15", "2020-04-30", "COVID"),
    ("2022-01-01", "2022-10-31", "Inflación/Tipos"),
]

figuras = []


def guardar(fig, nombre):
    ruta = os.path.join(DIR_FIGURES, nombre)
    fig.savefig(ruta)
    plt.close(fig)
    figuras.append(ruta)
    print(f"  Guardada: {ruta}")


# ── CORRECCIÓN 1 — AAII vs SPY (sin sombreados, solo líneas) ──

def fix_aaii_vs_spy():
    """Doble eje: AAII spread real (sin ffill) vs SPY acumulado."""
    print("\nCORRECCIÓN 1 — sentimiento_aaii_vs_spy.png...")

    # Datos AAII originales (sin forward-fill, una obs por semana real)
    aaii = pd.read_csv(os.path.join("data", "raw", "sentiment", "aaii_sentiment_clean.csv"),
                        parse_dates=["date"], index_col="date")
    bull_bear = aaii["bullish"] - aaii["bearish"]
    bull_bear_ma12 = bull_bear.rolling(12).mean()  # MA de 12 observaciones reales

    # SPY retorno acumulado desde precios semanales limpios
    etfs = pd.read_csv(os.path.join(DIR_INTERIM, "etfs_weekly_clean.csv"),
                        parse_dates=["date"], index_col="date")
    spy_log_ret = np.log(etfs["SPY"] / etfs["SPY"].shift(1))
    spy_cum = spy_log_ret.cumsum() * 100

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Eje izquierdo: AAII spread como scatter + MA12 como línea
    color_aaii = "#e67e22"
    color_aaii_ma = "#a04000"
    ax1.scatter(bull_bear.index, bull_bear, s=8, color=color_aaii,
                alpha=0.4, label="AAII Bull-Bear Spread (semanal)", zorder=2)
    ax1.plot(bull_bear_ma12.index, bull_bear_ma12, linewidth=2, color=color_aaii_ma,
             label="AAII Bull-Bear (MA 12 obs.)", zorder=3)
    ax1.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.set_ylabel("Spread Bull-Bear AAII", color=color_aaii_ma)
    ax1.tick_params(axis="y", labelcolor=color_aaii_ma)

    # Eje derecho: SPY retorno acumulado
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    color_spy = COLORES["SPY"]
    ax2.plot(spy_cum.index, spy_cum, linewidth=2, color=color_spy,
             label="SPY retorno acumulado (%)", zorder=1)
    ax2.set_ylabel("Retorno acumulado SPY (%)", color=color_spy)
    ax2.tick_params(axis="y", labelcolor=color_spy)

    ax1.set_title("Sentimiento AAII (Bull-Bear Spread) vs Retorno acumulado SPY",
                  fontweight="bold")
    ax1.set_xlabel("Fecha")

    # Leyenda combinada arriba
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    guardar(fig, "sentimiento_aaii_vs_spy.png")


# ── CORRECCIÓN 2 — Antes/después nulos (sentimiento) ──────────

def fix_antes_despues_nulos():
    """Heatmap de nulos usando sentimiento (4003 nulos = 40%, más visual)."""
    print("\nCORRECCIÓN 2 — antes_despues_nulos.png...")

    # Antes: sentiment_weekly_aligned.csv (con muchos nulos)
    sent_antes = pd.read_csv(os.path.join(DIR_INTERIM, "sentiment_weekly_aligned.csv"),
                              parse_dates=["date"], index_col="date")
    # Después: sentiment_weekly_clean.csv (limpio)
    sent_despues = pd.read_csv(os.path.join(DIR_INTERIM, "sentiment_weekly_clean.csv"),
                                parse_dates=["date"], index_col="date")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Panel izquierdo: ANTES
    nulos_antes = sent_antes.isnull().astype(int)
    sns.heatmap(nulos_antes.T, cmap=["#f0f0f0", "#e74c3c"], cbar=False, ax=ax1,
                yticklabels=True, xticklabels=False)
    ax1.set_title("ANTES de la limpieza\n(sentiment_weekly_aligned.csv)", fontweight="bold")
    ax1.set_xlabel(f"Semanas (n={len(sent_antes)})")
    ax1.set_ylabel("")
    total_antes = sent_antes.isnull().sum().sum()
    ax1.text(0.5, -0.08, f"Total nulos: {total_antes:,}  ({total_antes/(sent_antes.size)*100:.0f}%)",
             transform=ax1.transAxes, ha="center", fontsize=11, color="#e74c3c",
             fontweight="bold")

    # Panel derecho: DESPUÉS
    nulos_despues = sent_despues.isnull().astype(int)
    sns.heatmap(nulos_despues.T, cmap=["#f0f0f0", "#e74c3c"], cbar=False, ax=ax2,
                yticklabels=True, xticklabels=False)
    ax2.set_title("DESPUÉS de la limpieza\n(sentiment_weekly_clean.csv)", fontweight="bold")
    ax2.set_xlabel(f"Semanas (n={len(sent_despues)})")
    ax2.set_ylabel("")
    total_despues = sent_despues.isnull().sum().sum()
    ax2.text(0.5, -0.08, f"Total nulos: {total_despues}",
             transform=ax2.transAxes, ha="center", fontsize=11, color="#27ae60",
             fontweight="bold")

    fig.suptitle("Tratamiento de valores nulos — Antes vs Después (Sentimiento)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    guardar(fig, "antes_despues_nulos.png")


# ── CORRECCIÓN 3 — Google Trends recession (3 sombreados) ─────

def fix_google_recession():
    """Google Trends 'recession' con solo 3 sombreados de crisis etiquetados."""
    print("\nCORRECCIÓN 3 — google_trends_recession.png...")

    sentiment = pd.read_csv(os.path.join(DIR_INTERIM, "sentiment_weekly_clean.csv"),
                             parse_dates=["date"], index_col="date")
    recession = sentiment["recession"]

    fig, ax = plt.subplots(figsize=(14, 6))

    # 3 sombreados de crisis con etiquetas
    for inicio, fin, nombre in CRISIS:
        ax.axvspan(pd.Timestamp(inicio), pd.Timestamp(fin),
                   alpha=0.15, color="gray", zorder=0)
        # Etiqueta en el centro del sombreado, parte superior
        centro = pd.Timestamp(inicio) + (pd.Timestamp(fin) - pd.Timestamp(inicio)) / 2
        ax.text(centro, recession.max() * 0.95, nombre, fontsize=9,
                ha="center", color="#555555", style="italic", fontweight="bold")

    # Línea de Google Trends
    ax.plot(recession.index, recession, linewidth=1.2, color="#8e44ad",
            label='Google Trends: "recession"')

    ax.set_title('Búsquedas de "recession" en Google vs periodos de estrés del mercado',
                 fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Índice Google Trends")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    guardar(fig, "google_trends_recession.png")


# ── CORRECCIÓN 4 — ETFs evolución log (etiquetas centradas) ───

def fix_evolucion_log():
    """Precios base 100, escala log, etiquetas de crisis centradas verticalmente."""
    print("\nCORRECCIÓN 4 — etfs_evolucion_log.png...")

    etfs = pd.read_csv(os.path.join(DIR_INTERIM, "etfs_weekly_clean.csv"),
                        parse_dates=["date"], index_col="date")
    base100 = (etfs / etfs.iloc[0]) * 100

    fig, ax = plt.subplots(figsize=(14, 7))

    # Sombreados de crisis
    for inicio, fin, nombre in CRISIS:
        ax.axvspan(pd.Timestamp(inicio), pd.Timestamp(fin),
                   alpha=0.12, color="gray", zorder=0)

    # Líneas de precios
    for ticker in ETFS:
        ax.plot(base100.index, base100[ticker], linewidth=1.3,
                color=COLORES[ticker], label=NOMBRES[ticker])

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks([25, 50, 100, 200, 400, 800])
    ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())

    # Etiquetas de crisis centradas verticalmente en el sombreado
    # Usar coordenadas del eje para que queden en el centro vertical
    for inicio, fin, nombre in CRISIS:
        centro_x = pd.Timestamp(inicio) + (pd.Timestamp(fin) - pd.Timestamp(inicio)) / 2
        # transform=ax.get_xaxis_transform() pone y en coordenadas de eje (0-1)
        ax.text(centro_x, 0.5, nombre, fontsize=9, color="#555555",
                ha="center", va="center", style="italic", fontweight="bold",
                transform=ax.get_xaxis_transform(), alpha=0.7)

    ax.set_title("Evolución de precios de ETFs (base 100, escala logarítmica, 2007–2026)",
                 fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (base 100, escala log)")
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")

    guardar(fig, "etfs_evolucion_log.png")

    # Eliminar la versión base100 lineal (ya no necesaria)
    ruta_vieja = os.path.join(DIR_FIGURES, "etfs_evolucion_base100.png")
    if os.path.exists(ruta_vieja):
        os.remove(ruta_vieja)
        print(f"  Eliminado: {ruta_vieja}")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    fix_aaii_vs_spy()
    fix_antes_despues_nulos()
    fix_google_recession()
    fix_evolucion_log()

    print(f"\n{'='*60}")
    print(f"4 correcciones aplicadas:")
    for ruta in figuras:
        size = os.path.getsize(ruta) / 1024
        print(f"  {ruta} ({size:.0f} KB)")
    print(f"{'='*60}")
