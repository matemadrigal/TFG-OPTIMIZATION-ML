"""
EDA completo — Visualizaciones profesionales para la memoria del TFG.
Genera 10 figuras con estilo consistente en docs/figures/ a 300 DPI.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

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

# Paleta de colores fija por ETF
COLORES = {
    "SPY": "#1f77b4", "QQQ": "#2ca02c", "IWM": "#9467bd",
    "EFA": "#17becf", "EEM": "#ff7f0e",
    "AGG": "#7f7f7f", "LQD": "#8c564b", "TIP": "#bcbd22",
    "GLD": "#d4af37",
    "VNQ": "#d62728",
}

# Nombres descriptivos
NOMBRES = {
    "SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq 100 (QQQ)", "IWM": "Small Cap (IWM)",
    "EFA": "Desarrollados ex-US (EFA)", "EEM": "Emergentes (EEM)",
    "AGG": "Renta Fija US (AGG)", "LQD": "Corp. Inv. Grade (LQD)", "TIP": "TIPS Inflación (TIP)",
    "GLD": "Oro (GLD)",
    "VNQ": "Inmobiliario (VNQ)",
}

ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "LQD", "TIP", "GLD", "VNQ"]

# Periodos de crisis
CRISIS = [
    ("2008-09-01", "2009-03-31", "GFC 2008"),
    ("2020-02-15", "2020-04-30", "COVID"),
    ("2022-01-01", "2022-10-31", "Inflación/Tipos"),
]

figuras = []


def guardar(fig, nombre):
    """Guarda figura y la registra."""
    ruta = os.path.join(DIR_FIGURES, nombre)
    fig.savefig(ruta)
    plt.close(fig)
    figuras.append(ruta)
    print(f"  Guardada: {ruta}")


def sombrear_crisis(ax):
    """Añade sombreados de crisis a un eje."""
    for inicio, fin, nombre in CRISIS:
        ax.axvspan(pd.Timestamp(inicio), pd.Timestamp(fin),
                   alpha=0.12, color="gray", zorder=0)


# ── 1. Evolución de precios base 100 (escala log) ─────────────

def fig01_evolucion_log():
    """Precios normalizados base 100, escala logarítmica."""
    print("\n1. Evolución de precios base 100 (escala log)...")

    etfs = pd.read_csv(os.path.join(DIR_INTERIM, "etfs_weekly_clean.csv"),
                        parse_dates=["date"], index_col="date")
    base100 = (etfs / etfs.iloc[0]) * 100

    fig, ax = plt.subplots(figsize=(14, 7))
    sombrear_crisis(ax)

    for ticker in ETFS:
        ax.plot(base100.index, base100[ticker], linewidth=1.3,
                color=COLORES[ticker], label=NOMBRES[ticker])

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks([25, 50, 100, 200, 400, 800])
    ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())

    # Anotar crisis
    ax.text(pd.Timestamp("2008-11-01"), 22, "GFC", fontsize=8, color="gray",
            ha="center", style="italic")
    ax.text(pd.Timestamp("2020-03-15"), 22, "COVID", fontsize=8, color="gray",
            ha="center", style="italic")
    ax.text(pd.Timestamp("2022-06-01"), 22, "Inflación", fontsize=8, color="gray",
            ha="center", style="italic")

    ax.set_title("Evolución de precios de ETFs (base 100, escala logarítmica, 2007–2026)",
                 fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (base 100, escala log)")
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")

    guardar(fig, "etfs_evolucion_log.png")


# ── 2. Violin plots de retornos ───────────────────────────────

def fig02_violin_retornos():
    """Violin plots de retornos semanales por ETF."""
    print("\n2. Violin plots de retornos...")

    master = pd.read_csv(os.path.join(DIR_PROCESSED, "master_weekly_raw.csv"),
                          parse_dates=["date"], index_col="date")

    # Preparar datos en formato long
    datos = []
    for ticker in ETFS:
        col = f"{ticker}_log_ret"
        serie = master[col].dropna()
        for v in serie:
            datos.append({"ETF": ticker, "Log-return": v})
    df_long = pd.DataFrame(datos)

    fig, ax = plt.subplots(figsize=(14, 7))

    parts = ax.violinplot(
        [master[f"{t}_log_ret"].dropna().values for t in ETFS],
        positions=range(len(ETFS)),
        showmeans=True, showmedians=True, showextrema=False
    )

    # Colorear cada violín
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLORES[ETFS[i]])
        body.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("red")

    ax.set_xticks(range(len(ETFS)))
    ax.set_xticklabels(ETFS)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_title("Distribución de retornos semanales por ETF (violin plot)",
                 fontweight="bold")
    ax.set_xlabel("ETF")
    ax.set_ylabel("Log-return semanal")
    ax.grid(True, axis="y", alpha=0.3)

    # Leyenda manual
    from matplotlib.lines import Line2D
    leyenda = [
        Line2D([0], [0], color="black", linewidth=1.5, label="Media"),
        Line2D([0], [0], color="red", linewidth=1.5, label="Mediana"),
    ]
    ax.legend(handles=leyenda, loc="upper right")

    guardar(fig, "etfs_retornos_violin.png")


# ── 3. VIX histórico ──────────────────────────────────────────

def fig03_vix():
    """Evolución del VIX con niveles de referencia y picos anotados."""
    print("\n3. VIX histórico...")

    master = pd.read_csv(os.path.join(DIR_PROCESSED, "master_weekly_raw.csv"),
                          parse_dates=["date"], index_col="date")
    vix = master["vix_level"]

    fig, ax = plt.subplots(figsize=(14, 6))
    sombrear_crisis(ax)

    ax.plot(vix.index, vix, linewidth=1, color="#e74c3c", alpha=0.9)
    ax.fill_between(vix.index, 0, vix, alpha=0.15, color="#e74c3c")

    # Niveles de referencia
    ax.axhline(y=20, color="orange", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(vix.index[5], 21.5, 'VIX = 20 ("normal")', fontsize=8, color="orange")
    ax.axhline(y=30, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(vix.index[5], 31.5, 'VIX = 30 ("pánico")', fontsize=8, color="red")

    # Anotar picos
    # GFC: buscar máximo en 2008-2009
    mask_gfc = (vix.index >= "2008-09-01") & (vix.index <= "2009-03-31")
    if mask_gfc.any():
        idx_gfc = vix[mask_gfc].idxmax()
        val_gfc = vix[mask_gfc].max()
        ax.annotate(f"GFC: {val_gfc:.1f}", xy=(idx_gfc, val_gfc),
                    xytext=(idx_gfc + pd.Timedelta(days=120), val_gfc + 5),
                    fontsize=9, fontweight="bold", color="#c0392b",
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))

    # COVID: buscar máximo en feb-abr 2020
    mask_covid = (vix.index >= "2020-02-01") & (vix.index <= "2020-05-01")
    if mask_covid.any():
        idx_covid = vix[mask_covid].idxmax()
        val_covid = vix[mask_covid].max()
        ax.annotate(f"COVID: {val_covid:.1f}", xy=(idx_covid, val_covid),
                    xytext=(idx_covid + pd.Timedelta(days=120), val_covid + 5),
                    fontsize=9, fontweight="bold", color="#c0392b",
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))

    ax.set_title("Índice de Volatilidad VIX (2007–2026)", fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("VIX")
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)

    guardar(fig, "vix_historico.png")


# ── 4. Spread 10Y-2Y ──────────────────────────────────────────

def fig04_spread():
    """Spread Treasury 10Y-2Y con zonas de inversión."""
    print("\n4. Spread 10Y-2Y...")

    master = pd.read_csv(os.path.join(DIR_PROCESSED, "master_weekly_raw.csv"),
                          parse_dates=["date"], index_col="date")
    spread = master["spread_10y_2y"]

    fig, ax = plt.subplots(figsize=(14, 6))
    sombrear_crisis(ax)

    ax.plot(spread.index, spread, linewidth=1, color="#2c3e50")

    # Zona de inversión (spread < 0) sombreada en rojo
    ax.fill_between(spread.index, spread, 0,
                    where=(spread < 0), alpha=0.3, color="red",
                    label="Curva invertida (señal de recesión)")
    ax.fill_between(spread.index, spread, 0,
                    where=(spread >= 0), alpha=0.1, color="green")

    ax.axhline(y=0, color="black", linewidth=1)

    # Anotar periodos de inversión
    # Detectar inicio de periodos invertidos
    invertido = spread < 0
    cambios = invertido.astype(int).diff()
    inicios_inv = spread.index[cambios == 1]
    for inicio in inicios_inv:
        ax.annotate("Inversión", xy=(inicio, 0), xytext=(inicio, -0.8),
                    fontsize=7, color="red", alpha=0.7,
                    arrowprops=dict(arrowstyle="->", color="red", alpha=0.5, lw=0.8))

    ax.set_title("Spread Treasury 10 años – 2 años (indicador de recesión)",
                 fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Spread (pp)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    guardar(fig, "spread_10y2y.png")


# ── 5. Balance de la Fed ───────────────────────────────────────

def fig05_fed_balance():
    """Evolución del balance de la Fed con anotaciones de QE/QT."""
    print("\n5. Balance de la Fed (WALCL)...")

    liquidity = pd.read_csv(os.path.join(DIR_INTERIM, "liquidity_weekly_clean.csv"),
                             parse_dates=["date"], index_col="date")
    walcl = liquidity["WALCL"] / 1e6  # Convertir a billones (trillions US)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.fill_between(walcl.index, 0, walcl, alpha=0.3, color="#2980b9")
    ax.plot(walcl.index, walcl, linewidth=1.5, color="#2980b9")

    # Anotaciones de QE/QT
    anotaciones = [
        ("2009-03-01", "QE1\n(2008–2010)", 0.4),
        ("2010-11-01", "QE2\n(2010–2011)", 0.6),
        ("2012-09-01", "QE3\n(2012–2014)", 0.5),
        ("2020-04-01", "COVID QE\n(2020)", 0.85),
        ("2022-06-01", "QT\n(2022–)", 0.75),
    ]
    for fecha, texto, frac_y in anotaciones:
        ts = pd.Timestamp(fecha)
        if ts in walcl.index or ts <= walcl.index.max():
            y_pos = walcl.max() * frac_y
            ax.annotate(texto, xy=(ts, walcl.asof(ts) if ts <= walcl.index.max() else 0),
                        xytext=(ts, y_pos),
                        fontsize=9, fontweight="bold", ha="center", color="#2c3e50",
                        arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1))

    ax.set_title("Balance de la Reserva Federal — WALCL (2007–2026)",
                 fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Billones de USD (trillions)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    guardar(fig, "fed_balance.png")


# ── 6. AAII vs SPY ────────────────────────────────────────────

def fig06_aaii_vs_spy():
    """Doble eje: Spread Bull-Bear AAII vs retorno acumulado SPY."""
    print("\n6. Sentimiento AAII vs SPY...")

    master = pd.read_csv(os.path.join(DIR_PROCESSED, "master_weekly_raw.csv"),
                          parse_dates=["date"], index_col="date")

    bull_bear = master["aaii_bull_bear_spread"]
    spy_cum = master["SPY_log_ret"].cumsum()  # Retorno acumulado

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Eje izquierdo: AAII
    color_aaii = "#e67e22"
    ax1.fill_between(bull_bear.index, 0, bull_bear, where=(bull_bear >= 0),
                     alpha=0.15, color="green")
    ax1.fill_between(bull_bear.index, 0, bull_bear, where=(bull_bear < 0),
                     alpha=0.15, color="red")
    ax1.plot(bull_bear.index, bull_bear, linewidth=0.8, color=color_aaii,
             alpha=0.8, label="AAII Bull-Bear Spread")
    ax1.set_ylabel("Spread Bull-Bear AAII", color=color_aaii)
    ax1.tick_params(axis="y", labelcolor=color_aaii)
    ax1.axhline(y=0, color="gray", linewidth=0.5)

    # Eje derecho: SPY retorno acumulado
    ax2 = ax1.twinx()
    color_spy = "#1f77b4"
    ax2.plot(spy_cum.index, spy_cum * 100, linewidth=1.5, color=color_spy,
             alpha=0.9, label="SPY retorno acumulado")
    ax2.set_ylabel("Retorno acumulado SPY (%)", color=color_spy)
    ax2.tick_params(axis="y", labelcolor=color_spy)

    ax1.set_title("Sentimiento AAII (Bull-Bear Spread) vs Retorno acumulado SPY",
                  fontweight="bold")
    ax1.set_xlabel("Fecha")

    # Leyenda combinada
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(True, alpha=0.3)

    guardar(fig, "sentimiento_aaii_vs_spy.png")


# ── 7. Google Trends "recession" ──────────────────────────────

def fig07_google_recession():
    """Búsquedas de 'recession' vs periodos de estrés de SPY."""
    print("\n7. Google Trends 'recession'...")

    sentiment = pd.read_csv(os.path.join(DIR_INTERIM, "sentiment_weekly_clean.csv"),
                             parse_dates=["date"], index_col="date")
    etfs = pd.read_csv(os.path.join(DIR_INTERIM, "etfs_weekly_clean.csv"),
                        parse_dates=["date"], index_col="date")

    recession = sentiment["recession"]
    spy = etfs["SPY"]

    # Detectar periodos donde SPY cae >15% desde máximo reciente (rolling 52w)
    spy_max_rolling = spy.rolling(52, min_periods=1).max()
    drawdown_spy = (spy - spy_max_rolling) / spy_max_rolling
    estres = drawdown_spy < -0.15

    fig, ax = plt.subplots(figsize=(14, 6))

    # Sombrear periodos de estrés de SPY
    ax.fill_between(estres.index, 0, recession.max() * 1.1,
                    where=estres, alpha=0.15, color="red",
                    label="Caída SPY > 15%")

    ax.plot(recession.index, recession, linewidth=1.2, color="#8e44ad",
            label='Google Trends: "recession"')

    ax.set_title('Búsquedas de "recession" en Google vs periodos de estrés del S&P 500',
                 fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Índice Google Trends")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    guardar(fig, "google_trends_recession.png")


# ── 8. Distribución sentimiento NLP Refinitiv ──────────────────

def fig08_nlp_dist():
    """Distribución del sentimiento VADER sobre titulares de Refinitiv."""
    print("\n8. Distribución sentimiento NLP...")

    ruta = os.path.join("data", "raw", "sentiment", "all_news_refinitiv_scored.csv")
    df = pd.read_csv(ruta, parse_dates=["date"], index_col="date")
    scores = df["sentiment_score"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Histograma completo
    bins = np.linspace(-1, 1, 80)

    # Separar en 3 zonas: negativo, neutro, positivo
    neg = scores[scores < -0.05]
    neu = scores[(scores >= -0.05) & (scores <= 0.05)]
    pos = scores[scores > 0.05]

    ax.hist(neg, bins=bins, alpha=0.7, color="#e74c3c", label=f"Negativo ({len(neg):,})")
    ax.hist(neu, bins=bins, alpha=0.7, color="#95a5a6", label=f"Neutro ({len(neu):,})")
    ax.hist(pos, bins=bins, alpha=0.7, color="#27ae60", label=f"Positivo ({len(pos):,})")

    ax.axvline(x=0, color="black", linewidth=1, linestyle="-")
    ax.axvline(x=-0.05, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.axvline(x=0.05, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)

    ax.set_title("Distribución del sentimiento VADER sobre titulares de Refinitiv (17,181 titulares)",
                 fontweight="bold")
    ax.set_xlabel("Compound Score VADER")
    ax.set_ylabel("Frecuencia")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    guardar(fig, "nlp_sentiment_dist.png")


# ── 9. Antes/después nulos ─────────────────────────────────────

def fig09_antes_despues_nulos():
    """Heatmap de nulos antes y después de la limpieza."""
    print("\n9. Antes/después nulos (heatmap)...")

    # Antes: macro_weekly.csv (con nulos, antes de clean)
    macro_antes = pd.read_csv(os.path.join(DIR_INTERIM, "macro_weekly.csv"),
                               parse_dates=["date"], index_col="date")
    # Después: macro_weekly_clean.csv
    macro_despues = pd.read_csv(os.path.join(DIR_INTERIM, "macro_weekly_clean.csv"),
                                 parse_dates=["date"], index_col="date")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Panel izquierdo: ANTES
    nulos_antes = macro_antes.isnull().astype(int)
    sns.heatmap(nulos_antes.T, cmap=["#f0f0f0", "#e74c3c"], cbar=False, ax=ax1,
                yticklabels=True, xticklabels=False)
    ax1.set_title("ANTES de la limpieza\n(macro_weekly.csv)", fontweight="bold")
    ax1.set_xlabel(f"Semanas (n={len(macro_antes)})")
    ax1.set_ylabel("")
    # Añadir conteo de nulos
    total_antes = macro_antes.isnull().sum().sum()
    ax1.text(0.5, -0.08, f"Total nulos: {total_antes}",
             transform=ax1.transAxes, ha="center", fontsize=11, color="#e74c3c")

    # Panel derecho: DESPUÉS
    nulos_despues = macro_despues.isnull().astype(int)
    sns.heatmap(nulos_despues.T, cmap=["#f0f0f0", "#e74c3c"], cbar=False, ax=ax2,
                yticklabels=True, xticklabels=False)
    ax2.set_title("DESPUÉS de la limpieza\n(macro_weekly_clean.csv)", fontweight="bold")
    ax2.set_xlabel(f"Semanas (n={len(macro_despues)})")
    ax2.set_ylabel("")
    total_despues = macro_despues.isnull().sum().sum()
    ax2.text(0.5, -0.08, f"Total nulos: {total_despues}",
             transform=ax2.transAxes, ha="center", fontsize=11, color="#27ae60")

    fig.suptitle("Tratamiento de valores nulos — Antes vs Después",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    guardar(fig, "antes_despues_nulos.png")


# ── 10. Antes/después frecuencia ───────────────────────────────

def fig10_antes_despues_frecuencia():
    """Comparación visual de datos diarios vs semanales."""
    print("\n10. Antes/después frecuencia (diario vs semanal)...")

    # Datos diarios
    etfs_diario = pd.read_csv(os.path.join(DIR_INTERIM, "etfs_prices_daily.csv"),
                               parse_dates=["Date"], index_col="Date")
    # Datos semanales
    etfs_semanal = pd.read_csv(os.path.join(DIR_INTERIM, "etfs_weekly_clean.csv"),
                                parse_dates=["date"], index_col="date")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # Panel superior: diario
    ax1.plot(etfs_diario.index, etfs_diario["SPY"], linewidth=0.5,
             color=COLORES["SPY"], alpha=0.8)
    ax1.set_title("SPY — Precio diario (frecuencia original)", fontweight="bold")
    ax1.set_ylabel("Precio (USD)")
    ax1.grid(True, alpha=0.3)
    n_diario = len(etfs_diario)
    ax1.text(0.98, 0.05, f"n = {n_diario:,} observaciones",
             transform=ax1.transAxes, ha="right", fontsize=10, color="gray",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Panel inferior: semanal
    ax2.plot(etfs_semanal.index, etfs_semanal["SPY"], linewidth=1.2,
             color=COLORES["SPY"], alpha=0.9)
    ax2.scatter(etfs_semanal.index, etfs_semanal["SPY"], s=3,
                color=COLORES["SPY"], alpha=0.4)
    ax2.set_title("SPY — Precio semanal (W-FRI, tras resampleo)", fontweight="bold")
    ax2.set_xlabel("Fecha")
    ax2.set_ylabel("Precio (USD)")
    ax2.grid(True, alpha=0.3)
    n_semanal = len(etfs_semanal)
    ax2.text(0.98, 0.05, f"n = {n_semanal:,} observaciones",
             transform=ax2.transAxes, ha="right", fontsize=10, color="gray",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle("Resampleo de frecuencia diaria a semanal",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    guardar(fig, "antes_despues_frecuencia.png")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(DIR_FIGURES, exist_ok=True)

    fig01_evolucion_log()
    fig02_violin_retornos()
    fig03_vix()
    fig04_spread()
    fig05_fed_balance()
    fig06_aaii_vs_spy()
    fig07_google_recession()
    fig08_nlp_dist()
    fig09_antes_despues_nulos()
    fig10_antes_despues_frecuencia()

    print(f"\n{'='*65}")
    print(f"EDA completo. {len(figuras)} figuras generadas:")
    print(f"{'='*65}")
    for ruta in figuras:
        size = os.path.getsize(ruta) / 1024
        print(f"  {ruta} ({size:.0f} KB)")
    print(f"{'='*65}")
