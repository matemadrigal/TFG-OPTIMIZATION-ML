"""
EDA — Análisis de series temporales: estacionariedad, autocorrelación y outliers.
Genera 3 figuras para la memoria del TFG.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

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

ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "LQD", "TIP", "GLD", "VNQ"]

figuras = []


def guardar(fig, nombre):
    ruta = os.path.join(DIR_FIGURES, nombre)
    fig.savefig(ruta)
    plt.close(fig)
    figuras.append(ruta)
    print(f"  Guardada: {ruta}")


# ── BLOQUE 1 — Test ADF de estacionariedad ────────────────────

def bloque1_estacionariedad():
    """
    Test Augmented Dickey-Fuller sobre features clave.
    H0: la serie tiene raíz unitaria (NO es estacionaria).
    Si p < 0.05, rechazamos H0 → la serie ES estacionaria.
    Los modelos ML necesitan datos estacionarios para generalizar bien.
    """
    print("\nBLOQUE 1 — Test de estacionariedad ADF...")

    master = pd.read_csv(os.path.join(DIR_PROCESSED, "master_weekly_raw.csv"),
                          parse_dates=["date"], index_col="date")

    # Variables a testear, agrupadas por dimensión
    variables = {
        # Log-returns de ETFs (deberían ser estacionarios)
        **{f"{t}_log_ret": "ETF" for t in ETFS},
        # Macro
        "spread_10y_2y": "Macro",
        "cpi_change": "Macro",
        "unrate_change": "Macro",
        "umcsent_change": "Macro",
        # Riesgo
        "vix_level": "Riesgo",
        "vix_change": "Riesgo",
        "hy_spread_change": "Riesgo",
        "nfci_change": "Riesgo",
        # Sentimiento
        "aaii_bull_bear_spread": "Sentimiento",
    }

    resultados = []
    for col, dim in variables.items():
        serie = master[col].dropna()
        adf_stat, p_value, usedlag, nobs, crit, icbest = adfuller(serie, autolag="AIC")
        estacionaria = p_value < 0.05
        resultados.append({
            "Variable": col,
            "Dimensión": dim,
            "ADF Stat": f"{adf_stat:.3f}",
            "p-value": f"{p_value:.4f}",
            "Estacionaria": "Sí" if estacionaria else "NO",
            "_es_estacionaria": estacionaria,
        })

    df_res = pd.DataFrame(resultados)

    # Imprimir resumen
    n_est = df_res["_es_estacionaria"].sum()
    n_total = len(df_res)
    print(f"  Estacionarias: {n_est}/{n_total}")
    no_est = df_res[~df_res["_es_estacionaria"]]["Variable"].tolist()
    if no_est:
        print(f"  NO estacionarias: {no_est}")

    # Renderizar como tabla-figura
    cols_vis = ["Variable", "Dimensión", "ADF Stat", "p-value", "Estacionaria"]
    datos_vis = df_res[cols_vis]

    fig, ax = plt.subplots(figsize=(14, max(8, len(datos_vis) * 0.4 + 2)))
    ax.axis("off")

    tabla = ax.table(
        cellText=datos_vis.values,
        colLabels=cols_vis,
        cellLoc="center",
        loc="center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1, 1.5)

    # Cabecera azul
    for j in range(len(cols_vis)):
        tabla[0, j].set_facecolor("#4472C4")
        tabla[0, j].set_text_props(color="white", fontweight="bold")

    # Colorear filas según resultado
    for i, es_est in enumerate(df_res["_es_estacionaria"], 1):
        color = "#E2EFDA" if es_est else "#FCE4D6"
        for j in range(len(cols_vis)):
            tabla[i, j].set_facecolor(color)
        # Resaltar celda de resultado
        if es_est:
            tabla[i, 4].set_text_props(color="#27ae60", fontweight="bold")
        else:
            tabla[i, 4].set_text_props(color="#e74c3c", fontweight="bold")

    ax.set_title("Test de estacionariedad Augmented Dickey-Fuller\n"
                 "(p < 0.05 → estacionaria, verde | p ≥ 0.05 → no estacionaria, rojo)",
                 fontsize=13, fontweight="bold", pad=20)

    guardar(fig, "tabla_estacionariedad.png")


# ── BLOQUE 2 — Autocorrelación de retornos SPY ────────────────

def bloque2_autocorrelacion():
    """
    Función de autocorrelación (ACF) de los log-returns semanales de SPY.
    Si no hay barras significativas fuera de la banda azul,
    los retornos pasados NO predicen retornos futuros por sí solos.
    Esto justifica usar features externas (macro, sentimiento, etc.)
    como predictores en lugar de solo datos de precio.
    """
    print("\nBLOQUE 2 — Autocorrelación de retornos SPY...")

    master = pd.read_csv(os.path.join(DIR_PROCESSED, "master_weekly_raw.csv"),
                          parse_dates=["date"], index_col="date")
    spy_ret = master["SPY_log_ret"].dropna()

    fig, ax = plt.subplots(figsize=(12, 5))

    plot_acf(spy_ret, lags=20, ax=ax, alpha=0.05, color="#1f77b4",
             vlines_kwargs={"colors": "#1f77b4"})

    ax.set_title("Autocorrelación de retornos semanales — SPY (lags 1–20)",
                 fontweight="bold")
    ax.set_xlabel("Lag (semanas)")
    ax.set_ylabel("Autocorrelación")
    ax.grid(True, alpha=0.3)

    # Nota interpretativa
    ax.text(0.98, 0.95,
            "Barras dentro de la banda azul → no significativas\n"
            "→ Retornos pasados no predicen retornos futuros\n"
            "→ Justifica usar features externas (macro, sentimiento)",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      alpha=0.9, edgecolor="gray"))

    guardar(fig, "autocorrelacion_spy.png")


# ── BLOQUE 3 — Análisis de outliers ───────────────────────────

def bloque3_outliers():
    """
    Detección de outliers en retornos semanales usando método IQR.
    Outlier si: valor < Q1 - 1.5*IQR  o  valor > Q3 + 1.5*IQR.
    Los outliers se MANTIENEN: son eventos reales (crisis, COVID)
    y el modelo debe aprender de ellos.
    """
    print("\nBLOQUE 3 — Análisis de outliers...")

    master = pd.read_csv(os.path.join(DIR_PROCESSED, "master_weekly_raw.csv"),
                          parse_dates=["date"], index_col="date")

    # Calcular outliers por ETF
    stats_outliers = []
    for ticker in ETFS:
        col = f"{ticker}_log_ret"
        serie = master[col].dropna()
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = ((serie < lower) | (serie > upper)).sum()
        stats_outliers.append({
            "ETF": ticker,
            "Outliers": n_outliers,
            "Porcentaje": n_outliers / len(serie) * 100,
            "Lower": lower,
            "Upper": upper,
        })

    df_out = pd.DataFrame(stats_outliers)

    print(f"  Outliers totales: {df_out['Outliers'].sum()}")
    print(f"  Rango: {df_out['Outliers'].min()}–{df_out['Outliers'].max()} por ETF")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                     gridspec_kw={"height_ratios": [1, 1.5]})

    # Panel superior: barras de outliers por ETF
    colores_barra = [
        "#1f77b4", "#2ca02c", "#9467bd", "#17becf", "#ff7f0e",
        "#7f7f7f", "#8c564b", "#bcbd22", "#d4af37", "#d62728"
    ]
    bars = ax1.bar(df_out["ETF"], df_out["Outliers"], color=colores_barra, alpha=0.8)

    # Etiquetas sobre las barras
    for bar, row in zip(bars, df_out.itertuples()):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{row.Outliers}\n({row.Porcentaje:.1f}%)",
                 ha="center", va="bottom", fontsize=9)

    ax1.set_title("Número de outliers por ETF (método IQR)", fontweight="bold")
    ax1.set_ylabel("Nº de outliers")
    ax1.grid(True, axis="y", alpha=0.3)

    # Panel inferior: scatter temporal de outliers de SPY
    spy_ret = master["SPY_log_ret"]
    q1 = spy_ret.quantile(0.25)
    q3 = spy_ret.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    # Puntos normales (grises)
    normal = (spy_ret >= lower) & (spy_ret <= upper)
    ax2.scatter(spy_ret[normal].index, spy_ret[normal] * 100,
                s=5, color="lightgray", alpha=0.4, label="Normal", zorder=1)

    # Outliers positivos (verde)
    pos_out = spy_ret > upper
    ax2.scatter(spy_ret[pos_out].index, spy_ret[pos_out] * 100,
                s=25, color="#27ae60", alpha=0.8, label="Outlier positivo",
                edgecolors="darkgreen", linewidths=0.5, zorder=2)

    # Outliers negativos (rojo)
    neg_out = spy_ret < lower
    ax2.scatter(spy_ret[neg_out].index, spy_ret[neg_out] * 100,
                s=25, color="#e74c3c", alpha=0.8, label="Outlier negativo",
                edgecolors="darkred", linewidths=0.5, zorder=2)

    # Límites IQR
    ax2.axhline(y=upper * 100, color="green", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.axhline(y=lower * 100, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)

    ax2.set_title("Outliers temporales de SPY — retornos semanales", fontweight="bold")
    ax2.set_xlabel("Fecha")
    ax2.set_ylabel("Log-return semanal (%)")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    # Nota sobre decisión
    fig.text(0.5, -0.02,
             "Decisión: los outliers se MANTIENEN porque corresponden a eventos reales "
             "del mercado (crisis, COVID, etc.) y el modelo debe aprender de ellos.",
             ha="center", fontsize=10, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                       edgecolor="gray", alpha=0.9))

    fig.suptitle("Análisis de outliers en retornos semanales",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    guardar(fig, "outliers_analisis.png")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(DIR_FIGURES, exist_ok=True)

    bloque1_estacionariedad()
    bloque2_autocorrelacion()
    bloque3_outliers()

    print(f"\n{'='*60}")
    print(f"EDA series temporales completado. {len(figuras)} figuras:")
    for ruta in figuras:
        size = os.path.getsize(ruta) / 1024
        print(f"  {ruta} ({size:.0f} KB)")
    print(f"{'='*60}")
