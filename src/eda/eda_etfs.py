"""
EDA — Análisis exploratorio de ETFs.
Genera estadísticos descriptivos y visualizaciones para la memoria del TFG.
Todas las figuras se exportan a docs/figures/ en PNG a 300 DPI.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ── Configuración visual ──────────────────────────────────────

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("tab10")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DIR_FIGURES = os.path.join("docs", "figures")
DIR_PROCESSED = os.path.join("data", "processed")
DIR_INTERIM = os.path.join("data", "interim")

ETFS = ["AGG", "EEM", "EFA", "GLD", "IWM", "LQD", "QQQ", "SPY", "TIP", "VNQ"]

# Periodos de crisis para sombreados
CRISIS = [
    ("2007-10-01", "2009-03-31", "GFC 2008"),
    ("2020-02-15", "2020-04-30", "COVID"),
    ("2022-01-01", "2022-10-31", "Inflación/Tipos"),
]

figuras_generadas = []


# ── BLOQUE 1 — Tabla de estadísticos descriptivos ─────────────

def bloque1_estadisticos():
    """
    Genera tabla con estadísticos descriptivos de todas las features.
    Incluye: media, mediana, std, min, max, skewness, kurtosis.
    """
    print("BLOQUE 1 — Tabla de estadísticos descriptivos...")

    master = pd.read_csv(
        os.path.join(DIR_PROCESSED, "master_weekly_raw.csv"),
        parse_dates=["date"], index_col="date"
    )

    # Separar features de targets
    feature_cols = [c for c in master.columns if not c.startswith("target_")]
    target_cols = [c for c in master.columns if c.startswith("target_")]

    # Clasificar cada feature en su dimensión
    def clasificar(col):
        if any(col.startswith(t + "_") for t in ETFS):
            return "ETF"
        if col.startswith("target_"):
            return "Target"
        if col in ["spread_10y_2y", "cpi_change", "unrate_change", "umcsent_change"]:
            return "Macro"
        if col in ["vix_level", "vix_change", "hy_spread_change", "nfci_change"]:
            return "Riesgo"
        if col in ["fed_balance_change", "reverse_repo_change",
                    "bank_deposits_change", "tga_change"]:
            return "Liquidez"
        if "news" in col:
            return "Noticias NLP"
        return "Sentimiento"

    # Calcular estadísticos para features (sin NaN)
    filas = []
    for col in feature_cols + target_cols:
        serie = master[col].dropna()
        filas.append({
            "Variable": col,
            "Dimensión": clasificar(col),
            "Media": serie.mean(),
            "Mediana": serie.median(),
            "Std": serie.std(),
            "Min": serie.min(),
            "Max": serie.max(),
            "Skewness": serie.skew(),
            "Kurtosis": serie.kurtosis(),
        })

    tabla = pd.DataFrame(filas)

    # Resumen por dimensión
    resumen = tabla.groupby("Dimensión").size()
    print(f"  Variables por dimensión:")
    for dim, n in resumen.items():
        print(f"    {dim}: {n}")

    # Renderizar como imagen (tabla con las primeras filas de cada dimensión)
    # Seleccionar ~25 variables representativas para que la tabla sea legible
    representativas = []
    for dim in ["ETF", "Macro", "Riesgo", "Liquidez", "Sentimiento", "Noticias NLP", "Target"]:
        grupo = tabla[tabla["Dimensión"] == dim]
        representativas.append(grupo.head(4))
    tabla_vis = pd.concat(representativas)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("off")

    # Formatear números para la tabla visual
    datos_vis = tabla_vis.copy()
    for col_num in ["Media", "Mediana", "Std", "Min", "Max", "Skewness", "Kurtosis"]:
        datos_vis[col_num] = datos_vis[col_num].apply(lambda x: f"{x:.4f}")

    tabla_mpl = ax.table(
        cellText=datos_vis.values,
        colLabels=datos_vis.columns,
        cellLoc="center",
        loc="center",
    )
    tabla_mpl.auto_set_font_size(False)
    tabla_mpl.set_fontsize(7)
    tabla_mpl.scale(1, 1.4)

    # Colorear cabecera
    for j in range(len(datos_vis.columns)):
        tabla_mpl[0, j].set_facecolor("#4472C4")
        tabla_mpl[0, j].set_text_props(color="white", fontweight="bold")

    # Colorear filas por dimensión
    colores_dim = {
        "ETF": "#D6E4F0", "Macro": "#E2EFDA", "Riesgo": "#FCE4D6",
        "Liquidez": "#D9E2F3", "Sentimiento": "#FFF2CC",
        "Noticias NLP": "#F2DCDB", "Target": "#E2D9F3",
    }
    for i, (_, row) in enumerate(datos_vis.iterrows(), 1):
        color = colores_dim.get(row["Dimensión"], "white")
        for j in range(len(datos_vis.columns)):
            tabla_mpl[i, j].set_facecolor(color)

    ax.set_title("Estadísticos descriptivos — Variables representativas por dimensión",
                 fontsize=13, fontweight="bold", pad=20)

    ruta = os.path.join(DIR_FIGURES, "tabla_estadisticos.png")
    fig.savefig(ruta)
    plt.close(fig)
    figuras_generadas.append(ruta)
    print(f"  Guardada: {ruta}")

    # Exportar tabla completa como CSV también (útil para la memoria)
    ruta_csv = os.path.join(DIR_FIGURES, "tabla_estadisticos.csv")
    tabla.to_csv(ruta_csv, index=False)
    print(f"  Tabla completa en CSV: {ruta_csv}")


# ── BLOQUE 2 — Evolución de precios base 100 ──────────────────

def bloque2_evolucion_precios():
    """
    Normaliza precios de ETFs a base 100 desde la primera fecha.
    Muestra la evolución con sombreados de crisis.
    """
    print("\nBLOQUE 2 — Evolución de precios de ETFs (base 100)...")

    etfs = pd.read_csv(
        os.path.join(DIR_INTERIM, "etfs_weekly_clean.csv"),
        parse_dates=["date"], index_col="date"
    )

    # Normalizar a base 100
    base100 = (etfs / etfs.iloc[0]) * 100

    fig, ax = plt.subplots(figsize=(14, 7))

    # Sombreados de crisis
    for inicio, fin, nombre in CRISIS:
        ax.axvspan(pd.Timestamp(inicio), pd.Timestamp(fin),
                   alpha=0.15, color="gray", label=nombre)

    # Líneas de precios
    for ticker in ETFS:
        ax.plot(base100.index, base100[ticker], linewidth=1.2, label=ticker)

    ax.set_title("Evolución de precios de ETFs (base 100, 2007–2026)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (base 100)")
    ax.set_ylim(0, None)

    # Leyenda sin duplicar crisis
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            unique_handles.append(h)
            unique_labels.append(l)
    ax.legend(unique_handles, unique_labels, loc="upper left", ncol=3)

    ax.grid(True, alpha=0.3)

    ruta = os.path.join(DIR_FIGURES, "etfs_evolucion_base100.png")
    fig.savefig(ruta)
    plt.close(fig)
    figuras_generadas.append(ruta)
    print(f"  Guardada: {ruta}")


# ── BLOQUE 3 — Distribución de retornos ───────────────────────

def bloque3_distribucion_retornos():
    """
    Histogramas de retornos semanales con curva normal teórica superpuesta.
    Demuestra que los retornos reales tienen colas más pesadas (fat tails).
    """
    print("\nBLOQUE 3 — Distribución de retornos semanales vs normal...")

    master = pd.read_csv(
        os.path.join(DIR_PROCESSED, "master_weekly_raw.csv"),
        parse_dates=["date"], index_col="date"
    )

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()

    for i, ticker in enumerate(ETFS):
        ax = axes[i]
        col = f"{ticker}_log_ret"
        datos = master[col].dropna()

        # Histograma
        ax.hist(datos, bins=50, density=True, alpha=0.6, color="steelblue",
                edgecolor="white", linewidth=0.3)

        # Curva normal teórica
        mu, sigma = datos.mean(), datos.std()
        x = np.linspace(datos.min(), datos.max(), 200)
        normal = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, normal, "r-", linewidth=1.5, label="Normal teórica")

        # Estadísticos
        kurt = datos.kurtosis()
        skew = datos.skew()
        ax.set_title(f"{ticker}\nkurt={kurt:.1f}  skew={skew:.2f}", fontsize=10)

        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Distribución de retornos semanales vs distribución normal",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    ruta = os.path.join(DIR_FIGURES, "distribucion_retornos.png")
    fig.savefig(ruta)
    plt.close(fig)
    figuras_generadas.append(ruta)
    print(f"  Guardada: {ruta}")


# ── BLOQUE 4 — Drawdowns comparativos ─────────────────────────

def bloque4_drawdowns():
    """
    Drawdowns históricos de 5 ETFs representativos.
    Muestra cómo cada clase de activo responde a las crisis.
    """
    print("\nBLOQUE 4 — Drawdowns comparativos...")

    master = pd.read_csv(
        os.path.join(DIR_PROCESSED, "master_weekly_raw.csv"),
        parse_dates=["date"], index_col="date"
    )

    # 5 ETFs representativos de distintas clases de activo
    etfs_sel = {
        "SPY": "Renta variable US (SPY)",
        "QQQ": "Tecnología (QQQ)",
        "EEM": "Emergentes (EEM)",
        "AGG": "Renta fija (AGG)",
        "GLD": "Oro (GLD)",
    }

    fig, ax = plt.subplots(figsize=(14, 7))

    colores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d4a017"]

    for (ticker, nombre), color in zip(etfs_sel.items(), colores):
        col = f"{ticker}_drawdown"
        # Drawdown está en formato negativo (0 a -X)
        ax.fill_between(master.index, master[col] * 100, 0,
                        alpha=0.15, color=color)
        ax.plot(master.index, master[col] * 100,
                linewidth=1, label=nombre, color=color)

    ax.set_title("Drawdowns históricos de ETFs representativos",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    ruta = os.path.join(DIR_FIGURES, "drawdowns_comparativo.png")
    fig.savefig(ruta)
    plt.close(fig)
    figuras_generadas.append(ruta)
    print(f"  Guardada: {ruta}")


# ── BLOQUE 5 — Boxplots de retornos ───────────────────────────

def bloque5_boxplots():
    """
    Boxplots comparativos de retornos semanales por ETF.
    Muestra la dispersión y los outliers de cada activo.
    """
    print("\nBLOQUE 5 — Boxplots de retornos por ETF...")

    master = pd.read_csv(
        os.path.join(DIR_PROCESSED, "master_weekly_raw.csv"),
        parse_dates=["date"], index_col="date"
    )

    # Preparar datos para boxplot
    ret_cols = [f"{t}_log_ret" for t in ETFS]
    datos_ret = master[ret_cols].copy()
    datos_ret.columns = ETFS  # Nombres limpios para el eje X

    fig, ax = plt.subplots(figsize=(12, 7))

    bp = ax.boxplot(
        [datos_ret[t].dropna() for t in ETFS],
        labels=ETFS,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
    )

    # Colorear cajas
    colores = plt.cm.tab10(np.linspace(0, 1, 10))
    for patch, color in zip(bp["boxes"], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

    ax.set_title("Distribución de retornos semanales por ETF",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("ETF")
    ax.set_ylabel("Log-return semanal")
    ax.grid(True, axis="y", alpha=0.3)

    ruta = os.path.join(DIR_FIGURES, "boxplot_retornos.png")
    fig.savefig(ruta)
    plt.close(fig)
    figuras_generadas.append(ruta)
    print(f"  Guardada: {ruta}")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(DIR_FIGURES, exist_ok=True)

    bloque1_estadisticos()
    bloque2_evolucion_precios()
    bloque3_distribucion_retornos()
    bloque4_drawdowns()
    bloque5_boxplots()

    print(f"\n{'='*60}")
    print(f"EDA de ETFs completado. {len(figuras_generadas)} figuras generadas:")
    for ruta in figuras_generadas:
        size = os.path.getsize(ruta) / 1024
        print(f"  {ruta} ({size:.0f} KB)")
    print(f"{'='*60}")
