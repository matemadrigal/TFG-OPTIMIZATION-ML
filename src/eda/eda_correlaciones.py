"""
EDA — Análisis de correlaciones entre features.
Genera 5 figuras de correlaciones para la memoria del TFG.
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

ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "LQD", "TIP", "GLD", "VNQ"]

COLORES = {
    "SPY": "#1f77b4", "QQQ": "#2ca02c", "IWM": "#9467bd",
    "EFA": "#17becf", "EEM": "#ff7f0e",
    "AGG": "#7f7f7f", "LQD": "#8c564b", "TIP": "#bcbd22",
    "GLD": "#d4af37", "VNQ": "#d62728",
}

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


def clasificar(col):
    """Asigna cada feature a su dimensión."""
    if any(col.startswith(t + "_") for t in ETFS):
        return "ETF"
    if col in ["spread_10y_2y", "cpi_change", "unrate_change", "umcsent_change"]:
        return "Macro"
    if col in ["vix_level", "vix_change", "hy_spread_change", "nfci_change"]:
        return "Riesgo"
    if col in ["fed_balance_change", "reverse_repo_change",
                "bank_deposits_change", "tga_change"]:
        return "Liquidez"
    if "news" in col:
        return "Noticias"
    return "Sentimiento"


def cargar_features():
    """Carga master raw y filtra solo features (sin targets ni Refinitiv news)."""
    master = pd.read_csv(os.path.join(DIR_PROCESSED, "master_weekly_raw.csv"),
                          parse_dates=["date"], index_col="date")
    # Excluir targets y columnas de noticias NLP (tienen muchos NaN)
    cols = [c for c in master.columns
            if not c.startswith("target_") and "news" not in c]
    return master[cols].copy()


# ── GRÁFICO 1 — Clustermap de correlaciones ───────────────────

def fig01_heatmap():
    """
    Heatmap de correlaciones con clustering jerárquico.
    Agrupa automáticamente las features más correlacionadas.
    """
    print("\n1. Heatmap de correlaciones (clustermap)...")

    features = cargar_features()
    corr = features.corr()

    # Clustermap con dendrograma
    g = sns.clustermap(
        corr,
        cmap="coolwarm",
        vmin=-1, vmax=1,
        center=0,
        figsize=(20, 18),
        linewidths=0.1,
        linecolor="white",
        dendrogram_ratio=(0.08, 0.08),
        cbar_pos=(0.02, 0.82, 0.03, 0.15),
        xticklabels=True,
        yticklabels=True,
    )
    g.ax_heatmap.tick_params(labelsize=6)
    g.fig.suptitle("Matriz de correlaciones — Features del modelo (clustering jerárquico)",
                   fontsize=16, fontweight="bold", y=1.01)

    ruta = os.path.join(DIR_FIGURES, "correlacion_heatmap.png")
    g.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(g.fig)
    figuras.append(ruta)
    print(f"  Guardada: {ruta}")


# ── GRÁFICO 2 — Correlación media entre dimensiones ───────────

def fig02_entre_dimensiones():
    """
    Heatmap 5×5 de correlación media entre dimensiones.
    Resume de un vistazo qué dimensiones se relacionan más.
    """
    print("\n2. Correlación media entre dimensiones...")

    features = cargar_features()

    # Clasificar cada columna
    dims = {}
    for col in features.columns:
        dim = clasificar(col)
        if dim == "Noticias":
            continue  # Ya excluidas, pero por seguridad
        if dim not in dims:
            dims[dim] = []
        dims[dim].append(col)

    # Orden deseado
    orden = ["ETF", "Macro", "Riesgo", "Liquidez", "Sentimiento"]
    dims = {k: dims[k] for k in orden if k in dims}

    # Calcular correlación media entre cada par de dimensiones
    nombres_dim = list(dims.keys())
    n = len(nombres_dim)
    matriz = np.zeros((n, n))

    for i, dim_i in enumerate(nombres_dim):
        for j, dim_j in enumerate(nombres_dim):
            if i == j:
                # Correlación intra-dimensión: media de la triangular superior
                sub = features[dims[dim_i]].corr().values
                mask = np.triu(np.ones_like(sub, dtype=bool), k=1)
                matriz[i, j] = np.abs(sub[mask]).mean()
            else:
                # Correlación inter-dimensión: media absoluta de todas las parejas
                corrs = []
                for col_i in dims[dim_i]:
                    for col_j in dims[dim_j]:
                        corrs.append(abs(features[col_i].corr(features[col_j])))
                matriz[i, j] = np.mean(corrs)

    df_matriz = pd.DataFrame(matriz, index=nombres_dim, columns=nombres_dim)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(df_matriz, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=0.5, square=True, linewidths=1,
                linecolor="white", ax=ax,
                cbar_kws={"label": "Correlación media absoluta"})

    ax.set_title("Correlación media entre dimensiones",
                 fontweight="bold", pad=15)

    guardar(fig, "correlacion_entre_dimensiones.png")


# ── GRÁFICO 3 — Rolling correlation SPY vs AGG ────────────────

def fig03_rolling_spy_agg():
    """
    Correlación rolling 52 semanas entre SPY y AGG.
    Demuestra que la correlación acciones-bonos NO es estable:
    a veces es negativa (diversificación funciona) y a veces positiva (no funciona).
    Esto es clave para justificar un enfoque ML vs Markowitz estático.
    """
    print("\n3. Rolling correlation SPY vs AGG...")

    master = pd.read_csv(os.path.join(DIR_PROCESSED, "master_weekly_raw.csv"),
                          parse_dates=["date"], index_col="date")

    spy_ret = master["SPY_log_ret"]
    agg_ret = master["AGG_log_ret"]

    # Correlación rolling de 52 semanas (1 año)
    rolling_corr = spy_ret.rolling(52).corr(agg_ret)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Sombreados de crisis
    for inicio, fin, nombre in CRISIS:
        ax.axvspan(pd.Timestamp(inicio), pd.Timestamp(fin),
                   alpha=0.12, color="gray", zorder=0)
        centro = pd.Timestamp(inicio) + (pd.Timestamp(fin) - pd.Timestamp(inicio)) / 2
        ax.text(centro, 0.85, nombre, fontsize=8, ha="center",
                color="#555", style="italic", transform=ax.get_xaxis_transform())

    # Colorear zonas de correlación positiva (diversificación rota)
    ax.fill_between(rolling_corr.index, rolling_corr, 0,
                    where=(rolling_corr > 0), alpha=0.2, color="red",
                    label="Corr. positiva (diversificación rota)")
    ax.fill_between(rolling_corr.index, rolling_corr, 0,
                    where=(rolling_corr <= 0), alpha=0.2, color="green",
                    label="Corr. negativa (diversificación funciona)")

    ax.plot(rolling_corr.index, rolling_corr, linewidth=1.2, color="#2c3e50")
    ax.axhline(y=0, color="black", linewidth=0.8)

    ax.set_title("Correlación rolling 52 semanas: SPY vs AGG (acciones vs bonos)",
                 fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Correlación de Pearson")
    ax.set_ylim(-0.8, 0.8)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    guardar(fig, "rolling_corr_spy_agg.png")


# ── GRÁFICO 4 — Rolling correlations SPY vs varios ────────────

def fig04_rolling_multi():
    """
    Correlación rolling 52 semanas de SPY contra AGG, GLD, EEM, VNQ.
    Muestra cómo cambian las correlaciones con distintas clases de activos.
    """
    print("\n4. Rolling correlations SPY vs varias clases...")

    master = pd.read_csv(os.path.join(DIR_PROCESSED, "master_weekly_raw.csv"),
                          parse_dates=["date"], index_col="date")

    spy_ret = master["SPY_log_ret"]

    pares = {
        "AGG": ("Renta Fija (AGG)", COLORES["AGG"]),
        "GLD": ("Oro (GLD)", COLORES["GLD"]),
        "EEM": ("Emergentes (EEM)", COLORES["EEM"]),
        "VNQ": ("Inmobiliario (VNQ)", COLORES["VNQ"]),
    }

    fig, ax = plt.subplots(figsize=(14, 6))

    # Sombreados de crisis
    for inicio, fin, nombre in CRISIS:
        ax.axvspan(pd.Timestamp(inicio), pd.Timestamp(fin),
                   alpha=0.10, color="gray", zorder=0)

    for ticker, (nombre, color) in pares.items():
        rc = spy_ret.rolling(52).corr(master[f"{ticker}_log_ret"])
        ax.plot(rc.index, rc, linewidth=1.5, color=color, label=nombre)

    ax.axhline(y=0, color="black", linewidth=0.8)

    ax.set_title("Correlación rolling 52 semanas de SPY vs otras clases de activos",
                 fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Correlación de Pearson")
    ax.set_ylim(-0.8, 1.0)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    guardar(fig, "rolling_corr_multi.png")


# ── GRÁFICO 5 — Scatter VIX vs retornos SPY ───────────────────

def fig05_scatter_vix():
    """
    Scatter VIX vs retornos semanales de SPY, coloreado por periodo.
    Demuestra la relación negativa: VIX alto = retornos peores.
    """
    print("\n5. Scatter VIX vs retornos SPY...")

    master = pd.read_csv(os.path.join(DIR_PROCESSED, "master_weekly_raw.csv"),
                          parse_dates=["date"], index_col="date")

    vix = master["vix_level"]
    spy_ret = master["SPY_log_ret"]

    # Clasificar por periodo
    periodos = pd.cut(
        master.index.year,
        bins=[0, 2009, 2019, 2030],
        labels=["2007–2009", "2010–2019", "2020–2026"]
    )
    colores_periodo = {"2007–2009": "#1f77b4", "2010–2019": "#2ca02c", "2020–2026": "#d62728"}

    fig, ax = plt.subplots(figsize=(10, 8))

    for periodo, color in colores_periodo.items():
        mask = periodos == periodo
        ax.scatter(vix[mask], spy_ret[mask] * 100, s=15, alpha=0.4,
                   color=color, label=periodo, edgecolors="none")

    # Línea de regresión global
    mask_valid = vix.notna() & spy_ret.notna()
    slope, intercept, r, p, se = stats.linregress(vix[mask_valid], spy_ret[mask_valid] * 100)
    x_line = np.linspace(vix.min(), vix.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=2,
            linestyle="--", label=f"Regresión (R²={r**2:.3f})")

    ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)

    ax.set_title("Relación VIX — Retornos semanales SPY", fontweight="bold")
    ax.set_xlabel("VIX (nivel)")
    ax.set_ylabel("Log-return semanal SPY (%)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    guardar(fig, "scatter_vix_spy.png")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(DIR_FIGURES, exist_ok=True)

    fig01_heatmap()
    fig02_entre_dimensiones()
    fig03_rolling_spy_agg()
    fig04_rolling_multi()
    fig05_scatter_vix()

    print(f"\n{'='*60}")
    print(f"EDA de correlaciones completado. {len(figuras)} figuras:")
    for ruta in figuras:
        size = os.path.getsize(ruta) / 1024
        print(f"  {ruta} ({size:.0f} KB)")
    print(f"{'='*60}")
