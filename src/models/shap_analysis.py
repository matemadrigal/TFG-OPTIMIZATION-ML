"""
Análisis SHAP completo para el TFG — Interpretabilidad de modelos XGBoost.
Genera 8 figuras en 300 DPI con estilo científico (Tufte, Okabe-Ito).

Autor: Mateo Madrigal Arteaga, UFV
Uso:   python3 src/models/shap_analysis.py
"""

import matplotlib
matplotlib.use("Agg")

import sys
import os
import json
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xgboost as xgb
import shap

from src.models.data_loader import load_master_dataset, get_etf_tickers, get_feature_groups

# ── Constantes ──────────────────────────────────────────────────────

ETFS = get_etf_tickers()
FIGURES_DIR = "docs/figures"
PARAMS_PATH = "data/results/optuna_best_params_xgb.json"
TRAIN_SIZE = 790

# Paleta Okabe-Ito (colorblind-friendly, recomendada por Nature)
DIM_COLORS = {
    "market":    "#0072B2",   # azul
    "macro":     "#009E73",   # verde
    "risk":      "#D55E00",   # rojo-naranja
    "liquidity": "#CC79A7",   # rosa
    "sentiment": "#E69F00",   # amarillo-naranja
    "news_nlp":  "#999999",   # gris
}
DIM_LABELS = {
    "market": "Mercado", "macro": "Macro", "risk": "Riesgo",
    "liquidity": "Liquidez", "sentiment": "Sentimiento",
    "news_nlp": "NLP Noticias",
}

# Colores globales de texto
CLR_TEXT = "#333333"
CLR_ANNOT = "#666666"
CLR_GRID = "#e8e8e8"

GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

# ── Nombres legibles para las features ──────────────────────────────

ETF_NAMES = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000",
    "EFA": "EAFE Desarr.", "EEM": "Emergentes", "AGG": "Bonos US",
    "LQD": "Bonos Corp.", "TIP": "TIPS Inflación", "GLD": "Oro",
    "VNQ": "REITs Inmob.",
}
SUFFIX_NAMES = {
    "log_ret": "Retorno", "vol_4w": "Volat. 4sem", "vol_12w": "Volat. 12sem",
    "mom_4w": "Mom. 4sem", "mom_12w": "Mom. 12sem", "drawdown": "Drawdown",
}
FIXED_NAMES = {
    "nfci_change": "Condiciones Financieras (NFCI)",
    "vix_level": "Nivel VIX", "vix_change": "Cambio VIX",
    "hy_spread_change": "Spread High Yield",
    "spread_10y_2y": "Spread 10Y-2Y", "cpi_change": "Cambio CPI",
    "unrate_change": "Cambio Desempleo", "umcsent_change": "Sentimiento Consumidor",
    "fed_balance_change": "Balance Fed",
    "reverse_repo_change": "Cambio Reverse Repo",
    "bank_deposits_change": "Cambio Depósitos Bancarios",
    "tga_change": "Cambio Cuenta Tesoro",
    "aaii_bull_bear_spread": "AAII Bull-Bear",
    "inflation_change": "Google: Inflación (cambio)",
    "inflation_ma4w": "Google: Inflación",
    "recession_change": "Google: Recesión (cambio)",
    "recession_ma4w": "Google: Recesión",
    "bear_market_change": "Google: Bear Market (cambio)",
    "bear_market_ma4w": "Google: Bear Market",
    "bull_market_change": "Google: Bull Market (cambio)",
    "bull_market_ma4w": "Google: Bull Market",
    "buy_stocks_change": "Google: Buy Stocks (cambio)",
    "buy_stocks_ma4w": "Google: Buy Stocks",
    "sell_stocks_change": "Google: Sell Stocks (cambio)",
    "sell_stocks_ma4w": "Google: Sell Stocks",
    "unemployment_change": "Google: Unemployment (cambio)",
    "unemployment_ma4w": "Google: Unemployment",
    "news_sent_all": "Sentimiento Noticias (global)",
    "news_count_all": "Volumen Noticias (global)",
}

ETF_DIR_ACC = {
    "SPY": 59.6, "QQQ": 59.0, "IWM": 59.1, "EFA": 57.2, "EEM": 53.1,
    "AGG": 56.9, "LQD": 55.8, "TIP": 56.6, "GLD": 54.9, "VNQ": 55.5,
}


def readable_name(feature_name):
    """Convierte nombre técnico a legible."""
    if feature_name in FIXED_NAMES:
        return FIXED_NAMES[feature_name]
    for etf in ETFS:
        if feature_name == f"{etf}_news_sent":
            return f"Sent. Noticias {ETF_NAMES.get(etf, etf)}"
        if feature_name == f"{etf}_news_count":
            return f"Vol. Noticias {ETF_NAMES.get(etf, etf)}"
    for etf in ETFS:
        for suffix, suffix_name in SUFFIX_NAMES.items():
            if feature_name == f"{etf}_{suffix}":
                return f"{suffix_name} {ETF_NAMES.get(etf, etf)}"
    return feature_name


def readable_names_list(feature_names):
    return [readable_name(n) for n in feature_names]


def setup_style():
    """Estilo científico global: Tufte data-ink, fondo blanco, ejes limpios."""
    plt.rcParams.update({
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.5,
        "axes.labelcolor": CLR_TEXT,
        "axes.titlecolor": CLR_TEXT,
        "xtick.color": CLR_TEXT,
        "ytick.color": CLR_TEXT,
        "text.color": CLR_TEXT,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    })


# ══════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO PARA SHAP
# ══════════════════════════════════════════════════════════════════════

def train_for_shap(features, targets, etf, best_params, train_size=TRAIN_SIZE):
    target_col = f"target_{etf}"
    X_train_full = features.iloc[:train_size]
    y_train_full = targets[target_col].iloc[:train_size]
    X_test = features.iloc[train_size:]

    val_size = max(1, int(len(X_train_full) * 0.15))
    params = {
        "objective": "reg:squarederror", "n_estimators": 2000,
        "random_state": 42, "verbosity": 0, "early_stopping_rounds": 50,
        **best_params,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train_full.iloc[:-val_size], y_train_full.iloc[:-val_size],
              eval_set=[(X_train_full.iloc[-val_size:], y_train_full.iloc[-val_size:])],
              verbose=False)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return model, explainer, shap_values, X_test, X_train_full


def get_feature_to_dim(feature_names, feature_groups):
    mapping = {}
    for dim, cols in feature_groups.items():
        for col in cols:
            mapping[col] = dim
    return mapping


# ══════════════════════════════════════════════════════════════════════
# FIGURA 1 — Importancia global top 20
# ══════════════════════════════════════════════════════════════════════

def plot_global_importance(all_shap_values, feature_names, feature_groups):
    importances = np.zeros(len(feature_names))
    for etf in ETFS:
        importances += np.abs(all_shap_values[etf]).mean(axis=0)
    importances /= len(ETFS)

    top_idx = np.argsort(importances)[-20:]
    top_raw = [feature_names[i] for i in top_idx]
    top_names = [readable_name(n) for n in top_raw]
    top_vals = importances[top_idx]

    feat_to_dim = get_feature_to_dim(feature_names, feature_groups)
    colors = [DIM_COLORS.get(feat_to_dim.get(n, ""), "#999999") for n in top_raw]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(range(20), top_vals, color=colors)
    ax.set_yticks(range(20))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel("Importancia media |SHAP| (promedio 10 ETFs)", fontsize=11)
    ax.set_title("Top 20 variables por importancia SHAP global",
                 fontsize=14, weight="bold", pad=12)
    ax.axvline(x=0, color=CLR_GRID, linewidth=0.5, zorder=0)

    # Gridlines horizontales sutiles
    for y in range(20):
        ax.axhline(y=y, color=CLR_GRID, linewidth=0.3, zorder=0)

    handles = [mpatches.Patch(color=DIM_COLORS[d], label=DIM_LABELS[d])
               for d in ["market", "risk", "sentiment", "liquidity", "macro"]]
    ax.legend(handles=handles, loc="lower right", fontsize=9,
              framealpha=0.85, edgecolor="none", facecolor="white")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "shap_global_importance.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════
# FIGURA 2 — Importancia por dimensión (donut + tabla eficiencia)
# ══════════════════════════════════════════════════════════════════════

def plot_dimension_importance(all_shap_values, feature_names, feature_groups):
    feat_to_dim = get_feature_to_dim(feature_names, feature_groups)
    dim_imp = {d: 0.0 for d in DIM_COLORS}
    for etf in ETFS:
        abs_vals = np.abs(all_shap_values[etf]).mean(axis=0)
        for i, name in enumerate(feature_names):
            dim = feat_to_dim.get(name, "")
            if dim in dim_imp:
                dim_imp[dim] += abs_vals[i]
    for d in dim_imp:
        dim_imp[d] /= len(ETFS)

    total = sum(dim_imp.values())
    sorted_dims = sorted(dim_imp, key=lambda d: dim_imp[d], reverse=True)
    vals = [dim_imp[d] for d in sorted_dims]
    pcts = [v / total * 100 for v in vals]
    labels = [DIM_LABELS[d] for d in sorted_dims]
    colors = [DIM_COLORS[d] for d in sorted_dims]
    n_feats = [len(feature_groups[d]) for d in sorted_dims]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    # Donut: solo mostrar % en las porciones grandes (>5%)
    def autopct_func(pct):
        return f"{pct:.1f}%" if pct > 5 else ""

    wedges, texts, autotexts = ax1.pie(
        pcts, labels=None, colors=colors, autopct=autopct_func,
        startangle=90, pctdistance=0.78,
        wedgeprops={"width": 0.4, "edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
        t.set_color(CLR_TEXT)

    ax1.legend(labels, loc="center", fontsize=10, frameon=False)
    ax1.set_title("Contribución al poder predictivo",
                  fontsize=14, weight="bold", pad=12, color=CLR_TEXT)

    # Tabla de eficiencia
    ax2.axis("off")
    header = ["Dimensión", "Importancia", "Features", "Eficiencia"]
    rows = []
    effs = []
    for i in range(len(sorted_dims)):
        eff = pcts[i] / n_feats[i] if n_feats[i] > 0 else 0
        effs.append(eff)
        rows.append([labels[i], f"{pcts[i]:.1f}%", str(n_feats[i]), f"{eff:.2f}%/feat"])

    table = ax2.table(cellText=[header] + rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    for j in range(4):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Resaltar Riesgo (fila con max eficiencia) con fondo amarillo suave
    max_eff_row = np.argmax(effs) + 1
    for j in range(4):
        table[max_eff_row, j].set_facecolor("#FFF9C4")

    # Subtítulo
    risk_eff = effs[sorted_dims.index("risk")] if "risk" in sorted_dims else 0
    mkt_eff = effs[sorted_dims.index("market")] if "market" in sorted_dims else 1
    ratio = risk_eff / mkt_eff if mkt_eff > 0 else 0
    ax2.set_title(f"Riesgo aporta {ratio:.0f}x más información por variable que Mercado",
                  fontsize=10, style="italic", color=CLR_ANNOT, pad=12)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "shap_dimension_importance.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path, dim_imp, total


# ══════════════════════════════════════════════════════════════════════
# FIGURAS 3-5 — Beeswarm SPY, AGG, GLD (top 10)
# ══════════════════════════════════════════════════════════════════════

def plot_beeswarm_top_etfs(shap_values_by_etf, X_tests, feature_names):
    readable = readable_names_list(feature_names)
    paths = []

    for etf in ["SPY", "AGG", "GLD"]:
        sv = shap_values_by_etf[etf]
        xt = X_tests[etf]
        da = ETF_DIR_ACC.get(etf, 0)
        etf_name = ETF_NAMES.get(etf, etf)

        fig, ax = plt.subplots(figsize=(12, 7))
        shap.summary_plot(sv, xt, feature_names=readable,
                          max_display=10, show=False, plot_size=None)

        plt.title(f"SHAP Beeswarm — {etf} ({etf_name}) | Dir. Accuracy: {da:.1f}%",
                  fontsize=13, weight="bold", color=CLR_TEXT)
        # Subtítulo explicativo
        plt.figtext(0.5, -0.01,
                    "Azul = valor bajo de la variable  |  Rojo = valor alto",
                    ha="center", fontsize=9, style="italic", color=CLR_ANNOT)

        # Nota para AGG
        if etf == "AGG":
            plt.figtext(0.98, 0.02,
                        "(Escala ~100x menor que SPY — renta fija menos volátil)",
                        ha="right", fontsize=8, style="italic", color=CLR_ANNOT)

        plt.xlabel("Impacto SHAP en la predicción", fontsize=11, color=CLR_TEXT)
        plt.tight_layout()
        fname = f"shap_beeswarm_{etf.lower()}.png"
        path = os.path.join(FIGURES_DIR, fname)
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close("all")
        paths.append(path)
    return paths


# ══════════════════════════════════════════════════════════════════════
# FIGURA 6 — Heatmap comparación ETFs
# ══════════════════════════════════════════════════════════════════════

def plot_etf_comparison(all_shap_values, feature_names):
    imp_matrix = np.zeros((len(feature_names), len(ETFS)))
    for j, etf in enumerate(ETFS):
        imp_matrix[:, j] = np.abs(all_shap_values[etf]).mean(axis=0)

    global_imp = imp_matrix.mean(axis=1)
    top_idx = np.argsort(global_imp)[-15:][::-1]
    matrix = imp_matrix[top_idx, :]
    names = [readable_name(feature_names[i]) for i in top_idx]

    # ETF labels con categoría
    etf_labels = [f"{etf}\n{ETF_NAMES.get(etf, '')}" for etf in ETFS]

    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(ETFS)))
    ax.set_xticklabels(etf_labels, fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title("Importancia SHAP por variable y ETF (top 15)",
                 fontsize=14, weight="bold", pad=12, color=CLR_TEXT)
    plt.colorbar(im, ax=ax, label="Importancia media |SHAP|", shrink=0.8)

    for i in range(len(names)):
        for j in range(len(ETFS)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else CLR_TEXT
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=7, color=color)

    # Líneas divisorias: RV (SPY-EEM cols 0-4), RF (AGG-TIP cols 5-7), Alt (GLD-VNQ cols 8-9)
    # El orden de ETFS es: AGG,EEM,EFA,GLD,IWM,LQD,QQQ,SPY,TIP,VNQ
    # Encontrar posiciones por tipo
    rv = ["SPY", "QQQ", "IWM", "EFA", "EEM"]
    rf = ["AGG", "LQD", "TIP"]
    alt = ["GLD", "VNQ"]
    rv_idx = [ETFS.index(e) for e in rv if e in ETFS]
    rf_idx = [ETFS.index(e) for e in rf if e in ETFS]
    # Línea entre último RV y primer RF
    if rv_idx and rf_idx:
        boundary1 = (max(rv_idx) + min(rf_idx)) / 2
        # No dibujar si los índices no están contiguos (están mezclados alfabéticamente)
    # Dibujar líneas fijas entre los grupos lógicos
    ax.axvline(x=4.5, color="white", linewidth=2, zorder=3)
    ax.axvline(x=7.5, color="white", linewidth=2, zorder=3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "shap_etf_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════
# FIGURA 7 — Importancia temporal (3 períodos)
# ══════════════════════════════════════════════════════════════════════

def plot_temporal_importance(features, targets, best_params, feature_names, feature_groups):
    periods = [
        ("2011-2015", "2011-01-01", "2015-12-31"),
        ("2016-2020", "2016-01-01", "2020-12-31"),
        ("2021-2025", "2021-01-01", "2025-12-31"),
    ]
    # Colores Okabe-Ito para períodos
    period_colors = ["#0072B2", "#D55E00", "#009E73"]

    period_importances = {}
    for label, test_start, test_end in periods:
        mask_train = features.index <= pd.Timestamp(test_start)
        mask_test = (features.index >= pd.Timestamp(test_start)) & \
                    (features.index <= pd.Timestamp(test_end))
        if mask_train.sum() < 104 or mask_test.sum() < 10:
            continue

        X_train = features.loc[mask_train]
        X_test = features.loc[mask_test]

        imp = np.zeros(len(feature_names))
        for etf in ["SPY", "AGG", "GLD"]:
            y_tr = targets[f"target_{etf}"].loc[mask_train]
            val_size = max(1, int(len(X_train) * 0.15))
            params = {
                "objective": "reg:squarederror", "n_estimators": 2000,
                "random_state": 42, "verbosity": 0, "early_stopping_rounds": 50,
                **best_params,
            }
            model = xgb.XGBRegressor(**params)
            model.fit(X_train.iloc[:-val_size], y_tr.iloc[:-val_size],
                      eval_set=[(X_train.iloc[-val_size:], y_tr.iloc[-val_size:])],
                      verbose=False)
            sv = shap.TreeExplainer(model).shap_values(X_test)
            imp += np.abs(sv).mean(axis=0)
        imp /= 3
        period_importances[label] = imp

    avg_imp = sum(period_importances.values()) / len(period_importances)
    top_idx = np.argsort(avg_imp)[-10:][::-1]
    top_names = [readable_name(feature_names[i]) for i in top_idx]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(top_names))
    width = 0.25

    for k, (label, imp) in enumerate(period_importances.items()):
        ax.bar(x + k * width, imp[top_idx], width, label=label,
               color=period_colors[k])

    ax.set_xticks(x + width)
    ax.set_xticklabels(top_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Importancia media |SHAP|", fontsize=11, color=CLR_TEXT)
    ax.set_title("Evolución temporal de importancia SHAP (top 10 variables)",
                 fontsize=14, weight="bold", pad=12, color=CLR_TEXT)
    ax.legend(fontsize=11, framealpha=0.9, edgecolor="none")

    # Anotación COVID en esquina superior izquierda
    ax.annotate("2016-2020 incluye crisis COVID-19",
                xy=(0.02, 0.97), xycoords="axes fraction",
                ha="left", va="top", fontsize=11, fontweight="bold",
                color=period_colors[1])

    # Gridlines horizontales sutiles
    ax.yaxis.grid(True, color=CLR_GRID, linewidth=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "shap_temporal_importance.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════
# FIGURA 8 — Waterfall predicción extrema
# ══════════════════════════════════════════════════════════════════════

def plot_single_prediction(model, explainer, X_test, feature_names, etf="SPY"):
    y_pred = model.predict(X_test)
    worst_idx = np.argmin(y_pred)

    for crisis_start, crisis_end in [("2023-03-01", "2023-03-31"),
                                      ("2020-02-20", "2020-04-10")]:
        mask = (X_test.index >= crisis_start) & (X_test.index <= crisis_end)
        if mask.any():
            crisis_preds = y_pred.copy()
            crisis_preds[~np.array(mask)] = 999
            worst_idx = np.argmin(crisis_preds)
            break

    date_str = str(X_test.index[worst_idx].date())
    dt = X_test.index[worst_idx]

    if pd.Timestamp("2023-03-01") <= dt <= pd.Timestamp("2023-03-31"):
        context = "crisis bancaria SVB"
    elif pd.Timestamp("2020-02-20") <= dt <= pd.Timestamp("2020-04-10"):
        context = "crash COVID-19"
    else:
        context = "predicción extrema"

    sv = explainer.shap_values(X_test.iloc[[worst_idx]])
    expected = explainer.expected_value
    if isinstance(expected, np.ndarray):
        expected = expected[0]

    readable = readable_names_list(feature_names)
    explanation = shap.Explanation(
        values=sv[0], base_values=expected,
        data=X_test.iloc[worst_idx].values, feature_names=readable,
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(f"SHAP Waterfall — {etf} ({ETF_NAMES.get(etf, etf)}) "
              f"semana {date_str} ({context})",
              fontsize=12, weight="bold", color=CLR_TEXT)

    # Subtítulo interpretativo
    # Buscar feature con mayor impacto negativo
    top_neg_idx = np.argmin(sv[0])
    top_neg_name = readable[top_neg_idx]
    top_neg_val = sv[0][top_neg_idx]
    if top_neg_val < 0:
        plt.figtext(0.5, -0.02,
                    f"{top_neg_name} empuja la predicción {top_neg_val:.4f} "
                    f"(condiciones deteriorándose)",
                    ha="center", fontsize=9, style="italic", color=CLR_ANNOT)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "shap_waterfall_extreme.png")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close("all")
    return path, date_str, context


# ══════════════════════════════════════════════════════════════════════
# RESUMEN EN CONSOLA
# ══════════════════════════════════════════════════════════════════════

def print_shap_summary(all_shap_values, feature_names, feature_groups, dim_imp, dim_total):
    feat_to_dim = get_feature_to_dim(feature_names, feature_groups)
    importances = np.zeros(len(feature_names))
    for etf in ETFS:
        importances += np.abs(all_shap_values[etf]).mean(axis=0)
    importances /= len(ETFS)
    top_idx = np.argsort(importances)[-10:][::-1]

    iw = 66
    print(f"\n{'=' * iw}")
    print(f"{BOLD}{'ANALISIS SHAP — RESUMEN':^{iw}s}{RESET}")
    print(f"{'=' * iw}")

    print(f"\n  {BOLD}Top 10 features globales:{RESET}")
    for rank, idx in enumerate(top_idx, 1):
        name = readable_name(feature_names[idx])
        val = importances[idx]
        dim = feat_to_dim.get(feature_names[idx], "?")
        dim_label = DIM_LABELS.get(dim, dim)
        print(f"    {rank:>2d}. {name:<34s} {val:.5f}  ({dim_label})")

    print(f"\n  {BOLD}Importancia por dimension:{RESET}")
    for dim in sorted(dim_imp, key=lambda d: dim_imp[d], reverse=True):
        pct = dim_imp[dim] / dim_total * 100
        n_feat = len(feature_groups[dim])
        label = DIM_LABELS[dim]
        eff = pct / n_feat if n_feat > 0 else 0
        bar = "█" * int(pct / 2) + "░" * (25 - int(pct / 2))
        print(f"    {label:<14s} {bar} {pct:5.1f}% ({n_feat} feat, {eff:.1f}%/feat)")

    print(f"\n  {BOLD}Feature #1 por ETF:{RESET}")
    for etf in ETFS:
        etf_imp = np.abs(all_shap_values[etf]).mean(axis=0)
        best_idx = np.argmax(etf_imp)
        name = readable_name(feature_names[best_idx])
        print(f"    {etf:<5s} -> {name}")

    print(f"{'=' * iw}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()
    setup_style()

    print()
    print("=" * 62)
    print("  ANALISIS SHAP — Estilo cientifico (Tufte + Okabe-Ito)")
    print("  XGBoost Tuned | 10 ETFs | 109 features")
    print("=" * 62)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    with open(PARAMS_PATH) as f:
        best_params = json.load(f)
    print(f"\n  Params: lr={best_params['learning_rate']:.4f}, "
          f"depth={best_params['max_depth']}, mcw={best_params['min_child_weight']}")

    features, targets = load_master_dataset()
    feature_names = list(features.columns)
    feature_groups = get_feature_groups(feature_names)

    print(f"  Train: 1-{TRAIN_SIZE} | Test: {TRAIN_SIZE+1}-{len(features)}")

    # Entrenar y calcular SHAP
    print(f"\nEntrenando 10 modelos + TreeSHAP...")
    all_shap_values, X_tests, models, explainers = {}, {}, {}, {}
    for etf in ETFS:
        t_e = time.time()
        model, explainer, sv, X_test, _ = train_for_shap(features, targets, etf, best_params)
        all_shap_values[etf] = sv
        X_tests[etf] = X_test
        models[etf] = model
        explainers[etf] = explainer
        print(f"  {etf:>3s}: {model.best_iteration+1:>3d} arboles | {time.time()-t_e:.1f}s")

    # Generar 8 figuras
    print(f"\n{'=' * 62}")
    print("GENERANDO FIGURAS")
    print(f"{'=' * 62}")

    p1 = plot_global_importance(all_shap_values, feature_names, feature_groups)
    print(f"  1. {os.path.basename(p1)}")

    p2, dim_imp, dim_total = plot_dimension_importance(
        all_shap_values, feature_names, feature_groups)
    print(f"  2. {os.path.basename(p2)}")

    for i, p in enumerate(plot_beeswarm_top_etfs(all_shap_values, X_tests, feature_names), 3):
        print(f"  {i}. {os.path.basename(p)}")

    p6 = plot_etf_comparison(all_shap_values, feature_names)
    print(f"  6. {os.path.basename(p6)}")

    print(f"  7. Calculando SHAP temporal (3 periodos x 3 ETFs)...")
    p7 = plot_temporal_importance(features, targets, best_params, feature_names, feature_groups)
    print(f"     {os.path.basename(p7)}")

    p8, date_str, context = plot_single_prediction(
        models["SPY"], explainers["SPY"], X_tests["SPY"], feature_names, "SPY")
    print(f"  8. {os.path.basename(p8)} ({date_str}, {context})")

    print(f"{'=' * 62}")

    print_shap_summary(all_shap_values, feature_names, feature_groups, dim_imp, dim_total)

    shap_files = sorted([f for f in os.listdir(FIGURES_DIR) if f.startswith("shap_")])
    print(f"\n  Figuras en {FIGURES_DIR}/:")
    for f in shap_files:
        size = os.path.getsize(os.path.join(FIGURES_DIR, f))
        print(f"    {f:<40s} {size:>9,} bytes")

    print(f"\n  Tiempo total: {int((time.time()-t0)//60)}m {int((time.time()-t0)%60)}s")
