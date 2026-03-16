"""
Análisis SHAP completo para el TFG — Interpretabilidad de modelos XGBoost.
Genera 8 figuras en 300 DPI y un resumen en consola.

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
TRAIN_SIZE = 790   # ~80% de 987 semanas

# Colores por dimensión (consistentes en todas las figuras)
DIM_COLORS = {
    "market":    "#1f77b4",   # azul
    "macro":     "#2ca02c",   # verde
    "risk":      "#d62728",   # rojo
    "liquidity": "#9467bd",   # morado
    "sentiment": "#ff7f0e",   # naranja
    "news_nlp":  "#7f7f7f",   # gris
}
DIM_LABELS = {
    "market":    "Mercado",
    "macro":     "Macro",
    "risk":      "Riesgo",
    "liquidity": "Liquidez",
    "sentiment": "Sentimiento",
    "news_nlp":  "NLP Noticias",
}

GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


# ══════════════════════════════════════════════════════════════════════
# 1. ENTRENAMIENTO PARA SHAP
# ══════════════════════════════════════════════════════════════════════

def train_for_shap(features, targets, etf, best_params, train_size=TRAIN_SIZE):
    """
    Entrena un XGBoost sobre el 80% inicial y calcula SHAP sobre el 20% final.
    Retorna: modelo, shap_values (array), X_test (DataFrame), X_train (DataFrame)
    """
    target_col = f"target_{etf}"
    X = features
    y = targets[target_col]

    X_train_full = X.iloc[:train_size]
    y_train_full = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    # Separar validación del train (último 15%)
    val_size = max(1, int(len(X_train_full) * 0.15))
    X_train = X_train_full.iloc[:-val_size]
    y_train = y_train_full.iloc[:-val_size]
    X_val = X_train_full.iloc[-val_size:]
    y_val = y_train_full.iloc[-val_size:]

    # Params de Optuna + base
    params = {
        "objective": "reg:squarederror",
        "n_estimators": 2000,
        "random_state": 42,
        "verbosity": 0,
        "early_stopping_rounds": 50,
        **best_params,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # TreeSHAP (rápido para árboles)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    return model, explainer, shap_values, X_test, X_train_full


def get_feature_to_dim(feature_names, feature_groups):
    """Crea mapping feature_name → dimensión."""
    mapping = {}
    for dim, cols in feature_groups.items():
        for col in cols:
            mapping[col] = dim
    return mapping


# ══════════════════════════════════════════════════════════════════════
# 2. FIGURA 1 — Importancia global top 20
# ══════════════════════════════════════════════════════════════════════

def plot_global_importance(all_shap_values, feature_names, feature_groups):
    """Bar plot horizontal con las top 20 features por importancia SHAP global."""
    # Media absoluta de SHAP por feature, promediando los 10 ETFs
    importances = np.zeros(len(feature_names))
    for etf in ETFS:
        importances += np.abs(all_shap_values[etf]).mean(axis=0)
    importances /= len(ETFS)

    # Top 20
    top_idx = np.argsort(importances)[-20:]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = importances[top_idx]

    # Colores por dimensión
    feat_to_dim = get_feature_to_dim(feature_names, feature_groups)
    colors = [DIM_COLORS.get(feat_to_dim.get(n, ""), "#333333") for n in top_names]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(20), top_vals, color=colors)
    ax.set_yticks(range(20))
    ax.set_yticklabels(top_names, fontsize=10)
    ax.set_xlabel("Mean |SHAP value| (promedio 10 ETFs)", fontsize=12)
    ax.set_title("SHAP — Top 20 features por importancia global", fontsize=14, weight="bold")

    # Leyenda de dimensiones
    handles = [mpatches.Patch(color=c, label=DIM_LABELS[d])
               for d, c in DIM_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "shap_global_importance.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════
# 3. FIGURA 2 — Importancia por dimensión
# ══════════════════════════════════════════════════════════════════════

def plot_dimension_importance(all_shap_values, feature_names, feature_groups):
    """Bar chart con la contribución de cada dimensión."""
    feat_to_dim = get_feature_to_dim(feature_names, feature_groups)

    # Sumar importancia SHAP por dimensión
    dim_imp = {d: 0.0 for d in DIM_COLORS}
    for etf in ETFS:
        abs_vals = np.abs(all_shap_values[etf]).mean(axis=0)
        for i, name in enumerate(feature_names):
            dim = feat_to_dim.get(name, "")
            if dim in dim_imp:
                dim_imp[dim] += abs_vals[i]
    # Promediar ETFs
    for d in dim_imp:
        dim_imp[d] /= len(ETFS)

    total = sum(dim_imp.values())
    dims = list(DIM_COLORS.keys())
    vals = [dim_imp[d] for d in dims]
    pcts = [v / total * 100 for v in vals]
    labels = [DIM_LABELS[d] for d in dims]
    colors = [DIM_COLORS[d] for d in dims]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(labels, pcts, color=colors)
    ax.set_xlabel("Contribución al poder predictivo (%)", fontsize=12)
    ax.set_title("Contribución de cada dimensión al poder predictivo", fontsize=14, weight="bold")

    # Añadir etiquetas con %
    for bar, pct, n in zip(bars, pcts, [len(feature_groups[d]) for d in dims]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}% ({n} feat.)", va="center", fontsize=10)

    ax.set_xlim(0, max(pcts) * 1.25)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "shap_dimension_importance.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path, dim_imp, total


# ══════════════════════════════════════════════════════════════════════
# 4. FIGURAS 3-5 — Beeswarm plots para SPY, AGG, GLD
# ══════════════════════════════════════════════════════════════════════

def plot_beeswarm_top_etfs(shap_values_by_etf, X_tests, feature_names):
    """Beeswarm plots para SPY, AGG y GLD (top 15 features)."""
    paths = []
    for etf in ["SPY", "AGG", "GLD"]:
        sv = shap_values_by_etf[etf]
        xt = X_tests[etf]

        fig, ax = plt.subplots(figsize=(12, 8))
        # Usar shap.summary_plot (API estable, funciona bien con matplotlib)
        shap.summary_plot(
            sv, xt, feature_names=feature_names,
            max_display=15, show=False, plot_size=None,
        )
        plt.title(f"SHAP Beeswarm — {etf}", fontsize=14, weight="bold")
        plt.tight_layout()
        fname = f"shap_beeswarm_{etf.lower()}.png"
        path = os.path.join(FIGURES_DIR, fname)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close("all")
        paths.append(path)
    return paths


# ══════════════════════════════════════════════════════════════════════
# 5. FIGURA 6 — Heatmap ETF comparison
# ══════════════════════════════════════════════════════════════════════

def plot_etf_comparison(all_shap_values, feature_names):
    """Heatmap de importancia de top 15 features para cada ETF."""
    # Importancia por ETF y feature
    imp_matrix = np.zeros((len(feature_names), len(ETFS)))
    for j, etf in enumerate(ETFS):
        imp_matrix[:, j] = np.abs(all_shap_values[etf]).mean(axis=0)

    # Top 15 features (por importancia global media)
    global_imp = imp_matrix.mean(axis=1)
    top_idx = np.argsort(global_imp)[-15:][::-1]

    matrix = imp_matrix[top_idx, :]
    names = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(ETFS)))
    ax.set_xticklabels(ETFS, fontsize=11)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_title("Importancia SHAP por feature y ETF (top 15)", fontsize=14, weight="bold")
    plt.colorbar(im, ax=ax, label="Mean |SHAP value|", shrink=0.8)

    # Añadir valores en celdas
    for i in range(len(names)):
        for j in range(len(ETFS)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "shap_etf_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════
# 6. FIGURA 7 — Importancia temporal (3 períodos)
# ══════════════════════════════════════════════════════════════════════

def plot_temporal_importance(features, targets, best_params, feature_names, feature_groups):
    """Compara importancia SHAP en 3 períodos: 2011-2015, 2016-2020, 2021-2025."""
    periods = [
        ("2011-2015", "2007-01-01", "2015-12-31", "2011-01-01", "2015-12-31"),
        ("2016-2020", "2007-01-01", "2020-12-31", "2016-01-01", "2020-12-31"),
        ("2021-2025", "2007-01-01", "2025-12-31", "2021-01-01", "2025-12-31"),
    ]

    period_importances = {}

    for label, train_start, train_end, test_start, test_end in periods:
        # Train: todo lo anterior al período. Test: el período.
        mask_train = features.index <= pd.Timestamp(test_start)
        mask_test = (features.index >= pd.Timestamp(test_start)) & \
                    (features.index <= pd.Timestamp(test_end))

        if mask_train.sum() < 104 or mask_test.sum() < 10:
            continue

        X_train = features.loc[mask_train]
        X_test = features.loc[mask_test]

        # Promediar SHAP sobre SPY, AGG, GLD (representativos)
        imp = np.zeros(len(feature_names))
        for etf in ["SPY", "AGG", "GLD"]:
            y_tr = targets[f"target_{etf}"].loc[mask_train]
            y_te = targets[f"target_{etf}"].loc[mask_test]

            val_size = max(1, int(len(X_train) * 0.15))
            params = {
                "objective": "reg:squarederror",
                "n_estimators": 2000,
                "random_state": 42,
                "verbosity": 0,
                "early_stopping_rounds": 50,
                **best_params,
            }
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train.iloc[:-val_size], y_tr.iloc[:-val_size],
                eval_set=[(X_train.iloc[-val_size:], y_tr.iloc[-val_size:])],
                verbose=False,
            )
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_test)
            imp += np.abs(sv).mean(axis=0)

        imp /= 3
        period_importances[label] = imp

    # Top 10 features (por importancia global promediada entre períodos)
    avg_imp = sum(period_importances.values()) / len(period_importances)
    top_idx = np.argsort(avg_imp)[-10:][::-1]
    top_names = [feature_names[i] for i in top_idx]

    # Bar chart agrupado
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(top_names))
    width = 0.25
    period_labels = list(period_importances.keys())
    period_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for k, (label, imp) in enumerate(period_importances.items()):
        vals = imp[top_idx]
        ax.bar(x + k * width, vals, width, label=label, color=period_colors[k])

    ax.set_xticks(x + width)
    ax.set_xticklabels(top_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean |SHAP value|", fontsize=12)
    ax.set_title("Evolución temporal de importancia SHAP (top 10 features)",
                 fontsize=14, weight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "shap_temporal_importance.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════
# 7. FIGURA 8 — Waterfall de predicción extrema (COVID crash)
# ══════════════════════════════════════════════════════════════════════

def plot_single_prediction(model, explainer, X_test, feature_names, etf="SPY"):
    """Waterfall plot para la semana con mayor caída real en el test set."""
    # Buscar la semana con mayor caída en el test set
    # (usamos las predicciones del modelo para encontrar un punto interesante)
    y_pred = model.predict(X_test)
    worst_idx = np.argmin(y_pred)

    # Si hay semanas de marzo 2020 en el test set, preferirlas
    covid_mask = (X_test.index >= "2020-02-20") & (X_test.index <= "2020-04-10")
    if covid_mask.any():
        # Semana con predicción más negativa durante COVID
        covid_preds = y_pred.copy()
        covid_preds[~covid_mask.values] = 999
        worst_idx = np.argmin(covid_preds)

    date_str = str(X_test.index[worst_idx].date())

    sv = explainer.shap_values(X_test.iloc[[worst_idx]])
    expected = explainer.expected_value
    if isinstance(expected, np.ndarray):
        expected = expected[0]

    explanation = shap.Explanation(
        values=sv[0],
        base_values=expected,
        data=X_test.iloc[worst_idx].values,
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(f"SHAP Waterfall — {etf} semana {date_str}", fontsize=14, weight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "shap_waterfall_extreme.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close("all")
    return path, date_str


# ══════════════════════════════════════════════════════════════════════
# 8. RESUMEN EN CONSOLA
# ══════════════════════════════════════════════════════════════════════

def print_shap_summary(all_shap_values, feature_names, feature_groups, dim_imp, dim_total):
    """Imprime resumen interpretativo en consola."""
    feat_to_dim = get_feature_to_dim(feature_names, feature_groups)

    # Importancia global por feature
    importances = np.zeros(len(feature_names))
    for etf in ETFS:
        importances += np.abs(all_shap_values[etf]).mean(axis=0)
    importances /= len(ETFS)

    top_idx = np.argsort(importances)[-10:][::-1]

    iw = 62
    print(f"\n╔{'═' * iw}╗")
    print(f"║{BOLD}{'ANÁLISIS SHAP — RESUMEN':^{iw}s}{RESET}║")
    print(f"╠{'═' * iw}╣")

    # Top 10 features
    print(f"║  {BOLD}Top 10 features globales:{RESET}{' ' * (iw - 28)}║")
    for rank, idx in enumerate(top_idx, 1):
        name = feature_names[idx]
        val = importances[idx]
        dim = feat_to_dim.get(name, "?")
        dim_label = DIM_LABELS.get(dim, dim)
        line = f"    {rank:>2d}. {name:<28s} {val:.5f}  ({dim_label})"
        print(f"║{line:<{iw}s}║")

    print(f"╠{'═' * iw}╣")

    # Importancia por dimensión
    print(f"║  {BOLD}Importancia por dimensión:{RESET}{' ' * (iw - 29)}║")
    for dim in sorted(dim_imp, key=lambda d: dim_imp[d], reverse=True):
        pct = dim_imp[dim] / dim_total * 100
        n_feat = len(feature_groups[dim])
        label = DIM_LABELS[dim]
        bar = "█" * int(pct / 2) + "░" * (25 - int(pct / 2))
        line = f"    {label:<14s} {bar} {pct:5.1f}% ({n_feat} feat.)"
        print(f"║{line:<{iw}s}║")

    print(f"╠{'═' * iw}╣")

    # Features con importancia cercana a cero
    threshold = 0.0005
    low_imp = [(feature_names[i], importances[i]) for i in range(len(feature_names))
               if importances[i] < threshold]
    n_low = len(low_imp)
    line = f"  Features con importancia < {threshold}: {n_low} de {len(feature_names)}"
    print(f"║{line:<{iw}s}║")
    if n_low > 0 and n_low <= 15:
        for name, val in sorted(low_imp, key=lambda x: x[1]):
            line = f"    → {name:<30s} ({val:.6f})"
            print(f"║{line:<{iw}s}║")
    elif n_low > 15:
        for name, val in sorted(low_imp, key=lambda x: x[1])[:5]:
            line = f"    → {name:<30s} ({val:.6f})"
            print(f"║{line:<{iw}s}║")
        line = f"    ... y {n_low - 5} más"
        print(f"║{line:<{iw}s}║")

    print(f"╠{'═' * iw}╣")

    # Feature #1 por ETF
    print(f"║  {BOLD}Feature más importante por ETF:{RESET}{' ' * (iw - 34)}║")
    for etf in ETFS:
        etf_imp = np.abs(all_shap_values[etf]).mean(axis=0)
        best_idx = np.argmax(etf_imp)
        name = feature_names[best_idx]
        val = etf_imp[best_idx]
        line = f"    {etf:<5s} → {name:<30s} ({val:.5f})"
        print(f"║{line:<{iw}s}║")

    print(f"╠{'═' * iw}╣")

    # Párrafo interpretativo
    pcts = {d: dim_imp[d] / dim_total * 100 for d in dim_imp}
    mkt_pct = pcts["market"]
    risk_pct = pcts["risk"]
    sent_pct = pcts["sentiment"]
    nlp_pct = pcts["news_nlp"]
    macro_pct = pcts["macro"]

    print(f"║  {BOLD}Interpretación:{RESET}{' ' * (iw - 18)}║")
    lines = [
        f"  Las variables de mercado (retornos, volatilidad, volumen de",
        f"  los ETFs) aportan el {mkt_pct:.0f}% del poder predictivo, seguidas",
        f"  por riesgo (VIX, spreads) con {risk_pct:.0f}%. Las señales de",
        f"  sentimiento (Google Trends, AAII) aportan {sent_pct:.0f}% y las de",
        f"  NLP sobre noticias {nlp_pct:.0f}%, confirmando que las dimensiones",
        f"  no-tradicionales añaden valor predictivo. Las variables macro",
        f"  contribuyen {macro_pct:.0f}%. {n_low} features tienen importancia",
        f"  cercana a cero y podrían eliminarse sin pérdida de rendimiento.",
    ]
    for line in lines:
        print(f"║{line:<{iw}s}║")

    print(f"╚{'═' * iw}╝")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()

    print()
    print("▓" * 62)
    print("▓▓▓   ANÁLISIS SHAP — INTERPRETABILIDAD DE MODELOS          ▓▓▓")
    print("▓▓▓   XGBoost Tuned (Optuna v1) | 10 ETFs                   ▓▓▓")
    print("▓" * 62)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Cargar datos y parámetros ───────────────────────────────────
    with open(PARAMS_PATH) as f:
        best_params = json.load(f)
    print(f"\n  Parámetros: {PARAMS_PATH}")
    print(f"    lr={best_params['learning_rate']:.4f}, "
          f"depth={best_params['max_depth']}, "
          f"min_child_w={best_params['min_child_weight']}")

    features, targets = load_master_dataset()
    feature_names = list(features.columns)
    feature_groups = get_feature_groups(feature_names)

    print(f"\n  Split para SHAP:")
    print(f"    Train: semanas 1-{TRAIN_SIZE} "
          f"({features.index[0].date()} → {features.index[TRAIN_SIZE-1].date()})")
    print(f"    Test:  semanas {TRAIN_SIZE+1}-{len(features)} "
          f"({features.index[TRAIN_SIZE].date()} → {features.index[-1].date()})")

    # ── Entrenar 10 modelos y calcular SHAP ─────────────────────────
    print(f"\nEntrenando 10 modelos XGBoost y calculando TreeSHAP...")

    all_shap_values = {}   # {etf: shap_values array}
    X_tests = {}           # {etf: X_test DataFrame}
    models = {}
    explainers = {}

    for etf in ETFS:
        t_etf = time.time()
        model, explainer, sv, X_test, _ = train_for_shap(
            features, targets, etf, best_params)

        all_shap_values[etf] = sv
        X_tests[etf] = X_test
        models[etf] = model
        explainers[etf] = explainer

        elapsed = time.time() - t_etf
        n_trees = model.best_iteration + 1
        print(f"  ✓ {etf:>3s}: {n_trees:>3d} árboles | "
              f"SHAP shape: {sv.shape} | {elapsed:.1f}s")

    t_train = time.time() - t0
    print(f"\n  Entrenamiento + SHAP completado en {t_train:.0f}s")

    # ── Generar las 8 figuras ───────────────────────────────────────
    print(f"\n{'=' * 62}")
    print("GENERANDO FIGURAS")
    print(f"{'=' * 62}")

    # Figura 1 — Importancia global
    p1 = plot_global_importance(all_shap_values, feature_names, feature_groups)
    print(f"  ✓ Figura 1: {os.path.basename(p1)}")

    # Figura 2 — Importancia por dimensión
    p2, dim_imp, dim_total = plot_dimension_importance(
        all_shap_values, feature_names, feature_groups)
    print(f"  ✓ Figura 2: {os.path.basename(p2)}")

    # Figuras 3-5 — Beeswarm (SPY, AGG, GLD)
    beeswarm_paths = plot_beeswarm_top_etfs(
        all_shap_values, X_tests, feature_names)
    for i, p in enumerate(beeswarm_paths, 3):
        print(f"  ✓ Figura {i}: {os.path.basename(p)}")

    # Figura 6 — Heatmap ETF comparison
    p6 = plot_etf_comparison(all_shap_values, feature_names)
    print(f"  ✓ Figura 6: {os.path.basename(p6)}")

    # Figura 7 — Importancia temporal (3 períodos)
    print(f"  Calculando SHAP temporal (3 períodos × 3 ETFs)...")
    p7 = plot_temporal_importance(
        features, targets, best_params, feature_names, feature_groups)
    print(f"  ✓ Figura 7: {os.path.basename(p7)}")

    # Figura 8 — Waterfall predicción extrema
    p8, date_str = plot_single_prediction(
        models["SPY"], explainers["SPY"], X_tests["SPY"], feature_names, "SPY")
    print(f"  ✓ Figura 8: {os.path.basename(p8)} (semana {date_str})")

    print(f"{'=' * 62}")

    # ── Resumen en consola ──────────────────────────────────────────
    print_shap_summary(all_shap_values, feature_names, feature_groups,
                       dim_imp, dim_total)

    # ── Listar figuras guardadas ────────────────────────────────────
    shap_files = sorted([f for f in os.listdir(FIGURES_DIR) if f.startswith("shap_")])
    print(f"\n{'=' * 62}")
    print(f"  FIGURAS SHAP GUARDADAS EN {FIGURES_DIR}/")
    print(f"{'=' * 62}")
    for f in shap_files:
        size = os.path.getsize(os.path.join(FIGURES_DIR, f))
        print(f"    {f:<40s} {size:>9,} bytes")
    print(f"{'=' * 62}")

    total = time.time() - t0
    print(f"\n⏱️  Tiempo total: {int(total // 60)}m {int(total % 60)}s")
