"""
Curvas de aprendizaje — XGBoost Tuned vs LightGBM Tuned.
Fase 4 — Modelado | TFG Optimización de Carteras con ML
Autor: Mateo Madrigal Arteaga, UFV
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import os

# ── Configuración ─────────────────────────────────────────────────
ETF = "SPY"
SEED = 42
N_ESTIMATORS = 2000
EARLY_STOPPING = 50
SPLIT_RATIO = 0.85

XGB_PARAMS = dict(
    learning_rate=0.022,
    max_depth=7,
    subsample=0.687,
    colsample_bytree=0.825,
    min_child_weight=20,
    reg_alpha=2.0e-7,
    reg_lambda=1.3e-7,
    n_estimators=N_ESTIMATORS,
    early_stopping_rounds=EARLY_STOPPING,
    eval_metric="rmse",
    random_state=SEED,
)

LGB_PARAMS = dict(
    learning_rate=0.006,
    num_leaves=41,
    max_depth=9,
    subsample=0.581,
    colsample_bytree=0.967,
    reg_alpha=5.4e-4,
    reg_lambda=0.023,
    min_child_samples=33,
    n_estimators=N_ESTIMATORS,
    random_state=SEED,
    verbosity=-1,
)

OUT_DIR = "docs/figures"
OUT_PATH = os.path.join(OUT_DIR, "learning_curves.png")

# ── 1. Cargar datos ───────────────────────────────────────────────
print("Cargando dataset maestro...")
df = pd.read_csv("data/processed/master_weekly_raw.csv", index_col=0, parse_dates=True)

target_col = f"target_{ETF}"
feature_cols = [c for c in df.columns if not c.startswith("target_")]
X = df[feature_cols]
y = df[target_col]

split = int(len(df) * SPLIT_RATIO)
X_train, X_val = X.iloc[:split], X.iloc[split:]
y_train, y_val = y.iloc[:split], y.iloc[split:]

print(f"  Train: {len(X_train)} semanas | Val: {len(X_val)} semanas")
print(f"  ETF representativo: {ETF}")

# ── 2. Entrenar XGBoost con eval_set ─────────────────────────────
print("\nEntrenando XGBoost Tuned...")
xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False,
)

xgb_results = xgb_model.evals_result()
xgb_train_rmse = xgb_results["validation_0"]["rmse"]
xgb_val_rmse = xgb_results["validation_1"]["rmse"]
xgb_best_iter = xgb_model.best_iteration
xgb_best_rmse = xgb_val_rmse[xgb_best_iter]

print(f"  Best iteration: {xgb_best_iter} árboles")
print(f"  Best val RMSE:  {xgb_best_rmse:.6f}")

# ── 3. Entrenar LightGBM con eval_set ────────────────────────────
print("\nEntrenando LightGBM Tuned...")
lgb_model = lgb.LGBMRegressor(**LGB_PARAMS)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric="rmse",
    callbacks=[
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING),
        lgb.log_evaluation(period=0),  # silent
    ],
)

lgb_results = lgb_model._evals_result
lgb_train_rmse = lgb_results["training"]["rmse"]
lgb_val_rmse = lgb_results["valid_1"]["rmse"]
lgb_best_iter = lgb_model.best_iteration_
lgb_best_rmse = lgb_val_rmse[lgb_best_iter - 1]  # LGB is 1-indexed

print(f"  Best iteration: {lgb_best_iter} árboles")
print(f"  Best val RMSE:  {lgb_best_rmse:.6f}")

# ── 4. Figura ─────────────────────────────────────────────────────
print("\nGenerando figura...")

# Rango Y compartido para comparación directa
all_rmse = xgb_train_rmse + xgb_val_rmse + lgb_train_rmse + lgb_val_rmse
y_min = min(all_rmse) * 0.995
y_max = max(all_rmse) * 1.005

# Rango X compartido (máximo de árboles entrenados entre ambos)
x_max = max(len(xgb_train_rmse), len(lgb_train_rmse))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white",
                         sharey=True)

TRAIN_COLOR = "#2563EB"
VAL_COLOR = "#DC2626"
OVERFIT_COLOR = "#FEE2E2"
STOP_COLOR = "#6B7280"

models_data = [
    ("XGBoost Tuned", xgb_train_rmse, xgb_val_rmse, xgb_best_iter, xgb_best_rmse,
     xgb_train_rmse[xgb_best_iter]),
    ("LightGBM Tuned", lgb_train_rmse, lgb_val_rmse, lgb_best_iter, lgb_best_rmse,
     lgb_train_rmse[lgb_best_iter - 1]),
]

for ax, (title, train_rmse, val_rmse, best_iter, best_val_rmse, best_train_rmse) in zip(axes, models_data):
    n_iters = len(train_rmse)
    iters = np.arange(1, n_iters + 1)

    # Zona de overfitting (post-early-stop) sombreada
    ax.axvspan(best_iter, n_iters, color=OVERFIT_COLOR, alpha=0.5, label="Zona de overfitting")

    # Curvas principales
    ax.plot(iters, train_rmse, color=TRAIN_COLOR, linewidth=1.4, label="Train", alpha=0.9)
    ax.plot(iters, val_rmse, color=VAL_COLOR, linewidth=1.4, label="Validation", alpha=0.9)

    # Línea vertical de early stop
    ax.axvline(best_iter, color=STOP_COLOR, linestyle="--", linewidth=1.2, alpha=0.8)

    # Punto exacto de early stop sobre la curva de validation
    ax.plot(best_iter, best_val_rmse, "o", color=VAL_COLOR, markersize=7, zorder=5,
            markeredgecolor="white", markeredgewidth=1.5)

    # Anotación — posición adaptativa para no salirse
    text_x = best_iter * 0.42
    text_y = best_val_rmse + (y_max - y_min) * 0.25
    ax.annotate(
        f"Early stop: {best_iter} árboles\n"
        f"Val RMSE = {best_val_rmse:.6f}\n"
        f"Train RMSE = {best_train_rmse:.6f}",
        xy=(best_iter, best_val_rmse),
        xytext=(text_x, text_y),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="-|>", color=STOP_COLOR, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#D1D5DB",
                  alpha=0.95),
        ha="center",
    )

    ax.set_xlabel("Número de árboles (iteraciones)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_facecolor("white")
    ax.set_xlim(0, x_max * 1.02)
    ax.set_ylim(y_min, y_max)

axes[0].set_ylabel("RMSE", fontsize=10)

fig.suptitle(
    "Curvas de aprendizaje — XGBoost Tuned vs LightGBM Tuned",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()

os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

print(f"\n✓ Figura guardada en {OUT_PATH}")
print(f"  XGBoost: {xgb_best_iter} árboles (val RMSE = {xgb_best_rmse:.6f})")
print(f"  LightGBM: {lgb_best_iter} árboles (val RMSE = {lgb_best_rmse:.6f})")
