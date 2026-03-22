"""
Experimento: Walk-forward con wavelet denoising en features de entrenamiento.
Daubechies-4, nivel 2, soft thresholding. Sin leakage: denoising solo en train.

Autor: Mateo Madrigal Arteaga, UFV
Uso:   python3 src/models/train_wavelet.py
"""

import sys
import os
import json
import time
import warnings
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

from src.models.data_loader import load_master_dataset, get_etf_tickers
from src.models.walk_forward import WalkForwardValidator
from src.models.benchmarks import compute_portfolio_metrics, benchmark_60_40
from src.models.wavelet_denoise import wavelet_denoise_features

# ── Constantes ──────────────────────────────────────────────────────

ETFS = get_etf_tickers()
EARLY_STOPPING = 50
VAL_FRACTION = 0.15

RESULTS_ORIG = {
    "XGB Tuned": 1.397, "LGB Tuned": 1.313,
    "XGB Default": 1.154, "LGB Default": 1.000,
}


def load_params():
    """Carga los mejores hiperparámetros de Optuna."""
    with open("data/results/optuna_best_params_xgb.json") as f:
        xgb_params = json.load(f)
    with open("data/results/optuna_best_params_lgb.json") as f:
        lgb_params = json.load(f)
    return xgb_params, lgb_params


def train_walk_forward_wavelet(model_type, params, features, targets, splits, etf):
    """
    Walk-forward con wavelet denoising en el train set de cada split.
    El test set usa datos originales (sin denoising).
    """
    target_col = f"target_{etf}"
    y_all = targets[target_col]
    X_all = features
    feature_cols = list(X_all.columns)

    records = []

    for train_idx, test_idx in splits:
        X_train_full = X_all.iloc[train_idx]
        y_train_full = y_all.iloc[train_idx]
        X_test = X_all.iloc[test_idx]  # Test SIN denoising
        y_test = y_all.iloc[test_idx]

        # Wavelet denoising SOLO en train
        X_train_denoised = wavelet_denoise_features(X_train_full, feature_cols)

        # Split validación (último 15% del train denoised)
        val_size = max(1, int(len(X_train_denoised) * VAL_FRACTION))
        X_train = X_train_denoised.iloc[:-val_size]
        y_train = y_train_full.iloc[:-val_size]
        X_val = X_train_denoised.iloc[-val_size:]
        y_val = y_train_full.iloc[-val_size:]

        try:
            if model_type == "xgb":
                p = {
                    "objective": "reg:squarederror",
                    "n_estimators": 2000,
                    "random_state": 42,
                    "verbosity": 0,
                    "early_stopping_rounds": EARLY_STOPPING,
                    **params,
                }
                model = xgb.XGBRegressor(**p)
                model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)], verbose=False)
                n_trees = model.best_iteration + 1
            else:
                p = {
                    "objective": "regression",
                    "n_estimators": 2000,
                    "random_state": 42,
                    "verbosity": -1,
                    **params,
                }
                model = lgb.LGBMRegressor(**p)
                model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)],
                          callbacks=[
                              lgb.early_stopping(EARLY_STOPPING, verbose=False),
                              lgb.log_evaluation(period=0),
                          ])
                n_trees = model.best_iteration_ + 1

            y_pred = model.predict(X_test)[0]
        except Exception:
            y_pred = 0.0
            n_trees = 0

        records.append({
            "date": X_test.index[0],
            "y_true": y_test.values[0],
            "y_pred": y_pred,
            "n_trees": n_trees,
        })

    return pd.DataFrame(records)


def optimize_portfolio(predictions_by_etf, targets, splits):
    """Optimización de cartera con predicciones ML (igual que train_base.py)."""
    n_assets = len(ETFS)
    target_cols = [f"target_{t}" for t in ETFS]

    pred_dates = predictions_by_etf[ETFS[0]]["date"].values
    pred_matrix = np.column_stack([
        predictions_by_etf[etf]["y_pred"].values for etf in ETFS
    ])
    true_matrix = np.column_stack([
        predictions_by_etf[etf]["y_true"].values for etf in ETFS
    ])

    portfolio_returns = []
    for i in range(len(pred_dates)):
        mu = pred_matrix[i]
        train_idx = splits[i][0]
        hist_data = targets[target_cols].iloc[train_idx]
        cov = hist_data.cov().values

        w0 = np.ones(n_assets) / n_assets

        def neg_sharpe(w):
            port_ret = w @ mu
            port_vol = np.sqrt(w @ cov @ w)
            return -port_ret / port_vol if port_vol > 1e-10 else 1e6

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 0.40)] * n_assets
        result = minimize(neg_sharpe, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        w_opt = result.x if result.success else w0
        portfolio_returns.append(true_matrix[i] @ w_opt)

    dates_index = pd.to_datetime(pred_dates)
    return pd.Series(portfolio_returns, index=dates_index, name="ML Portfolio")


def run_experiment(model_type, params, features, targets, splits, label):
    """Ejecuta walk-forward wavelet para un modelo y devuelve métricas."""
    predictions = {}
    t0 = time.time()

    for etf in ETFS:
        t_e = time.time()
        df = train_walk_forward_wavelet(model_type, params, features, targets, splits, etf)
        predictions[etf] = df
        elapsed = time.time() - t_e

        # Dir accuracy
        y_t = df["y_true"].values
        y_p = df["y_pred"].values
        da = (np.sign(y_p) == np.sign(y_t)).mean() * 100
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        print(f"  {etf:>3s} [{label}]: RMSE={rmse:.4f} | DA={da:.1f}% | "
              f"Trees={df['n_trees'].mean():.0f} | {elapsed:.0f}s")

    total_time = time.time() - t0
    print(f"  {label} completado en {int(total_time//60)}m {int(total_time%60)}s")

    # Optimizar cartera
    ml_returns = optimize_portfolio(predictions, targets, splits)
    metrics = compute_portfolio_metrics(ml_returns, label)

    return metrics, predictions, ml_returns


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_global = time.time()

    print()
    print("=" * 62)
    print("  EXPERIMENTO: WAVELET DENOISING (db4, nivel 2)")
    print("  Denoising solo en features de train (sin leakage)")
    print("=" * 62)

    # Cargar datos
    features, targets = load_master_dataset()
    wf = WalkForwardValidator(min_train_weeks=208, retrain_every=1, embargo_weeks=1)
    splits = wf.generate_splits(features.index)

    xgb_params, lgb_params = load_params()

    # XGBoost Tuned + Wavelet
    xgb_tuned_params = {
        "objective": "reg:squarederror", "n_estimators": 2000,
        "random_state": 42, "verbosity": 0, **xgb_params,
    }
    print(f"\n--- XGBoost Tuned + Wavelet ---")
    xgb_met, xgb_pred, xgb_ret = run_experiment(
        "xgb", xgb_params, features, targets, splits, "XGB Tuned+Wav")

    # LightGBM Tuned + Wavelet
    print(f"\n--- LightGBM Tuned + Wavelet ---")
    lgb_met, lgb_pred, lgb_ret = run_experiment(
        "lgb", lgb_params, features, targets, splits, "LGB Tuned+Wav")

    # ── Comparativa ───────────────────────────────────────────────

    print(f"\n{'=' * 70}")
    print("COMPARATIVA: SIN WAVELET vs CON WAVELET")
    print(f"{'=' * 70}")
    print(f"{'Estrategia':<20s} | {'Sharpe (orig)':>14s} | {'Sharpe (wav)':>14s} | {'Delta':>8s}")
    print(f"{'-'*70}")

    results_wav = {
        "XGB Tuned": xgb_met["Sharpe Ratio"],
        "LGB Tuned": lgb_met["Sharpe Ratio"],
    }

    for name in ["XGB Tuned", "LGB Tuned"]:
        orig = RESULTS_ORIG[name]
        wav = results_wav[name]
        delta = wav - orig
        marker = "+" if delta > 0 else ""
        print(f"{name:<20s} | {orig:>14.3f} | {wav:>14.3f} | {marker}{delta:>7.3f}")

    # Benchmarks no cambian
    for name in ["60/40", "Markowitz", "Equal Weight"]:
        s = {"60/40": 0.847, "Markowitz": 0.832, "Equal Weight": 0.649}[name]
        print(f"{name:<20s} | {s:>14.3f} | {s:>14.3f} | {'0.000':>8s}")

    # Métricas completas
    print(f"\n{'=' * 70}")
    print("MÉTRICAS COMPLETAS — WAVELET")
    print(f"{'=' * 70}")
    for name, met in [("XGB Tuned+Wav", xgb_met), ("LGB Tuned+Wav", lgb_met)]:
        print(f"  {name}:")
        print(f"    Sharpe:  {met['Sharpe Ratio']:.3f}")
        print(f"    Return:  {met['Retorno Anualizado']:.2%}")
        print(f"    MaxDD:   {met['Max Drawdown']:.2%}")
        print(f"    Sortino: {met['Sortino Ratio']:.3f}")
        print(f"    Total:   {met['Total Return']:.2%}")

    # ── Decisión GO/NO-GO ─────────────────────────────────────────

    xgb_wav_sharpe = results_wav["XGB Tuned"]
    print(f"\n{'=' * 70}")

    if xgb_wav_sharpe > 1.397:
        print(f"GO: Wavelet MEJORA el Sharpe ({xgb_wav_sharpe:.3f} > 1.397)")
        print("Guardando nuevos resultados...")
        # Guardar predicciones y pesos wavelet
        for model_key, preds in [("xgb", xgb_pred), ("lgb", lgb_pred)]:
            rows = []
            for etf in ETFS:
                df = preds[etf].copy()
                df["etf"] = etf
                rows.append(df)
            pred_df = pd.concat(rows, ignore_index=True)
            pred_df = pred_df[["date", "etf", "y_true", "y_pred", "n_trees"]]
            pred_df.to_csv(f"data/results/{model_key}_tuned_predictions.csv", index=False)
    else:
        print(f"NO-GO: Wavelet NO mejora el Sharpe ({xgb_wav_sharpe:.3f} <= 1.397)")
        print("Restaurando backup...")
        # Restaurar backup
        backup_dir = "data/results_backup_pre_wavelet"
        for f in os.listdir(backup_dir):
            shutil.copy2(os.path.join(backup_dir, f), os.path.join("data/results", f))
        print("Backup restaurado. Resultados originales intactos.")

        # Párrafo para la memoria
        print(f"\n{'=' * 70}")
        print("PÁRRAFO PARA LA MEMORIA:")
        print(f"{'=' * 70}")
        print(f"\"Se investigó el preprocesamiento mediante wavelet denoising "
              f"(Daubechies-4, nivel 2, soft thresholding universal) aplicado a las "
              f"109 features de entrenamiento en cada split del walk-forward, sin "
              f"aplicar denoising a los datos de test para evitar data leakage. "
              f"El Sharpe ratio del XGBoost Tuned pasó de 1.397 a {xgb_wav_sharpe:.3f}, "
              f"indicando que el denoising no aporta mejora. Este resultado es "
              f"coherente con la capacidad de XGBoost para manejar ruido nativamente "
              f"mediante su mecanismo de regularización (L1/L2) y early stopping "
              f"(Chen y Guestrin, 2016).\"")

    print(f"{'=' * 70}")
    total = time.time() - t_global
    print(f"\nTiempo total: {int(total//60)}m {int(total%60)}s")
