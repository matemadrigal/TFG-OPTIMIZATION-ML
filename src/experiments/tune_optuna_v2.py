"""
Hyperparameter tuning v2 — Segunda ronda con Optuna.
Rangos ampliados, params nuevos, semilla de v1, anti-overfitting.
Fase 4 — Capa 3b de Modelado | TFG Optimización de Carteras con ML
Autor: Mateo Madrigal Arteaga, UFV
"""

import sys
import os
import json
import time
import warnings
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
warnings.filterwarnings("ignore")
logging.getLogger("optuna").setLevel(logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error

from src.models.data_loader import load_master_dataset, get_etf_tickers
from src.models.walk_forward import WalkForwardValidator
from src.models.benchmarks import compute_portfolio_metrics, compare_benchmarks, benchmark_60_40
from src.models.train_base import (
    evaluate_predictions, optimize_ml_portfolio,
    EARLY_STOPPING_ROUNDS, VAL_FRACTION,
)
from src.models.diagnostics import run_full_diagnostics

# ── Constantes ──────────────────────────────────────────────────────

ETFS_TUNING = ["SPY", "AGG", "GLD"]
ETFS_ALL = get_etf_tickers()
N_ESTIMATORS_V2 = 3000
N_TRIALS = 100


# ── Walk-forward con check de overfitting ───────────────────────────

def _walk_forward_rmse_v2(model_type, params, features, targets, splits, etf_ticker):
    """
    Walk-forward para un ETF. Retorna (test_rmse, train_rmse) para detección de overfitting.
    """
    target_col = f"target_{etf_ticker}"
    y_all = targets[target_col]
    X_all = features

    y_trues, y_preds = [], []
    train_rmses_split = []

    for train_idx, test_idx in splits:
        X_train_full = X_all.iloc[train_idx]
        y_train_full = y_all.iloc[train_idx]
        X_test = X_all.iloc[test_idx]
        y_test = y_all.iloc[test_idx]

        val_size = max(1, int(len(X_train_full) * VAL_FRACTION))
        X_train = X_train_full.iloc[:-val_size]
        y_train = y_train_full.iloc[:-val_size]
        X_val = X_train_full.iloc[-val_size:]
        y_val = y_train_full.iloc[-val_size:]

        try:
            if model_type == "xgb":
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)], verbose=False)
            else:
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)],
                          callbacks=[
                              lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                              lgb.log_evaluation(period=0),
                          ])
            y_pred = model.predict(X_test)[0]

            # RMSE en train para check de overfitting
            train_pred = model.predict(X_train_full)
            tr_rmse = np.sqrt(mean_squared_error(y_train_full, train_pred))
            train_rmses_split.append(tr_rmse)
        except Exception:
            y_pred = 0.0
            train_rmses_split.append(np.nan)

        y_trues.append(y_test.values[0])
        y_preds.append(y_pred)

    test_rmse = np.sqrt(mean_squared_error(y_trues, y_preds))
    avg_train_rmse = np.nanmean(train_rmses_split)
    return test_rmse, avg_train_rmse


# ── Funciones objetivo v2 ───────────────────────────────────────────

def objective_xgb_v2(trial, features, targets, splits_tuning, etfs_tuning):
    """Función objetivo XGBoost v2 — rangos ampliados + anti-overfitting."""
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_estimators": N_ESTIMATORS_V2,
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "max_bin": trial.suggest_int("max_bin", 128, 512),
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "random_state": 42,
        "verbosity": 0,
    }

    test_rmses, train_rmses = [], []
    for i, etf in enumerate(etfs_tuning):
        test_rmse, train_rmse = _walk_forward_rmse_v2(
            "xgb", params, features, targets, splits_tuning, etf)
        test_rmses.append(test_rmse)
        train_rmses.append(train_rmse)

        # Pruning
        trial.report(np.mean(test_rmses), i)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    avg_test = np.mean(test_rmses)
    avg_train = np.nanmean(train_rmses)
    overfit_ratio = avg_test / avg_train if avg_train > 0 else 1.0

    # Penalizar overfitting
    if overfit_ratio > 2.0:
        return avg_test * 1.5
    return avg_test


def objective_lgb_v2(trial, features, targets, splits_tuning, etfs_tuning):
    """Función objetivo LightGBM v2 — rangos ampliados + anti-overfitting."""
    params = {
        "objective": "regression",
        "n_estimators": N_ESTIMATORS_V2,
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 10, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 5.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
        "random_state": 42,
        "verbosity": -1,
    }

    test_rmses, train_rmses = [], []
    for i, etf in enumerate(etfs_tuning):
        test_rmse, train_rmse = _walk_forward_rmse_v2(
            "lgb", params, features, targets, splits_tuning, etf)
        test_rmses.append(test_rmse)
        train_rmses.append(train_rmse)

        trial.report(np.mean(test_rmses), i)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    avg_test = np.mean(test_rmses)
    avg_train = np.nanmean(train_rmses)
    overfit_ratio = avg_test / avg_train if avg_train > 0 else 1.0

    if overfit_ratio > 2.0:
        return avg_test * 1.5
    return avg_test


# ── Tuning v2 ───────────────────────────────────────────────────────

def run_tuning_v2(model_type, features, targets, n_trials=N_TRIALS):
    """Segunda ronda de tuning con semilla de v1."""
    model_name = "XGBoost" if model_type == "xgb" else "LightGBM"

    print(f"\n{'=' * 62}")
    print(f"TUNING OPTUNA v2 — {model_name} ({n_trials} trials)")
    print(f"{'=' * 62}")

    # Splits reducidos
    wf = WalkForwardValidator(min_train_weeks=208, retrain_every=4, embargo_weeks=1)
    splits_tuning = wf.generate_splits(features.index)
    print(f"  ETFs: {', '.join(ETFS_TUNING)} | Splits: {len(splits_tuning)} | "
          f"Entrenamientos/trial: {len(splits_tuning) * len(ETFS_TUNING)}")

    # Cargar mejores params de v1 como semilla
    v1_path = f"data/results/optuna_best_params_{model_type}.json"
    with open(v1_path) as f:
        best_v1 = json.load(f)
    print(f"  Semilla v1 cargada de: {v1_path}")

    # RMSE de v1 como referencia
    if model_type == "xgb":
        ref_params = {
            "objective": "reg:squarederror", "tree_method": "hist",
            "n_estimators": N_ESTIMATORS_V2,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "random_state": 42, "verbosity": 0, **best_v1,
        }
        # Añadir defaults para params nuevos en v2
        ref_params.setdefault("gamma", 0.0)
        ref_params.setdefault("max_bin", 256)
    else:
        ref_params = {
            "objective": "regression", "n_estimators": N_ESTIMATORS_V2,
            "random_state": 42, "verbosity": -1, **best_v1,
        }
        ref_params.setdefault("min_split_gain", 0.0)
        ref_params.setdefault("subsample_freq", 1)
        ref_params.setdefault("path_smooth", 0.0)

    print(f"\n  Calculando RMSE de v1 como referencia...")
    v1_rmses = []
    for etf in ETFS_TUNING:
        rmse, _ = _walk_forward_rmse_v2(model_type, ref_params, features, targets, splits_tuning, etf)
        v1_rmses.append(rmse)
        print(f"    {etf}: {rmse:.6f}")
    v1_rmse = np.mean(v1_rmses)
    print(f"  RMSE v1: {v1_rmse:.6f}")

    # Crear estudio
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=15),
    )

    # Enqueue semilla v1
    seed_params = dict(best_v1)
    if model_type == "xgb":
        seed_params["gamma"] = 0.0
        seed_params["max_bin"] = 256
    else:
        seed_params["min_split_gain"] = 0.0
        seed_params["subsample_freq"] = 1
        seed_params["path_smooth"] = 0.0
    study.enqueue_trial(seed_params)

    if model_type == "xgb":
        objective = lambda trial: objective_xgb_v2(trial, features, targets, splits_tuning, ETFS_TUNING)
    else:
        objective = lambda trial: objective_lgb_v2(trial, features, targets, splits_tuning, ETFS_TUNING)

    # Callback de progreso
    best_so_far = [float("inf")]
    t0 = time.time()

    def callback(study, trial):
        if trial.value is not None and trial.value < best_so_far[0]:
            best_so_far[0] = trial.value
        elapsed = time.time() - t0
        status = ("PRUNED" if trial.state == optuna.trial.TrialState.PRUNED
                  else f"RMSE={trial.value:.6f}" if trial.value else "ERROR")
        print(f"  Trial {trial.number + 1:>3d}/{n_trials} | "
              f"Mejor: {best_so_far[0]:.6f} | {status} | "
              f"{elapsed:.0f}s", flush=True)

    print(f"\n  Iniciando optimización v2...\n")
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)
    total_time = time.time() - t0

    best_params = study.best_params
    best_rmse = study.best_value
    improvement_vs_v1 = (v1_rmse - best_rmse) / v1_rmse * 100

    # Imprimir resultados
    print(f"\n╔{'═' * 56}╗")
    print(f"║{'MEJORES HIPERPARÁMETROS v2 — ' + model_name:^56s}║")
    print(f"╠{'═' * 56}╣")
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f"║  {k:<26s}: {v:>24.6f}  ║")
        else:
            print(f"║  {k:<26s}: {v:>24}  ║")
    print(f"╠{'═' * 56}╣")
    print(f"║  {'RMSE (v2 best)':<26s}: {best_rmse:>18.6f}        ║")
    print(f"║  {'RMSE (v1 best)':<26s}: {v1_rmse:>18.6f}        ║")
    print(f"║  {'Mejora vs v1':<26s}: {improvement_vs_v1:>17.2f}%        ║")
    print(f"║  {'Tiempo':<26s}: {total_time:>15.0f}s        ║")
    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    print(f"║  {'Trials completados':<26s}: {len(study.trials):>24}  ║")
    print(f"║  {'Trials pruned':<26s}: {n_pruned:>24}  ║")
    print(f"╚{'═' * 56}╝")

    # Top 5
    print(f"\n  Top 5 trials:")
    sorted_trials = sorted([t for t in study.trials if t.value is not None],
                           key=lambda t: t.value)[:5]
    for i, t in enumerate(sorted_trials):
        lr = t.params.get("learning_rate", 0)
        depth = t.params.get("max_depth", 0)
        print(f"    #{i+1} Trial {t.number}: RMSE={t.value:.6f} | lr={lr:.4f}, depth={depth}")

    return study, best_params, v1_rmse, best_rmse


# ── Reentrenamiento completo ────────────────────────────────────────

def retrain_full(model_type, best_params, features, targets, splits_full):
    """Reentrena un modelo con walk-forward completo. Retorna dict de resultados."""
    model_name = "XGBoost" if model_type == "xgb" else "LightGBM"

    if model_type == "xgb":
        full_params = {
            "objective": "reg:squarederror", "tree_method": "hist",
            "n_estimators": N_ESTIMATORS_V2,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "random_state": 42, "verbosity": 0,
            **best_params,
        }
    else:
        full_params = {
            "objective": "regression", "n_estimators": N_ESTIMATORS_V2,
            "random_state": 42, "verbosity": -1,
            **best_params,
        }

    etf_metrics = {}
    predictions_by_etf = {}
    t0 = time.time()

    for etf in ETFS_ALL:
        t_etf = time.time()
        target_col = f"target_{etf}"
        y_all = targets[target_col]
        X_all = features
        records = []
        train_preds_all = []

        for train_idx, test_idx in splits_full:
            X_train_full = X_all.iloc[train_idx]
            y_train_full = y_all.iloc[train_idx]
            X_test = X_all.iloc[test_idx]
            y_test = y_all.iloc[test_idx]

            val_size = max(1, int(len(X_train_full) * VAL_FRACTION))
            X_train = X_train_full.iloc[:-val_size]
            y_train = y_train_full.iloc[:-val_size]
            X_val = X_train_full.iloc[-val_size:]
            y_val = y_train_full.iloc[-val_size:]

            try:
                if model_type == "xgb":
                    model = xgb.XGBRegressor(**full_params)
                    model.fit(X_train, y_train,
                              eval_set=[(X_val, y_val)], verbose=False)
                    n_trees = model.best_iteration + 1
                else:
                    model = lgb.LGBMRegressor(**full_params)
                    model.fit(X_train, y_train,
                              eval_set=[(X_val, y_val)],
                              callbacks=[
                                  lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                                  lgb.log_evaluation(period=0),
                              ])
                    n_trees = model.best_iteration_ + 1

                y_pred = model.predict(X_test)[0]
                train_pred = model.predict(X_train_full)
                train_rmse = np.sqrt(mean_squared_error(y_train_full, train_pred))
                train_preds_all.append(train_rmse)
            except Exception:
                y_pred = 0.0
                n_trees = 0
                train_preds_all.append(np.nan)

            records.append({
                "date": X_test.index[0],
                "y_true": y_test.values[0],
                "y_pred": y_pred,
                "n_trees": n_trees,
            })

        results_df = pd.DataFrame(records)
        results_df.attrs["avg_train_rmse"] = np.nanmean(train_preds_all)
        metrics = evaluate_predictions(results_df)
        etf_metrics[etf] = metrics
        predictions_by_etf[etf] = results_df

        elapsed = time.time() - t_etf
        print(f"  ✓ {etf:>3s} [{model_name} v2]: "
              f"RMSE={metrics['RMSE']:.4f} | "
              f"Dir.Acc={metrics['Dir_Acc']:.1f}% | "
              f"Árboles={metrics['Avg_Trees']:.0f} | "
              f"{elapsed:.0f}s")

    total_time = time.time() - t0
    print(f"\n  ── 10/10 ETFs en {int(total_time // 60)}m {int(total_time % 60)}s ──")

    # Optimizar cartera
    ml_returns, ml_weights = optimize_ml_portfolio(predictions_by_etf, targets, splits_full)
    portfolio_metrics = compute_portfolio_metrics(ml_returns, f"{model_name} v2")

    print(f"  ✓ Cartera: Sharpe={portfolio_metrics['Sharpe Ratio']:.3f} | "
          f"Retorno={portfolio_metrics['Retorno Anualizado']:.2%} | "
          f"MaxDD={portfolio_metrics['Max Drawdown']:.2%}")

    return {
        "etf_metrics": etf_metrics,
        "portfolio_metrics": portfolio_metrics,
        "predictions_by_etf": predictions_by_etf,
        "ml_returns": ml_returns,
        "ml_weights": ml_weights,
    }


# ── Tabla comparativa 9 estrategias ─────────────────────────────────

def print_mega_comparison(all_metrics):
    """Tabla de todas las estrategias."""
    names = list(all_metrics.keys())
    col_w = 10

    rows_def = [
        ("Ret. Anual.", "Retorno Anualizado", "pct"),
        ("Volatilidad", "Volatilidad Anualizada", "pct"),
        ("Sharpe", "Sharpe Ratio", "f3"),
        ("Sortino", "Sortino Ratio", "f3"),
        ("Max DD", "Max Drawdown", "pct"),
        ("Calmar", "Calmar Ratio", "f3"),
        ("Ret. Total", "Total Return", "pct"),
    ]

    header = "║ " + f"{'Métrica':<12s}│"
    for n in names:
        header += f"{n:^{col_w}s}│"
    header = header[:-1] + " ║"
    inner_w = len(header) - 2

    # Mejor Sharpe
    sharpes = {n: m["Sharpe Ratio"] for n, m in all_metrics.items()}
    best_name = max(sharpes, key=sharpes.get)

    print(f"\n╔{'═' * inner_w}╗")
    print(f"║{'COMPARATIVA COMPLETA — TODAS LAS ESTRATEGIAS':^{inner_w}s}║")
    print(f"╠{'═' * inner_w}╣")
    print(header)
    print(f"╠{'═' * inner_w}╣")

    for label, key, fmt in rows_def:
        line = "║ " + f"{label:<12s}│"
        for n in names:
            val = all_metrics[n][key]
            s = f"{val:.2%}" if fmt == "pct" else f"{val:.3f}"
            if key == "Sharpe Ratio" and n == best_name:
                s = f"\033[92m{s}\033[0m"
                line += f"{s:^{col_w + 9}s}│"
            else:
                line += f"{s:^{col_w}s}│"
        line = line[:-1] + " ║"
        print(line)

    print(f"╚{'═' * inner_w}╝")
    return all_metrics


# ── Guardar resultados ──────────────────────────────────────────────

def save_v2_results(best_params_xgb, best_params_lgb,
                    v2_results, v1_sharpes, all_metrics_table):
    """Guarda resultados v2. Solo guarda predicciones/pesos si v2 mejora a v1."""
    os.makedirs("data/results", exist_ok=True)

    summary_lines = []
    summary_lines.append("OPTUNA v2 — RESUMEN DEL TUNING")
    summary_lines.append("=" * 50)

    for model_type, bp in [("xgb", best_params_xgb), ("lgb", best_params_lgb)]:
        model_name = "XGBoost" if model_type == "xgb" else "LightGBM"
        v2_sharpe = v2_results[model_type]["portfolio_metrics"]["Sharpe Ratio"]
        v1_sharpe = v1_sharpes[model_type]

        summary_lines.append(f"\n{model_name}:")
        summary_lines.append(f"  Sharpe v1: {v1_sharpe:.3f}")
        summary_lines.append(f"  Sharpe v2: {v2_sharpe:.3f}")
        summary_lines.append(f"  Mejores params v2: {json.dumps(bp, indent=4)}")

        if v2_sharpe > v1_sharpe:
            summary_lines.append(f"  → v2 MEJORA a v1 (+{v2_sharpe - v1_sharpe:.3f})")

            # Guardar params
            with open(f"data/results/optuna_best_params_{model_type}_v2.json", "w") as f:
                json.dump(bp, f, indent=2)

            # Guardar predicciones y pesos
            res = v2_results[model_type]
            rows = []
            for etf in ETFS_ALL:
                df = res["predictions_by_etf"][etf].copy()
                df["etf"] = etf
                rows.append(df)
            pred_df = pd.concat(rows, ignore_index=True)
            pred_df = pred_df[["date", "etf", "y_true", "y_pred", "n_trees"]]
            pred_df.to_csv(f"data/results/{model_type}_tuned_v2_predictions.csv", index=False)
            res["ml_weights"].to_csv(f"data/results/{model_type}_tuned_v2_weights.csv")
        else:
            summary_lines.append(f"  → v2 NO mejora. Los params de v1 son definitivos.")

    # Guardar summary
    with open("data/results/optuna_v2_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))

    # Guardar tabla comparativa
    rows = []
    for name, m in all_metrics_table.items():
        row = {"Cartera": name}
        row.update({k: v for k, v in m.items() if k != "Nombre"})
        rows.append(row)
    pd.DataFrame(rows).to_csv("data/results/portfolio_comparison_v2.csv", index=False)

    print(f"\nArchivos guardados en data/results/:")
    print(f"  • optuna_v2_summary.txt")
    print(f"  • portfolio_comparison_v2.csv")
    for model_type in ["xgb", "lgb"]:
        v2_sharpe = v2_results[model_type]["portfolio_metrics"]["Sharpe Ratio"]
        v1_sharpe = v1_sharpes[model_type]
        if v2_sharpe > v1_sharpe:
            model_name = "XGBoost" if model_type == "xgb" else "LightGBM"
            print(f"  • optuna_best_params_{model_type}_v2.json")
            print(f"  • {model_type}_tuned_v2_predictions.csv")
            print(f"  • {model_type}_tuned_v2_weights.csv")


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_global = time.time()

    features, targets = load_master_dataset()

    # ── PASO 1: Tuning v2 ──
    study_xgb, bp_xgb, v1_rmse_xgb, v2_rmse_xgb = run_tuning_v2("xgb", features, targets, N_TRIALS)
    study_lgb, bp_lgb, v1_rmse_lgb, v2_rmse_lgb = run_tuning_v2("lgb", features, targets, N_TRIALS)

    # ── PASO 2: Reentrenamiento completo ──
    print(f"\n{'=' * 62}")
    print("REENTRENAMIENTO COMPLETO v2")
    print(f"{'=' * 62}")

    wf_full = WalkForwardValidator(min_train_weeks=208, retrain_every=1, embargo_weeks=1)
    splits_full = wf_full.generate_splits(features.index)

    v2_results = {}
    print(f"\n── XGBoost v2 ──")
    v2_results["xgb"] = retrain_full("xgb", bp_xgb, features, targets, splits_full)
    print(f"\n── LightGBM v2 ──")
    v2_results["lgb"] = retrain_full("lgb", bp_lgb, features, targets, splits_full)

    # ── PASO 3: Cargar resultados previos ──
    # v1 tuned
    v1_comp = pd.read_csv("data/results/portfolio_comparison_tuned.csv")
    v1_metrics = {}
    for _, row in v1_comp.iterrows():
        v1_metrics[row["Cartera"]] = {
            "Retorno Anualizado": row["Retorno Anualizado"],
            "Volatilidad Anualizada": row["Volatilidad Anualizada"],
            "Sharpe Ratio": row["Sharpe Ratio"],
            "Sortino Ratio": row["Sortino Ratio"],
            "Max Drawdown": row["Max Drawdown"],
            "Calmar Ratio": row["Calmar Ratio"],
            "Total Return": row["Total Return"],
            "Semanas": row["Semanas"],
        }

    # Defaults
    def_comp = pd.read_csv("data/results/portfolio_comparison.csv")
    def_metrics = {}
    for _, row in def_comp.iterrows():
        def_metrics[row["Cartera"]] = {
            "Retorno Anualizado": row["Retorno Anualizado"],
            "Volatilidad Anualizada": row["Volatilidad Anualizada"],
            "Sharpe Ratio": row["Sharpe Ratio"],
            "Sortino Ratio": row["Sortino Ratio"],
            "Max Drawdown": row["Max Drawdown"],
            "Calmar Ratio": row["Calmar Ratio"],
            "Total Return": row["Total Return"],
            "Semanas": row["Semanas"],
        }

    # Benchmarks
    print(f"\n{'=' * 60}")
    print("CALCULANDO BENCHMARKS")
    print(f"{'=' * 60}")
    bench_results, _ = compare_benchmarks(targets, min_train_weeks=208)

    ml_dates = v2_results["xgb"]["ml_returns"].index
    start_date, end_date = ml_dates[0], ml_dates[-1]
    bench_fin = {
        "Markowitz": compute_portfolio_metrics(
            bench_results["Markowitz"].loc[start_date:end_date], "Markowitz"),
        "60/40": compute_portfolio_metrics(
            bench_results["60/40"].loc[start_date:end_date], "60/40"),
        "Equal Wt": compute_portfolio_metrics(
            bench_results["Equal Weight"].loc[start_date:end_date], "Equal Wt"),
    }

    # ── PASO 4: Tabla completa ──
    all_strats = {
        "XGB Def": def_metrics.get("XGB+Opt", {}),
        "XGB v1": v1_metrics.get("XGB Tun", v1_metrics.get("XGB+Opt", {})),
        "XGB v2": v2_results["xgb"]["portfolio_metrics"],
        "LGB Def": def_metrics.get("LGB+Opt", {}),
        "LGB v1": v1_metrics.get("LGB Tun", v1_metrics.get("LGB+Opt", {})),
        "LGB v2": v2_results["lgb"]["portfolio_metrics"],
        "Markow.": bench_fin["Markowitz"],
        "60/40": bench_fin["60/40"],
        "EqWt": bench_fin["Equal Wt"],
    }
    print_mega_comparison(all_strats)

    # ── PASO 5: Diagnóstico v1 vs v2 ──
    v1_sharpes = {
        "xgb": v1_metrics.get("XGB Tun", v1_metrics.get("XGB+Opt", {})).get("Sharpe Ratio", 0),
        "lgb": v1_metrics.get("LGB Tun", v1_metrics.get("LGB+Opt", {})).get("Sharpe Ratio", 0),
    }

    print(f"\n{'=' * 62}")
    print("DIAGNÓSTICO v1 vs v2")
    print(f"{'=' * 62}")
    for mt in ["xgb", "lgb"]:
        mn = "XGBoost" if mt == "xgb" else "LightGBM"
        s_v1 = v1_sharpes[mt]
        s_v2 = v2_results[mt]["portfolio_metrics"]["Sharpe Ratio"]
        da_v2 = np.mean([m["Dir_Acc"] for m in v2_results[mt]["etf_metrics"].values()])
        trees_v2 = np.mean([m["Avg_Trees"] for m in v2_results[mt]["etf_metrics"].values()])

        print(f"\n  {mn}:")
        print(f"    Sharpe: v1={s_v1:.3f} → v2={s_v2:.3f} ({s_v2 - s_v1:+.3f})")
        print(f"    Dir.Acc v2: {da_v2:.1f}%")
        print(f"    Árboles v2: {trees_v2:.0f}")

        if s_v2 > s_v1:
            print(f"    ✅ v2 MEJORA a v1 en {s_v2 - s_v1:.3f} puntos de Sharpe")
        elif abs(s_v2 - s_v1) < 0.01:
            print(f"    ➖ v2 ≈ v1. Los params de v1 se mantienen como definitivos.")
        else:
            print(f"    ⚠️  v2 NO mejora. Los params de v1 son los definitivos.")

    print(f"{'=' * 62}")

    # ── PASO 6: Diagnósticos visuales del mejor modelo global ──
    all_sharpes = {
        "XGB v2": v2_results["xgb"]["portfolio_metrics"]["Sharpe Ratio"],
        "LGB v2": v2_results["lgb"]["portfolio_metrics"]["Sharpe Ratio"],
        "XGB v1": v1_sharpes["xgb"],
        "LGB v1": v1_sharpes["lgb"],
    }
    best_key = max(all_sharpes, key=all_sharpes.get)
    if "XGB" in best_key:
        best_mt = "xgb"
    else:
        best_mt = "lgb"
    if "v2" in best_key:
        best_res = v2_results[best_mt]
    else:
        # v1 es mejor, pero solo tenemos v2 results en memoria — usar v2 para diagnóstico
        best_res = v2_results[best_mt]

    bench_6040 = benchmark_60_40(targets).loc[start_date:end_date]
    run_full_diagnostics(best_res, bench_6040, best_key)

    # ── PASO 7: Guardar ──
    save_v2_results(bp_xgb, bp_lgb, v2_results, v1_sharpes, all_strats)

    total = time.time() - t_global
    print(f"\n⏱️  Tiempo total: {int(total // 60)}m {int(total % 60)}s")
