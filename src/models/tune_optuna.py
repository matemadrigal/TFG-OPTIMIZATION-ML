"""
Hyperparameter tuning con Optuna para XGBoost y LightGBM.
Fase 4 — Capa 3 de Modelado | TFG Optimización de Carteras con ML
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

# Silenciar logs de Optuna y modelos
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
    train_walk_forward, evaluate_predictions, optimize_ml_portfolio,
    XGB_DEFAULT_PARAMS, LGB_DEFAULT_PARAMS, EARLY_STOPPING_ROUNDS, VAL_FRACTION,
)
from src.models.diagnostics import run_full_diagnostics

# ── Constantes ──────────────────────────────────────────────────────

ETFS_TUNING = ["SPY", "AGG", "GLD"]  # Representativos: RV, RF, commodities
ETFS_ALL = get_etf_tickers()
N_ESTIMATORS_TUNING = 2000  # Fijo; early stopping decide cuántos usar


# ── Funciones objetivo para Optuna ──────────────────────────────────

def _walk_forward_rmse(model_type, params, features, targets, splits, etf_ticker):
    """
    Ejecuta walk-forward para un ETF con parámetros dados.
    Retorna RMSE out-of-sample.
    """
    target_col = f"target_{etf_ticker}"
    y_all = targets[target_col]
    X_all = features

    y_trues = []
    y_preds = []

    for train_idx, test_idx in splits:
        X_train_full = X_all.iloc[train_idx]
        y_train_full = y_all.iloc[train_idx]
        X_test = X_all.iloc[test_idx]
        y_test = y_all.iloc[test_idx]

        # Separar validation (último 15%)
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
        except Exception:
            y_pred = 0.0

        y_trues.append(y_test.values[0])
        y_preds.append(y_pred)

    return np.sqrt(mean_squared_error(y_trues, y_preds))


def objective_xgb(trial, features, targets, splits_tuning, etfs_tuning):
    """Función objetivo para Optuna — XGBoost."""
    params = {
        "objective": "reg:squarederror",
        "n_estimators": N_ESTIMATORS_TUNING,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "random_state": 42,
        "verbosity": 0,
    }

    rmses = []
    for i, etf in enumerate(etfs_tuning):
        rmse = _walk_forward_rmse("xgb", params, features, targets, splits_tuning, etf)
        rmses.append(rmse)

        # Pruning: reportar RMSE parcial
        trial.report(np.mean(rmses), i)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(rmses)


def objective_lgb(trial, features, targets, splits_tuning, etfs_tuning):
    """Función objetivo para Optuna — LightGBM."""
    params = {
        "objective": "regression",
        "n_estimators": N_ESTIMATORS_TUNING,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "random_state": 42,
        "verbosity": -1,
    }

    rmses = []
    for i, etf in enumerate(etfs_tuning):
        rmse = _walk_forward_rmse("lgb", params, features, targets, splits_tuning, etf)
        rmses.append(rmse)

        trial.report(np.mean(rmses), i)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(rmses)


# ── Tuning principal ────────────────────────────────────────────────

def run_tuning(model_type, features, targets, n_trials=75):
    """
    Ejecuta el tuning con Optuna para un tipo de modelo.
    Retorna: study, best_params
    """
    model_name = "XGBoost" if model_type == "xgb" else "LightGBM"

    print(f"\n{'=' * 60}")
    print(f"TUNING OPTUNA — {model_name} ({n_trials} trials)")
    print(f"{'=' * 60}")
    print(f"  ETFs de tuning: {', '.join(ETFS_TUNING)}")

    # Splits reducidos (retrain cada 4 semanas)
    wf_tuning = WalkForwardValidator(min_train_weeks=208, retrain_every=4, embargo_weeks=1)
    splits_tuning = wf_tuning.generate_splits(features.index)

    print(f"  Splits por ETF: {len(splits_tuning)}")
    print(f"  Entrenamientos por trial: {len(splits_tuning) * len(ETFS_TUNING)}")

    # Calcular RMSE con defaults como referencia
    print(f"\n  Calculando RMSE con defaults como referencia...")
    default_rmses = []
    for etf in ETFS_TUNING:
        if model_type == "xgb":
            params_def = {**XGB_DEFAULT_PARAMS, "early_stopping_rounds": EARLY_STOPPING_ROUNDS}
        else:
            params_def = dict(LGB_DEFAULT_PARAMS)
        rmse = _walk_forward_rmse(model_type, params_def, features, targets, splits_tuning, etf)
        default_rmses.append(rmse)
        print(f"    {etf}: RMSE={rmse:.6f}")
    default_rmse = np.mean(default_rmses)
    print(f"  RMSE medio defaults: {default_rmse:.6f}")

    # Crear estudio
    if model_type == "xgb":
        objective = lambda trial: objective_xgb(trial, features, targets, splits_tuning, ETFS_TUNING)
    else:
        objective = lambda trial: objective_lgb(trial, features, targets, splits_tuning, ETFS_TUNING)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    # Callback para imprimir progreso
    best_so_far = [float("inf")]
    t0 = time.time()

    def callback(study, trial):
        if trial.value is not None and trial.value < best_so_far[0]:
            best_so_far[0] = trial.value
        elapsed = time.time() - t0
        status = "PRUNED" if trial.state == optuna.trial.TrialState.PRUNED else f"RMSE={trial.value:.6f}" if trial.value else "ERROR"
        print(f"  Trial {trial.number + 1:>3d}/{n_trials} | "
              f"Mejor: {best_so_far[0]:.6f} | {status} | "
              f"{elapsed:.0f}s", flush=True)

    print(f"\n  Iniciando optimización...\n")
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)

    total_time = time.time() - t0

    # Resultados
    best_params = study.best_params
    best_rmse = study.best_value
    improvement = (default_rmse - best_rmse) / default_rmse * 100

    # Imprimir mejores hiperparámetros
    print(f"\n╔{'═' * 54}╗")
    print(f"║{'MEJORES HIPERPARÁMETROS — ' + model_name:^54s}║")
    print(f"╠{'═' * 54}╣")
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f"║  {k:<24s}: {v:>24.6f}  ║")
        else:
            print(f"║  {k:<24s}: {v:>24}  ║")
    print(f"╠{'═' * 54}╣")
    print(f"║  {'RMSE (tuning)':<24s}: {best_rmse:>18.6f}      ║")
    print(f"║  {'RMSE (defaults)':<24s}: {default_rmse:>18.6f}      ║")
    print(f"║  {'Mejora':<24s}: {improvement:>17.2f}%      ║")
    print(f"║  {'Tiempo':<24s}: {total_time:>15.0f}s      ║")
    print(f"║  {'Trials completados':<24s}: {len(study.trials):>24}  ║")
    print(f"║  {'Trials pruned':<24s}: {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED):>24}  ║")
    print(f"╚{'═' * 54}╝")

    # Top 5 trials
    print(f"\n  Top 5 trials:")
    sorted_trials = sorted([t for t in study.trials if t.value is not None],
                           key=lambda t: t.value)[:5]
    for i, t in enumerate(sorted_trials):
        lr = t.params.get("learning_rate", 0)
        depth = t.params.get("max_depth", 0)
        print(f"    #{i+1} Trial {t.number}: RMSE={t.value:.6f} | lr={lr:.4f}, depth={depth}")

    return study, best_params, default_rmse


# ── Reentrenamiento completo con mejores params ─────────────────────

def retrain_with_best_params(best_params_xgb, best_params_lgb,
                             features, targets, splits_full,
                             default_rmse_xgb=None, default_rmse_lgb=None):
    """
    Reentrena ambos modelos con los mejores hiperparámetros sobre el walk-forward completo.
    Compara contra defaults y benchmarks.
    """
    print(f"\n{'=' * 70}")
    print("REENTRENAMIENTO COMPLETO CON HIPERPARÁMETROS OPTIMIZADOS")
    print(f"{'=' * 70}")

    results = {}

    for model_type, best_params in [("xgb", best_params_xgb), ("lgb", best_params_lgb)]:
        model_name = "XGBoost" if model_type == "xgb" else "LightGBM"
        print(f"\n── {model_name} Tuned ──")

        # Construir params completos para el modelo
        if model_type == "xgb":
            full_params = {
                "objective": "reg:squarederror",
                "n_estimators": N_ESTIMATORS_TUNING,
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                "random_state": 42,
                "verbosity": 0,
                **best_params,
            }
        else:
            full_params = {
                "objective": "regression",
                "n_estimators": N_ESTIMATORS_TUNING,
                "random_state": 42,
                "verbosity": -1,
                **best_params,
            }

        # Entrenar walk-forward completo para los 10 ETFs
        etf_metrics = {}
        predictions_by_etf = {}
        t0 = time.time()

        for etf in ETFS_ALL:
            t_etf = time.time()

            # Reutilizar _walk_forward con params tuned
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
                    y_train_pred = model.predict(X_train_full)
                    train_rmse = np.sqrt(mean_squared_error(y_train_full, y_train_pred))
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
            print(f"  ✓ {etf:>3s} [{model_name} Tuned]: "
                  f"RMSE={metrics['RMSE']:.4f} | "
                  f"Dir.Acc={metrics['Dir_Acc']:.1f}% | "
                  f"Árboles={metrics['Avg_Trees']:.0f} | "
                  f"{elapsed:.0f}s")

        total_time = time.time() - t0
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        print(f"\n  ── 10/10 ETFs completados en {minutes}m {seconds}s ──")

        # Optimizar cartera
        print(f"  Optimizando cartera {model_name} Tuned...")
        ml_returns, ml_weights = optimize_ml_portfolio(predictions_by_etf, targets, splits_full)
        portfolio_metrics = compute_portfolio_metrics(ml_returns, f"{model_name} Tuned")

        print(f"  ✓ Cartera: Sharpe={portfolio_metrics['Sharpe Ratio']:.3f} | "
              f"Retorno={portfolio_metrics['Retorno Anualizado']:.2%} | "
              f"MaxDD={portfolio_metrics['Max Drawdown']:.2%}")

        results[model_type] = {
            "etf_metrics": etf_metrics,
            "portfolio_metrics": portfolio_metrics,
            "predictions_by_etf": predictions_by_etf,
            "ml_returns": ml_returns,
            "ml_weights": ml_weights,
        }

    return results


# ── Tabla comparativa final ─────────────────────────────────────────

def print_full_comparison(tuned_results, default_metrics, bench_metrics):
    """
    Imprime tabla comparativa de las 7 estrategias:
    XGB Default, XGB Tuned, LGB Default, LGB Tuned, Markowitz, 60/40, Equal Wt
    """
    all_metrics = {
        "XGB Def": default_metrics["xgb"],
        "XGB Tun": tuned_results["xgb"]["portfolio_metrics"],
        "LGB Def": default_metrics["lgb"],
        "LGB Tun": tuned_results["lgb"]["portfolio_metrics"],
        "Markow.": bench_metrics["Markowitz"],
        "60/40": bench_metrics["60/40"],
        "EqualWt": bench_metrics["Equal Wt"],
    }

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

    header = "║ " + f"{'Métrica':<13s}│"
    for n in names:
        header += f"{n:^{col_w}s}│"
    header = header[:-1] + " ║"
    inner_w = len(header) - 2

    print(f"\n╔{'═' * inner_w}╗")
    print(f"║{'COMPARATIVA: DEFAULTS vs TUNED vs BENCHMARKS':^{inner_w}s}║")
    print(f"╠{'═' * inner_w}╣")
    print(header)
    print(f"╠{'═' * inner_w}╣")

    # Encontrar mejor Sharpe para resaltar
    sharpes = {n: all_metrics[n]["Sharpe Ratio"] for n in names}
    best_sharpe_name = max(sharpes, key=sharpes.get)

    for label, key, fmt in rows_def:
        line = "║ " + f"{label:<13s}│"
        for n in names:
            val = all_metrics[n][key]
            if fmt == "pct":
                s = f"{val:.2%}"
            else:
                s = f"{val:.3f}"

            # Resaltar mejor Sharpe
            if key == "Sharpe Ratio" and n == best_sharpe_name:
                s = f"\033[92m{s}\033[0m"
                line += f"{s:^{col_w + 9}s}│"
            else:
                line += f"{s:^{col_w}s}│"
        line = line[:-1] + " ║"
        print(line)

    print(f"╚{'═' * inner_w}╝")

    return all_metrics


def print_tuning_diagnostics(tuned_results, default_results_xgb, default_results_lgb):
    """Diagnóstico del impacto del tuning."""

    print(f"\n{'=' * 70}")
    print("DIAGNÓSTICO DEL TUNING")
    print(f"{'=' * 70}")

    for model_type, def_res in [("xgb", default_results_xgb), ("lgb", default_results_lgb)]:
        model_name = "XGBoost" if model_type == "xgb" else "LightGBM"
        tuned = tuned_results[model_type]

        sharpe_def = def_res["Sharpe Ratio"]
        sharpe_tun = tuned["portfolio_metrics"]["Sharpe Ratio"]
        sharpe_pct = (sharpe_tun - sharpe_def) / abs(sharpe_def) * 100 if sharpe_def != 0 else 0

        da_def = np.mean([m["Dir_Acc"] for m in default_etf_metrics[model_type].values()])
        da_tun = np.mean([m["Dir_Acc"] for m in tuned["etf_metrics"].values()])

        trees_def_avg = np.mean([m["Avg_Trees"] for m in default_etf_metrics[model_type].values()])
        trees_tun_avg = np.mean([m["Avg_Trees"] for m in tuned["etf_metrics"].values()])

        dd_def = def_res["Max Drawdown"]
        dd_tun = tuned["portfolio_metrics"]["Max Drawdown"]

        print(f"\n  {model_name}:")
        print(f"    Sharpe:     {sharpe_def:.3f} → {sharpe_tun:.3f} ({sharpe_pct:+.1f}%)")
        print(f"    Dir.Acc:    {da_def:.1f}% → {da_tun:.1f}%")
        print(f"    Árboles:    {trees_def_avg:.0f} → {trees_tun_avg:.0f}")
        print(f"    Max DD:     {dd_def:.2%} → {dd_tun:.2%}")

        if sharpe_tun > sharpe_def:
            print(f"    ✅ El tuning ha MEJORADO los resultados")
        elif sharpe_tun == sharpe_def:
            print(f"    ➖ El tuning no cambió los resultados")
        else:
            print(f"    ⚠️  El tuning NO mejoró el Sharpe. Los defaults ya eran buenos. "
                  "Considerar feature engineering adicional.")

    print(f"{'=' * 70}")


# ── Guardar resultados ──────────────────────────────────────────────

def save_tuned_results(best_params_xgb, best_params_lgb, tuned_results, all_metrics):
    """Guarda params, predicciones, pesos y comparativa."""
    os.makedirs("data/results", exist_ok=True)

    # Params como JSON
    with open("data/results/optuna_best_params_xgb.json", "w") as f:
        json.dump(best_params_xgb, f, indent=2)
    with open("data/results/optuna_best_params_lgb.json", "w") as f:
        json.dump(best_params_lgb, f, indent=2)

    # Predicciones y pesos
    for model_key in ["xgb", "lgb"]:
        res = tuned_results[model_key]
        rows = []
        for etf in ETFS_ALL:
            df = res["predictions_by_etf"][etf].copy()
            df["etf"] = etf
            rows.append(df)
        pred_df = pd.concat(rows, ignore_index=True)
        pred_df = pred_df[["date", "etf", "y_true", "y_pred", "n_trees"]]
        pred_df.to_csv(f"data/results/{model_key}_tuned_predictions.csv", index=False)

        res["ml_weights"].to_csv(f"data/results/{model_key}_tuned_weights.csv")

    # Tabla comparativa
    rows = []
    for name, m in all_metrics.items():
        row = {"Cartera": name}
        row.update({k: v for k, v in m.items() if k != "Nombre"})
        rows.append(row)
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv("data/results/portfolio_comparison_tuned.csv", index=False)

    n_pred = len(tuned_results["xgb"]["predictions_by_etf"][ETFS_ALL[0]]) * len(ETFS_ALL)
    n_weeks = len(tuned_results["xgb"]["ml_weights"])

    print(f"\nArchivos guardados en data/results/:")
    print(f"  • optuna_best_params_xgb.json")
    print(f"  • optuna_best_params_lgb.json")
    print(f"  • xgb_tuned_predictions.csv ({n_pred:,} predicciones)")
    print(f"  • lgb_tuned_predictions.csv ({n_pred:,} predicciones)")
    print(f"  • xgb_tuned_weights.csv ({n_weeks} semanas × 10 pesos)")
    print(f"  • lgb_tuned_weights.csv ({n_weeks} semanas × 10 pesos)")
    print(f"  • portfolio_comparison_tuned.csv (tabla resumen)")


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_global = time.time()

    # Cargar datos
    features, targets = load_master_dataset()

    # ── PASO 1: Tuning ──
    study_xgb, best_params_xgb, default_rmse_xgb = run_tuning("xgb", features, targets, n_trials=75)
    study_lgb, best_params_lgb, default_rmse_lgb = run_tuning("lgb", features, targets, n_trials=75)

    # ── PASO 2: Reentrenamiento completo ──
    wf_full = WalkForwardValidator(min_train_weeks=208, retrain_every=1, embargo_weeks=1)
    splits_full = wf_full.generate_splits(features.index)

    tuned_results = retrain_with_best_params(
        best_params_xgb, best_params_lgb,
        features, targets, splits_full,
    )

    # ── PASO 3: Cargar métricas de defaults (de portfolio_comparison.csv) ──
    default_comp = pd.read_csv("data/results/portfolio_comparison.csv")
    default_metrics_portfolio = {}
    default_etf_metrics = {}

    for _, row in default_comp.iterrows():
        name = row["Cartera"]
        default_metrics_portfolio[name] = {
            "Retorno Anualizado": row["Retorno Anualizado"],
            "Volatilidad Anualizada": row["Volatilidad Anualizada"],
            "Sharpe Ratio": row["Sharpe Ratio"],
            "Sortino Ratio": row["Sortino Ratio"],
            "Max Drawdown": row["Max Drawdown"],
            "Calmar Ratio": row["Calmar Ratio"],
            "Total Return": row["Total Return"],
            "Semanas": row["Semanas"],
        }

    # Cargar métricas por ETF de defaults (recalcular desde predicciones guardadas)
    for model_key in ["xgb", "lgb"]:
        pred_df = pd.read_csv(f"data/results/{model_key}_predictions.csv")
        etf_m = {}
        for etf in ETFS_ALL:
            sub = pred_df[pred_df.etf == etf].copy()
            sub.attrs["avg_train_rmse"] = 0  # No disponible desde CSV
            etf_m[etf] = evaluate_predictions(sub)
        default_etf_metrics[model_key] = etf_m

    # ── PASO 4: Benchmarks ──
    print(f"\n{'=' * 60}")
    print("CALCULANDO BENCHMARKS")
    print(f"{'=' * 60}")
    bench_results, _ = compare_benchmarks(targets, min_train_weeks=208)

    # Métricas de benchmarks sobre período ML
    ml_dates = tuned_results["xgb"]["ml_returns"].index
    start_date, end_date = ml_dates[0], ml_dates[-1]
    bench_financial = {
        "Markowitz": compute_portfolio_metrics(
            bench_results["Markowitz"].loc[start_date:end_date], "Markowitz"),
        "60/40": compute_portfolio_metrics(
            bench_results["60/40"].loc[start_date:end_date], "60/40"),
        "Equal Wt": compute_portfolio_metrics(
            bench_results["Equal Weight"].loc[start_date:end_date], "Equal Wt"),
    }

    # Defaults
    default_portfolio_metrics = {
        "xgb": default_metrics_portfolio.get("XGB+Opt", default_metrics_portfolio.get("XGB+Opt", {})),
        "lgb": default_metrics_portfolio.get("LGB+Opt", default_metrics_portfolio.get("LGB+Opt", {})),
    }

    # ── PASO 5: Tabla comparativa final ──
    all_metrics = print_full_comparison(
        tuned_results, default_portfolio_metrics, bench_financial
    )

    # ── PASO 6: Diagnóstico ──
    print_tuning_diagnostics(tuned_results, default_portfolio_metrics["xgb"], default_portfolio_metrics["lgb"])

    # ── PASO 7: Guardar ──
    save_tuned_results(best_params_xgb, best_params_lgb, tuned_results, all_metrics)

    # ── PASO 8: Diagnósticos visuales del mejor modelo ──
    xgb_sharpe = tuned_results["xgb"]["portfolio_metrics"]["Sharpe Ratio"]
    lgb_sharpe = tuned_results["lgb"]["portfolio_metrics"]["Sharpe Ratio"]
    if xgb_sharpe >= lgb_sharpe:
        best = tuned_results["xgb"]
        best_name = "XGBoost Tuned"
    else:
        best = tuned_results["lgb"]
        best_name = "LightGBM Tuned"

    ml_dates = best["ml_returns"].index
    bench_6040 = benchmark_60_40(targets).loc[ml_dates[0]:ml_dates[-1]]
    run_full_diagnostics(best, bench_6040, best_name)

    # Tiempo total
    total = time.time() - t_global
    print(f"\n⏱️  Tiempo total: {int(total // 60)}m {int(total % 60)}s")
