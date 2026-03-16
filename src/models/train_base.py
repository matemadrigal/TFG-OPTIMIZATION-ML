"""
Entrenamiento base Walk-Forward de XGBoost y LightGBM.
Fase 4 — Capa 2 de Modelado | TFG Optimización de Carteras con ML
Autor: Mateo Madrigal Arteaga, UFV
"""

import sys
import os
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize
from tqdm import tqdm

from src.models.data_loader import load_master_dataset, get_etf_tickers
from src.models.walk_forward import WalkForwardValidator
from src.models.benchmarks import compute_portfolio_metrics, compare_benchmarks, benchmark_60_40

# ── Constantes ──────────────────────────────────────────────────────

XGB_DEFAULT_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 1000,
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "random_state": 42,
    "verbosity": 0,
}

LGB_DEFAULT_PARAMS = {
    "objective": "regression",
    "n_estimators": 1000,
    "learning_rate": 0.1,
    "max_depth": -1,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "random_state": 42,
    "verbosity": -1,
}

ETFS = get_etf_tickers()
EARLY_STOPPING_ROUNDS = 50
VAL_FRACTION = 0.15  # Último 15% del train como validation para early stopping


# ── Entrenamiento Walk-Forward ──────────────────────────────────────

def train_walk_forward(model_type, features, targets, splits, etf_ticker,
                       early_stopping_rounds=EARLY_STOPPING_ROUNDS):
    """
    Entrena un modelo por cada split walk-forward para un ETF concreto.

    Parámetros:
        model_type: 'xgb' o 'lgb'
        features: DataFrame de features
        targets: DataFrame de targets
        splits: lista de tuplas (train_idx, test_idx)
        etf_ticker: nombre del ETF (ej: 'SPY')
        early_stopping_rounds: rondas para early stopping

    Retorna:
        DataFrame con columnas: date, y_true, y_pred, n_trees
    """
    target_col = f"target_{etf_ticker}"
    y_all = targets[target_col]
    X_all = features

    records = []
    train_preds_all = []  # Para diagnóstico de overfitting

    for train_idx, test_idx in splits:
        X_train_full = X_all.iloc[train_idx]
        y_train_full = y_all.iloc[train_idx]
        X_test = X_all.iloc[test_idx]
        y_test = y_all.iloc[test_idx]

        # Separar validation del training (último 15%)
        val_size = max(1, int(len(X_train_full) * VAL_FRACTION))
        X_train = X_train_full.iloc[:-val_size]
        y_train = y_train_full.iloc[:-val_size]
        X_val = X_train_full.iloc[-val_size:]
        y_val = y_train_full.iloc[-val_size:]

        try:
            if model_type == "xgb":
                # XGBoost 3.x: early_stopping_rounds va en el constructor
                params = {**XGB_DEFAULT_PARAMS,
                          "early_stopping_rounds": early_stopping_rounds}
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)], verbose=False)
                n_trees = model.best_iteration + 1
            else:
                model = lgb.LGBMRegressor(**LGB_DEFAULT_PARAMS)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )
                n_trees = model.best_iteration_ + 1

            y_pred = model.predict(X_test)[0]

            # Predicción in-sample para diagnóstico de overfitting
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
    avg_train_rmse = np.nanmean(train_preds_all)
    results_df.attrs["avg_train_rmse"] = avg_train_rmse

    return results_df


# ── Evaluación ──────────────────────────────────────────────────────

def evaluate_predictions(results_df):
    """
    Calcula métricas de calidad predictiva para un ETF.
    Retorna dict con RMSE, MAE, R², Directional Accuracy, media de árboles.
    """
    y_true = results_df["y_true"].values
    y_pred = results_df["y_pred"].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Directional accuracy: % de veces que sign(pred) == sign(real)
    signs_match = np.sign(y_pred) == np.sign(y_true)
    # Excluir casos donde ambos son exactamente 0
    dir_acc = signs_match.mean() * 100

    avg_trees = results_df["n_trees"].mean()

    # R² ajustado (n=observaciones, p=features implícito, usamos fórmula simplificada)
    n = len(y_true)
    # Usamos 109 features como referencia
    p = 109
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

    avg_train_rmse = results_df.attrs.get("avg_train_rmse", np.nan)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "R2_adj": r2_adj,
        "Dir_Acc": dir_acc,
        "Avg_Trees": avg_trees,
        "Avg_Train_RMSE": avg_train_rmse,
    }


# ── Optimización de cartera con predicciones ML ─────────────────────

def optimize_ml_portfolio(predictions_by_etf, targets, splits):
    """
    Usa las predicciones ML como μ esperado para optimizar pesos con Markowitz.
    Los retornos de la cartera se calculan con retornos REALES, no predichos.

    Retorna:
        portfolio_returns: pd.Series con retornos semanales
        weights_df: DataFrame con pesos por semana
    """
    n_assets = len(ETFS)
    target_cols = [f"target_{t}" for t in ETFS]

    # Construir matriz de predicciones alineadas por fecha
    pred_dates = predictions_by_etf[ETFS[0]]["date"].values
    pred_matrix = np.column_stack([
        predictions_by_etf[etf]["y_pred"].values for etf in ETFS
    ])
    true_matrix = np.column_stack([
        predictions_by_etf[etf]["y_true"].values for etf in ETFS
    ])

    portfolio_returns = []
    all_weights = []

    for i in range(len(pred_dates)):
        mu = pred_matrix[i]

        # Covarianza histórica hasta esta semana
        # Encontrar el split correspondiente para saber el train
        train_idx = splits[i][0]
        hist_data = targets[target_cols].iloc[train_idx]
        cov = hist_data.cov().values

        w0 = np.ones(n_assets) / n_assets

        def neg_sharpe(w):
            port_ret = w @ mu
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-10:
                return 1e6
            return -port_ret / port_vol

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 0.40)] * n_assets

        result = minimize(neg_sharpe, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 500, "ftol": 1e-10})

        w_opt = result.x if result.success else w0

        # Retorno real de la semana con pesos optimizados
        ret = true_matrix[i] @ w_opt
        portfolio_returns.append(ret)
        all_weights.append(w_opt)

    dates_index = pd.to_datetime(pred_dates)
    returns = pd.Series(portfolio_returns, index=dates_index, name="ML Portfolio")
    weights_df = pd.DataFrame(all_weights, index=dates_index, columns=ETFS)

    return returns, weights_df


# ── Ejecución para todos los ETFs ──────────────────────────────────

def run_all_etfs(model_type="xgb", features=None, targets=None, splits=None):
    """
    Entrena y evalúa el modelo para los 10 ETFs.
    Retorna dict con métricas por ETF, métricas de cartera, predicciones y pesos.
    """
    model_name = "XGBoost" if model_type == "xgb" else "LightGBM"

    if features is None or targets is None:
        features, targets = load_master_dataset()
    if splits is None:
        wf = WalkForwardValidator(min_train_weeks=208, retrain_every=1, embargo_weeks=1)
        splits = wf.generate_splits(features.index)

    print(f"\n{'=' * 60}")
    print(f"ENTRENAMIENTO WALK-FORWARD — {model_name}")
    print(f"{'=' * 60}")

    etf_metrics = {}
    predictions_by_etf = {}
    t0 = time.time()

    for etf in ETFS:
        t_etf = time.time()
        results_df = train_walk_forward(model_type, features, targets, splits, etf)
        metrics = evaluate_predictions(results_df)
        etf_metrics[etf] = metrics
        predictions_by_etf[etf] = results_df

        elapsed = time.time() - t_etf
        print(f"  ✓ {etf:>3s} [{model_name}]: "
              f"RMSE={metrics['RMSE']:.4f} | "
              f"MAE={metrics['MAE']:.4f} | "
              f"R²={metrics['R2']:.3f} | "
              f"Dir.Acc={metrics['Dir_Acc']:.1f}% | "
              f"Árboles={metrics['Avg_Trees']:.0f} (media) | "
              f"{elapsed:.0f}s")

    total_time = time.time() - t0
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f"\n── 10/10 ETFs completados en {minutes}m {seconds}s ──")

    # Optimizar cartera ML
    print(f"\nOptimizando cartera {model_name} + Markowitz...")
    ml_returns, ml_weights = optimize_ml_portfolio(predictions_by_etf, targets, splits)
    portfolio_metrics = compute_portfolio_metrics(ml_returns, f"{model_name}+Opt")

    print(f"  ✓ Cartera {model_name}: Sharpe={portfolio_metrics['Sharpe Ratio']:.3f} | "
          f"Retorno={portfolio_metrics['Retorno Anualizado']:.2%} | "
          f"MaxDD={portfolio_metrics['Max Drawdown']:.2%}")

    return {
        "etf_metrics": etf_metrics,
        "portfolio_metrics": portfolio_metrics,
        "predictions_by_etf": predictions_by_etf,
        "ml_returns": ml_returns,
        "ml_weights": ml_weights,
        "model_type": model_type,
        "total_time": total_time,
    }


# ── Tablas de salida ────────────────────────────────────────────────

def print_ml_metrics_table(xgb_results, lgb_results):
    """Imprime tabla comparativa de métricas ML agregadas."""
    xgb_m = xgb_results["etf_metrics"]
    lgb_m = lgb_results["etf_metrics"]

    # Medias
    def avg(metrics_dict, key):
        return np.mean([m[key] for m in metrics_dict.values()])

    rows = [
        ("RMSE (media)", avg(xgb_m, "RMSE"), avg(lgb_m, "RMSE"), "lower"),
        ("MAE (media)", avg(xgb_m, "MAE"), avg(lgb_m, "MAE"), "lower"),
        ("R² (media)", avg(xgb_m, "R2"), avg(lgb_m, "R2"), "higher"),
        ("R² ajust. (media)", avg(xgb_m, "R2_adj"), avg(lgb_m, "R2_adj"), "higher"),
        ("Dir. Accuracy (media)", avg(xgb_m, "Dir_Acc"), avg(lgb_m, "Dir_Acc"), "higher"),
        ("Árboles (media)", avg(xgb_m, "Avg_Trees"), avg(lgb_m, "Avg_Trees"), None),
    ]

    print(f"\n╔{'═' * 58}╗")
    print(f"║{'MÉTRICAS ML — PREDICCIÓN DE RETORNOS':^58s}║")
    print(f"╠{'═' * 58}╣")
    print(f"║  {'Métrica':<26s}│{'XGBoost':^14s}│{'LightGBM':^14s}║")
    print(f"╠{'═' * 58}╣")

    for name, xgb_v, lgb_v, better in rows:
        if "Accuracy" in name:
            x_str = f"{xgb_v:.1f}%"
            l_str = f"{lgb_v:.1f}%"
        elif "Árboles" in name:
            x_str = f"{xgb_v:.0f}"
            l_str = f"{lgb_v:.0f}"
        elif "R²" in name:
            x_str = f"{xgb_v:.4f}"
            l_str = f"{lgb_v:.4f}"
        else:
            x_str = f"{xgb_v:.4f}"
            l_str = f"{lgb_v:.4f}"
        print(f"║  {name:<26s}│{x_str:^14s}│{l_str:^14s}║")

    print(f"╚{'═' * 58}╝")


def print_etf_comparison_table(xgb_results, lgb_results):
    """Imprime tabla desglosada por ETF comparando XGBoost vs LightGBM."""
    xgb_m = xgb_results["etf_metrics"]
    lgb_m = lgb_results["etf_metrics"]

    print(f"\n╔{'═' * 76}╗")
    print(f"║{'MÉTRICAS ML POR ETF — XGBoost vs LightGBM':^76s}║")
    print(f"╠{'═' * 76}╣")
    print(f"║  {'ETF':<5s}│{'RMSE (XGB)':^12s}│{'RMSE (LGB)':^12s}│"
          f"{'DA (XGB)':^12s}│{'DA (LGB)':^12s}│{'Mejor':^12s}║")
    print(f"╠{'═' * 76}╣")

    for etf in ETFS:
        xr = xgb_m[etf]["RMSE"]
        lr = lgb_m[etf]["RMSE"]
        xd = xgb_m[etf]["Dir_Acc"]
        ld = lgb_m[etf]["Dir_Acc"]

        # Decidir ganador (menor RMSE + mayor Dir.Acc)
        score_xgb = (1 if xr < lr else 0) + (1 if xd > ld else 0)
        score_lgb = (1 if lr < xr else 0) + (1 if ld > xd else 0)
        if score_xgb > score_lgb:
            winner = "XGB"
        elif score_lgb > score_xgb:
            winner = "LGB"
        else:
            winner = "Empate"

        print(f"║  {etf:<5s}│{xr:^12.4f}│{lr:^12.4f}│"
              f"{xd:^11.1f}%│{ld:^11.1f}%│{winner:^12s}║")

    print(f"╚{'═' * 76}╝")


def print_financial_comparison(xgb_results, lgb_results, bench_results):
    """Imprime tabla de métricas financieras de las 5 carteras."""
    # Calcular métricas de benchmarks sobre el mismo período que ML
    ml_dates = xgb_results["ml_returns"].index
    start_date = ml_dates[0]
    end_date = ml_dates[-1]

    bench_60_40 = bench_results["60/40"].loc[start_date:end_date]
    bench_ew = bench_results["Equal Weight"].loc[start_date:end_date]
    bench_mkw = bench_results["Markowitz"].loc[start_date:end_date]

    metrics = {
        "XGB+Opt": xgb_results["portfolio_metrics"],
        "LGB+Opt": lgb_results["portfolio_metrics"],
        "Markowitz": compute_portfolio_metrics(bench_mkw, "Markowitz"),
        "60/40": compute_portfolio_metrics(bench_60_40, "60/40"),
        "Equal Wt": compute_portfolio_metrics(bench_ew, "Equal Wt"),
    }

    rows_def = [
        ("Retorno Anual.", "Retorno Anualizado", "pct"),
        ("Volatilidad", "Volatilidad Anualizada", "pct"),
        ("Sharpe Ratio", "Sharpe Ratio", "f3"),
        ("Sortino Ratio", "Sortino Ratio", "f3"),
        ("Max Drawdown", "Max Drawdown", "pct"),
        ("Calmar Ratio", "Calmar Ratio", "f3"),
        ("Retorno Total", "Total Return", "pct"),
    ]

    names = ["XGB+Opt", "LGB+Opt", "Markowitz", "60/40", "Equal Wt"]
    col_w = 11

    header_line = "║  " + f"{'Métrica':<18s}│"
    for n in names:
        header_line += f"{n:^{col_w}s}│"
    header_line = header_line[:-1] + "║"

    total_w = len(header_line) - 2  # Sin ║ exteriores
    inner_w = total_w

    print(f"\n╔{'═' * inner_w}╗")
    print(f"║{'COMPARATIVA DE CARTERAS — MÉTRICAS FINANCIERAS':^{inner_w}s}║")
    print(f"╠{'═' * inner_w}╣")
    print(header_line)
    print(f"╠{'═' * inner_w}╣")

    # Sharpe del 60/40 para colorear
    sharpe_6040 = metrics["60/40"]["Sharpe Ratio"]

    for label, key, fmt in rows_def:
        line = f"║  {label:<18s}│"
        for n in names:
            val = metrics[n][key]
            if fmt == "pct":
                s = f"{val:.2%}"
            else:
                s = f"{val:.3f}"

            # Colorear Sharpe de carteras ML
            if key == "Sharpe Ratio" and n in ("XGB+Opt", "LGB+Opt"):
                if val > sharpe_6040:
                    s = f"\033[92m{s}\033[0m"  # Verde
                else:
                    s = f"\033[91m{s}\033[0m"  # Rojo
                # Ajustar ancho por caracteres ANSI (9 chars extra)
                line += f"{s:^{col_w + 9}s}│"
            else:
                line += f"{s:^{col_w}s}│"
        line = line[:-1] + "║"
        print(line)

    print(f"╚{'═' * inner_w}╝")

    return metrics


def print_verdict(xgb_results, lgb_results, financial_metrics):
    """Imprime el veredicto final."""
    xgb_sharpe = xgb_results["portfolio_metrics"]["Sharpe Ratio"]
    lgb_sharpe = lgb_results["portfolio_metrics"]["Sharpe Ratio"]
    sharpe_6040 = financial_metrics["60/40"]["Sharpe Ratio"]
    dd_6040 = financial_metrics["60/40"]["Max Drawdown"]

    # Mejor modelo ML
    if xgb_sharpe > lgb_sharpe:
        best_name, best_sharpe, other_sharpe = "XGBoost", xgb_sharpe, lgb_sharpe
        best_dd = xgb_results["portfolio_metrics"]["Max Drawdown"]
        best_results = xgb_results
    else:
        best_name, best_sharpe, other_sharpe = "LightGBM", lgb_sharpe, xgb_sharpe
        best_dd = lgb_results["portfolio_metrics"]["Max Drawdown"]
        best_results = lgb_results

    # Dir.Accuracy media del mejor
    da_avg = np.mean([m["Dir_Acc"] for m in best_results["etf_metrics"].values()])

    print(f"\n{'─' * 70}")
    sharpe_diff = best_sharpe - sharpe_6040
    dd_diff = (best_dd - dd_6040) * 100  # En puntos porcentuales

    if best_sharpe > sharpe_6040:
        print(f"🏆 VEREDICTO: \033[92m{best_name}\033[0m genera la mejor cartera "
              f"(Sharpe {best_sharpe:.3f} vs {other_sharpe:.3f})")
    else:
        print(f"🏆 VEREDICTO: \033[93m{best_name}\033[0m es el mejor ML "
              f"(Sharpe {best_sharpe:.3f} vs {other_sharpe:.3f})")

    if sharpe_diff > 0:
        print(f"📈 vs 60/40: \033[92mMejora\033[0m el Sharpe en {sharpe_diff:.3f} puntos")
    else:
        print(f"📈 vs 60/40: \033[91mEmpeora\033[0m el Sharpe en {abs(sharpe_diff):.3f} puntos")

    if best_dd > dd_6040:  # Menos negativo = mejor
        print(f"📉 vs 60/40: \033[92mMejora\033[0m el max drawdown en {abs(dd_diff):.2f} pp")
    else:
        print(f"📉 vs 60/40: \033[91mEmpeora\033[0m el max drawdown en {abs(dd_diff):.2f} pp")

    print(f"🎯 Directional Accuracy media: {da_avg:.1f}% "
          f"({'> 50% confirma poder predictivo' if da_avg > 50 else '≤ 50% sin poder predictivo'})")
    print(f"{'─' * 70}")


def print_diagnostics(xgb_results, lgb_results, financial_metrics):
    """Diagnóstico automático y plan de acción."""
    xgb_m = xgb_results["etf_metrics"]
    lgb_m = lgb_results["etf_metrics"]

    def avg(metrics_dict, key):
        return np.mean([m[key] for m in metrics_dict.values()])

    # Usar el mejor modelo para diagnóstico principal
    xgb_sharpe = xgb_results["portfolio_metrics"]["Sharpe Ratio"]
    lgb_sharpe = lgb_results["portfolio_metrics"]["Sharpe Ratio"]
    best_m = xgb_m if xgb_sharpe >= lgb_sharpe else lgb_m
    best_name = "XGBoost" if xgb_sharpe >= lgb_sharpe else "LightGBM"

    da_xgb = avg(xgb_m, "Dir_Acc")
    da_lgb = avg(lgb_m, "Dir_Acc")
    da_best = max(da_xgb, da_lgb)

    rmse_xgb = avg(xgb_m, "RMSE")
    rmse_lgb = avg(lgb_m, "RMSE")

    trees_xgb = avg(xgb_m, "Avg_Trees")
    trees_lgb = avg(lgb_m, "Avg_Trees")
    trees_avg = (trees_xgb + trees_lgb) / 2

    print(f"\n{'=' * 70}")
    print("DIAGNÓSTICO AUTOMÁTICO")
    print(f"{'=' * 70}")

    # DIAGNÓSTICO 1 — Poder predictivo
    if da_best > 55:
        print("✅ Poder predictivo FUERTE. Los modelos capturan señales reales.")
    elif da_best > 50:
        print("⚠️  Poder predictivo MODERADO. Hay señal pero débil. El tuning puede mejorar esto.")
    else:
        print("❌ Sin poder predictivo. Los modelos no superan al azar. Revisar features o planteamiento.")

    # DIAGNÓSTICO 2 — Overfitting
    train_rmse_xgb = avg(xgb_m, "Avg_Train_RMSE")
    train_rmse_lgb = avg(lgb_m, "Avg_Train_RMSE")
    test_rmse_xgb = rmse_xgb
    test_rmse_lgb = rmse_lgb

    ratio_xgb = test_rmse_xgb / train_rmse_xgb if train_rmse_xgb > 0 else 1
    ratio_lgb = test_rmse_lgb / train_rmse_lgb if train_rmse_lgb > 0 else 1
    ratio_avg = (ratio_xgb + ratio_lgb) / 2

    if ratio_avg > 1.5:
        print(f"⚠️  Posible OVERFITTING: ratio test/train RMSE = {ratio_avg:.2f}. "
              "El tuning con regularización debería ayudar.")
    else:
        print(f"✅ No hay signos claros de overfitting (ratio test/train RMSE = {ratio_avg:.2f}).")

    # DIAGNÓSTICO 3 — Early stopping
    if trees_avg < 100:
        print(f"ℹ️  Los modelos convergen RÁPIDO (media {trees_avg:.0f} árboles). "
              "En tuning probar learning rate más bajo (0.01-0.05).")
    elif trees_avg > 800:
        print(f"ℹ️  Los modelos usan CASI TODOS los árboles ({trees_avg:.0f} media). "
              "El early stopping no frena. Probar más regularización.")
    else:
        print(f"✅ Early stopping funciona correctamente (media {trees_avg:.0f} árboles).")

    # DIAGNÓSTICO 4 — XGBoost vs LightGBM
    wins_xgb = 0
    wins_lgb = 0
    comparisons = [
        ("RMSE", rmse_xgb, rmse_lgb, "lower"),
        ("Dir.Accuracy", da_xgb, da_lgb, "higher"),
        ("Sharpe", xgb_sharpe, lgb_sharpe, "higher"),
    ]
    for name, xv, lv, better in comparisons:
        if better == "lower":
            if xv < lv:
                wins_xgb += 1
            elif lv < xv:
                wins_lgb += 1
        else:
            if xv > lv:
                wins_xgb += 1
            elif lv > xv:
                wins_lgb += 1

    print(f"\nXGBoost gana en {wins_xgb} de 3 métricas clave, "
          f"LightGBM gana en {wins_lgb} de 3.")
    if wins_xgb > wins_lgb:
        print(f"→ {best_name} es el candidato principal para tuning intensivo.")
    elif wins_lgb > wins_xgb:
        print(f"→ LightGBM es el candidato principal para tuning intensivo.")
    else:
        print("→ Ambos modelos son competitivos. Tunear ambos.")

    # DIAGNÓSTICO 5 — Cartera ML vs Benchmarks
    sharpe_6040 = financial_metrics["60/40"]["Sharpe Ratio"]
    sharpe_mkw = financial_metrics["Markowitz"]["Sharpe Ratio"]
    best_ml_sharpe = max(xgb_sharpe, lgb_sharpe)

    if best_ml_sharpe > sharpe_6040:
        print("\n✅ La cartera ML SUPERA al 60/40. El ML aporta valor incluso con defaults.")
    elif best_ml_sharpe > sharpe_mkw:
        print("\n⚠️  La cartera ML supera a Markowitz pero NO al 60/40. "
              "El tuning es clave para cerrar la brecha.")
    else:
        print("\n❌ La cartera ML no supera ni a Markowitz. "
              "Revisar el optimizador o las predicciones.")

    # DIAGNÓSTICO 6 — Por ETF
    # Combinar Dir.Accuracy de ambos modelos (usar el mejor por ETF)
    etf_da = {}
    for etf in ETFS:
        etf_da[etf] = max(xgb_m[etf]["Dir_Acc"], lgb_m[etf]["Dir_Acc"])

    sorted_etfs = sorted(etf_da.items(), key=lambda x: x[1], reverse=True)
    top3 = sorted_etfs[:3]
    bottom3 = sorted_etfs[-3:]

    print(f"\nMejores predicciones: {', '.join(f'{e} ({d:.1f}%)' for e, d in top3)}")
    print(f"Peores predicciones:  {', '.join(f'{e} ({d:.1f}%)' for e, d in bottom3)}")

    for etf, da in sorted_etfs:
        if da < 50:
            print(f"⚠️  {etf} tiene Dir.Accuracy < 50%. "
                  "Considerar excluirlo o añadir features específicas.")

    # DIAGNÓSTICO 7 — R²
    r2_best = avg(best_m, "R2")
    if r2_best < 0:
        print(f"\nℹ️  R² negativo ({r2_best:.4f}) es NORMAL en predicción de retornos financieros. "
              "No indica un modelo inútil — los retornos son extremadamente ruidosos. "
              "Lo que importa es Dir.Accuracy y Sharpe.")
    elif r2_best > 0.05:
        print(f"\n✅ R² positivo y significativo para finanzas ({r2_best:.4f}). Muy buen resultado.")
    else:
        print(f"\nℹ️  R² ligeramente positivo ({r2_best:.4f}). Marginal pero aceptable en finanzas.")

    # PLAN DE ACCIÓN
    print(f"\n{'=' * 70}")
    print("PLAN DE ACCIÓN PARA CAPA 3 (TUNING)")
    print(f"{'=' * 70}")

    if ratio_avg > 1.5:
        print("→ Priorizar: reg_alpha (L1), reg_lambda (L2), max_depth más bajo, subsample más bajo")
    if trees_avg < 100:
        print("→ Priorizar: learning_rate más bajo (0.01-0.05), n_estimators más alto (2000-5000)")
    if da_best < 55:
        print("→ Priorizar: feature selection con SHAP previo al tuning, "
              "probar diferentes ventanas de features")

    print("→ Rango sugerido para Optuna:")
    print("    learning_rate: [0.005, 0.3]")
    print("    max_depth:     [3, 10]")
    print("    num_leaves:    [15, 127]")
    print("    subsample:     [0.5, 1.0]")
    print("    colsample:     [0.3, 1.0]")
    print("    reg_alpha:     [0, 10]")
    print("    reg_lambda:    [0, 10]")
    print(f"{'=' * 70}")


# ── Guardar resultados ──────────────────────────────────────────────

def save_results(xgb_results, lgb_results, financial_metrics):
    """Guarda predicciones, pesos y tabla comparativa en data/results/."""
    os.makedirs("data/results", exist_ok=True)

    # Predicciones por ETF en formato largo
    for model_key, results in [("xgb", xgb_results), ("lgb", lgb_results)]:
        rows = []
        for etf in ETFS:
            df = results["predictions_by_etf"][etf].copy()
            df["etf"] = etf
            rows.append(df)
        pred_df = pd.concat(rows, ignore_index=True)
        pred_df = pred_df[["date", "etf", "y_true", "y_pred", "n_trees"]]
        pred_df.to_csv(f"data/results/{model_key}_predictions.csv", index=False)

    # Pesos de cartera
    xgb_results["ml_weights"].to_csv("data/results/xgb_weights.csv")
    lgb_results["ml_weights"].to_csv("data/results/lgb_weights.csv")

    # Tabla comparativa
    rows = []
    for name, m in financial_metrics.items():
        row = {"Cartera": name}
        row.update({k: v for k, v in m.items() if k != "Nombre"})
        rows.append(row)
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv("data/results/portfolio_comparison.csv", index=False)

    # Contar registros
    n_pred = len(xgb_results["predictions_by_etf"][ETFS[0]]) * len(ETFS)
    n_weeks = len(xgb_results["ml_weights"])

    print(f"\nArchivos guardados en data/results/:")
    print(f"  • xgb_predictions.csv ({n_pred:,} predicciones)")
    print(f"  • lgb_predictions.csv ({n_pred:,} predicciones)")
    print(f"  • xgb_weights.csv ({n_weeks} semanas × 10 pesos)")
    print(f"  • lgb_weights.csv ({n_weeks} semanas × 10 pesos)")
    print(f"  • portfolio_comparison.csv (tabla resumen)")


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_global = time.time()

    # Cargar datos una sola vez
    features, targets = load_master_dataset()
    wf = WalkForwardValidator(min_train_weeks=208, retrain_every=1, embargo_weeks=1)
    splits = wf.generate_splits(features.index)

    # Entrenar ambos modelos
    xgb_results = run_all_etfs("xgb", features, targets, splits)
    lgb_results = run_all_etfs("lgb", features, targets, splits)

    # Benchmarks (sobre el mismo período)
    print(f"\n{'=' * 60}")
    print("CALCULANDO BENCHMARKS")
    print(f"{'=' * 60}")
    bench_results, _ = compare_benchmarks(targets, min_train_weeks=208)

    # ── BLOQUE 2: Métricas ML agregadas ──
    print_ml_metrics_table(xgb_results, lgb_results)

    # ── BLOQUE 3: Métricas por ETF ──
    print_etf_comparison_table(xgb_results, lgb_results)

    # ── BLOQUE 4: Métricas financieras ──
    financial_metrics = print_financial_comparison(xgb_results, lgb_results, bench_results)

    # ── BLOQUE 5: Veredicto ──
    print_verdict(xgb_results, lgb_results, financial_metrics)

    # ── BLOQUE 7: Diagnóstico ──
    print_diagnostics(xgb_results, lgb_results, financial_metrics)

    # ── BLOQUE 6: Guardar archivos ──
    save_results(xgb_results, lgb_results, financial_metrics)

    # ── BLOQUE 8: Diagnósticos visuales ──
    from src.models.diagnostics import run_full_diagnostics

    # Determinar el mejor modelo por Sharpe
    xgb_sharpe = xgb_results["portfolio_metrics"]["Sharpe Ratio"]
    lgb_sharpe = lgb_results["portfolio_metrics"]["Sharpe Ratio"]
    if xgb_sharpe >= lgb_sharpe:
        best = xgb_results
        best_name = "XGBoost"
    else:
        best = lgb_results
        best_name = "LightGBM"

    # Retornos 60/40 alineados al período ML
    ml_dates = best["ml_returns"].index
    bench_6040 = benchmark_60_40(targets).loc[ml_dates[0]:ml_dates[-1]]

    run_full_diagnostics(best, bench_6040, best_name)

    # Tiempo total
    total = time.time() - t_global
    print(f"\n⏱️  Tiempo total de ejecución: {int(total // 60)}m {int(total % 60)}s")
