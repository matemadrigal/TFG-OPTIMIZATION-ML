"""
Benchmarks de carteras para comparar contra el modelo ML.
Fase 4 — Modelado | TFG Optimización de Carteras con ML
Autor: Mateo Madrigal Arteaga, UFV
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

from src.models.data_loader import load_master_dataset, get_etf_tickers


# ── Benchmarks ──────────────────────────────────────────────────────

def benchmark_60_40(targets):
    """
    Cartera clásica 60/40: 60% SPY + 40% AGG.
    Retorna pd.Series con el retorno semanal de la cartera.
    """
    returns = 0.6 * targets["target_SPY"] + 0.4 * targets["target_AGG"]
    returns.name = "60/40"
    return returns


def benchmark_equal_weight(targets):
    """
    Cartera equiponderada: 10% en cada uno de los 10 ETFs.
    Retorna pd.Series con el retorno semanal (media de los 10 targets).
    """
    returns = targets.mean(axis=1)
    returns.name = "Equal Weight"
    return returns


def benchmark_markowitz(targets, min_train_weeks=208):
    """
    Walk-forward con Markowitz clásico (maximización de Sharpe ratio).

    Para cada semana t >= min_train_weeks:
        - Estima μ y Σ con datos históricos [0:t]
        - Optimiza pesos: max Sharpe, long-only, max 40% por activo
        - Calcula retorno de la semana t con esos pesos

    Retorna:
        returns: pd.Series con retornos semanales de la cartera
        weights_df: DataFrame con pesos óptimos por semana
    """
    tickers = get_etf_tickers()
    target_cols = [f"target_{t}" for t in tickers]
    n_assets = len(tickers)
    data = targets[target_cols]

    # Almacenar resultados
    portfolio_returns = []
    portfolio_dates = []
    all_weights = []

    print(f"\nOptimizando Markowitz walk-forward ({len(data) - min_train_weeks} semanas)...")

    for t in tqdm(range(min_train_weeks, len(data)), desc="  Markowitz"):
        # Datos históricos hasta t (sin incluir t)
        hist = data.iloc[:t]
        mu = hist.mean().values
        cov = hist.cov().values

        # Pesos iniciales: equiponderados
        w0 = np.ones(n_assets) / n_assets

        # Función objetivo: Sharpe negativo (minimizar)
        def neg_sharpe(w):
            port_ret = w @ mu
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-10:
                return 1e6
            return -port_ret / port_vol

        # Restricciones y límites
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 0.40)] * n_assets

        result = minimize(neg_sharpe, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 500, "ftol": 1e-10})

        w_opt = result.x if result.success else w0

        # Retorno de la semana t con los pesos optimizados
        ret_t = data.iloc[t].values @ w_opt
        portfolio_returns.append(ret_t)
        portfolio_dates.append(data.index[t])
        all_weights.append(w_opt)

    returns = pd.Series(portfolio_returns, index=portfolio_dates, name="Markowitz")
    weights_df = pd.DataFrame(all_weights, index=portfolio_dates, columns=tickers)

    return returns, weights_df


# ── Métricas ────────────────────────────────────────────────────────

def compute_portfolio_metrics(returns, name="Portfolio", annual_factor=52):
    """
    Calcula métricas de rendimiento para una serie de retornos semanales (log-returns).

    Retorna un dict con las métricas principales.
    """
    mean_ret = returns.mean()
    std_ret = returns.std()

    # Retorno y volatilidad anualizados
    ann_ret = mean_ret * annual_factor
    ann_vol = std_ret * np.sqrt(annual_factor)

    # Sharpe ratio (rf = 0)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Sortino ratio (solo downside deviation)
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 0.0
    ann_downside = downside_std * np.sqrt(annual_factor)
    sortino = ann_ret / ann_downside if ann_downside > 0 else 0.0

    # Max drawdown sobre curva de equity acumulada
    equity = np.exp(returns.cumsum())
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    # Calmar ratio
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0 else 0.0

    # Retorno total acumulado
    total_ret = np.exp(returns.sum()) - 1

    return {
        "Nombre": name,
        "Retorno Anualizado": ann_ret,
        "Volatilidad Anualizada": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
        "Total Return": total_ret,
        "Semanas": len(returns),
    }


# ── Comparación ─────────────────────────────────────────────────────

def compare_benchmarks(targets, min_train_weeks=208):
    """
    Ejecuta los 3 benchmarks sobre el período de test (desde min_train_weeks)
    y compara sus métricas en una tabla formateada.

    Retorna:
        results: dict con las series de retornos de cada benchmark
        metrics_df: DataFrame con las métricas comparativas
    """
    # Recortar targets al período de test (mismo rango para todos)
    test_targets = targets.iloc[min_train_weeks:]

    print(f"\nPeriodo de evaluación: {test_targets.index[0].date()} → {test_targets.index[-1].date()}")
    print(f"Semanas de test: {len(test_targets)}\n")

    # Calcular benchmarks
    ret_60_40 = benchmark_60_40(test_targets)
    ret_ew = benchmark_equal_weight(test_targets)
    ret_mkw, weights_mkw = benchmark_markowitz(targets, min_train_weeks)

    # Métricas
    metrics = [
        compute_portfolio_metrics(ret_60_40, "60/40"),
        compute_portfolio_metrics(ret_ew, "Equal Weight"),
        compute_portfolio_metrics(ret_mkw, "Markowitz"),
    ]
    metrics_df = pd.DataFrame(metrics).set_index("Nombre")

    # Imprimir tabla
    print("\n" + "=" * 80)
    print("COMPARACIÓN DE BENCHMARKS")
    print("=" * 80)
    print(f"{'Métrica':<25s} {'60/40':>14s} {'Equal Weight':>14s} {'Markowitz':>14s}")
    print("-" * 80)

    fmt = {
        "Retorno Anualizado": "{:>14.2%}",
        "Volatilidad Anualizada": "{:>14.2%}",
        "Sharpe Ratio": "{:>14.3f}",
        "Sortino Ratio": "{:>14.3f}",
        "Max Drawdown": "{:>14.2%}",
        "Calmar Ratio": "{:>14.3f}",
        "Total Return": "{:>14.2%}",
        "Semanas": "{:>14.0f}",
    }

    for metric_name, f in fmt.items():
        vals = [f.format(metrics_df.loc[name, metric_name])
                for name in ["60/40", "Equal Weight", "Markowitz"]]
        print(f"{metric_name:<25s} {''.join(vals)}")

    print("=" * 80)

    # Pesos medios de Markowitz
    print(f"\nPesos medios Markowitz:")
    mean_w = weights_mkw.mean()
    for ticker, w in mean_w.items():
        bar = "█" * int(w * 50)
        print(f"  {ticker:>4s}: {w:6.1%} {bar}")

    results = {
        "60/40": ret_60_40,
        "Equal Weight": ret_ew,
        "Markowitz": ret_mkw,
        "Markowitz Weights": weights_mkw,
    }

    return results, metrics_df


# ── Ejecución directa ──────────────────────────────────────────────

if __name__ == "__main__":
    features, targets = load_master_dataset()
    results, metrics_df = compare_benchmarks(targets, min_train_weeks=208)
