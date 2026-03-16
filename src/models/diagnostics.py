"""
Diagnósticos visuales para evaluación de modelos ML.
Fase 4 — Modelado | TFG Optimización de Carteras con ML
Autor: Mateo Madrigal Arteaga, UFV
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from src.models.data_loader import get_etf_tickers
from src.models.benchmarks import compute_portfolio_metrics

ETFS = get_etf_tickers()

# Colores ANSI
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


# ── BLOQUE 1: Matriz direccional ────────────────────────────────────

def _directional_matrix(y_true, y_pred):
    """Calcula TP, FP, FN, TN de dirección (sube/baja)."""
    real_up = y_true > 0
    pred_up = y_pred > 0

    tp = int(np.sum(real_up & pred_up))
    fn = int(np.sum(real_up & ~pred_up))
    fp = int(np.sum(~real_up & pred_up))
    tn = int(np.sum(~real_up & ~pred_up))

    return tp, fp, fn, tn


def print_directional_matrices(predictions_by_etf, model_name="XGBoost Tuned"):
    """
    Imprime la matriz direccional 2×2 para cada ETF y la agregada.
    predictions_by_etf: dict {etf: DataFrame con y_true, y_pred}
    """
    print(f"\n{'═' * 62}")
    print(f"{BOLD}  MATRICES DIRECCIONALES — {model_name}{RESET}")
    print(f"{'═' * 62}")

    # Acumuladores para la matriz agregada
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    for etf in ETFS:
        df = predictions_by_etf[etf]
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values
        tp, fp, fn, tn = _directional_matrix(y_true, y_pred)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        _print_single_matrix(etf, tp, fp, fn, tn)

    # Matriz agregada
    print(f"\n{'═' * 62}")
    print(f"{BOLD}  MATRIZ AGREGADA — 10 ETFs ({model_name}){RESET}")
    print(f"{'═' * 62}")
    _print_single_matrix("TOTAL", total_tp, total_fp, total_fn, total_tn, show_border=True)


def _print_single_matrix(label, tp, fp, fn, tn, show_border=False):
    """Imprime una matriz 2×2 con métricas."""
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total * 100 if total > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Acierto cuando sube / cuando baja
    up_acc = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    down_acc = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0

    print(f"\n  {label:>5s}   {'Pred ↑':>10s}  {'Pred ↓':>10s}")
    print(f"  Real ↑  {tp:>10d}  {fn:>10d}    ({up_acc:.1f}% acierto cuando sube)")
    print(f"  Real ↓  {fp:>10d}  {tn:>10d}    ({down_acc:.1f}% acierto cuando baja)")
    print(f"  Acc={accuracy:.1f}% | Prec={precision:.1f}% | Recall={recall:.1f}% | F1={f1:.1f}%")

    if show_border:
        print(f"{'═' * 62}")


# ── BLOQUE 2: Diagnóstico de overfitting ────────────────────────────

def print_overfitting_diagnostic(etf_metrics, model_name="XGBoost Tuned"):
    """
    Tabla de overfitting: Train RMSE vs Test RMSE por ETF.
    etf_metrics: dict {etf: dict con RMSE, Avg_Train_RMSE, ...}
    """
    print(f"\n{'═' * 62}")
    print(f"{BOLD}  DIAGNÓSTICO DE OVERFITTING — {model_name}{RESET}")
    print(f"{'═' * 62}")
    print(f"  {'ETF':<6s} {'Train RMSE':>12s} {'Test RMSE':>12s} {'Ratio':>8s}  Estado")
    print(f"  {'─' * 56}")

    ratios = []
    for etf in ETFS:
        m = etf_metrics[etf]
        test_rmse = m["RMSE"]
        train_rmse = m.get("Avg_Train_RMSE", 0)

        if train_rmse > 0 and not np.isnan(train_rmse):
            ratio = test_rmse / train_rmse
        else:
            ratio = float("nan")

        ratios.append(ratio)

        if np.isnan(ratio):
            estado = "  ❓ Sin datos"
            color = YELLOW
        elif ratio > 2.0:
            estado = f"  {RED}❌ OVERFITTING{RESET}"
            color = RED
        elif ratio > 1.5:
            estado = f"  {YELLOW}⚠️  RIESGO{RESET}"
            color = YELLOW
        else:
            estado = f"  {GREEN}✅ OK{RESET}"
            color = GREEN

        ratio_str = f"{ratio:.2f}" if not np.isnan(ratio) else "  N/A"
        print(f"  {etf:<6s} {train_rmse:>12.4f} {test_rmse:>12.4f} {ratio_str:>8s}{estado}")

    valid_ratios = [r for r in ratios if not np.isnan(r)]
    if valid_ratios:
        avg_ratio = np.mean(valid_ratios)
        print(f"  {'─' * 56}")
        print(f"  {'MEDIA':<6s} {'':>12s} {'':>12s} {avg_ratio:>8.2f}", end="")
        if avg_ratio > 1.5:
            print(f"  {YELLOW}⚠️  Riesgo global{RESET}")
        else:
            print(f"  {GREEN}✅ Global OK{RESET}")
    print(f"{'═' * 62}")


# ── BLOQUE 3: Report Card ──────────────────────────────────────────

def _progress_bar(value, max_val, width=14):
    """Genera una barra ████░░░░ proporcional."""
    if max_val <= 0:
        filled = 0
    else:
        filled = int(min(value / max_val, 1.0) * width)
    return "█" * filled + "░" * (width - filled)


def print_report_card(etf_metrics, portfolio_metrics, benchmark_6040_sharpe,
                      model_name="XGBoost Tuned"):
    """
    Report card visual tipo 'nota global' del modelo.
    """
    avg_da = np.mean([m["Dir_Acc"] for m in etf_metrics.values()])
    avg_trees = np.mean([m["Avg_Trees"] for m in etf_metrics.values()])
    sharpe = portfolio_metrics["Sharpe Ratio"]
    max_dd = portfolio_metrics["Max Drawdown"]
    sharpe_diff = sharpe - benchmark_6040_sharpe

    # Ratio overfitting medio
    valid_ratios = []
    for m in etf_metrics.values():
        tr = m.get("Avg_Train_RMSE", 0)
        if tr > 0 and not np.isnan(tr):
            valid_ratios.append(m["RMSE"] / tr)
    avg_ratio = np.mean(valid_ratios) if valid_ratios else float("nan")

    # Evaluar cada métrica
    def eval_da(v):
        if v > 57: return (GREEN, "✅ FUERTE")
        if v > 52: return (YELLOW, "⚠️  MODERADO")
        return (RED, "❌ DÉBIL")

    def eval_sharpe(v):
        if v > 1.2: return (GREEN, "✅ EXCELENTE")
        if v > 0.8: return (GREEN, "✅ BUENO")
        if v > 0.5: return (YELLOW, "⚠️  MODERADO")
        return (RED, "❌ BAJO")

    def eval_dd(v):
        if v > -0.15: return (GREEN, "✅ EXCELENTE")
        if v > -0.20: return (GREEN, "✅ BUENO")
        if v > -0.30: return (YELLOW, "⚠️  MODERADO")
        return (RED, "❌ ALTO")

    def eval_overfit(v):
        if np.isnan(v): return (YELLOW, "❓ SIN DATOS")
        if v < 1.3: return (GREEN, "✅ BAJO")
        if v < 1.5: return (YELLOW, "⚠️  MODERADO")
        return (RED, "❌ ALTO")

    def eval_trees(v):
        if v > 80: return (GREEN, "✅ BIEN")
        if v > 30: return (YELLOW, "⚠️  POCOS")
        return (RED, "❌ MUY POCOS")

    def eval_vs_bench(v):
        if v > 0.15: return (GREEN, "✅ SUPERA")
        if v > 0: return (GREEN, "✅ SUPERA")
        if v > -0.1: return (YELLOW, "⚠️  SIMILAR")
        return (RED, "❌ INFERIOR")

    inner_w = 60
    print(f"\n╔{'═' * inner_w}╗")
    print(f"║{BOLD}{'REPORT CARD — ' + model_name:^{inner_w}s}{RESET}║")
    print(f"╠{'═' * inner_w}╣")

    # Cada línea: nombre, barra, valor, evaluación
    lines = [
        ("Poder predictivo", _progress_bar(avg_da, 100), f"{avg_da:.1f}%", eval_da(avg_da)),
        ("Sharpe ratio", _progress_bar(sharpe, 2.0), f"{sharpe:.3f}", eval_sharpe(sharpe)),
        ("Max drawdown", _progress_bar(abs(max_dd), 0.50), f"{max_dd:.1%}", eval_dd(max_dd)),
        ("Overfitting risk", _progress_bar(avg_ratio if not np.isnan(avg_ratio) else 0, 3.0),
         f"{avg_ratio:.2f}x" if not np.isnan(avg_ratio) else "N/A", eval_overfit(avg_ratio)),
        ("Árboles usados", _progress_bar(avg_trees, 500), f"{avg_trees:.0f} avg", eval_trees(avg_trees)),
        ("vs 60/40", _progress_bar(max(sharpe_diff + 0.5, 0), 1.0),
         f"{sharpe_diff:+.3f}", eval_vs_bench(sharpe_diff)),
    ]

    for name, bar, val_str, (color, status) in lines:
        print(f"║  {name:<20s}{bar}  {val_str:<8s} {color}{status}{RESET}{'':>{inner_w - 20 - 14 - 10 - len(status) - 2}}║")

    # Nota global
    scores = {
        "da": 1 if avg_da > 55 else (0.5 if avg_da > 50 else 0),
        "sharpe": 1 if sharpe > 1.0 else (0.5 if sharpe > 0.7 else 0),
        "dd": 1 if max_dd > -0.20 else (0.5 if max_dd > -0.30 else 0),
        "overfit": 1 if (not np.isnan(avg_ratio) and avg_ratio < 1.5) else 0.5,
        "bench": 1 if sharpe_diff > 0 else (0.5 if sharpe_diff > -0.1 else 0),
    }
    total_score = sum(scores.values()) / len(scores)

    if total_score >= 0.9:
        grade, desc = "A", "excepcional, supera todos los benchmarks"
    elif total_score >= 0.75:
        grade, desc = "A-", "muy bueno, supera benchmarks con solidez"
    elif total_score >= 0.6:
        grade, desc = "B+", "supera benchmarks, margen de mejora"
    elif total_score >= 0.45:
        grade, desc = "B", "competitivo, mejoras necesarias"
    elif total_score >= 0.3:
        grade, desc = "C+", "moderado, necesita tuning significativo"
    else:
        grade, desc = "C", "insuficiente, revisar planteamiento"

    print(f"╠{'═' * inner_w}╣")
    grade_line = f"NOTA GLOBAL:  {grade} ({desc})"
    print(f"║  {BOLD}{grade_line}{RESET}{'':>{inner_w - len(grade_line) - 2}}║")
    print(f"╚{'═' * inner_w}╝")


# ── BLOQUE 4: Evolución temporal ────────────────────────────────────

def print_temporal_evolution(ml_returns, bench_6040_returns, model_name="XGBoost Tuned"):
    """
    Sharpe rolling por año del modelo ML vs 60/40.
    ml_returns: pd.Series con retornos semanales del modelo
    bench_6040_returns: pd.Series con retornos semanales del 60/40
    """
    print(f"\n{'═' * 62}")
    print(f"{BOLD}  EVOLUCIÓN TEMPORAL — Sharpe Anual ML vs 60/40{RESET}")
    print(f"{'═' * 62}")

    # Alinear ambas series por fechas comunes
    common_idx = ml_returns.index.intersection(bench_6040_returns.index)
    ml = ml_returns.loc[common_idx]
    bench = bench_6040_returns.loc[common_idx]

    # Agrupar por año
    ml_yearly = ml.groupby(ml.index.year)
    bench_yearly = bench.groupby(bench.index.year)

    years_ml_wins = 0
    years_total = 0

    print(f"  {'Año':>6s}  {'ML Sharpe':>10s}  {'60/40 Sharpe':>13s}  Resultado")
    print(f"  {'─' * 52}")

    for year in sorted(ml.index.year.unique()):
        ml_y = ml_yearly.get_group(year)
        if year not in bench_yearly.groups:
            continue
        bench_y = bench_yearly.get_group(year)

        # Sharpe anualizado del año
        ml_sharpe = (ml_y.mean() * 52) / (ml_y.std() * np.sqrt(52)) if ml_y.std() > 0 else 0
        bench_sharpe = (bench_y.mean() * 52) / (bench_y.std() * np.sqrt(52)) if bench_y.std() > 0 else 0

        years_total += 1
        if ml_sharpe > bench_sharpe:
            years_ml_wins += 1
            result = f"{GREEN}✅ ML mejor{RESET}"
        else:
            result = f"{RED}❌ ML peor{RESET}"

        print(f"  {year:>6d}  {ml_sharpe:>10.2f}  {bench_sharpe:>13.2f}  {result}")

    print(f"  {'─' * 52}")
    pct = years_ml_wins / years_total * 100 if years_total > 0 else 0
    color = GREEN if pct >= 60 else (YELLOW if pct >= 50 else RED)
    print(f"  Años donde ML gana: {color}{years_ml_wins}/{years_total} ({pct:.0f}%){RESET}")
    print(f"  Años donde ML pierde: {years_total - years_ml_wins}/{years_total}")
    print(f"{'═' * 62}")


# ── Función principal que ejecuta los 4 bloques ─────────────────────

def run_full_diagnostics(best_results, bench_6040_returns, model_name="XGBoost Tuned"):
    """
    Ejecuta los 4 bloques de diagnóstico visual.

    Parámetros:
        best_results: dict con claves:
            - etf_metrics: dict {etf: métricas}
            - portfolio_metrics: dict con métricas de cartera
            - predictions_by_etf: dict {etf: DataFrame}
            - ml_returns: pd.Series con retornos de cartera
        bench_6040_returns: pd.Series con retornos del benchmark 60/40
        model_name: nombre del modelo para los títulos
    """
    print(f"\n\n{'▓' * 62}")
    print(f"{'▓' * 5}{'DIAGNÓSTICOS VISUALES':^52s}{'▓' * 5}")
    print(f"{'▓' * 62}")

    # Bloque 1: Matrices direccionales
    print_directional_matrices(best_results["predictions_by_etf"], model_name)

    # Bloque 2: Overfitting
    print_overfitting_diagnostic(best_results["etf_metrics"], model_name)

    # Bloque 3: Report Card
    bench_sharpe = compute_portfolio_metrics(bench_6040_returns, "60/40")["Sharpe Ratio"]
    print_report_card(
        best_results["etf_metrics"],
        best_results["portfolio_metrics"],
        bench_sharpe,
        model_name,
    )

    # Bloque 4: Evolución temporal
    print_temporal_evolution(best_results["ml_returns"], bench_6040_returns, model_name)
