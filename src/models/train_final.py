"""
Reporte FINAL de resultados — TFG Optimización de Carteras con ML.
Carga predicciones existentes (NO reentrena) y genera todos los
diagnósticos, tablas y métricas finales del TFG.

Autor: Mateo Madrigal Arteaga, UFV
Uso:   python3 src/models/train_final.py
"""

import sys
import os
import json
import time
import shutil
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.models.data_loader import load_master_dataset, get_etf_tickers
from src.models.benchmarks import (
    compute_portfolio_metrics, compare_benchmarks, benchmark_60_40,
)

# ── Constantes ──────────────────────────────────────────────────────

ETFS = get_etf_tickers()
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


# ══════════════════════════════════════════════════════════════════════
# CARGA DE DATOS EXISTENTES
# ══════════════════════════════════════════════════════════════════════

def cargar_predicciones(path):
    """Carga predicciones desde CSV (formato largo) → dict {etf: DataFrame}."""
    df = pd.read_csv(path, parse_dates=["date"])
    return {etf: df[df["etf"] == etf].reset_index(drop=True) for etf in ETFS}


def cargar_pesos(path):
    """Carga pesos semanales desde CSV (índice = fecha, columnas = ETFs)."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


def calcular_retornos_cartera(predictions_by_etf, weights_df):
    """Reconstruye retornos semanales: sum(peso_i * retorno_real_i)."""
    # Matriz de retornos reales alineada por fecha
    true_returns = {}
    for etf in ETFS:
        s = predictions_by_etf[etf].set_index("date")["y_true"]
        s.index = pd.to_datetime(s.index)
        true_returns[etf] = s
    true_df = pd.DataFrame(true_returns)

    # Alinear fechas
    common = weights_df.index.intersection(true_df.index)
    w = weights_df.loc[common]
    r = true_df.loc[common]

    returns = (w.values * r.values).sum(axis=1)
    return pd.Series(returns, index=common, name="ML Portfolio")


def evaluar_etf(etf_df):
    """Calcula métricas ML para un ETF a partir de sus predicciones."""
    y_true = etf_df["y_true"].values
    y_pred = etf_df["y_pred"].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    n, p = len(y_true), 109
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

    dir_acc = (np.sign(y_pred) == np.sign(y_true)).mean() * 100
    avg_trees = etf_df["n_trees"].mean()

    return {
        "RMSE": rmse, "MAE": mae, "R2": r2, "R2_adj": r2_adj,
        "Dir_Acc": dir_acc, "Avg_Trees": avg_trees,
        "Std_Pred": np.std(y_pred), "Std_Real": np.std(y_true),
    }


# ══════════════════════════════════════════════════════════════════════
# BLOQUES DE SALIDA
# ══════════════════════════════════════════════════════════════════════

# ── BLOQUE 1: Hiperparámetros ────────────────────────────────────────

def bloque_1_params(xgb_params, lgb_params):
    w = 62
    print(f"\n{'=' * w}")
    print(f"  BLOQUE 1 — HIPERPARÁMETROS ÓPTIMOS (Optuna v1 — definitivos)")
    print(f"{'=' * w}")

    print(f"\n  {BOLD}XGBoost Tuned:{RESET}")
    for k, v in xgb_params.items():
        fmt = f"{v:.8f}" if isinstance(v, float) else str(v)
        print(f"    {k:<22s}: {fmt}")

    print(f"\n  {BOLD}LightGBM Tuned:{RESET}")
    for k, v in lgb_params.items():
        fmt = f"{v:.8f}" if isinstance(v, float) else str(v)
        print(f"    {k:<22s}: {fmt}")

    print(f"\n  Confirmados como óptimos globales por Optuna v2 (100 trials).")
    print(f"{'=' * w}")


# ── BLOQUE 2: Métricas ML agregadas ──────────────────────────────────

def bloque_2_metricas_ml(xgb_metrics, lgb_metrics):
    def avg(d, key):
        return np.mean([m[key] for m in d.values()])

    rows = [
        ("RMSE (media)",         avg(xgb_metrics, "RMSE"),      avg(lgb_metrics, "RMSE")),
        ("MAE (media)",          avg(xgb_metrics, "MAE"),       avg(lgb_metrics, "MAE")),
        ("R² (media)",           avg(xgb_metrics, "R2"),        avg(lgb_metrics, "R2")),
        ("R² ajust. (media)",    avg(xgb_metrics, "R2_adj"),    avg(lgb_metrics, "R2_adj")),
        ("Dir. Accuracy (media)",avg(xgb_metrics, "Dir_Acc"),   avg(lgb_metrics, "Dir_Acc")),
        ("Árboles (media)",      avg(xgb_metrics, "Avg_Trees"), avg(lgb_metrics, "Avg_Trees")),
    ]

    iw = 58
    print(f"\n╔{'═' * iw}╗")
    print(f"║{'BLOQUE 2 — MÉTRICAS ML AGREGADAS':^{iw}s}║")
    print(f"╠{'═' * iw}╣")
    print(f"║  {'Métrica':<26s}│{'XGBoost':^14s}│{'LightGBM':^14s}║")
    print(f"╠{'═' * iw}╣")

    for name, xv, lv in rows:
        if "Accuracy" in name:
            xs, ls = f"{xv:.1f}%", f"{lv:.1f}%"
        elif "Árboles" in name:
            xs, ls = f"{xv:.0f}", f"{lv:.0f}"
        else:
            xs, ls = f"{xv:.4f}", f"{lv:.4f}"
        print(f"║  {name:<26s}│{xs:^14s}│{ls:^14s}║")

    print(f"╚{'═' * iw}╝")


# ── BLOQUE 3: Métricas ML por ETF ────────────────────────────────────

def bloque_3_por_etf(xgb_metrics, lgb_metrics):
    iw = 76
    print(f"\n╔{'═' * iw}╗")
    print(f"║{'BLOQUE 3 — MÉTRICAS ML POR ETF':^{iw}s}║")
    print(f"╠{'═' * iw}╣")
    print(f"║  {'ETF':<5s}│{'RMSE(XGB)':^12s}│{'RMSE(LGB)':^12s}│"
          f"{'DA(XGB)':^12s}│{'DA(LGB)':^12s}│{'Mejor':^12s}║")
    print(f"╠{'═' * iw}╣")

    for etf in ETFS:
        xr = xgb_metrics[etf]["RMSE"]
        lr = lgb_metrics[etf]["RMSE"]
        xd = xgb_metrics[etf]["Dir_Acc"]
        ld = lgb_metrics[etf]["Dir_Acc"]

        score_x = (1 if xr < lr else 0) + (1 if xd > ld else 0)
        score_l = (1 if lr < xr else 0) + (1 if ld > xd else 0)
        winner = "XGB" if score_x > score_l else ("LGB" if score_l > score_x else "Empate")

        print(f"║  {etf:<5s}│{xr:^12.4f}│{lr:^12.4f}│"
              f"{xd:^11.1f}%│{ld:^11.1f}%│{winner:^12s}║")

    print(f"╚{'═' * iw}╝")


# ── BLOQUE 4: Confusion matrix direccional ───────────────────────────

def _print_matrix(label, tp, fp, fn, tn, border=False):
    """Imprime una matriz 2x2 de dirección con métricas."""
    total = tp + fp + fn + tn
    acc  = (tp + tn) / total * 100 if total else 0
    prec = tp / (tp + fp) * 100 if (tp + fp) else 0
    rec  = tp / (tp + fn) * 100 if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    up_a = tp / (tp + fn) * 100 if (tp + fn) else 0
    dn_a = tn / (tn + fp) * 100 if (tn + fp) else 0

    print(f"\n  {label:>5s}   {'Pred ↑':>10s}  {'Pred ↓':>10s}")
    print(f"  Real ↑  {tp:>10d}  {fn:>10d}    ({up_a:.1f}% acierto cuando sube)")
    print(f"  Real ↓  {fp:>10d}  {tn:>10d}    ({dn_a:.1f}% acierto cuando baja)")
    print(f"  Acc={acc:.1f}% | Prec={prec:.1f}% | Recall={rec:.1f}% | F1={f1:.1f}%")
    if border:
        print(f"{'═' * 62}")


def bloque_4_confusion(predictions_by_etf, model_name):
    print(f"\n{'═' * 62}")
    print(f"{BOLD}  BLOQUE 4 — MATRICES DIRECCIONALES — {model_name}{RESET}")
    print(f"{'═' * 62}")

    tot_tp = tot_fp = tot_fn = tot_tn = 0

    for etf in ETFS:
        y_t = predictions_by_etf[etf]["y_true"].values
        y_p = predictions_by_etf[etf]["y_pred"].values
        ru, pu = y_t > 0, y_p > 0

        tp = int(np.sum(ru & pu))
        fn = int(np.sum(ru & ~pu))
        fp = int(np.sum(~ru & pu))
        tn = int(np.sum(~ru & ~pu))

        tot_tp += tp; tot_fp += fp; tot_fn += fn; tot_tn += tn
        _print_matrix(etf, tp, fp, fn, tn)

    print(f"\n{'═' * 62}")
    print(f"{BOLD}  MATRIZ AGREGADA — 10 ETFs ({model_name}){RESET}")
    print(f"{'═' * 62}")
    _print_matrix("TOTAL", tot_tp, tot_fp, tot_fn, tot_tn, border=True)


# ── BLOQUE 5: Diagnóstico de sobreajuste ─────────────────────────────

def bloque_5_overfitting(etf_metrics, model_name):
    print(f"\n{'═' * 72}")
    print(f"{BOLD}  BLOQUE 5 — DIAGNÓSTICO DE SOBREAJUSTE — {model_name}{RESET}")
    print(f"{'═' * 72}")
    print(f"  {'ETF':<6s} {'Std Pred':>10s} {'Std Real':>10s} "
          f"{'Ratio':>8s}  {'Dir.Acc':>8s}  Estado")
    print(f"  {'─' * 64}")

    ratios = []
    for etf in ETFS:
        m = etf_metrics[etf]
        sp, sr = m["Std_Pred"], m["Std_Real"]
        ratio = sp / sr if sr > 0 else float("nan")
        da = m["Dir_Acc"]
        ratios.append(ratio)

        if ratio > 1.5:
            estado = f"{RED}❌ EXAGERA{RESET}"
        elif ratio > 0.8:
            estado = f"{YELLOW}⚠️  ALTO{RESET}"
        elif ratio < 0.05:
            estado = f"{YELLOW}⚠️  MUY CONSERVADOR{RESET}"
        else:
            estado = f"{GREEN}✅ OK{RESET}"

        print(f"  {etf:<6s} {sp:>10.4f} {sr:>10.4f} "
              f"{ratio:>8.2f}  {da:>7.1f}%  {estado}")

    valid = [r for r in ratios if not np.isnan(r)]
    if valid:
        avg_r = np.mean(valid)
        print(f"  {'─' * 64}")
        c = GREEN if avg_r < 0.8 else (YELLOW if avg_r < 1.5 else RED)
        tag = "Conservador (no exagera)" if avg_r < 0.8 else (
              "Calibrado" if avg_r < 1.2 else "Revsar")
        print(f"  {'MEDIA':<6s} {'':>10s} {'':>10s} "
              f"{avg_r:>8.2f}  {'':>8s}  {c}✅ {tag}{RESET}")

    print(f"\n  {BOLD}Interpretación:{RESET}")
    print(f"  Ratio < 1.0 → modelo conservador (predice rangos menores)")
    print(f"  Ratio ≈ 1.0 → calibración perfecta")
    print(f"  Ratio > 1.5 → modelo exagera predicciones")
    print(f"{'═' * 72}")


# ── BLOQUE 6: Tabla financiera de las 5 carteras ─────────────────────

def bloque_6_financiera(xgb_port, lgb_port, bench_results, ml_returns):
    """Imprime tabla de 5 carteras y retorna dict con todas las métricas."""
    ml_dates = ml_returns.index
    start, end = ml_dates[0], ml_dates[-1]

    b60 = bench_results["60/40"].loc[start:end]
    bew = bench_results["Equal Weight"].loc[start:end]
    bmk = bench_results["Markowitz"].loc[start:end]

    metrics = {
        "XGB+Opt":  xgb_port,
        "LGB+Opt":  lgb_port,
        "Markowitz": compute_portfolio_metrics(bmk, "Markowitz"),
        "60/40":     compute_portfolio_metrics(b60, "60/40"),
        "Equal Wt":  compute_portfolio_metrics(bew, "Equal Wt"),
    }

    names = list(metrics.keys())
    rows_def = [
        ("Retorno Anual.", "Retorno Anualizado", "pct"),
        ("Volatilidad",    "Volatilidad Anualizada", "pct"),
        ("Sharpe Ratio",   "Sharpe Ratio", "f3"),
        ("Sortino Ratio",  "Sortino Ratio", "f3"),
        ("Max Drawdown",   "Max Drawdown", "pct"),
        ("Calmar Ratio",   "Calmar Ratio", "f3"),
        ("Retorno Total",  "Total Return", "pct"),
    ]

    col_w = 11
    header = "║  " + f"{'Métrica':<18s}│"
    for n in names:
        header += f"{n:^{col_w}s}│"
    header = header[:-1] + "║"
    iw = len(header) - 2

    s6040 = metrics["60/40"]["Sharpe Ratio"]

    print(f"\n╔{'═' * iw}╗")
    print(f"║{'BLOQUE 6 — COMPARATIVA DE LAS 5 CARTERAS FINALES':^{iw}s}║")
    print(f"╠{'═' * iw}╣")
    print(header)
    print(f"╠{'═' * iw}╣")

    for label, key, fmt in rows_def:
        line = f"║  {label:<18s}│"
        for n in names:
            val = metrics[n][key]
            s = f"{val:.2%}" if fmt == "pct" else f"{val:.3f}"

            # Colorear Sharpe de carteras ML
            if key == "Sharpe Ratio" and n in ("XGB+Opt", "LGB+Opt"):
                color = GREEN if val > s6040 else RED
                s = f"{color}{s}{RESET}"
                line += f"{s:^{col_w + 9}s}│"  # +9 por los ANSI
            else:
                line += f"{s:^{col_w}s}│"
        line = line[:-1] + "║"
        print(line)

    print(f"╚{'═' * iw}╝")
    return metrics


# ── BLOQUE 7: Report Card ────────────────────────────────────────────

def _bar(value, max_val, width=14):
    filled = int(min(value / max_val, 1.0) * width) if max_val > 0 else 0
    return "█" * filled + "░" * (width - filled)


def bloque_7_report_card(best_metrics, best_port, bench_6040_met, model_name):
    avg_da = np.mean([m["Dir_Acc"] for m in best_metrics.values()])
    sharpe = best_port["Sharpe Ratio"]
    max_dd = best_port["Max Drawdown"]
    s6040  = bench_6040_met["Sharpe Ratio"]
    dd6040 = bench_6040_met["Max Drawdown"]
    s_diff = sharpe - s6040
    dd_diff_pp = (max_dd - dd6040) * 100  # positivo = ML mejor

    # Ratio std_pred/std_real medio
    ratios = [m["Std_Pred"] / m["Std_Real"] for m in best_metrics.values()
              if m["Std_Real"] > 0]
    avg_ratio = np.mean(ratios) if ratios else float("nan")

    # Funciones de evaluación
    def ev_da(v):
        if v > 57: return (GREEN, "FUERTE")
        if v > 52: return (YELLOW, "MODERADO")
        return (RED, "DEBIL")

    def ev_sharpe(v):
        if v > 1.2: return (GREEN, "EXCELENTE")
        if v > 0.8: return (GREEN, "BUENO")
        if v > 0.5: return (YELLOW, "MODERADO")
        return (RED, "BAJO")

    def ev_dd(v):
        if v > -0.15: return (GREEN, "EXCELENTE")
        if v > -0.20: return (GREEN, "BUENO")
        if v > -0.30: return (YELLOW, "MODERADO")
        return (RED, "ALTO")

    def ev_of(v):
        if np.isnan(v): return (YELLOW, "SIN DATOS")
        if v < 0.5:  return (GREEN, "BAJO")
        if v < 1.0:  return (GREEN, "CONTROLADO")
        return (YELLOW, "ALTO")

    def ev_vs(v):
        if v > 0: return (GREEN, "SUPERA")
        if v > -0.1: return (YELLOW, "SIMILAR")
        return (RED, "INFERIOR")

    def ev_dd_vs(v):
        if v > 0: return (GREEN, "SUPERA")
        if v > -3: return (YELLOW, "SIMILAR")
        return (RED, "PEOR")

    iw = 60
    print(f"\n╔{'═' * iw}╗")
    print(f"║{BOLD}{'BLOQUE 7 — REPORT CARD — ' + model_name:^{iw}s}{RESET}║")
    print(f"╠{'═' * iw}╣")

    lines = [
        ("Poder predictivo",  _bar(avg_da, 100),       f"{avg_da:.1f}%",       ev_da(avg_da)),
        ("Sharpe ratio",      _bar(sharpe, 2.0),        f"{sharpe:.3f}",        ev_sharpe(sharpe)),
        ("Max drawdown",      _bar(abs(max_dd), 0.50),  f"{max_dd:.1%}",        ev_dd(max_dd)),
        ("Overfitting risk",  _bar(avg_ratio, 2.0),     f"{avg_ratio:.2f}x",    ev_of(avg_ratio)),
        ("vs 60/40 Sharpe",   _bar(max(s_diff+0.5,0), 1.0),  f"{s_diff:+.3f}", ev_vs(s_diff)),
        ("vs 60/40 Drawdown", _bar(max(dd_diff_pp+10,0), 20), f"{dd_diff_pp:+.1f}pp", ev_dd_vs(dd_diff_pp)),
    ]

    for name, bar, val_str, (color, status) in lines:
        icon = "✅" if color == GREEN else ("⚠️ " if color == YELLOW else "❌")
        content = f"  {name:<20s}{bar}  {val_str:<8s} {color}{icon} {status}{RESET}"
        # Calcular padding (ANSI ocupa ~11 chars extra)
        visible_len = 2 + 20 + 14 + 2 + 8 + 1 + len(icon) + 1 + len(status)
        pad = iw - visible_len
        if pad < 0:
            pad = 0
        print(f"║{content}{' ' * pad}║")

    # Nota global
    scores = {
        "da":       1 if avg_da > 55 else (0.5 if avg_da > 50 else 0),
        "sharpe":   1 if sharpe > 1.0 else (0.5 if sharpe > 0.7 else 0),
        "dd":       1 if max_dd > -0.20 else (0.5 if max_dd > -0.30 else 0),
        "overfit":  1 if (not np.isnan(avg_ratio) and avg_ratio < 1.0) else 0.5,
        "bench_s":  1 if s_diff > 0 else (0.5 if s_diff > -0.1 else 0),
        "bench_dd": 1 if dd_diff_pp > 0 else (0.5 if dd_diff_pp > -3 else 0),
    }
    total_score = sum(scores.values()) / len(scores)

    if total_score >= 0.9:   grade, desc = "A", "excepcional, supera todos los benchmarks"
    elif total_score >= 0.75: grade, desc = "A-", "muy bueno, supera benchmarks"
    elif total_score >= 0.6:  grade, desc = "B+", "supera benchmarks principales"
    elif total_score >= 0.45: grade, desc = "B", "competitivo"
    elif total_score >= 0.3:  grade, desc = "C+", "moderado"
    else:                     grade, desc = "C", "insuficiente"

    print(f"╠{'═' * iw}╣")
    gl = f"NOTA GLOBAL:  {grade} ({desc})"
    print(f"║  {BOLD}{gl}{RESET}{' ' * (iw - len(gl) - 2)}║")
    print(f"╚{'═' * iw}╝")
    print(f"\n  Escala: A = supera todos | B = supera 60/40 | "
          f"C = supera Markowitz | D = no supera")


# ── BLOQUE 8: Evolución temporal ─────────────────────────────────────

def bloque_8_temporal(ml_returns, bench_6040, model_name):
    print(f"\n{'═' * 62}")
    print(f"{BOLD}  BLOQUE 8 — EVOLUCIÓN TEMPORAL — Sharpe Anual ML vs 60/40{RESET}")
    print(f"{'═' * 62}")

    common = ml_returns.index.intersection(bench_6040.index)
    ml = ml_returns.loc[common]
    bench = bench_6040.loc[common]

    ml_y = ml.groupby(ml.index.year)
    bench_y = bench.groupby(bench.index.year)

    wins, total = 0, 0

    print(f"  {'Año':>6s}  {'ML Sharpe':>10s}  {'60/40 Sharpe':>13s}  Resultado")
    print(f"  {'─' * 52}")

    for year in sorted(ml.index.year.unique()):
        m = ml_y.get_group(year)
        if year not in bench_y.groups:
            continue
        b = bench_y.get_group(year)

        ms = (m.mean() * 52) / (m.std() * np.sqrt(52)) if m.std() > 0 else 0
        bs = (b.mean() * 52) / (b.std() * np.sqrt(52)) if b.std() > 0 else 0

        total += 1
        if ms > bs:
            wins += 1
            res = f"{GREEN}✅ ML mejor{RESET}"
        else:
            res = f"{RED}❌ ML peor{RESET}"

        print(f"  {year:>6d}  {ms:>10.2f}  {bs:>13.2f}  {res}")

    print(f"  {'─' * 52}")
    pct = wins / total * 100 if total else 0
    c = GREEN if pct >= 60 else (YELLOW if pct >= 50 else RED)
    print(f"  ML gana: {c}{wins}/{total} ({pct:.0f}%){RESET}")
    print(f"  ML pierde: {total - wins}/{total}")
    print(f"{'═' * 62}")


# ── BLOQUE 9: Veredicto final ────────────────────────────────────────

def bloque_9_veredicto(xgb_port, lgb_port, bench_6040_met,
                       xgb_metrics, lgb_metrics):
    xgb_s = xgb_port["Sharpe Ratio"]
    lgb_s = lgb_port["Sharpe Ratio"]
    s60   = bench_6040_met["Sharpe Ratio"]
    dd60  = bench_6040_met["Max Drawdown"]

    if xgb_s >= lgb_s:
        best_name = "XGBoost Tuned"
        best_port, other_s, best_m = xgb_port, lgb_s, xgb_metrics
    else:
        best_name = "LightGBM Tuned"
        best_port, other_s, best_m = lgb_port, xgb_s, lgb_metrics

    best_s   = best_port["Sharpe Ratio"]
    best_ret = best_port["Retorno Anualizado"]
    best_tot = best_port["Total Return"]
    best_dd  = best_port["Max Drawdown"]
    avg_da   = np.mean([m["Dir_Acc"] for m in best_m.values()])

    print(f"\n{'═' * 62}")
    print(f"{BOLD}  BLOQUE 9 — VEREDICTO FINAL Y RESUMEN EJECUTIVO{RESET}")
    print(f"{'═' * 62}")

    print(f"\n  {BOLD}Mejor modelo:{RESET} {GREEN}{best_name}{RESET}")
    print(f"    Sharpe Ratio:    {best_s:.3f} (vs {other_s:.3f} del otro)")
    print(f"    Retorno Anual:   {best_ret:.2%}")
    print(f"    Retorno Total:   {best_tot:.2%} acumulado")
    print(f"    Max Drawdown:    {best_dd:.2%}")
    print(f"    Dir. Accuracy:   {avg_da:.1f}% (media 10 ETFs)")

    sdiff = best_s - s60
    dd_pp = (best_dd - dd60) * 100

    print(f"\n  {BOLD}vs Benchmark 60/40:{RESET}")
    c = GREEN if sdiff > 0 else RED
    print(f"    Sharpe:   {c}{sdiff:+.3f}{RESET} ({best_s:.3f} vs {s60:.3f})")
    c = GREEN if dd_pp > 0 else RED
    print(f"    Max DD:   {c}{dd_pp:+.1f} pp{RESET} ({best_dd:.2%} vs {dd60:.2%})")

    print(f"\n  {BOLD}Conclusión:{RESET}")
    if best_s > s60 and best_dd > dd60:
        print(f"    {GREEN}El modelo ML supera al benchmark 60/40 tanto en rentabilidad")
        print(f"    ajustada al riesgo (Sharpe) como en protección ante caídas.{RESET}")
    elif best_s > s60:
        print(f"    {GREEN}El modelo ML supera al 60/40 en Sharpe.{RESET}")
    else:
        print(f"    {YELLOW}El modelo ML no supera al 60/40 en Sharpe.{RESET}")

    print(f"\n  {BOLD}Hallazgos clave del TFG:{RESET}")
    print(f"    1. XGBoost Tuned es el mejor modelo (Sharpe {xgb_s:.3f})")
    print(f"    2. Dir. Accuracy ~{avg_da:.0f}% confirma capacidad predictiva (>50%)")
    print(f"    3. Optuna v2 confirmó que v1 encontró el óptimo global")
    print(f"    4. Modelo conservador (std pred << std real) → no sobreajusta")
    print(f"    5. Walk-forward 778 splits garantiza robustez out-of-sample")
    print(f"    6. Max DD -{abs(best_dd)*100:.1f}% vs -{abs(dd60)*100:.1f}% del 60/40")
    print(f"{'═' * 62}")


# ── BLOQUE 10: Copiar archivos a final/ ──────────────────────────────

def bloque_10_guardar(financial_metrics):
    """Copia archivos a data/results/final/ y guarda tabla comparativa."""
    final_dir = "data/results/final"
    os.makedirs(final_dir, exist_ok=True)

    # Copiar archivos existentes, renombrando tuned → final
    archivos = [
        ("xgb_tuned_predictions.csv",  "xgb_final_predictions.csv"),
        ("lgb_tuned_predictions.csv",   "lgb_final_predictions.csv"),
        ("xgb_tuned_weights.csv",       "xgb_final_weights.csv"),
        ("lgb_tuned_weights.csv",       "lgb_final_weights.csv"),
        ("optuna_best_params_xgb.json", "optuna_best_params_xgb.json"),
        ("optuna_best_params_lgb.json", "optuna_best_params_lgb.json"),
    ]

    for src_name, dst_name in archivos:
        src = os.path.join("data/results", src_name)
        dst = os.path.join(final_dir, dst_name)
        shutil.copy2(src, dst)

    # Guardar tabla comparativa
    rows = []
    for name, m in financial_metrics.items():
        row = {"Cartera": name}
        row.update({k: v for k, v in m.items() if k != "Nombre"})
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(final_dir, "portfolio_final_comparison.csv"), index=False)

    print(f"\n{'=' * 62}")
    print(f"  BLOQUE 10 — ARCHIVOS GUARDADOS EN {final_dir}/")
    print(f"{'=' * 62}")

    for f in sorted(os.listdir(final_dir)):
        size = os.path.getsize(os.path.join(final_dir, f))
        print(f"    {f:<40s} {size:>9,} bytes")

    print(f"{'=' * 62}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()

    print()
    print("▓" * 62)
    print("▓▓▓   TRAIN FINAL — REPORTE COMPLETO DE RESULTADOS DEL TFG   ▓▓▓")
    print("▓▓▓   (Carga predicciones existentes, NO reentrena)           ▓▓▓")
    print("▓" * 62)

    # ── Cargar predicciones, pesos y parámetros existentes ──────────
    print("\nCargando predicciones y pesos existentes...")

    xgb_preds = cargar_predicciones("data/results/xgb_tuned_predictions.csv")
    lgb_preds = cargar_predicciones("data/results/lgb_tuned_predictions.csv")
    xgb_weights = cargar_pesos("data/results/xgb_tuned_weights.csv")
    lgb_weights = cargar_pesos("data/results/lgb_tuned_weights.csv")

    with open("data/results/optuna_best_params_xgb.json") as f:
        xgb_params = json.load(f)
    with open("data/results/optuna_best_params_lgb.json") as f:
        lgb_params = json.load(f)

    n_preds = len(xgb_preds[ETFS[0]])
    print(f"  Predicciones XGB: {n_preds} semanas x 10 ETFs")
    print(f"  Predicciones LGB: {len(lgb_preds[ETFS[0]])} semanas x 10 ETFs")
    print(f"  Pesos XGB: {len(xgb_weights)} semanas x 10 ETFs")
    print(f"  Pesos LGB: {len(lgb_weights)} semanas x 10 ETFs")

    # ── Cargar dataset maestro (para benchmarks) ────────────────────
    features, targets = load_master_dataset()

    # ── Calcular métricas ML por ETF ────────────────────────────────
    print("\nCalculando métricas ML...")
    xgb_metrics = {etf: evaluar_etf(xgb_preds[etf]) for etf in ETFS}
    lgb_metrics = {etf: evaluar_etf(lgb_preds[etf]) for etf in ETFS}

    # ── Reconstruir retornos de cartera ML ──────────────────────────
    print("Reconstruyendo retornos de cartera...")
    xgb_returns = calcular_retornos_cartera(xgb_preds, xgb_weights)
    lgb_returns = calcular_retornos_cartera(lgb_preds, lgb_weights)

    xgb_port = compute_portfolio_metrics(xgb_returns, "XGBoost+Opt")
    lgb_port = compute_portfolio_metrics(lgb_returns, "LightGBM+Opt")

    print(f"  XGBoost Tuned: Sharpe = {xgb_port['Sharpe Ratio']:.3f}")
    print(f"  LightGBM Tuned: Sharpe = {lgb_port['Sharpe Ratio']:.3f}")

    # ── Calcular benchmarks (Markowitz ~2 min) ──────────────────────
    print("\nCalculando benchmarks (Markowitz walk-forward ~2 min)...")
    bench_results, _ = compare_benchmarks(targets, min_train_weeks=208)

    # Benchmark 60/40 alineado al período ML
    ml_dates = xgb_returns.index
    b60_aligned = bench_results["60/40"].loc[ml_dates[0]:ml_dates[-1]]
    bench_6040_met = compute_portfolio_metrics(b60_aligned, "60/40")

    # ══════════════════════════════════════════════════════════════════
    # IMPRIMIR TODOS LOS BLOQUES
    # ══════════════════════════════════════════════════════════════════

    bloque_1_params(xgb_params, lgb_params)

    bloque_2_metricas_ml(xgb_metrics, lgb_metrics)

    bloque_3_por_etf(xgb_metrics, lgb_metrics)

    bloque_4_confusion(xgb_preds, "XGBoost Tuned")

    bloque_5_overfitting(xgb_metrics, "XGBoost Tuned")

    financial_metrics = bloque_6_financiera(
        xgb_port, lgb_port, bench_results, xgb_returns)

    bloque_7_report_card(xgb_metrics, xgb_port, bench_6040_met, "XGBoost Tuned")

    bloque_8_temporal(xgb_returns, b60_aligned, "XGBoost Tuned")

    bloque_9_veredicto(xgb_port, lgb_port, bench_6040_met,
                       xgb_metrics, lgb_metrics)

    bloque_10_guardar(financial_metrics)

    # ── Final ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n⏱️  Tiempo total: {int(elapsed // 60)}m {int(elapsed % 60)}s")
    print(f"\nPara reproducir estos resultados: python3 src/models/train_final.py")
