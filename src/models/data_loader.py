"""
Módulo de carga y preparación del dataset maestro.
Fase 4 — Modelado | TFG Optimización de Carteras con ML
Autor: Mateo Madrigal Arteaga, UFV
"""

import pandas as pd

# ── Constantes ──────────────────────────────────────────────────────

ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "LQD", "TIP", "GLD", "VNQ"]

DEFAULT_PATH = "data/processed/master_weekly_raw.csv"


# ── Funciones públicas ──────────────────────────────────────────────

def get_etf_tickers():
    """Retorna la lista de los 10 ETFs del universo de inversión."""
    return list(ETFS)


def get_feature_groups(feature_cols):
    """
    Agrupa las columnas de features por dimensión.
    Recibe la lista de nombres de columnas de features.
    Retorna un dict {dimensión: [columnas]}.
    Verifica que no haya duplicados ni columnas faltantes.
    """
    groups = {
        "market": [],
        "macro": [],
        "risk": [],
        "liquidity": [],
        "sentiment": [],
        "news_nlp": [],
        "market_structure": [],
    }

    # Columnas fijas por dimensión
    macro_cols = {"spread_10y_2y", "cpi_change", "unrate_change", "umcsent_change",
                  "wei_level", "wei_change", "icsa_change", "ccsa_change",
                  "spread_10y_3m"}
    risk_cols = {"vix_level", "vix_change", "hy_spread_change", "nfci_change",
                 "stlfsi4_level", "stlfsi4_change", "move_level", "move_change"}
    liquidity_cols = {"fed_balance_change", "reverse_repo_change",
                      "bank_deposits_change", "tga_change"}
    market_structure_cols = {"vix_term_structure", "spy_agg_corr_52w",
                             "etf_return_dispersion"}

    for col in feature_cols:
        # NLP de noticias (news_sent_all, news_count_all, y *_news_sent, *_news_count)
        if col.startswith("news_") or col.endswith("_news_sent") or col.endswith("_news_count"):
            groups["news_nlp"].append(col)
        # Market Structure (variables internas calculadas)
        elif col in market_structure_cols:
            groups["market_structure"].append(col)
        # Macro
        elif col in macro_cols:
            groups["macro"].append(col)
        # Riesgo
        elif col in risk_cols:
            groups["risk"].append(col)
        # Liquidez
        elif col in liquidity_cols:
            groups["liquidity"].append(col)
        # Sentimiento (AAII, Google Trends)
        elif ("aaii" in col or "recession" in col or "inflation" in col
              or "bear_market" in col or "bull_market" in col
              or "buy_stocks" in col or "sell_stocks" in col
              or "unemployment" in col):
            groups["sentiment"].append(col)
        # Market (contiene algún ticker de ETF)
        elif any(col.startswith(etf + "_") for etf in ETFS):
            groups["market"].append(col)
        else:
            print(f"  [AVISO] Columna sin grupo asignado: {col}")

    # Verificación de integridad
    todas = []
    for cols in groups.values():
        todas.extend(cols)

    set_todas = set(todas)
    set_features = set(feature_cols)

    duplicados = len(todas) - len(set_todas)
    faltantes = set_features - set_todas
    sobrantes = set_todas - set_features

    if duplicados > 0:
        print(f"  [ERROR] {duplicados} columnas duplicadas entre grupos")
    if faltantes:
        print(f"  [ERROR] Columnas sin asignar: {faltantes}")
    if sobrantes:
        print(f"  [ERROR] Columnas en grupos que no existen: {sobrantes}")
    if duplicados == 0 and not faltantes and not sobrantes:
        print(f"  ✓ Verificación OK: {len(todas)} features en {len(groups)} grupos, sin duplicados ni faltantes")

    return groups


def load_master_dataset(path=DEFAULT_PATH):
    """
    Carga el dataset maestro y separa features de targets.

    Retorna:
        features (DataFrame): columnas de features con índice datetime
        targets  (DataFrame): columnas target_* con índice datetime
    """
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")

    # Eliminar filas donde TODOS los targets son NaN (última fila por el shift)
    target_cols = [c for c in df.columns if c.startswith("target_")]
    mask_all_nan = df[target_cols].isna().all(axis=1)
    n_dropped = mask_all_nan.sum()
    df = df[~mask_all_nan]

    # Separar features y targets
    feature_cols = [c for c in df.columns if not c.startswith("target_")]
    features = df[feature_cols]
    targets = df[target_cols]

    # ── Resumen ──
    print("=" * 60)
    print("DATASET MAESTRO CARGADO")
    print("=" * 60)
    print(f"  Ruta: {path}")
    print(f"  Periodo: {features.index.min().date()} → {features.index.max().date()}")
    print(f"  Filas eliminadas (targets NaN): {n_dropped}")
    print(f"  Features: {features.shape[0]} filas × {features.shape[1]} columnas")
    print(f"  Targets:  {targets.shape[0]} filas × {targets.shape[1]} columnas")

    # Nulos por dimensión
    groups = get_feature_groups(list(feature_cols))
    print(f"\n  Nulos por dimensión:")
    for dim, cols in groups.items():
        n_nulls = features[cols].isnull().sum().sum()
        n_total = features[cols].size
        pct = n_nulls / n_total * 100 if n_total > 0 else 0
        print(f"    {dim:12s}: {n_nulls:>6} nulos ({pct:.1f}%)")

    total_nulls = features.isnull().sum().sum()
    print(f"    {'TOTAL':12s}: {total_nulls:>6} nulos")

    print(f"\n  Targets (10 ETFs): {', '.join(c.replace('target_', '') for c in target_cols)}")
    print("=" * 60)

    return features, targets


# ── Ejecución directa ──────────────────────────────────────────────

if __name__ == "__main__":
    features, targets = load_master_dataset()
    print(f"\nPrimeras 3 filas de features:")
    print(features.head(3))
    print(f"\nPrimeras 3 filas de targets:")
    print(targets.head(3))
