"""
Integra 6 variables externas nuevas + 3 variables internas al dataset maestro.
No re-ejecuta el pipeline completo: carga el master existente y añade columnas.

Variables externas (FRED + Yahoo Finance):
  - WEI: Weekly Economic Index (actividad real semanal)
  - CCSA: Continued Claims desempleo
  - ICSA: Initial Claims desempleo (cambio %)
  - T10Y3M: Spread 10Y-3M Treasury (señal de recesión)
  - STLFSI4: St. Louis Financial Stress Index
  - MOVE: MOVE Index (volatilidad implícita de bonos)

Variables internas (calculadas del master existente):
  - vix_term_structure: VIX / media móvil 12 semanas (>1 = pánico)
  - spy_agg_corr_52w: correlación rolling SPY-AGG 52 semanas
  - etf_return_dispersion: dispersión semanal de retornos entre los 10 ETFs

Autor: Mateo Madrigal Arteaga, UFV
Uso:   python3 src/transformers/add_new_features.py
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Configuración ──────────────────────────────────────────────────

DIR_INTERIM = os.path.join("data", "interim")
DIR_PROCESSED = os.path.join("data", "processed")

ETFS = ["AGG", "EEM", "EFA", "GLD", "IWM", "LQD", "QQQ", "SPY", "TIP", "VNQ"]

# CSVs subidos a interim (formato FRED: observation_date, valor)
FRED_CSVS = {
    "WEI":     os.path.join(DIR_INTERIM, "WEI.csv"),
    "CCSA":    os.path.join(DIR_INTERIM, "CCSA.csv"),
    "ICSA":    os.path.join(DIR_INTERIM, "ICSA.csv"),
    "T10Y3M":  os.path.join(DIR_INTERIM, "T10Y3M.csv"),
    "STLFSI4": os.path.join(DIR_INTERIM, "STLFSI4.csv"),
}

# Columnas nuevas que se van a crear
NUEVAS_EXTERNAS = [
    "wei_level", "wei_change",
    "icsa_change", "ccsa_change",
    "stlfsi4_level", "stlfsi4_change",
    "move_level", "move_change",
    "spread_10y_3m",
]
NUEVAS_INTERNAS = [
    "vix_term_structure",
    "spy_agg_corr_52w",
    "etf_return_dispersion",
]


# ══════════════════════════════════════════════════════════════════════
# PASO 1: Cargar dataset maestro actual
# ══════════════════════════════════════════════════════════════════════

def cargar_master():
    """Carga el dataset maestro actual."""
    print("PASO 1 — Cargando dataset maestro...")
    ruta = os.path.join(DIR_PROCESSED, "master_weekly_raw.csv")
    master = pd.read_csv(ruta, parse_dates=["date"], index_col="date")
    print(f"  Master actual: {master.shape[0]} filas × {master.shape[1]} cols")
    print(f"  Rango: {master.index.min().date()} → {master.index.max().date()}")

    # Eliminar columnas nuevas previas si existen (re-ejecución segura)
    cols_previas = [c for c in NUEVAS_EXTERNAS + NUEVAS_INTERNAS
                    if c in master.columns]
    if cols_previas:
        master = master.drop(columns=cols_previas)
        print(f"  Eliminadas {len(cols_previas)} columnas de ejecución previa")

    return master


# ══════════════════════════════════════════════════════════════════════
# PASO 2: Descargar MOVE Index
# ══════════════════════════════════════════════════════════════════════

def descargar_move():
    """Descarga MOVE Index desde Yahoo Finance y lo resamplea a semanal."""
    print("\nPASO 2 — Descargando MOVE Index (^MOVE) desde Yahoo Finance...")
    import yfinance as yf

    ruta_cache = os.path.join(DIR_INTERIM, "MOVE_daily.csv")

    # Intentar descargar; si falla, usar cache
    try:
        df = yf.download("^MOVE", start="2007-01-01", auto_adjust=True, progress=False)
        if len(df) == 0:
            raise ValueError("yfinance devolvió 0 filas")
        # Guardar cache
        df[["Close"]].to_csv(ruta_cache)
        print(f"  Descargado: {len(df)} filas ({df.index.min().date()} → {df.index.max().date()})")
    except Exception as e:
        print(f"  Error descargando: {e}")
        if os.path.exists(ruta_cache):
            df = pd.read_csv(ruta_cache, parse_dates=[0], index_col=0)
            print(f"  Usando cache: {ruta_cache} ({len(df)} filas)")
        else:
            print(f"  No hay cache. MOVE no estará disponible.")
            return None

    # Resamplear a semanal (viernes) — último valor de la semana
    move_weekly = df["Close"].resample("W-FRI").last()
    move_weekly = move_weekly.dropna()
    move_weekly.name = "MOVE"
    print(f"  Semanal: {len(move_weekly)} semanas "
          f"({move_weekly.index.min().date()} → {move_weekly.index.max().date()})")
    return move_weekly


# ══════════════════════════════════════════════════════════════════════
# PASO 3: Cargar y alinear CSVs de FRED a semanal
# ══════════════════════════════════════════════════════════════════════

def cargar_fred_csv(path, nombre):
    """Carga un CSV de FRED (observation_date, valor) y lo alinea a W-FRI."""
    df = pd.read_csv(path, parse_dates=["observation_date"], index_col="observation_date")
    df.index.name = "date"
    col = df.columns[0]
    serie = df[col].dropna()

    # Detectar frecuencia
    if len(serie) > 1:
        mediana_dias = serie.index.to_series().diff().dt.days.median()
    else:
        mediana_dias = 7

    if mediana_dias <= 2:
        # Diario → ffill a diario y resamplear a viernes
        serie = serie.resample("D").last().ffill()
        semanal = serie.resample("W-FRI").last()
    else:
        # Semanal → alinear al viernes más cercano
        semanal = serie.resample("W-FRI").last().ffill()

    semanal = semanal.dropna()
    semanal.name = nombre
    print(f"  {nombre:<8s}: {len(serie):>5d} obs ({mediana_dias:.0f}d) → "
          f"{len(semanal):>4d} semanas "
          f"({semanal.index.min().date()} → {semanal.index.max().date()})")
    return semanal


def cargar_todos_fred():
    """Carga todos los CSVs de FRED y los alinea a semanal."""
    print("\nPASO 3 — Cargando y alineando CSVs de FRED a W-FRI...")
    series = {}
    for nombre, path in FRED_CSVS.items():
        if os.path.exists(path):
            series[nombre] = cargar_fred_csv(path, nombre)
        else:
            print(f"  {nombre}: ARCHIVO NO ENCONTRADO ({path})")
    return series


# ══════════════════════════════════════════════════════════════════════
# PASO 4: Calcular features derivadas de las nuevas variables externas
# ══════════════════════════════════════════════════════════════════════

def calcular_features_externas(series_fred, move_weekly):
    """
    Calcula las features derivadas de cada variable externa.
    Retorna DataFrame indexado por fecha con todas las features.
    """
    print("\nPASO 4 — Calculando features externas...")
    features = pd.DataFrame()

    # WEI: nivel y cambio semanal
    if "WEI" in series_fred:
        wei = series_fred["WEI"]
        features["wei_level"] = wei
        features["wei_change"] = wei.diff()
        print(f"  ✓ wei_level, wei_change")

    # ICSA: cambio porcentual semanal (initial claims)
    if "ICSA" in series_fred:
        icsa = series_fred["ICSA"]
        features["icsa_change"] = icsa.pct_change()
        print(f"  ✓ icsa_change")

    # CCSA: cambio porcentual semanal (continued claims)
    if "CCSA" in series_fred:
        ccsa = series_fred["CCSA"]
        features["ccsa_change"] = ccsa.pct_change()
        print(f"  ✓ ccsa_change")

    # STLFSI4: nivel y cambio semanal
    if "STLFSI4" in series_fred:
        stlfsi = series_fred["STLFSI4"]
        features["stlfsi4_level"] = stlfsi
        features["stlfsi4_change"] = stlfsi.diff()
        print(f"  ✓ stlfsi4_level, stlfsi4_change")

    # MOVE: nivel y cambio semanal
    if move_weekly is not None:
        features["move_level"] = move_weekly
        features["move_change"] = move_weekly.diff()
        print(f"  ✓ move_level, move_change")

    # T10Y3M: nivel del spread (ya es un spread, no calcular cambio)
    if "T10Y3M" in series_fred:
        features["spread_10y_3m"] = series_fred["T10Y3M"]
        print(f"  ✓ spread_10y_3m")

    print(f"  Total: {features.shape[1]} features externas")
    return features


# ══════════════════════════════════════════════════════════════════════
# PASO 5: Calcular las 3 variables internas
# ══════════════════════════════════════════════════════════════════════

def calcular_features_internas(master):
    """
    Calcula 3 variables de estructura de mercado a partir del master existente.
    """
    print("\nPASO 5 — Calculando features internas (market structure)...")
    features = pd.DataFrame(index=master.index)

    # a) VIX Term Structure: VIX actual / media móvil 12 semanas
    #    > 1 = curva invertida (pánico, VIX por encima de lo normal)
    #    < 1 = curva normal (calma)
    if "vix_level" in master.columns:
        vix = master["vix_level"]
        vix_ma12 = vix.rolling(12).mean()
        features["vix_term_structure"] = vix / vix_ma12
        print(f"  ✓ vix_term_structure (VIX / VIX_MA12w)")
    else:
        print(f"  ✗ vix_term_structure: no existe vix_level en master")

    # b) Correlación rolling SPY-AGG 52 semanas
    #    Positiva alta = diversificación rota (ambos caen juntos, ej: 2022)
    #    Negativa = diversificación funciona (flight to quality)
    if "SPY_log_ret" in master.columns and "AGG_log_ret" in master.columns:
        spy_ret = master["SPY_log_ret"]
        agg_ret = master["AGG_log_ret"]
        features["spy_agg_corr_52w"] = spy_ret.rolling(52).corr(agg_ret)
        print(f"  ✓ spy_agg_corr_52w (rolling 52w)")
    else:
        print(f"  ✗ spy_agg_corr_52w: faltan columnas de retornos")

    # c) Dispersión de retornos ETFs
    #    Desviación estándar cross-seccional de los 10 retornos cada semana.
    #    Alta dispersión = oportunidades de selección (stock picking)
    #    Baja dispersión = mercado correlacionado (beta driven)
    ret_cols = [f"{etf}_log_ret" for etf in ETFS]
    available = [c for c in ret_cols if c in master.columns]
    if len(available) == 10:
        features["etf_return_dispersion"] = master[available].std(axis=1)
        print(f"  ✓ etf_return_dispersion (std cross-seccional de 10 ETFs)")
    else:
        print(f"  ✗ etf_return_dispersion: solo {len(available)}/10 columnas de retornos")

    print(f"  Total: {features.shape[1]} features internas")
    return features


# ══════════════════════════════════════════════════════════════════════
# PASO 6: Merge con el master
# ══════════════════════════════════════════════════════════════════════

def merge_con_master(master, feat_externas, feat_internas):
    """
    Hace left join del master con las nuevas features.
    El master conserva todas sus filas; las features se alinean por fecha.
    """
    print("\nPASO 6 — Merge con dataset maestro (left join)...")
    cols_antes = master.shape[1]

    # Añadir features externas (left join por fecha)
    master = master.join(feat_externas, how="left")

    # Añadir features internas (ya están alineadas por índice)
    for col in feat_internas.columns:
        master[col] = feat_internas[col]

    cols_despues = master.shape[1]
    n_nuevas = cols_despues - cols_antes

    print(f"  Columnas antes: {cols_antes}")
    print(f"  Columnas añadidas: {n_nuevas}")
    print(f"  Columnas después: {cols_despues}")
    print(f"  Filas: {master.shape[0]} (sin cambios)")

    # Verificar NaN en las nuevas columnas
    print(f"\n  NaN en nuevas columnas:")
    for col in NUEVAS_EXTERNAS + NUEVAS_INTERNAS:
        if col in master.columns:
            n_nan = master[col].isna().sum()
            n_total = len(master)
            pct = n_nan / n_total * 100
            estado = "✅" if pct < 5 else ("⚠️ " if pct < 50 else "ℹ️ ")
            print(f"    {estado} {col:<26s}: {n_nan:>4d} NaN ({pct:.1f}%)")

    return master


# ══════════════════════════════════════════════════════════════════════
# PASO 7: Guardar
# ══════════════════════════════════════════════════════════════════════

def guardar(master):
    """Guarda el dataset maestro actualizado."""
    print("\nPASO 7 — Guardando dataset maestro actualizado...")

    ruta_raw = os.path.join(DIR_PROCESSED, "master_weekly_raw.csv")
    master.to_csv(ruta_raw)
    size = os.path.getsize(ruta_raw) / (1024 * 1024)
    print(f"  Guardado: {ruta_raw} ({size:.2f} MB)")

    return master


# ══════════════════════════════════════════════════════════════════════
# PASO 8: Resumen final
# ══════════════════════════════════════════════════════════════════════

def resumen_final(master):
    """Imprime resumen completo del dataset actualizado."""
    target_cols = [c for c in master.columns if c.startswith("target_")]
    feature_cols = [c for c in master.columns if not c.startswith("target_")]

    print(f"\n{'=' * 70}")
    print("RESUMEN FINAL — DATASET MAESTRO ACTUALIZADO")
    print(f"{'=' * 70}")
    print(f"\n  Filas:    {master.shape[0]}")
    print(f"  Columnas: {master.shape[1]} ({len(feature_cols)} features + {len(target_cols)} targets)")
    print(f"  Rango:    {master.index.min().date()} → {master.index.max().date()}")

    # Desglose por dimensión
    dims = {
        "Mercado (ETFs)": [c for c in feature_cols
                           if any(c.startswith(t + "_") for t in ETFS)],
        "Macro": [c for c in feature_cols
                  if c in ["spread_10y_2y", "cpi_change", "unrate_change",
                           "umcsent_change", "wei_level", "wei_change",
                           "icsa_change", "ccsa_change", "spread_10y_3m"]],
        "Riesgo": [c for c in feature_cols
                   if c in ["vix_level", "vix_change", "hy_spread_change",
                            "nfci_change", "stlfsi4_level", "stlfsi4_change",
                            "move_level", "move_change"]],
        "Liquidez": [c for c in feature_cols
                     if c in ["fed_balance_change", "reverse_repo_change",
                              "bank_deposits_change", "tga_change"]],
        "Sentimiento": [c for c in feature_cols
                        if "aaii" in c or any(t in c for t in
                           ["recession", "inflation", "bear_market",
                            "bull_market", "buy_stocks", "sell_stocks",
                            "unemployment"])
                        and not c.startswith("target_")],
        "NLP Noticias": [c for c in feature_cols
                         if "news_" in c or c.endswith("_news_sent")
                         or c.endswith("_news_count")],
        "Market Structure": [c for c in feature_cols
                             if c in ["vix_term_structure", "spy_agg_corr_52w",
                                      "etf_return_dispersion"]],
    }

    print(f"\n  Desglose por dimensión:")
    total_classified = 0
    for dim, cols in dims.items():
        print(f"    {dim:<22s}: {len(cols):>3d} features")
        total_classified += len(cols)

    print(f"    {'TOTAL':22s}: {total_classified:>3d} features + {len(target_cols)} targets")

    # Nuevas columnas añadidas
    print(f"\n  Nuevas columnas añadidas ({len(NUEVAS_EXTERNAS) + len(NUEVAS_INTERNAS)}):")
    for col in NUEVAS_EXTERNAS + NUEVAS_INTERNAS:
        if col in master.columns:
            n_ok = master[col].notna().sum()
            print(f"    ✓ {col:<28s} ({n_ok}/{len(master)} con datos)")
        else:
            print(f"    ✗ {col:<28s} (no creada)")

    print(f"{'=' * 70}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("  INTEGRACIÓN DE NUEVAS VARIABLES AL DATASET MAESTRO")
    print("  6 externas (FRED + Yahoo) + 3 internas (market structure)")
    print("=" * 70)

    # Paso 1: Cargar master actual
    master = cargar_master()

    # Paso 2: Descargar MOVE Index
    move_weekly = descargar_move()

    # Paso 3: Cargar CSVs de FRED
    series_fred = cargar_todos_fred()

    # Paso 4: Features externas
    feat_externas = calcular_features_externas(series_fred, move_weekly)

    # Paso 5: Features internas
    feat_internas = calcular_features_internas(master)

    # Paso 6: Merge
    master = merge_con_master(master, feat_externas, feat_internas)

    # Paso 7: Guardar
    master = guardar(master)

    # Paso 8: Resumen
    resumen_final(master)

    print(f"\nDataset maestro actualizado con {len(NUEVAS_EXTERNAS) + len(NUEVAS_INTERNAS)} nuevas features.")
