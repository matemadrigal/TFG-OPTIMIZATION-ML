"""
Feature Engineering: creación de variables derivadas para el modelo ML.
Transforma los datos limpios semanales en features con significado financiero.
"""

import os

import numpy as np
import pandas as pd

# ── Configuración ──────────────────────────────────────────────

DIR_INTERIM = os.path.join("data", "interim")

ETFS = ["AGG", "EEM", "EFA", "GLD", "IWM", "LQD", "QQQ", "SPY", "TIP", "VNQ"]


# ── Funciones ──────────────────────────────────────────────────

def cargar_datos():
    """Paso 1: Carga los 5 CSVs limpios."""
    print("PASO 1 — Cargando datos limpios...")

    archivos = {
        "etfs":      "etfs_weekly_clean.csv",
        "macro":     "macro_weekly_clean.csv",
        "risk":      "risk_weekly_clean.csv",
        "liquidity": "liquidity_weekly_clean.csv",
        "sentiment": "sentiment_weekly_clean.csv",
    }

    datos = {}
    for nombre, archivo in archivos.items():
        df = pd.read_csv(os.path.join(DIR_INTERIM, archivo),
                         parse_dates=["date"], index_col="date")
        print(f"  {nombre:<12} {df.shape[0]} filas × {df.shape[1]} cols")
        datos[nombre] = df

    return datos


def features_etfs(etfs):
    """
    Paso 2: Variables derivadas de precios de ETFs.
    Para cada ETF se calculan 6 métricas estándar de análisis cuantitativo.
    """
    print("\nPASO 2 — Features de ETFs...")
    feat = pd.DataFrame(index=etfs.index)

    for ticker in ETFS:
        precio = etfs[ticker]

        # Log-return semanal: ln(P_t / P_{t-1})
        # Preferido en finanzas porque es aditivo en el tiempo
        # y aproxima bien rendimientos pequeños.
        feat[f"{ticker}_log_ret"] = np.log(precio / precio.shift(1))

        # Volatilidad rolling 4 semanas (~1 mes):
        # Desviación estándar de log-returns recientes.
        # Mide el riesgo a corto plazo del activo.
        feat[f"{ticker}_vol_4w"] = feat[f"{ticker}_log_ret"].rolling(4).std()

        # Volatilidad rolling 12 semanas (~3 meses):
        # Misma idea pero horizonte más largo.
        # Captura cambios de régimen de volatilidad.
        feat[f"{ticker}_vol_12w"] = feat[f"{ticker}_log_ret"].rolling(12).std()

        # Momentum 4 semanas: retorno acumulado en las últimas 4 semanas.
        # Señal de tendencia corta — los activos que suben tienden a seguir
        # subiendo a corto plazo (efecto momentum documentado en la literatura).
        feat[f"{ticker}_mom_4w"] = np.log(precio / precio.shift(4))

        # Momentum 12 semanas: retorno acumulado en ~3 meses.
        # Señal de tendencia media — captura tendencias más establecidas.
        feat[f"{ticker}_mom_12w"] = np.log(precio / precio.shift(12))

        # Drawdown: caída porcentual desde el máximo histórico acumulado.
        # Mide cuánto ha perdido el activo desde su pico.
        # Útil para detectar crisis y periodos de estrés.
        maximo_historico = precio.cummax()
        feat[f"{ticker}_drawdown"] = (precio - maximo_historico) / maximo_historico

    print(f"  Generadas {feat.shape[1]} columnas para {len(ETFS)} ETFs")
    print(f"  Métricas: log_ret, vol_4w, vol_12w, mom_4w, mom_12w, drawdown")
    return feat


def features_macro(macro):
    """
    Paso 3: Variables derivadas macroeconómicas.
    Se extraen señales de cambio que anticipan movimientos del mercado.
    """
    print("\nPASO 3 — Features de Macro...")
    feat = pd.DataFrame(index=macro.index)

    # Spread 10Y-2Y: diferencia entre tipos a 10 y 2 años.
    # Indicador clásico de recesión: cuando se invierte (negativo),
    # históricamente anticipa recesión en 12-18 meses.
    feat["spread_10y_2y"] = macro["DGS10"] - macro["DGS2"]

    # Variación semanal del CPI (Core):
    # Tras ffill, un cambio indica publicación de dato nuevo.
    # Captura sorpresas inflacionarias que mueven mercados.
    feat["cpi_change"] = macro["CPILFESL"].pct_change()

    # Variación semanal del desempleo:
    # Subidas inesperadas señalan debilitamiento económico.
    feat["unrate_change"] = macro["UNRATE"].diff()

    # Variación semanal de UMCSENT (sentimiento del consumidor Michigan):
    # Cambios reflejan confianza del consumidor, que anticipa gasto y PIB.
    feat["umcsent_change"] = macro["UMCSENT"].diff()

    print(f"  Generadas {feat.shape[1]} columnas")
    print(f"  Variables: spread_10y_2y, cpi_change, unrate_change, umcsent_change")
    return feat


def features_riesgo(risk):
    """
    Paso 4: Variables derivadas de riesgo.
    Cambios semanales en indicadores de estrés del mercado.
    """
    print("\nPASO 4 — Features de Riesgo...")
    feat = pd.DataFrame(index=risk.index)

    # VIX nivel: índice de volatilidad implícita del S&P 500.
    # Conocido como "índice del miedo". Niveles altos = mercado estresado.
    feat["vix_level"] = risk["VIXCLS"]

    # Variación semanal del VIX:
    # Subidas bruscas señalan aumento de incertidumbre.
    feat["vix_change"] = risk["VIXCLS"].diff()

    # Variación semanal del HY spread (BAMLH0A0HYM2):
    # Spread de bonos high yield sobre Treasuries.
    # Ampliaciones señalan aversión al riesgo crediticio.
    feat["hy_spread_change"] = risk["BAMLH0A0HYM2"].diff()

    # Variación semanal del NFCI (National Financial Conditions Index):
    # Valores positivos = condiciones financieras restrictivas.
    # Cambios capturan endurecimiento o relajación del crédito.
    feat["nfci_change"] = risk["NFCI"].diff()

    print(f"  Generadas {feat.shape[1]} columnas")
    print(f"  Variables: vix_level, vix_change, hy_spread_change, nfci_change")
    return feat


def features_liquidez(liquidity):
    """
    Paso 5: Variables derivadas de liquidez.
    Cambios en indicadores de liquidez del sistema financiero.
    """
    print("\nPASO 5 — Features de Liquidez...")
    feat = pd.DataFrame(index=liquidity.index)

    # Variación semanal de WALCL (balance de la Fed):
    # Expansión del balance = inyección de liquidez (QE).
    # Contracción = drenaje de liquidez (QT).
    feat["fed_balance_change"] = liquidity["WALCL"].pct_change()

    # Variación semanal de RRPONTSYD (reverse repo):
    # El reverse repo absorbe liquidez del sistema.
    # Cambios grandes indican movimientos de liquidez entre bancos y la Fed.
    feat["reverse_repo_change"] = liquidity["RRPONTSYD"].diff()

    # Variación semanal de depósitos bancarios (DPSACBW027SBOG):
    # Caídas en depósitos pueden señalar estrés bancario (ej: SVB 2023).
    feat["bank_deposits_change"] = liquidity["DPSACBW027SBOG"].pct_change()

    # Variación semanal de TGA (WTREGEN - Treasury General Account):
    # La cuenta del Tesoro en la Fed. Cuando sube, drena liquidez del mercado.
    # Cuando baja, inyecta liquidez.
    feat["tga_change"] = liquidity["WTREGEN"].pct_change()

    print(f"  Generadas {feat.shape[1]} columnas")
    print(f"  Variables: fed_balance_change, reverse_repo_change, bank_deposits_change, tga_change")
    return feat


def features_sentimiento(sentiment):
    """
    Paso 6: Variables derivadas de sentimiento.
    Combina encuesta AAII y Google Trends para captar el ánimo del mercado.
    """
    print("\nPASO 6 — Features de Sentimiento...")
    feat = pd.DataFrame(index=sentiment.index)

    # Spread Bull-Bear de AAII:
    # Diferencia entre % de alcistas y bajistas en la encuesta AAII.
    # Valores muy positivos = euforia (contrarian bearish).
    # Valores muy negativos = pánico (contrarian bullish).
    feat["aaii_bull_bear_spread"] = sentiment["aaii_bullish"] - sentiment["aaii_bearish"]

    # Google Trends: variación semanal y media móvil de 4 semanas.
    # Los términos de búsqueda reflejan el interés/preocupación del público.
    # La variación captura cambios bruscos de atención.
    # La media móvil suaviza el ruido semanal de Google Trends.
    trends_cols = ["recession", "inflation", "bear_market", "bull_market",
                   "buy_stocks", "sell_stocks", "unemployment"]

    for col in trends_cols:
        # Variación semanal: cambio respecto a la semana anterior
        feat[f"{col}_change"] = sentiment[col].diff()

        # Media móvil 4 semanas: suaviza ruido del dato semanal
        feat[f"{col}_ma4w"] = sentiment[col].rolling(4).mean()

    print(f"  Generadas {feat.shape[1]} columnas")
    print(f"  Variables: aaii_bull_bear_spread + 7 trends × (change + ma4w)")
    return feat


def eliminar_nulos_iniciales(features):
    """
    Paso 7: Elimina las primeras filas con NaN.
    Las ventanas rolling (especialmente 12 semanas) generan nulos al inicio.
    """
    print("\nPASO 7 — Eliminando filas iniciales con NaN...")
    filas_antes = len(features)

    # dropna: eliminar filas donde CUALQUIER columna tenga NaN
    features = features.dropna()

    filas_despues = len(features)
    filas_perdidas = filas_antes - filas_despues
    print(f"  Filas antes: {filas_antes}")
    print(f"  Filas después: {filas_despues}")
    print(f"  Filas eliminadas: {filas_perdidas} (ventanas rolling de inicio)")

    return features


def guardar_features(features):
    """Paso 8: Guarda el DataFrame de features como CSV."""
    print("\nPASO 8 — Guardando features...")
    features.index.name = "date"
    ruta = os.path.join(DIR_INTERIM, "features_weekly.csv")
    features.to_csv(ruta)
    print(f"  Guardado: {ruta}")


def resumen_final(features):
    """Paso 9: Imprime resumen completo del DataFrame de features."""
    print(f"\n{'='*70}")
    print("PASO 9 — RESUMEN FINAL")
    print(f"{'='*70}")

    print(f"\n  Total features: {features.shape[1]}")
    print(f"  Total filas:    {features.shape[0]}")
    print(f"  Rango:          {features.index.min().strftime('%Y-%m-%d')} → "
          f"{features.index.max().strftime('%Y-%m-%d')}")
    print(f"  Nulos:          {features.isnull().sum().sum()}")

    # Desglose por categoría
    categorias = {
        "ETFs (returns, vol, momentum, drawdown)": [c for c in features.columns
            if any(c.startswith(t + "_") for t in ETFS)],
        "Macro (spreads, cambios)": [c for c in features.columns
            if c in ["spread_10y_2y", "cpi_change", "unrate_change", "umcsent_change"]],
        "Riesgo (VIX, spreads, NFCI)": [c for c in features.columns
            if c in ["vix_level", "vix_change", "hy_spread_change", "nfci_change"]],
        "Liquidez (Fed, repo, depósitos, TGA)": [c for c in features.columns
            if c in ["fed_balance_change", "reverse_repo_change",
                      "bank_deposits_change", "tga_change"]],
        "Sentimiento (AAII, Google Trends)": [c for c in features.columns
            if "aaii_bull" in c or c.endswith("_change") and any(t in c for t in
               ["recession", "inflation", "bear_market", "bull_market",
                "buy_stocks", "sell_stocks", "unemployment"])
            or c.endswith("_ma4w")],
    }

    print(f"\n  Desglose por categoría:")
    for cat, cols in categorias.items():
        print(f"    {cat}: {len(cols)} features")

    # Lista completa de columnas
    print(f"\n  Columnas ({features.shape[1]} total):")
    for i, col in enumerate(features.columns, 1):
        print(f"    {i:>3}. {col}")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Paso 1: Cargar datos limpios
    datos = cargar_datos()

    # Paso 2-6: Crear features por dimensión
    feat_etfs = features_etfs(datos["etfs"])
    feat_macro = features_macro(datos["macro"])
    feat_riesgo = features_riesgo(datos["risk"])
    feat_liquidez = features_liquidez(datos["liquidity"])
    feat_sentimiento = features_sentimiento(datos["sentiment"])

    # Unir todas las features en un solo DataFrame
    features = pd.concat([
        feat_etfs, feat_macro, feat_riesgo, feat_liquidez, feat_sentimiento
    ], axis=1)
    print(f"\n  Total antes de limpieza: {features.shape[1]} features × {features.shape[0]} filas")

    # Paso 7: Eliminar filas con NaN del inicio
    features = eliminar_nulos_iniciales(features)

    # Paso 8: Guardar
    guardar_features(features)

    # Paso 9: Resumen
    resumen_final(features)

    print(f"\n{'='*70}")
    print("Feature engineering completado.")
