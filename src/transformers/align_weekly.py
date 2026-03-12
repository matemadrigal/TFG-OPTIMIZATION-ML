"""
Alineación temporal de todas las dimensiones a frecuencia semanal (viernes).
Toma los consolidados intermedios (data/interim/) y los resamplea a W-FRI,
recortando al periodo común 2007–presente.
"""

import os

import pandas as pd

# ── Configuración ──────────────────────────────────────────────

DIR_INTERIM = os.path.join("data", "interim")

FECHA_INICIO = "2007-01-05"  # Primer viernes de 2007


# ── Funciones ──────────────────────────────────────────────────

def cargar_consolidados():
    """Paso 1: Carga los 5 consolidados intermedios."""
    print("PASO 1 — Cargando consolidados intermedios...")

    etfs = pd.read_csv(
        os.path.join(DIR_INTERIM, "etfs_prices_daily.csv"),
        parse_dates=["Date"], index_col="Date"
    )
    print(f"  ETFs:       {etfs.shape[0]} filas × {etfs.shape[1]} cols")

    macro = pd.read_csv(
        os.path.join(DIR_INTERIM, "macro_daily.csv"),
        parse_dates=["date"], index_col="date"
    )
    print(f"  Macro:      {macro.shape[0]} filas × {macro.shape[1]} cols")

    risk = pd.read_csv(
        os.path.join(DIR_INTERIM, "risk_daily.csv"),
        parse_dates=["date"], index_col="date"
    )
    print(f"  Riesgo:     {risk.shape[0]} filas × {risk.shape[1]} cols")

    liquidity = pd.read_csv(
        os.path.join(DIR_INTERIM, "liquidity_daily.csv"),
        parse_dates=["date"], index_col="date"
    )
    print(f"  Liquidez:   {liquidity.shape[0]} filas × {liquidity.shape[1]} cols")

    sentiment = pd.read_csv(
        os.path.join(DIR_INTERIM, "sentiment_weekly.csv"),
        parse_dates=["date"], index_col="date"
    )
    print(f"  Sentimiento:{sentiment.shape[0]} filas × {sentiment.shape[1]} cols")

    return etfs, macro, risk, liquidity, sentiment


def resamplear_etfs(etfs):
    """Paso 2: Resamplea ETFs a semanal (último valor del viernes)."""
    print("\nPASO 2 — Resampleando ETFs a semanal (W-FRI, último valor)...")
    etfs_w = etfs.resample("W-FRI").last()
    print(f"  {etfs.shape[0]} filas diarias → {etfs_w.shape[0]} filas semanales")
    return etfs_w


def resamplear_macro(macro):
    """
    Paso 3: Resamplea macro a semanal.
    - Series diarias: tomar último valor de la semana.
    - Series mensuales: forward-fill primero para expandir, luego resamplear.
    """
    print("\nPASO 3 — Resampleando macro a semanal...")

    # Forward-fill para expandir series mensuales/semanales a diario
    print("  Aplicando forward-fill para expandir series mensuales...")
    macro_filled = macro.ffill()

    # Resamplear a semanal (último valor del viernes)
    macro_w = macro_filled.resample("W-FRI").last()
    print(f"  {macro.shape[0]} filas → {macro_w.shape[0]} filas semanales")
    return macro_w


def resamplear_riesgo(risk):
    """Paso 4: Resamplea riesgo a semanal (ffill + last)."""
    print("\nPASO 4 — Resampleando riesgo a semanal...")

    risk_filled = risk.ffill()
    risk_w = risk_filled.resample("W-FRI").last()
    print(f"  {risk.shape[0]} filas → {risk_w.shape[0]} filas semanales")
    return risk_w


def resamplear_liquidez(liquidity):
    """
    Paso 5: Resamplea liquidez a semanal.
    RRPONTSYD: rellenar con 0 antes de 2013 (la facilidad no existía).
    """
    print("\nPASO 5 — Resampleando liquidez a semanal...")

    # Rellenar RRPONTSYD con 0 antes de su inicio (pre-2013)
    if "RRPONTSYD" in liquidity.columns:
        primer_valor = liquidity["RRPONTSYD"].first_valid_index()
        if primer_valor is not None:
            liquidity.loc[:primer_valor, "RRPONTSYD"] = (
                liquidity.loc[:primer_valor, "RRPONTSYD"].fillna(0)
            )
            print(f"  RRPONTSYD: rellenado con 0 antes de {primer_valor.date()}")

    liquidity_filled = liquidity.ffill()
    liquidity_w = liquidity_filled.resample("W-FRI").last()
    print(f"  {liquidity.shape[0]} filas → {liquidity_w.shape[0]} filas semanales")
    return liquidity_w


def alinear_sentimiento(sentiment, calendario):
    """
    Paso 6: Alinea sentimiento al calendario W-FRI.
    Usa reindex con ffill para propagar valores al viernes más cercano.
    """
    print("\nPASO 6 — Alineando sentimiento al calendario semanal...")

    # Reindexar al calendario de viernes y propagar valores
    sentiment_w = sentiment.reindex(calendario, method="ffill")
    print(f"  {sentiment.shape[0]} filas → {sentiment_w.shape[0]} filas alineadas")
    return sentiment_w


def recortar_periodo(etfs_w, macro_w, risk_w, liquidity_w, sentiment_w):
    """
    Paso 7: Recorta todas las dimensiones al periodo común.
    Desde 2007-01-05 hasta la última fecha disponible en todas.
    """
    print("\nPASO 7 — Recortando al periodo común...")

    inicio = pd.Timestamp(FECHA_INICIO)

    # Encontrar la última fecha común a todas las dimensiones
    fin = min(
        etfs_w.index.max(),
        macro_w.index.max(),
        risk_w.index.max(),
        liquidity_w.index.max(),
        sentiment_w.index.max(),
    )
    print(f"  Periodo común: {inicio.date()} → {fin.date()}")

    etfs_w = etfs_w.loc[inicio:fin]
    macro_w = macro_w.loc[inicio:fin]
    risk_w = risk_w.loc[inicio:fin]
    liquidity_w = liquidity_w.loc[inicio:fin]
    sentiment_w = sentiment_w.loc[inicio:fin]

    return etfs_w, macro_w, risk_w, liquidity_w, sentiment_w


def guardar_resultados(etfs_w, macro_w, risk_w, liquidity_w, sentiment_w):
    """Paso 8: Guarda cada dimensión alineada como CSV."""
    print("\nPASO 8 — Guardando CSVs alineados...")

    archivos = {
        "etfs_weekly.csv": etfs_w,
        "macro_weekly.csv": macro_w,
        "risk_weekly.csv": risk_w,
        "liquidity_weekly.csv": liquidity_w,
        "sentiment_weekly_aligned.csv": sentiment_w,
    }

    for nombre, df in archivos.items():
        df.index.name = "date"
        ruta = os.path.join(DIR_INTERIM, nombre)
        df.to_csv(ruta)
        print(f"  Guardado: {ruta}")


def imprimir_resumen(etfs_w, macro_w, risk_w, liquidity_w, sentiment_w):
    """Paso 9: Imprime resumen final de cada dimensión alineada."""
    print(f"\n{'='*70}")
    print("PASO 9 — RESUMEN FINAL")
    print(f"{'='*70}")

    dimensiones = {
        "ETFs":        etfs_w,
        "Macro":       macro_w,
        "Riesgo":      risk_w,
        "Liquidez":    liquidity_w,
        "Sentimiento": sentiment_w,
    }

    print(f"\n{'Dimensión':<14} {'Filas':>6} {'Cols':>5} {'Inicio':>12} {'Fin':>12} {'Nulos%':>7}")
    print("-" * 60)

    for nombre, df in dimensiones.items():
        filas = df.shape[0]
        cols = df.shape[1]
        inicio = df.index.min().strftime("%Y-%m-%d")
        fin = df.index.max().strftime("%Y-%m-%d")
        total = filas * cols
        nulos = df.isnull().sum().sum()
        pct = (nulos / total * 100) if total > 0 else 0
        print(f"{nombre:<14} {filas:>6} {cols:>5} {inicio:>12} {fin:>12} {pct:>6.2f}%")

    # Verificar que todas tienen el mismo número de filas
    filas_set = {df.shape[0] for df in dimensiones.values()}
    if len(filas_set) == 1:
        print(f"\nTodas las dimensiones tienen {filas_set.pop()} semanas. Alineación correcta.")
    else:
        print(f"\n¡ATENCIÓN! Filas distintas: {filas_set}")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Paso 1: Cargar datos
    etfs, macro, risk, liquidity, sentiment = cargar_consolidados()

    # Paso 2-5: Resamplear cada dimensión a semanal
    etfs_w = resamplear_etfs(etfs)
    macro_w = resamplear_macro(macro)
    risk_w = resamplear_riesgo(risk)
    liquidity_w = resamplear_liquidez(liquidity)

    # Paso 6: Alinear sentimiento al calendario de viernes de ETFs
    calendario = etfs_w.index
    sentiment_w = alinear_sentimiento(sentiment, calendario)

    # Paso 7: Recortar al periodo común
    etfs_w, macro_w, risk_w, liquidity_w, sentiment_w = recortar_periodo(
        etfs_w, macro_w, risk_w, liquidity_w, sentiment_w
    )

    # Paso 8: Guardar
    guardar_resultados(etfs_w, macro_w, risk_w, liquidity_w, sentiment_w)

    # Paso 9: Resumen
    imprimir_resumen(etfs_w, macro_w, risk_w, liquidity_w, sentiment_w)

    print(f"\n{'='*70}")
    print("Alineación semanal completada.")
