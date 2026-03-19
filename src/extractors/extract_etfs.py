"""
Extractor de datos históricos de ETFs e índices desde Yahoo Finance.
Descarga precios diarios de 10 ETFs + MOVE Index y genera CSVs.
"""

import os
from datetime import date

import pandas as pd
import yfinance as yf

# ── Configuración ──────────────────────────────────────────────
TICKERS = ["AGG", "EEM", "EFA", "GLD", "IWM", "LQD", "QQQ", "SPY", "TIP", "VNQ"]

# Índices adicionales (no son ETFs del universo, se usan como features)
INDICES_EXTRA = ["^MOVE"]  # MOVE Index (volatilidad implícita de bonos del Tesoro)

FECHA_INICIO = "2007-01-01"
FECHA_FIN = date.today().strftime("%Y-%m-%d")

# Rutas de salida (relativas a la raíz del proyecto)
DIR_RAW = os.path.join("data", "raw", "etfs")
DIR_INTERIM = os.path.join("data", "interim")


# ── Funciones ──────────────────────────────────────────────────

def descargar_etf(ticker, inicio, fin):
    """Descarga datos históricos diarios de un ETF desde Yahoo Finance."""
    print(f"\nDescargando {ticker}...")
    df = yf.download(ticker, start=inicio, end=fin, auto_adjust=True, progress=False)
    print(f"  Filas descargadas: {len(df)}")

    if len(df) > 0:
        print(f"  Rango de fechas: {df.index.min().date()} → {df.index.max().date()}")
        nulos = df.isnull().sum().sum()
        print(f"  Valores nulos: {nulos}")
    else:
        print("  ⚠ No se obtuvieron datos")

    return df


def guardar_csv_individual(df, ticker, directorio):
    """Guarda el DataFrame de un ETF como CSV individual."""
    ruta = os.path.join(directorio, f"{ticker}_daily.csv")
    df.to_csv(ruta)
    print(f"  Guardado en: {ruta}")


def crear_consolidado(diccionario_precios, directorio):
    """
    Crea un CSV consolidado con el precio de cierre ajustado de todos los ETFs.
    Columnas = tickers, filas = fechas.
    """
    # Unir todos los precios de cierre en un solo DataFrame
    consolidado = pd.DataFrame()
    for ticker, df in diccionario_precios.items():
        if len(df) > 0 and "Close" in df.columns:
            consolidado[ticker] = df["Close"]

    # Ordenar por fecha
    consolidado = consolidado.sort_index()

    # Guardar
    ruta = os.path.join(directorio, "etfs_prices_daily.csv")
    consolidado.to_csv(ruta)

    print(f"\n{'='*50}")
    print(f"Consolidado guardado en: {ruta}")
    print(f"  Dimensiones: {consolidado.shape[0]} filas × {consolidado.shape[1]} columnas")
    print(f"  Rango: {consolidado.index.min().date()} → {consolidado.index.max().date()}")
    print(f"  Nulos totales: {consolidado.isnull().sum().sum()}")

    return consolidado


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Periodo de descarga: {FECHA_INICIO} → {FECHA_FIN}")
    print(f"ETFs a descargar: {', '.join(TICKERS)}")
    print(f"Índices extra: {', '.join(INDICES_EXTRA)}")

    # Crear directorios si no existen
    os.makedirs(DIR_RAW, exist_ok=True)
    os.makedirs(DIR_INTERIM, exist_ok=True)

    # Descargar cada ETF y guardar su CSV individual
    precios = {}
    for ticker in TICKERS:
        df = descargar_etf(ticker, FECHA_INICIO, FECHA_FIN)
        guardar_csv_individual(df, ticker, DIR_RAW)
        precios[ticker] = df

    # Crear y guardar el CSV consolidado de ETFs
    crear_consolidado(precios, DIR_INTERIM)

    # Descargar índices extra (MOVE, etc.) como CSVs separados
    for ticker in INDICES_EXTRA:
        df = descargar_etf(ticker, FECHA_INICIO, FECHA_FIN)
        # Nombre limpio para archivo: ^MOVE → MOVE
        nombre = ticker.replace("^", "")
        ruta = os.path.join(DIR_RAW, f"{nombre}_daily.csv")
        df.to_csv(ruta)
        print(f"  Guardado en: {ruta}")

        # Guardar también en interim como serie de cierre
        if len(df) > 0 and "Close" in df.columns:
            serie = df[["Close"]].copy()
            serie.columns = [nombre]
            ruta_interim = os.path.join(DIR_INTERIM, f"{nombre}_daily.csv")
            serie.to_csv(ruta_interim)
            print(f"  Interim: {ruta_interim} ({len(serie)} filas)")

    print(f"\n{'='*50}")
    print("Extracción completada.")
