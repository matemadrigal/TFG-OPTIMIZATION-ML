"""
Extractor de titulares de noticias de ETFs desde Refinitiv/LSEG Workspace.
Descarga titulares paginando en lotes de 1000 y genera CSVs por ETF y consolidado.

REQUISITOS:
- LSEG Workspace abierto en Windows
- Proxy local corriendo: python src/extractors/refinitiv_proxy.py &
"""

import os
import time

import pandas as pd
import refinitiv.data as rd

# ── Configuración ──────────────────────────────────────────────

TICKERS = ["SPY", "QQQ", "IWM", "EEM", "EFA", "AGG", "LQD", "GLD", "TIP", "VNQ"]

# Número de lotes de 1000 titulares por ETF (5 lotes ≈ 5000 titulares ≈ 6 meses)
LOTES_POR_ETF = 5
TITULARES_POR_LOTE = 1000

# Rutas de salida
DIR_RAW = os.path.join("data", "raw", "sentiment")


# ── Funciones ──────────────────────────────────────────────────

def descargar_titulares_etf(ticker, lotes=5, titulares_por_lote=1000, max_reintentos=3):
    """
    Descarga titulares de noticias de un ETF paginando hacia atrás.
    Cada lote trae hasta 1000 titulares; se encadenan usando la fecha
    más antigua del lote anterior como tope del siguiente.
    """
    print(f"\nDescargando noticias de {ticker}...")
    todos_los_lotes = []
    end_date = None  # None = desde ahora hacia atrás

    for i in range(lotes):
        # Preparar parámetros de la petición
        kwargs = {"query": ticker, "count": titulares_por_lote}
        if end_date:
            kwargs["end"] = end_date

        # Intentar descargar con reintentos
        df = None
        for intento in range(1, max_reintentos + 1):
            try:
                df = rd.news.get_headlines(**kwargs)
                break
            except Exception as e:
                if intento < max_reintentos:
                    print(f"    Error en intento {intento}: {e}")
                    print(f"    Reintentando en 3s...")
                    time.sleep(3)
                else:
                    print(f"    ERROR: No se pudo descargar lote {i+1} tras {max_reintentos} intentos")

        # Si no hay datos, parar
        if df is None or len(df) == 0:
            print(f"  Lote {i+1}: sin datos, parando")
            break

        print(f"  Lote {i+1}/{lotes}: {len(df)} titulares "
              f"({df.index.min().strftime('%Y-%m-%d')} → {df.index.max().strftime('%Y-%m-%d')})")
        todos_los_lotes.append(df)

        # Calcular fecha tope del siguiente lote (1 segundo antes del más antiguo)
        end_date = (df.index.min() - pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S")

        # Pausa entre lotes
        time.sleep(1)

    # Concatenar todos los lotes y eliminar duplicados
    if todos_los_lotes:
        df_completo = pd.concat(todos_los_lotes)
        df_completo = df_completo[~df_completo.index.duplicated(keep="first")]
        df_completo = df_completo.sort_index()

        # Añadir columna con el ticker
        df_completo["ticker"] = ticker

        # Renombrar columnas para claridad
        df_completo.index.name = "date"

        nulos = df_completo["headline"].isnull().sum()
        print(f"  Total {ticker}: {len(df_completo)} titulares únicos")
        print(f"  Rango: {df_completo.index.min().strftime('%Y-%m-%d')} → "
              f"{df_completo.index.max().strftime('%Y-%m-%d')}")
        print(f"  Nulos en headline: {nulos}")

        return df_completo
    else:
        print(f"  No se obtuvieron titulares para {ticker}")
        return None


def guardar_csv_individual(df, ticker, directorio):
    """Guarda los titulares de un ETF como CSV individual."""
    ruta = os.path.join(directorio, f"{ticker}_news_refinitiv.csv")
    df.to_csv(ruta)
    print(f"  Guardado en: {ruta}")


def crear_consolidado(diccionario_noticias, directorio):
    """Une todos los titulares en un solo CSV."""
    partes = [df for df in diccionario_noticias.values() if df is not None]

    if not partes:
        print("No hay datos para consolidar.")
        return None

    consolidado = pd.concat(partes)
    consolidado = consolidado.sort_index()
    consolidado.index.name = "date"

    ruta = os.path.join(directorio, "all_news_refinitiv.csv")
    consolidado.to_csv(ruta)

    print(f"\n{'='*50}")
    print(f"Consolidado guardado en: {ruta}")
    print(f"  Titulares totales: {len(consolidado)}")
    print(f"  Rango: {consolidado.index.min().strftime('%Y-%m-%d')} → "
          f"{consolidado.index.max().strftime('%Y-%m-%d')}")
    print(f"  Titulares por ETF:")
    for ticker, count in consolidado["ticker"].value_counts().sort_index().items():
        print(f"    {ticker}: {count}")

    return consolidado


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"ETFs a descargar: {', '.join(TICKERS)}")
    print(f"Lotes por ETF: {LOTES_POR_ETF} × {TITULARES_POR_LOTE} = "
          f"hasta {LOTES_POR_ETF * TITULARES_POR_LOTE} titulares")

    os.makedirs(DIR_RAW, exist_ok=True)

    # Abrir sesión de Refinitiv
    print("\nConectando a Refinitiv...")
    rd.open_session()

    # Descargar titulares de cada ETF
    noticias = {}
    for ticker in TICKERS:
        df = descargar_titulares_etf(ticker, LOTES_POR_ETF, TITULARES_POR_LOTE)
        if df is not None:
            guardar_csv_individual(df, ticker, DIR_RAW)
            noticias[ticker] = df
        # Pausa entre ETFs
        time.sleep(2)

    # Cerrar sesión
    rd.close_session()

    # Crear consolidado
    crear_consolidado(noticias, DIR_RAW)

    print(f"\n{'='*50}")
    print("Extracción de noticias completada.")
