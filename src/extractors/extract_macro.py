"""
Extractor de series macroeconómicas desde la API de FRED.
Descarga 10 indicadores macro y genera CSVs individuales y consolidado.
"""

import os
import time
from datetime import date

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

# ── Configuración ──────────────────────────────────────────────

# Cargar la API key desde el archivo .env
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

FECHA_INICIO = "2007-01-01"
FECHA_FIN = date.today().strftime("%Y-%m-%d")

# Series a descargar: código FRED → descripción
SERIES = {
    "BAMLH0A0HYM2": "HY OAS spread",
    "CCSA":          "Continued claims desempleo (semanal)",
    "CPILFESL":      "Core CPI",
    "DGS2":          "Treasury 2 años",
    "DGS10":         "Treasury 10 años",
    "ICSA":          "Initial claims desempleo",
    "PCEPILFE":      "Core PCE",
    "T5YIFR":        "5Y forward inflation",
    "T10Y3M":        "Spread 10Y-3M Treasury (inversión = recesión)",
    "UMCSENT":       "Sentimiento consumidor UMich",
    "UNRATE":        "Tasa desempleo",
    "VIXCLS":        "VIX",
    "WEI":           "Weekly Economic Index (actividad real semanal)",
}

# Rutas de salida
DIR_RAW = os.path.join("data", "raw", "macro")
DIR_INTERIM = os.path.join("data", "interim")


# ── Funciones ──────────────────────────────────────────────────

def descargar_serie(fred, codigo, descripcion, inicio, fin, max_reintentos=3):
    """Descarga una serie temporal desde FRED y muestra info por consola."""
    print(f"\nDescargando {codigo} ({descripcion})...")

    # Reintentar si FRED devuelve error temporal (500, timeout, etc.)
    for intento in range(1, max_reintentos + 1):
        try:
            serie = fred.get_series(codigo, observation_start=inicio, observation_end=fin)
            break
        except Exception as e:
            if intento < max_reintentos:
                espera = intento * 5
                print(f"  Error en intento {intento}: {e}")
                print(f"  Reintentando en {espera}s...")
                time.sleep(espera)
            else:
                print(f"  ERROR: No se pudo descargar {codigo} tras {max_reintentos} intentos")
                return None

    # Convertir a DataFrame con columna nombrada
    df = serie.to_frame(name=codigo)
    df.index.name = "date"

    # Detectar frecuencia aproximada por la mediana de días entre observaciones
    if len(df) > 1:
        dias_entre = pd.Series(df.index).diff().dt.days.median()
        if dias_entre <= 1:
            frecuencia = "diaria"
        elif dias_entre <= 7:
            frecuencia = "semanal"
        else:
            frecuencia = "mensual"
    else:
        frecuencia = "desconocida"

    nulos = df[codigo].isnull().sum()
    print(f"  Frecuencia: {frecuencia}")
    print(f"  Filas: {len(df)}")
    print(f"  Rango: {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Nulos: {nulos}")

    return df


def guardar_csv_individual(df, codigo, directorio):
    """Guarda la serie como CSV individual con sufijo _api."""
    ruta = os.path.join(directorio, f"{codigo}_api.csv")
    df.to_csv(ruta)
    print(f"  Guardado en: {ruta}")


def crear_consolidado(diccionario_series, directorio):
    """
    Une todas las series en un solo DataFrame.
    Usa outer join para conservar todas las fechas de todas las frecuencias.
    """
    consolidado = pd.DataFrame()
    for codigo, df in diccionario_series.items():
        consolidado[codigo] = df[codigo]

    # Ordenar por fecha
    consolidado = consolidado.sort_index()
    consolidado.index.name = "date"

    # Guardar
    ruta = os.path.join(directorio, "macro_daily.csv")
    consolidado.to_csv(ruta)

    print(f"\n{'='*50}")
    print(f"Consolidado guardado en: {ruta}")
    print(f"  Dimensiones: {consolidado.shape[0]} filas × {consolidado.shape[1]} columnas")
    print(f"  Rango: {consolidado.index.min().date()} → {consolidado.index.max().date()}")
    print(f"  Nulos totales: {consolidado.isnull().sum().sum()}")

    return consolidado


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Verificar que la API key está configurada
    if not FRED_API_KEY:
        print("ERROR: No se encontró FRED_API_KEY en el archivo .env")
        print("Crea un archivo .env en la raíz del proyecto con: FRED_API_KEY=tu_clave")
        exit(1)

    print(f"Periodo de descarga: {FECHA_INICIO} → {FECHA_FIN}")
    print(f"Series a descargar: {len(SERIES)}")

    # Crear directorios si no existen
    os.makedirs(DIR_RAW, exist_ok=True)
    os.makedirs(DIR_INTERIM, exist_ok=True)

    # Conectar a FRED
    fred = Fred(api_key=FRED_API_KEY)

    # Descargar cada serie y guardar CSV individual
    series_descargadas = {}
    for codigo, descripcion in SERIES.items():
        df = descargar_serie(fred, codigo, descripcion, FECHA_INICIO, FECHA_FIN)
        if df is not None:
            guardar_csv_individual(df, codigo, DIR_RAW)
            series_descargadas[codigo] = df
        # Pausa de 1s entre peticiones para no saturar la API
        time.sleep(1)

    # Crear y guardar el consolidado
    crear_consolidado(series_descargadas, DIR_INTERIM)

    print(f"\n{'='*50}")
    print("Extracción macro completada.")
