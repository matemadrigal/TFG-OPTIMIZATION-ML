"""
Combina los datos de Google Trends de dos fuentes:
- CSVs de la API (semanales, parciales por bloqueo de Google)
- CSVs descargados manualmente (mensuales, más completos)

Prioriza los datos de API donde haya solapamiento.
Los datos manuales (mensuales) se resamplean a mensual para rellenar huecos.
El resultado final mantiene frecuencia semanal donde hay datos de API,
y mensual donde solo hay datos manuales.
"""

import os
import glob

import pandas as pd

# ── Configuración ──────────────────────────────────────────────

# Mapeo: término → nombre del archivo manual (ojo: "unemployement" con typo en el original)
TERMINOS = {
    "recession":    "recession.csv",
    "inflation":    "inflation.csv",
    "bear market":  "bear market.csv",
    "bull market":  "bull market.csv",
    "buy stocks":   "buy stocks.csv",
    "sell stocks":  "sell stocks.csv",
    "unemployment": "unemployement.csv",  # Typo en el archivo original
}

DIR_RAW = os.path.join("data", "raw", "sentiment")
DIR_INTERIM = os.path.join("data", "interim")


# ── Funciones ──────────────────────────────────────────────────

def leer_csv_api(termino, directorio):
    """
    Lee el CSV descargado con pytrends (semanal).
    Formato: date,termino con fechas YYYY-MM-DD.
    """
    nombre = termino.replace(" ", "_") + "_trends_api.csv"
    ruta = os.path.join(directorio, nombre)

    if not os.path.exists(ruta):
        return None

    df = pd.read_csv(ruta, parse_dates=["date"], index_col="date")
    return df


def leer_csv_manual(nombre_archivo, directorio):
    """
    Lee el CSV descargado manualmente desde Google Trends.
    Formato especial: línea 1 = categoría, línea 2 = vacía, línea 3+ = datos.
    Las fechas son mensuales (YYYY-MM).
    """
    ruta = os.path.join(directorio, nombre_archivo)

    if not os.path.exists(ruta):
        return None

    # Saltar las 2 primeras líneas (categoría + línea vacía)
    df = pd.read_csv(ruta, skiprows=2)

    # Renombrar columnas
    col_fecha = df.columns[0]   # "Mes"
    col_valor = df.columns[1]   # "termino: (Todo el mundo)"

    # Convertir fecha mensual YYYY-MM al primer día del mes
    df[col_fecha] = pd.to_datetime(df[col_fecha], format="%Y-%m")
    df = df.rename(columns={col_fecha: "date", col_valor: "value"})
    df = df.set_index("date")

    # Convertir a numérico (Google Trends pone "<1" a veces)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df


def combinar_fuentes(df_api, df_manual, termino):
    """
    Combina datos de API (semanales) con datos manuales (mensuales).
    - Donde hay datos de API, se usan esos (son semanales, más granulares).
    - Donde solo hay datos manuales, se usan los mensuales para cubrir el hueco.
    """
    nombre_col = termino.replace(" ", "_")

    # Si solo existe una fuente, devolverla directamente
    if df_api is None and df_manual is None:
        return None
    if df_api is not None and df_manual is None:
        df_api.columns = [nombre_col]
        return df_api
    if df_api is None and df_manual is not None:
        df_manual.columns = [nombre_col]
        return df_manual

    # Ambas fuentes existen: combinar
    # Determinar el rango que cubre la API
    api_inicio = df_api.index.min()
    api_fin = df_api.index.max()

    # Tomar datos manuales solo donde la API NO tiene cobertura
    # (antes del inicio de la API o después del fin de la API)
    manual_antes = df_manual[df_manual.index < api_inicio].copy()
    manual_despues = df_manual[df_manual.index > api_fin].copy()

    # Renombrar columnas para que coincidan
    df_api.columns = [nombre_col]
    manual_antes.columns = [nombre_col]
    manual_despues.columns = [nombre_col]

    # Concatenar: manual_antes + API + manual_despues
    partes = []
    if len(manual_antes) > 0:
        partes.append(manual_antes)
    partes.append(df_api)
    if len(manual_despues) > 0:
        partes.append(manual_despues)

    resultado = pd.concat(partes)
    resultado = resultado.sort_index()
    resultado = resultado[~resultado.index.duplicated(keep="first")]

    return resultado


def crear_consolidado(diccionario_merged, directorio):
    """
    Crea el CSV consolidado de sentimiento con todos los términos
    de Google Trends + AAII Sentiment Survey.
    """
    consolidado = pd.DataFrame()

    # Añadir cada término de Google Trends
    for termino, df in diccionario_merged.items():
        nombre_col = termino.replace(" ", "_")
        consolidado[nombre_col] = df[nombre_col]

    # Añadir AAII si existe
    ruta_aaii = os.path.join("data", "raw", "sentiment", "aaii_sentiment_clean.csv")
    if os.path.exists(ruta_aaii):
        df_aaii = pd.read_csv(ruta_aaii, parse_dates=["date"], index_col="date")
        for col in df_aaii.columns:
            consolidado[f"aaii_{col}"] = df_aaii[col]

    consolidado = consolidado.sort_index()
    consolidado.index.name = "date"

    ruta = os.path.join(directorio, "sentiment_weekly.csv")
    consolidado.to_csv(ruta)

    print(f"\n{'='*60}")
    print(f"Consolidado guardado en: {ruta}")
    print(f"  Dimensiones: {consolidado.shape[0]} filas × {consolidado.shape[1]} columnas")
    print(f"  Rango: {consolidado.index.min().date()} → {consolidado.index.max().date()}")
    print(f"  Nulos totales: {consolidado.isnull().sum().sum()}")

    return consolidado


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Combinando fuentes de Google Trends (API + manual)")
    print(f"Términos: {len(TERMINOS)}")

    os.makedirs(DIR_INTERIM, exist_ok=True)
    merged = {}

    print(f"\n{'Término':<16} {'API':>6} {'Manual':>8} {'Combinado':>10}   Rango final")
    print("-" * 75)

    for termino, archivo_manual in TERMINOS.items():
        # Leer ambas fuentes
        df_api = leer_csv_api(termino, DIR_RAW)
        df_manual = leer_csv_manual(archivo_manual, DIR_RAW)

        filas_api = len(df_api) if df_api is not None else 0
        filas_manual = len(df_manual) if df_manual is not None else 0

        # Combinar
        df_merged = combinar_fuentes(df_api, df_manual, termino)

        if df_merged is not None:
            # Guardar CSV combinado
            nombre_salida = termino.replace(" ", "_") + "_trends_merged.csv"
            ruta_salida = os.path.join(DIR_RAW, nombre_salida)
            df_merged.index.name = "date"
            df_merged.to_csv(ruta_salida)

            merged[termino] = df_merged

            rango = f"{df_merged.index.min().date()} → {df_merged.index.max().date()}"
            print(f"{termino:<16} {filas_api:>6} {filas_manual:>8} {len(df_merged):>10}   {rango}")
        else:
            print(f"{termino:<16} {filas_api:>6} {filas_manual:>8}          0   SIN DATOS")

    # Crear consolidado
    crear_consolidado(merged, DIR_INTERIM)

    print(f"\n{'='*60}")
    print("Merge de Google Trends completado.")
