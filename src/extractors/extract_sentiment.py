"""
Extractor de datos de sentimiento de mercado.
Parte 1: Google Trends (interés de búsqueda semanal de 7 términos).
Parte 2: AAII Sentiment Survey (lectura y limpieza del archivo .xls existente).
"""

import os
import time
from datetime import date

import pandas as pd

# ── Configuración ──────────────────────────────────────────────

FECHA_INICIO = "2007-01-01"
FECHA_FIN = date.today().strftime("%Y-%m-%d")

# Términos de búsqueda para Google Trends
TERMINOS = [
    "recession",
    "inflation",
    "bear market",
    "bull market",
    "buy stocks",
    "sell stocks",
    "unemployment",
]

# Rutas de salida
DIR_RAW = os.path.join("data", "raw", "sentiment")
DIR_INTERIM = os.path.join("data", "interim")

# Ruta del archivo AAII existente
RUTA_AAII_XLS = os.path.join(DIR_RAW, "aai bull bear sentiment survey.xls")


# ── Parte 1: Google Trends ────────────────────────────────────

def generar_tramos(inicio, fin, anos_por_tramo=5):
    """
    Divide el periodo en tramos de N años para Google Trends.
    Google Trends solo devuelve datos semanales si el rango es <= 5 años.
    """
    from datetime import datetime

    tramos = []
    actual = datetime.strptime(inicio, "%Y-%m-%d")
    final = datetime.strptime(fin, "%Y-%m-%d")

    while actual < final:
        fin_tramo = actual.replace(year=actual.year + anos_por_tramo)
        if fin_tramo > final:
            fin_tramo = final
        tramos.append((actual.strftime("%Y-%m-%d"), fin_tramo.strftime("%Y-%m-%d")))
        actual = fin_tramo

    return tramos


def descargar_trends_termino(termino, inicio, fin):
    """
    Descarga el interés semanal de un término de Google Trends.
    Divide en tramos de 5 años y concatena los resultados.
    """
    from pytrends.request import TrendReq

    print(f"\nDescargando Google Trends: '{termino}'...")
    pytrends = TrendReq(hl="es", tz=360)

    tramos = generar_tramos(inicio, fin)
    partes = []

    for i, (t_inicio, t_fin) in enumerate(tramos):
        print(f"  Tramo {i+1}/{len(tramos)}: {t_inicio} → {t_fin}")

        try:
            pytrends.build_payload(
                kw_list=[termino],
                timeframe=f"{t_inicio} {t_fin}",
                geo="",  # Todo el mundo
            )
            df_tramo = pytrends.interest_over_time()

            if not df_tramo.empty:
                # Quitar la columna 'isPartial' que añade pytrends
                if "isPartial" in df_tramo.columns:
                    df_tramo = df_tramo.drop(columns=["isPartial"])
                partes.append(df_tramo)
                print(f"    → {len(df_tramo)} filas obtenidas")
            else:
                print(f"    → Sin datos para este tramo")

        except Exception as e:
            print(f"    → Error: {e}")

        # Pausa de 5s entre peticiones para evitar bloqueo de Google
        if i < len(tramos) - 1:
            print(f"    Esperando 5s...")
            time.sleep(5)

    # Concatenar todos los tramos
    if partes:
        df_completo = pd.concat(partes)
        # Eliminar duplicados por fecha (solapamientos entre tramos)
        df_completo = df_completo[~df_completo.index.duplicated(keep="last")]
        df_completo = df_completo.sort_index()
        df_completo.index.name = "date"

        print(f"  Total: {len(df_completo)} filas, "
              f"rango {df_completo.index.min().date()} → {df_completo.index.max().date()}")
        return df_completo
    else:
        print(f"  No se obtuvieron datos para '{termino}'")
        return None


def descargar_todos_los_trends(terminos, inicio, fin, directorio):
    """
    Descarga todos los términos de Google Trends y guarda CSVs individuales.
    Si Google bloquea la descarga, reutiliza el CSV previo si existe.
    """
    trends_descargados = {}

    for termino in terminos:
        nombre_archivo = termino.replace(" ", "_") + "_trends_api.csv"
        ruta = os.path.join(directorio, nombre_archivo)

        df = descargar_trends_termino(termino, inicio, fin)

        if df is not None:
            # Descarga exitosa: guardar nuevo CSV
            df.to_csv(ruta)
            print(f"  Guardado en: {ruta}")
            trends_descargados[termino] = df
        elif os.path.exists(ruta):
            # Descarga fallida pero existe CSV previo: reutilizarlo
            df_previo = pd.read_csv(ruta, parse_dates=[0], index_col=0)
            trends_descargados[termino] = df_previo
            print(f"  Usando CSV previo: {ruta} ({len(df_previo)} filas)")

        # Pausa entre términos
        print(f"  Esperando 5s antes del siguiente término...")
        time.sleep(5)

    return trends_descargados


# ── Parte 2: AAII Sentiment Survey ────────────────────────────

def limpiar_aaii(ruta_xls, directorio_salida):
    """
    Lee el archivo .xls de AAII y extrae las columnas relevantes:
    Date, Bullish, Neutral, Bearish.
    Guarda el resultado como CSV limpio.
    """
    print(f"\n{'='*50}")
    print("Procesando AAII Sentiment Survey...")
    print(f"  Leyendo: {ruta_xls}")

    # Leer sin header (la estructura del xls tiene cabeceras en fila 3)
    df_raw = pd.read_excel(ruta_xls, header=None)

    # Los datos empiezan en la fila 5 (índice 5)
    # Columnas: 0=Date, 1=Bullish, 2=Neutral, 3=Bearish
    df = df_raw.iloc[5:, [0, 1, 2, 3]].copy()
    df.columns = ["date", "bullish", "neutral", "bearish"]

    # Convertir fecha a datetime y descartar filas sin fecha válida
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Convertir porcentajes a numérico
    df["bullish"] = pd.to_numeric(df["bullish"], errors="coerce")
    df["neutral"] = pd.to_numeric(df["neutral"], errors="coerce")
    df["bearish"] = pd.to_numeric(df["bearish"], errors="coerce")

    # Filtrar desde 2007
    df = df[df["date"] >= "2007-01-01"]
    df = df.set_index("date").sort_index()

    # Guardar
    ruta_salida = os.path.join(directorio_salida, "aaii_sentiment_clean.csv")
    df.to_csv(ruta_salida)

    nulos = df.isnull().sum().sum()
    print(f"  Filas: {len(df)}")
    print(f"  Rango: {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Nulos: {nulos}")
    print(f"  Guardado en: {ruta_salida}")

    return df


# ── Consolidado de sentimiento ─────────────────────────────────

def crear_consolidado(trends_dict, df_aaii, directorio):
    """
    Crea un CSV consolidado con todos los indicadores de sentimiento.
    Combina Google Trends (semanal) con AAII (semanal).
    """
    consolidado = pd.DataFrame()

    # Añadir cada término de Google Trends
    for termino, df in trends_dict.items():
        nombre_col = termino.replace(" ", "_")
        consolidado[nombre_col] = df.iloc[:, 0]  # Primera columna de cada df

    # Añadir AAII
    if df_aaii is not None:
        for col in df_aaii.columns:
            consolidado[f"aaii_{col}"] = df_aaii[col]

    consolidado = consolidado.sort_index()
    consolidado.index.name = "date"

    # Guardar
    ruta = os.path.join(directorio, "sentiment_weekly.csv")
    consolidado.to_csv(ruta)

    print(f"\n{'='*50}")
    print(f"Consolidado guardado en: {ruta}")
    print(f"  Dimensiones: {consolidado.shape[0]} filas × {consolidado.shape[1]} columnas")
    if len(consolidado) > 0:
        print(f"  Rango: {consolidado.index.min().date()} → {consolidado.index.max().date()}")
    print(f"  Nulos totales: {consolidado.isnull().sum().sum()}")

    return consolidado


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Periodo de descarga: {FECHA_INICIO} → {FECHA_FIN}")

    os.makedirs(DIR_RAW, exist_ok=True)
    os.makedirs(DIR_INTERIM, exist_ok=True)

    # Parte 1: Google Trends
    print(f"\n{'='*50}")
    print("PARTE 1: Google Trends")
    print(f"Términos a descargar: {len(TERMINOS)}")
    trends = descargar_todos_los_trends(TERMINOS, FECHA_INICIO, FECHA_FIN, DIR_RAW)

    # Parte 2: AAII Sentiment Survey
    df_aaii = None
    if os.path.exists(RUTA_AAII_XLS):
        df_aaii = limpiar_aaii(RUTA_AAII_XLS, DIR_RAW)
    else:
        print(f"\nArchivo AAII no encontrado en: {RUTA_AAII_XLS}")
        print("Se omite la parte de AAII.")

    # Consolidado
    crear_consolidado(trends, df_aaii, DIR_INTERIM)

    print(f"\n{'='*50}")
    print("Extracción de sentimiento completada.")
