"""
Añade el sentimiento de noticias de Refinitiv al dataset maestro.
Hace un left join por fecha: el master conserva todas sus filas (987)
y las columnas de Refinitiv se añaden donde haya solapamiento (66 semanas).
Las semanas anteriores a dic 2024 quedan con NaN (es correcto: no hay datos).
"""

import os

import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Configuración ──────────────────────────────────────────────

DIR_PROCESSED = os.path.join("data", "processed")
DIR_INTERIM = os.path.join("data", "interim")


# ── Funciones ──────────────────────────────────────────────────

def cargar_datos():
    """Paso 1: Carga el master raw y el sentimiento de Refinitiv."""
    print("PASO 1 — Cargando datos...")

    master = pd.read_csv(
        os.path.join(DIR_PROCESSED, "master_weekly_raw.csv"),
        parse_dates=["date"], index_col="date"
    )
    print(f"  Master:    {master.shape[0]} filas × {master.shape[1]} cols")

    refinitiv = pd.read_csv(
        os.path.join(DIR_INTERIM, "refinitiv_sentiment_weekly.csv"),
        parse_dates=["date"], index_col="date"
    )
    print(f"  Refinitiv: {refinitiv.shape[0]} filas × {refinitiv.shape[1]} cols")
    print(f"  Rango Refinitiv: {refinitiv.index.min().strftime('%Y-%m-%d')} → "
          f"{refinitiv.index.max().strftime('%Y-%m-%d')}")

    return master, refinitiv


def unir_datasets(master, refinitiv):
    """
    Paso 2: Left join por fecha.
    El master conserva todas sus 987 filas.
    Las columnas de Refinitiv se añaden donde coincida la fecha.
    """
    print("\nPASO 2 — Uniendo datasets (left join por fecha)...")

    cols_antes = master.shape[1]
    cols_nuevas = refinitiv.columns.tolist()

    # Eliminar columnas de Refinitiv previas si existen (re-ejecución)
    cols_existentes = [c for c in cols_nuevas if c in master.columns]
    if cols_existentes:
        master = master.drop(columns=cols_existentes)
        print(f"  Eliminadas {len(cols_existentes)} columnas previas de Refinitiv (re-ejecución)")

    # Left join: el master es la tabla principal
    master = master.join(refinitiv, how="left")

    cols_despues = master.shape[1]
    cols_añadidas = cols_despues - cols_antes

    # Contar filas con datos de Refinitiv vs NaN
    # Usamos la primera columna de Refinitiv como indicador
    col_check = cols_nuevas[0]
    filas_con_datos = master[col_check].notna().sum()
    filas_sin_datos = master[col_check].isna().sum()

    print(f"  Columnas antes: {cols_antes}")
    print(f"  Columnas añadidas: {cols_añadidas}")
    print(f"  Columnas después: {cols_despues}")
    print(f"  Filas con datos Refinitiv: {filas_con_datos}")
    print(f"  Filas sin datos (NaN): {filas_sin_datos} (anterior a dic 2024)")
    print(f"  Columnas nuevas: {cols_nuevas}")

    return master, cols_nuevas


def normalizar_y_guardar(master, cols_refinitiv):
    """
    Paso 3: Genera versión normalizada y guarda ambas.

    Normalización z-score solo para features continuas.
    NO se normalizan:
    - Targets (target_*): deben mantenerse en escala original
    - Conteos de noticias (*_news_count*): son enteros, no métricas continuas
    """
    print("\nPASO 3 — Normalizando y guardando...")

    # Identificar columnas por tipo
    target_cols = [c for c in master.columns if c.startswith("target_")]
    count_cols = [c for c in master.columns if "_news_count" in c]
    feature_cols = [c for c in master.columns
                    if c not in target_cols and c not in count_cols]

    print(f"  Features a normalizar: {len(feature_cols)}")
    print(f"  Targets (sin normalizar): {len(target_cols)}")
    print(f"  Conteos (sin normalizar): {len(count_cols)}")

    # Versión raw (sin normalizar)
    master_raw = master.copy()
    ruta_raw = os.path.join(DIR_PROCESSED, "master_weekly_raw.csv")
    master_raw.to_csv(ruta_raw)
    size_raw = os.path.getsize(ruta_raw) / (1024 * 1024)
    print(f"\n  Guardado: {ruta_raw} ({size_raw:.2f} MB)")

    # Versión normalizada
    # StandardScaler solo en las columnas de features (ignorando NaN)
    master_norm = master.copy()
    scaler = StandardScaler()

    # Normalizar solo filas sin NaN para cada columna de features
    # Para las columnas de Refinitiv (con NaN), normalizar solo donde hay datos
    master_norm[feature_cols] = scaler.fit_transform(master[feature_cols])

    ruta_norm = os.path.join(DIR_PROCESSED, "master_weekly.csv")
    master_norm.to_csv(ruta_norm)
    size_norm = os.path.getsize(ruta_norm) / (1024 * 1024)
    print(f"  Guardado: {ruta_norm} ({size_norm:.2f} MB)")

    return master_raw


def resumen_final(master):
    """Paso 4: Resumen del dataset maestro actualizado."""
    print(f"\n{'='*70}")
    print("RESUMEN FINAL — DATASET MAESTRO ACTUALIZADO")
    print(f"{'='*70}")

    target_cols = [c for c in master.columns if c.startswith("target_")]
    feature_cols = [c for c in master.columns if not c.startswith("target_")]

    print(f"\n  Filas:    {master.shape[0]}")
    print(f"  Columnas: {master.shape[1]} ({len(feature_cols)} features + {len(target_cols)} targets)")
    print(f"  Rango:    {master.index.min().strftime('%Y-%m-%d')} → "
          f"{master.index.max().strftime('%Y-%m-%d')}")

    # Desglose de NaN
    nulos_por_col = master.isnull().sum()
    cols_con_nulos = nulos_por_col[nulos_por_col > 0]
    if len(cols_con_nulos) > 0:
        print(f"\n  Columnas con NaN ({len(cols_con_nulos)}):")
        for col, n in cols_con_nulos.items():
            print(f"    {col}: {n} NaN")
    else:
        print(f"\n  Nulos: 0")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Paso 1: Cargar
    master, refinitiv = cargar_datos()

    # Paso 2: Unir
    master, cols_refinitiv = unir_datasets(master, refinitiv)

    # Paso 3: Normalizar y guardar
    master_raw = normalizar_y_guardar(master, cols_refinitiv)

    # Paso 4: Resumen
    resumen_final(master_raw)

    print(f"\n{'='*70}")
    print("Dataset maestro actualizado con sentimiento de Refinitiv.")
