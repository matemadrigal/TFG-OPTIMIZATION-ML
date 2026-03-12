"""
Construcción del dataset maestro para modelado ML.
Une features derivadas + targets (retorno futuro de cada ETF).
Genera versión normalizada y sin normalizar.

NOTA SOBRE DATA LEAKAGE:
La normalización aquí usa todos los datos (fit global).
En la fase de modelado se debe hacer train/test split ANTES de normalizar,
ajustando el scaler solo con datos de entrenamiento.
Esta versión normalizada es para exploración; la versión raw es la definitiva.
"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Configuración ──────────────────────────────────────────────

DIR_INTERIM = os.path.join("data", "interim")
DIR_PROCESSED = os.path.join("data", "processed")

ETFS = ["AGG", "EEM", "EFA", "GLD", "IWM", "LQD", "QQQ", "SPY", "TIP", "VNQ"]


# ── Funciones ──────────────────────────────────────────────────

def cargar_datos():
    """Paso 1: Carga features derivadas y precios originales de ETFs."""
    print("PASO 1 — Cargando datos...")

    features = pd.read_csv(
        os.path.join(DIR_INTERIM, "features_weekly.csv"),
        parse_dates=["date"], index_col="date"
    )
    print(f"  Features: {features.shape[0]} filas × {features.shape[1]} cols")

    etfs = pd.read_csv(
        os.path.join(DIR_INTERIM, "etfs_weekly_clean.csv"),
        parse_dates=["date"], index_col="date"
    )
    print(f"  ETFs:     {etfs.shape[0]} filas × {etfs.shape[1]} cols")

    return features, etfs


def crear_targets(features):
    """
    Paso 2: Crea columnas target = log-return de la SIGUIENTE semana.

    target_SPY(t) = log_return_SPY(t+1)

    Esto es CRÍTICO para evitar data leakage:
    - Las features contienen información del presente y pasado (t, t-1, t-2...)
    - El target es el retorno FUTURO (t+1)
    - El modelo aprende a predecir el futuro usando solo información pasada
    """
    print("\nPASO 2 — Creando targets (retorno de la semana siguiente)...")

    targets = pd.DataFrame(index=features.index)

    for ticker in ETFS:
        col_ret = f"{ticker}_log_ret"
        col_target = f"target_{ticker}"

        # shift(-1): mover el retorno una posición hacia atrás
        # Así en cada fila t, el target contiene el retorno de t+1
        targets[col_target] = features[col_ret].shift(-1)

    # La última fila queda con NaN (no conocemos el retorno futuro)
    nulos = targets.isnull().sum().sum()
    print(f"  Targets creados: {targets.shape[1]} columnas")
    print(f"  NaN en última fila: {nulos} (esperado: {len(ETFS)})")

    return targets


def juntar_dataset(features, targets):
    """
    Paso 3: Une features + targets en un solo DataFrame.
    Elimina la última fila (targets con NaN porque no hay semana siguiente).
    """
    print("\nPASO 3 — Juntando features + targets...")

    master = pd.concat([features, targets], axis=1)
    filas_antes = len(master)

    # Eliminar última fila (no tiene target)
    master = master.dropna()

    print(f"  Filas antes: {filas_antes}")
    print(f"  Filas después: {len(master)} (eliminada 1 fila sin target futuro)")
    print(f"  Columnas: {master.shape[1]} ({features.shape[1]} features + {targets.shape[1]} targets)")

    return master


def normalizar_features(master, feature_cols, target_cols):
    """
    Paso 4: Normaliza las features con StandardScaler (z-score).

    z = (x - media) / desviación_estándar

    SOLO se normalizan las features, NUNCA los targets.
    Los targets deben mantenerse en escala original (log-returns)
    para que las predicciones sean interpretables.

    NOTA: Este scaler usa todos los datos (fit global).
    En el modelado se debe re-hacer con split train/test.
    """
    print("\nPASO 4 — Normalizando features (z-score)...")

    # Versión sin normalizar (copia completa)
    master_raw = master.copy()

    # Normalizar solo las columnas de features
    scaler = StandardScaler()
    master_norm = master.copy()
    master_norm[feature_cols] = scaler.fit_transform(master[feature_cols])

    # Verificar
    medias = master_norm[feature_cols].mean().abs()
    stds = master_norm[feature_cols].std()
    print(f"  Features normalizadas: {len(feature_cols)}")
    print(f"  Media absoluta promedio: {medias.mean():.6f} (esperado ≈ 0)")
    print(f"  Std promedio: {stds.mean():.4f} (esperado ≈ 1)")
    print(f"  Targets SIN normalizar: {len(target_cols)}")

    return master_norm, master_raw


def guardar_datasets(master_norm, master_raw):
    """Paso 5: Guarda ambas versiones en data/processed/."""
    print("\nPASO 5 — Guardando datasets...")

    os.makedirs(DIR_PROCESSED, exist_ok=True)

    # Versión normalizada
    ruta_norm = os.path.join(DIR_PROCESSED, "master_weekly.csv")
    master_norm.to_csv(ruta_norm)
    size_norm = os.path.getsize(ruta_norm) / (1024 * 1024)
    print(f"  Guardado: {ruta_norm} ({size_norm:.2f} MB)")

    # Versión sin normalizar
    ruta_raw = os.path.join(DIR_PROCESSED, "master_weekly_raw.csv")
    master_raw.to_csv(ruta_raw)
    size_raw = os.path.getsize(ruta_raw) / (1024 * 1024)
    print(f"  Guardado: {ruta_raw} ({size_raw:.2f} MB)")


def resumen_final(master, feature_cols, target_cols):
    """Paso 6: Imprime resumen completo del dataset maestro."""
    print(f"\n{'='*70}")
    print("PASO 6 — RESUMEN FINAL DEL DATASET MAESTRO")
    print(f"{'='*70}")

    print(f"\n  Filas:      {master.shape[0]}")
    print(f"  Columnas:   {master.shape[1]} ({len(feature_cols)} features + {len(target_cols)} targets)")
    print(f"  Rango:      {master.index.min().strftime('%Y-%m-%d')} → "
          f"{master.index.max().strftime('%Y-%m-%d')}")
    print(f"  Nulos:      {master.isnull().sum().sum()}")

    print(f"\n  FEATURES ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"    {i:>3}. {col}")

    print(f"\n  TARGETS ({len(target_cols)}):")
    for i, col in enumerate(target_cols, 1):
        print(f"    {i:>3}. {col}")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Paso 1: Cargar
    features, etfs = cargar_datos()

    # Paso 2: Crear targets
    targets = crear_targets(features)

    # Paso 3: Juntar
    master = juntar_dataset(features, targets)

    # Identificar columnas de features y targets
    feature_cols = [c for c in master.columns if not c.startswith("target_")]
    target_cols = [c for c in master.columns if c.startswith("target_")]

    # Paso 4: Normalizar
    master_norm, master_raw = normalizar_features(master, feature_cols, target_cols)

    # Paso 5: Guardar
    guardar_datasets(master_norm, master_raw)

    # Paso 6: Resumen
    resumen_final(master_raw, feature_cols, target_cols)

    print(f"\n{'='*70}")
    print("Dataset maestro construido correctamente.")
    print("RECORDATORIO: En el modelado, normalizar SOLO con datos de train.")
