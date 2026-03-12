"""
Limpieza de datos semanales alineados.
Rellena nulos, elimina duplicados entre dimensiones y verifica rangos lógicos.
"""

import os

import pandas as pd

# ── Configuración ──────────────────────────────────────────────

DIR_INTERIM = os.path.join("data", "interim")


# ── Funciones ──────────────────────────────────────────────────

def cargar_datos():
    """Paso 1: Carga los 5 CSVs semanales alineados."""
    print("PASO 1 — Cargando datos semanales alineados...")

    archivos = {
        "etfs":       "etfs_weekly.csv",
        "macro":      "macro_weekly.csv",
        "risk":       "risk_weekly.csv",
        "liquidity":  "liquidity_weekly.csv",
        "sentiment":  "sentiment_weekly_aligned.csv",
    }

    datos = {}
    for nombre, archivo in archivos.items():
        df = pd.read_csv(os.path.join(DIR_INTERIM, archivo), parse_dates=["date"], index_col="date")
        nulos = df.isnull().sum().sum()
        print(f"  {nombre:<12} {df.shape[0]} filas × {df.shape[1]} cols, nulos: {nulos}")
        datos[nombre] = df

    return datos


def limpiar_nulos_macro(df):
    """Paso 2: Limpia nulos de macro con ffill + bfill."""
    print("\nPASO 2 — Limpiando nulos de MACRO...")
    nulos_antes = df.isnull().sum().sum()

    # Forward-fill: propagar último valor conocido hacia adelante
    df = df.ffill()
    # Backward-fill: cubrir los primeros valores sin dato (inicio de serie)
    df = df.bfill()

    nulos_despues = df.isnull().sum().sum()
    print(f"  Nulos antes: {nulos_antes} → después: {nulos_despues}")
    return df


def limpiar_nulos_riesgo(df):
    """Paso 3: Limpia nulos de riesgo con ffill + bfill."""
    print("\nPASO 3 — Limpiando nulos de RIESGO...")
    nulos_antes = df.isnull().sum().sum()

    df = df.ffill()
    df = df.bfill()

    nulos_despues = df.isnull().sum().sum()
    print(f"  Nulos antes: {nulos_antes} → después: {nulos_despues}")
    return df


def limpiar_nulos_sentimiento(df):
    """
    Paso 4: Limpia nulos de sentimiento.
    - AAII: ffill (encuesta del jueves vale para el viernes)
    - Google Trends: ffill (dato mensual vigente hasta el siguiente)
    - bfill para cubrir el inicio
    """
    print("\nPASO 4 — Limpiando nulos de SENTIMIENTO...")
    nulos_antes = df.isnull().sum().sum()

    # Mostrar nulos por columna antes
    print("  Nulos por columna (antes):")
    for col in df.columns:
        n = df[col].isnull().sum()
        if n > 0:
            print(f"    {col}: {n}")

    # Forward-fill: el último valor conocido se mantiene vigente
    df = df.ffill()
    # Backward-fill: cubrir el inicio donde no hay dato previo
    df = df.bfill()

    nulos_despues = df.isnull().sum().sum()
    print(f"  Nulos totales: {nulos_antes} → {nulos_despues}")
    return df


def eliminar_duplicados(datos):
    """
    Paso 5: Elimina series duplicadas entre dimensiones.
    - VIXCLS: se queda en riesgo, se elimina de macro
    - BAMLH0A0HYM2: se queda en riesgo, se elimina de macro
    - RRPONTSYD: se queda en liquidez, se elimina de riesgo
    """
    print("\nPASO 5 — Eliminando columnas duplicadas entre dimensiones...")

    eliminaciones = [
        ("macro", "VIXCLS",       "riesgo"),
        ("macro", "BAMLH0A0HYM2", "riesgo"),
        ("risk",  "RRPONTSYD",    "liquidez"),
    ]

    for origen, columna, destino in eliminaciones:
        df = datos[origen]
        if columna in df.columns:
            df = df.drop(columns=[columna])
            datos[origen] = df
            print(f"  {columna} eliminado de {origen} (se queda en {destino})")
        else:
            print(f"  {columna} no encontrado en {origen}, nada que eliminar")

    # Mostrar columnas finales de cada dimensión
    for nombre, df in datos.items():
        print(f"  {nombre}: {list(df.columns)}")

    return datos


def verificar_rangos(datos):
    """
    Paso 6: Verifica que los valores estén dentro de rangos lógicos.
    Imprime alertas si hay valores fuera de rango.
    """
    print("\nPASO 6 — Verificando rangos lógicos...")
    alertas = 0

    # ETFs: precios > 0
    etfs = datos["etfs"]
    negativos = (etfs <= 0).sum().sum()
    if negativos > 0:
        print(f"  ALERTA: {negativos} precios de ETFs <= 0")
        alertas += 1
    else:
        print(f"  ETFs: todos los precios > 0 ✓")

    # VIX: entre 0 y 100
    risk = datos["risk"]
    if "VIXCLS" in risk.columns:
        vix = risk["VIXCLS"]
        fuera = ((vix < 0) | (vix > 100)).sum()
        if fuera > 0:
            print(f"  ALERTA: {fuera} valores de VIX fuera de [0, 100]")
            alertas += 1
        else:
            print(f"  VIX: todos los valores en [0, 100] ✓")

    # UNRATE: entre 0 y 30
    macro = datos["macro"]
    if "UNRATE" in macro.columns:
        unrate = macro["UNRATE"]
        fuera = ((unrate < 0) | (unrate > 30)).sum()
        if fuera > 0:
            print(f"  ALERTA: {fuera} valores de UNRATE fuera de [0, 30]")
            alertas += 1
        else:
            print(f"  UNRATE: todos los valores en [0, 30] ✓")

    # AAII: verificar formato (0-1 o 0-100)
    sentiment = datos["sentiment"]
    for col in ["aaii_bullish", "aaii_neutral", "aaii_bearish"]:
        if col in sentiment.columns:
            vals = sentiment[col]
            maximo = vals.max()
            if maximo > 1:
                # Formato 0-100, verificar rango
                fuera = ((vals < 0) | (vals > 100)).sum()
                rango_str = "[0, 100]"
            else:
                # Formato 0-1, verificar rango
                fuera = ((vals < 0) | (vals > 1)).sum()
                rango_str = "[0, 1]"
            if fuera > 0:
                print(f"  ALERTA: {fuera} valores de {col} fuera de {rango_str}")
                alertas += 1
            else:
                print(f"  {col}: formato {rango_str}, valores correctos ✓")

    if alertas == 0:
        print("  Sin alertas. Todos los rangos son correctos.")
    else:
        print(f"  {alertas} alerta(s) encontrada(s).")

    return alertas


def guardar_limpios(datos):
    """Paso 7: Guarda los DataFrames limpios como CSV."""
    print("\nPASO 7 — Guardando datos limpios...")

    archivos = {
        "etfs":      "etfs_weekly_clean.csv",
        "macro":     "macro_weekly_clean.csv",
        "risk":      "risk_weekly_clean.csv",
        "liquidity": "liquidity_weekly_clean.csv",
        "sentiment": "sentiment_weekly_clean.csv",
    }

    for nombre, archivo in archivos.items():
        df = datos[nombre]
        df.index.name = "date"
        ruta = os.path.join(DIR_INTERIM, archivo)
        df.to_csv(ruta)
        print(f"  Guardado: {ruta}")


def resumen_final(datos):
    """Paso 8: Imprime resumen final de todos los DataFrames limpios."""
    print(f"\n{'='*65}")
    print("PASO 8 — RESUMEN FINAL")
    print(f"{'='*65}")

    print(f"\n{'Dimensión':<14} {'Filas':>6} {'Cols':>5} {'Inicio':>12} {'Fin':>12} {'Nulos':>6}")
    print("-" * 58)

    todo_limpio = True
    for nombre, df in datos.items():
        filas = df.shape[0]
        cols = df.shape[1]
        inicio = df.index.min().strftime("%Y-%m-%d")
        fin = df.index.max().strftime("%Y-%m-%d")
        nulos = df.isnull().sum().sum()
        if nulos > 0:
            todo_limpio = False
        print(f"{nombre:<14} {filas:>6} {cols:>5} {inicio:>12} {fin:>12} {nulos:>6}")

    if todo_limpio:
        print(f"\n0 nulos en total. Datos completamente limpios.")
    else:
        print(f"\nQuedan nulos residuales. Revisar.")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Paso 1: Cargar
    datos = cargar_datos()

    # Pasos 2-4: Limpiar nulos
    datos["macro"] = limpiar_nulos_macro(datos["macro"])
    datos["risk"] = limpiar_nulos_riesgo(datos["risk"])
    datos["sentiment"] = limpiar_nulos_sentimiento(datos["sentiment"])

    # Paso 5: Eliminar duplicados
    datos = eliminar_duplicados(datos)

    # Paso 6: Verificar rangos
    verificar_rangos(datos)

    # Paso 7: Guardar
    guardar_limpios(datos)

    # Paso 8: Resumen
    resumen_final(datos)

    print(f"\n{'='*65}")
    print("Limpieza completada.")
