"""
Análisis de sentimiento NLP sobre titulares de noticias de Refinitiv.
Usa VADER (Valence Aware Dictionary and sEntiment Reasoner) para puntuar
cada titular y agrega los scores a frecuencia semanal por ETF.

¿Por qué VADER?
- Diseñado específicamente para textos cortos (titulares, tweets).
- No requiere entrenamiento: usa un lexicón validado por humanos.
- Rápido y determinista: ideal para ~17k titulares.
- Ampliamente citado en la literatura de finanzas cuantitativas.
- El compound score (-1 a +1) resume sentimiento en un solo número.
"""

import os

import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ── Configuración ──────────────────────────────────────────────

DIR_RAW_SENT = os.path.join("data", "raw", "sentiment")
DIR_INTERIM = os.path.join("data", "interim")


# ── Funciones ──────────────────────────────────────────────────

def preparar_vader():
    """Descarga el lexicón de VADER si no está disponible."""
    print("Descargando lexicón de VADER...")
    nltk.download("vader_lexicon", quiet=True)
    print("  Lexicón listo.")
    return SentimentIntensityAnalyzer()


def cargar_titulares():
    """Paso 1: Carga los titulares de Refinitiv."""
    print("\nPASO 1 — Cargando titulares de noticias...")

    ruta = os.path.join(DIR_RAW_SENT, "all_news_refinitiv.csv")
    df = pd.read_csv(ruta, parse_dates=["date"], index_col="date")

    # Eliminar filas sin titular (si las hay)
    nulos_antes = df["headline"].isnull().sum()
    if nulos_antes > 0:
        df = df.dropna(subset=["headline"])
        print(f"  Eliminados {nulos_antes} titulares vacíos")

    print(f"  Titulares cargados: {len(df)}")
    print(f"  Rango: {df.index.min().strftime('%Y-%m-%d')} → {df.index.max().strftime('%Y-%m-%d')}")
    print(f"  ETFs: {sorted(df['ticker'].unique())}")

    return df


def analizar_sentimiento(df, sia):
    """
    Paso 2: Calcula el compound score de VADER para cada titular.

    El compound score combina las puntuaciones positiva, negativa y neutra
    en un valor normalizado entre -1 (muy negativo) y +1 (muy positivo).
    Es la métrica estándar de VADER para análisis de sentimiento.
    """
    print("\nPASO 2 — Analizando sentimiento con VADER...")

    # Aplicar VADER a cada titular
    scores = df["headline"].apply(lambda x: sia.polarity_scores(str(x))["compound"])
    df["sentiment_score"] = scores

    # Estadísticas generales
    print(f"  Titulares analizados: {len(df)}")
    print(f"  Score medio: {scores.mean():.4f}")
    print(f"  Score mediana: {scores.median():.4f}")
    print(f"  Rango: [{scores.min():.4f}, {scores.max():.4f}]")

    # Distribución por categoría
    positivos = (scores > 0.05).sum()
    negativos = (scores < -0.05).sum()
    neutros = len(scores) - positivos - negativos
    print(f"  Positivos (>0.05): {positivos} ({positivos/len(scores)*100:.1f}%)")
    print(f"  Neutros:           {neutros} ({neutros/len(scores)*100:.1f}%)")
    print(f"  Negativos (<-0.05):{negativos} ({negativos/len(scores)*100:.1f}%)")

    # Ejemplos para verificación
    print("\n  3 titulares MÁS POSITIVOS:")
    top_pos = df.nlargest(3, "sentiment_score")
    for _, row in top_pos.iterrows():
        print(f"    [{row['sentiment_score']:+.4f}] {row['headline'][:90]}")

    print("\n  3 titulares MÁS NEGATIVOS:")
    top_neg = df.nsmallest(3, "sentiment_score")
    for _, row in top_neg.iterrows():
        print(f"    [{row['sentiment_score']:+.4f}] {row['headline'][:90]}")

    print("\n  3 titulares NEUTROS (más cercanos a 0):")
    idx_neutros = df["sentiment_score"].abs().nsmallest(3).index
    for idx in idx_neutros:
        row = df.loc[idx]
        # Si hay duplicados en el índice, tomar el primero
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        print(f"    [{row['sentiment_score']:+.4f}] {row['headline'][:90]}")

    return df


def agregar_semanal_por_etf(df):
    """
    Paso 3: Agrega sentimiento a frecuencia semanal por ETF.

    Para cada ETF y cada semana (W-FRI) se calcula:
    - sentiment_mean: media del compound score → sentimiento promedio
    - sentiment_count: número de titulares → proxy de atención mediática ("buzz")
    """
    print("\nPASO 3 — Agregando a frecuencia semanal por ETF...")

    # Agrupar por ticker y semana (viernes)
    df_reset = df.reset_index()
    # Resamplear a semana W-FRI: usa el viernes como fecha de cierre
    # .normalize() elimina la parte horaria para que coincida con el master (00:00:00)
    df_reset["week"] = df_reset["date"].dt.to_period("W-FRI").apply(
        lambda x: x.end_time.normalize()
    )

    grouped = df_reset.groupby(["ticker", "week"]).agg(
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_count=("sentiment_score", "count"),
    ).reset_index()

    # Pivotar: una fila por semana, columnas por ETF
    sent_pivot = grouped.pivot(index="week", columns="ticker", values="sentiment_mean")
    sent_pivot.columns = [f"{col}_news_sent" for col in sent_pivot.columns]

    count_pivot = grouped.pivot(index="week", columns="ticker", values="sentiment_count")
    count_pivot.columns = [f"{col}_news_count" for col in count_pivot.columns]

    # Unir sentimiento y conteo
    resultado = pd.concat([sent_pivot, count_pivot], axis=1)
    resultado.index.name = "date"
    resultado.index = pd.to_datetime(resultado.index)

    print(f"  Semanas con datos: {len(resultado)}")
    print(f"  Columnas: {list(resultado.columns)}")

    # Mostrar media de sentimiento por ETF
    print("\n  Sentimiento medio por ETF:")
    for col in sorted([c for c in resultado.columns if c.endswith("_news_sent")]):
        ticker = col.replace("_news_sent", "")
        media = resultado[col].mean()
        print(f"    {ticker}: {media:+.4f}")

    return resultado


def crear_indice_agregado(df, resultado):
    """
    Paso 4: Índice de sentimiento agregado (todos los ETFs juntos).

    news_sent_all: sentimiento medio de TODOS los titulares de la semana.
    news_count_all: total de titulares de la semana (atención mediática global).
    """
    print("\nPASO 4 — Creando índice de sentimiento agregado...")

    df_reset = df.reset_index()
    df_reset["week"] = df_reset["date"].dt.to_period("W-FRI").apply(
        lambda x: x.end_time.normalize()
    )

    agg_all = df_reset.groupby("week").agg(
        news_sent_all=("sentiment_score", "mean"),
        news_count_all=("sentiment_score", "count"),
    )
    agg_all.index = pd.to_datetime(agg_all.index)
    agg_all.index.name = "date"

    # Añadir al resultado
    resultado = resultado.join(agg_all)

    print(f"  Sentimiento global medio: {resultado['news_sent_all'].mean():+.4f}")
    print(f"  Titulares por semana (media): {resultado['news_count_all'].mean():.0f}")

    return resultado


def guardar_resultados(df_scored, resultado):
    """Paso 5: Guarda titulares con score y sentimiento semanal."""
    print("\nPASO 5 — Guardando resultados...")

    # Titulares individuales con score
    ruta_scored = os.path.join(DIR_RAW_SENT, "all_news_refinitiv_scored.csv")
    df_scored.to_csv(ruta_scored)
    print(f"  Guardado: {ruta_scored} ({len(df_scored)} titulares)")

    # Sentimiento semanal agregado
    ruta_weekly = os.path.join(DIR_INTERIM, "refinitiv_sentiment_weekly.csv")
    resultado.to_csv(ruta_weekly)
    print(f"  Guardado: {ruta_weekly} ({len(resultado)} semanas)")


def resumen_final(resultado):
    """Paso 6: Imprime resumen del sentimiento semanal."""
    print(f"\n{'='*70}")
    print("PASO 6 — RESUMEN FINAL")
    print(f"{'='*70}")

    print(f"\n  Semanas: {len(resultado)}")
    print(f"  Rango:   {resultado.index.min().strftime('%Y-%m-%d')} → "
          f"{resultado.index.max().strftime('%Y-%m-%d')}")
    print(f"  Columnas: {resultado.shape[1]}")

    # Sentimiento medio por ETF
    sent_cols = [c for c in resultado.columns if c.endswith("_news_sent") and c != "news_sent_all"]
    medias = {col.replace("_news_sent", ""): resultado[col].mean() for col in sent_cols}

    etf_mas_positivo = max(medias, key=medias.get)
    etf_mas_negativo = min(medias, key=medias.get)

    print(f"\n  Sentimiento global medio: {resultado['news_sent_all'].mean():+.4f}")
    print(f"  ETF más positivo: {etf_mas_positivo} ({medias[etf_mas_positivo]:+.4f})")
    print(f"  ETF más negativo: {etf_mas_negativo} ({medias[etf_mas_negativo]:+.4f})")

    print(f"\n  Nulos: {resultado.isnull().sum().sum()}")
    if resultado.isnull().sum().sum() > 0:
        print("  (Nulos esperados: no todos los ETFs tienen noticias cada semana)")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Preparar VADER
    sia = preparar_vader()

    # Paso 1: Cargar titulares
    df = cargar_titulares()

    # Paso 2: Analizar sentimiento
    df = analizar_sentimiento(df, sia)

    # Paso 3: Agregar por ETF y semana
    resultado = agregar_semanal_por_etf(df)

    # Paso 4: Índice agregado
    resultado = crear_indice_agregado(df, resultado)

    # Paso 5: Guardar
    guardar_resultados(df, resultado)

    # Paso 6: Resumen
    resumen_final(resultado)

    print(f"\n{'='*70}")
    print("Análisis de sentimiento NLP completado.")
