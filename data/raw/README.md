# data/raw/ — Datos crudos

Esta carpeta contiene dos tipos de archivos:

## 1. Generados por scripts (fuente principal)
Archivos descargados automáticamente por los scripts de `src/extractors/`.
Se identifican por su sufijo:
- `*_api.csv` — Series descargadas de FRED API (macro, riesgo, liquidez)
- `*_daily.csv` — Precios diarios de ETFs descargados de Yahoo Finance
- `*_trends_api.csv` — Google Trends descargados con pytrends
- `*_trends_merged.csv` — Google Trends combinando API + manual
- `*_news_refinitiv.csv` — Titulares de noticias de Refinitiv/LSEG
- `all_news_refinitiv.csv` — Consolidado de todos los titulares
- `all_news_refinitiv_scored.csv` — Titulares con score de sentimiento VADER
- `aaii_sentiment_clean.csv` — Encuesta AAII limpiada desde el Excel original

## 2. Descargados manualmente (respaldo)
Archivos descargados a mano desde las fuentes originales antes de automatizar
la extracción. Se mantienen como respaldo y referencia:
- `etfs/` — CSVs con nombre tipo "SPY ETF Stock Price History.csv"
- `macro/` — CSVs con nombre de la serie FRED (ej. "UNRATE.csv")
- `risk/` — CSVs con nombre de la serie FRED (ej. "VIXCLS.csv")
- `liquidity/` — CSVs con nombre de la serie FRED (ej. "WALCL.csv")
- `sentiment/` — CSVs de Google Trends manual (ej. "recession.csv") y Excel de AAII

Los scripts del pipeline solo usan los archivos generados automáticamente.
Los manuales no se borran por trazabilidad.
