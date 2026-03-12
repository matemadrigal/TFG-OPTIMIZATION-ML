# FASE 2 — TRANSFORMACIÓN Y LIMPIEZA: Notas para la Memoria del TFG

**Proyecto:** Optimización de Carteras de Inversión mediante Predicción de Retornos con XGBoost y Señales Macroeconómicas, de Riesgo, de Sentimiento Aplicado a ETFs.

**Autor:** Mateo Madrigal Arteaga

**Fecha de esta fase:** Marzo 2026

---

## 1. Visión general

La Fase 2 transforma los datos crudos extraídos en la Fase 1 en un dataset maestro unificado, limpio y listo para alimentar modelos de Machine Learning. El proceso se organiza en cuatro bloques secuenciales: alineación temporal, limpieza, generación de variables derivadas (feature engineering) y consolidación final.

El resultado es un archivo `master_weekly.csv` donde cada fila representa una semana (viernes) y cada columna una variable predictora o un target, cubriendo el periodo 2007-03-30 a 2026-02-20.

---

## 2. Bloque 1 — Alineación temporal

**Script:** `src/transformers/align_weekly.py`

### Problema
Los datos de la Fase 1 tienen frecuencias heterogéneas:
- ETFs: diaria (~4,828 filas)
- Series FRED macro/riesgo: mezcla de diaria, semanal y mensual
- Liquidez: mayormente semanal
- Sentimiento (Google Trends): semanal/mensual
- AAII: semanal (jueves de publicación)

### Decisiones tomadas
- **Frecuencia objetivo: semanal (viernes).** Se eligió el viernes como cierre semanal por ser el estándar en finanzas (último día de negociación de la semana bursátil). La frecuencia semanal reduce el ruido de los datos diarios y es coherente con la mayoría de las señales de sentimiento y liquidez.
- **Periodo: desde 2007-01-05** (primer viernes de 2007) hasta la última fecha disponible en todas las dimensiones.

### Método de resampleo por tipo de frecuencia
- **Series diarias → semanal:** Se toma el último valor de cada semana (`resample('W-FRI').last()`). Para precios de ETFs esto equivale al cierre del viernes.
- **Series mensuales → semanal:** Se aplica forward-fill primero (el dato mensual publicado se mantiene vigente hasta la siguiente publicación) y luego se resamplea a viernes.
- **Series semanales:** Se alinean al calendario de viernes mediante `reindex` + `ffill`.

### Resultado
Todas las dimensiones quedaron con exactamente 1,000 filas (semanas), periodo 2007-01-05 → 2026-02-27.

---

## 3. Bloque 2 — Limpieza

**Script:** `src/transformers/clean_data.py`

### Tratamiento de nulos

| Dimensión | Nulos antes | Método | Nulos después |
|-----------|-------------|--------|---------------|
| ETFs | 0 | — | 0 |
| Macro | 13 (0.13%) | ffill + bfill | 0 |
| Riesgo | 16 (0.32%) | ffill + bfill | 0 |
| Liquidez | 0 | — | 0 |
| Sentimiento | 4,003 (40%) | ffill + bfill | 0 |

**Justificación del forward-fill:** En series económicas y financieras, el valor más reciente publicado sigue vigente hasta que se publica uno nuevo. Por ejemplo, si la tasa de desempleo publicada en enero es 4.1%, ese valor es la mejor estimación disponible hasta que se publique el dato de febrero. Esto no es "inventar datos", sino reflejar cómo los inversores realmente trabajan con la información disponible.

**Caso especial — RRPONTSYD (Reverse Repo):** Esta serie tiene valores nulos antes de ~2013 porque el programa de reverse repo overnight de la Reserva Federal no existía antes de esa fecha. Se rellenaron con 0, lo cual es correcto: si la facilidad no existía, el volumen de operaciones era literalmente cero.

**Caso especial — Sentimiento (40% nulos):** Los nulos elevados se debían a dos factores: la encuesta AAII se publica los jueves y al alinear a viernes no siempre coincidía, y los datos mensuales de Google Trends generaban huecos entre meses. El forward-fill resuelve ambos: el sentimiento medido el jueves sigue vigente el viernes, y el dato mensual de búsquedas se mantiene hasta el siguiente mes.

### Eliminación de duplicados entre dimensiones

Algunas series aparecían en más de una dimensión. Se decidió mantener cada serie en la dimensión donde tiene mayor relevancia conceptual:

| Serie | Se queda en | Se elimina de | Justificación |
|-------|-------------|---------------|---------------|
| VIXCLS (VIX) | Riesgo | Macro | Es primariamente un indicador de riesgo/volatilidad |
| BAMLH0A0HYM2 (HY spread) | Riesgo | Macro | Mide riesgo crediticio, no macro |
| RRPONTSYD (Reverse Repo) | Liquidez | Riesgo | Es un instrumento de liquidez del sistema |

Tras la deduplicación: macro pasó de 10 a 8 columnas, riesgo de 5 a 4, liquidez se mantuvo en 4.

### Verificación de rangos
Se verificó que no hubiera valores absurdos:
- Precios de ETFs > 0: ✓
- VIX entre 0 y 100: ✓
- Tasa de desempleo entre 0 y 30%: ✓
- Porcentajes AAII entre 0 y 1: ✓

---

## 4. Bloque 3 — Feature Engineering (Variables Derivadas)

**Script:** `src/transformers/feature_engineering.py`

### Filosofía
Los datos crudos (precios, niveles de indicadores) por sí solos no son informativo para un modelo predictivo. Lo que aporta poder predictivo son las transformaciones que capturan dinámica, tendencia, riesgo y cambio. Cada variable derivada tiene una justificación financiera.

### Variables derivadas por dimensión

#### ETFs (60 features: 6 por cada uno de los 10 ETFs)

| Variable | Fórmula | Justificación financiera |
|----------|---------|-------------------------|
| Log-return semanal | ln(P_t / P_{t-1}) | Variación porcentual semanal. Se usa log-return porque es aditivo en el tiempo y estabiliza la varianza. |
| Volatilidad rolling 4 semanas | std(log_ret, ventana=4) | Riesgo reciente (~1 mes). Captura episodios de alta/baja volatilidad. |
| Volatilidad rolling 12 semanas | std(log_ret, ventana=12) | Riesgo a medio plazo (~3 meses). Complementa la ventana corta. |
| Momentum 4 semanas | Retorno acumulado 4 semanas | Señal de tendencia corta. Si es positivo, el activo tiene impulso alcista. |
| Momentum 12 semanas | Retorno acumulado 12 semanas | Señal de tendencia a medio plazo. Usado en estrategias de momentum clásicas. |
| Drawdown | (P_t - max_histórico) / max_histórico | Caída desde el máximo. Mide cuánto ha perdido un activo desde su pico, proxy de estrés. |

#### Macro (4 features)

| Variable | Justificación |
|----------|---------------|
| Spread 10Y-2Y (DGS10 - DGS2) | Indicador clásico de recesión. Cuando se invierte (negativo), históricamente anticipa recesiones. |
| Variación semanal CPI | Captura el momento exacto en que se publica un nuevo dato de inflación. |
| Variación semanal UNRATE | Señal de deterioro o mejora del mercado laboral. |
| Variación semanal UMCSENT | Cambios en la confianza del consumidor. |

#### Riesgo (4 features)

| Variable | Justificación |
|----------|---------------|
| VIX nivel | Nivel absoluto de volatilidad implícita. Por encima de 30 indica pánico. |
| Variación semanal VIX | Cambio en la percepción de riesgo semana a semana. |
| Variación semanal HY spread | Ampliación del spread indica aversión al riesgo crediticio. |
| Variación semanal NFCI | Cambio en las condiciones financieras (endurecimiento o relajación). |

#### Liquidez (4 features)

| Variable | Justificación |
|----------|---------------|
| Var. semanal WALCL | Expansión/contracción del balance de la Fed. Indicador de política monetaria cuantitativa. |
| Var. semanal RRPONTSYD | Cambios en la facilidad de reverse repo. Proxy de exceso de liquidez en el sistema. |
| Var. semanal depósitos | Flujos de depósitos bancarios. Retiradas masivas indican estrés financiero. |
| Var. semanal TGA | Movimientos de la cuenta del Tesoro. Afecta a la liquidez del sistema financiero. |

#### Sentimiento (15 features)

| Variable | Justificación |
|----------|---------------|
| Spread Bull-Bear AAII | Bullish - Bearish. Indicador clásico de sentimiento retail. Extremos suelen ser señales contrarias. |
| Var. semanal de 7 términos Google Trends | Captura cambios bruscos en el interés de búsqueda (ej. pico en "recession" puede anticipar caídas). |
| Media móvil 4 sem. de 7 términos Google Trends | Suaviza el ruido de las búsquedas semanales para captar tendencias. |

### Tratamiento de NaN iniciales
Las ventanas rolling de 12 semanas generan 12 filas con NaN al inicio del dataset. Se eliminaron, dejando el periodo efectivo en 2007-03-30 → 2026-02-27 (988 filas → 987 tras el shift del target).

### Resultado
**87 features** generadas, 0 nulos, 988 filas.

---

## 5. Bloque 4 — NLP sobre titulares de Refinitiv

**Script:** `src/transformers/nlp_sentiment.py`

### Método
Se aplicó análisis de sentimiento con VADER (Valence Aware Dictionary and sEntiment Reasoner) sobre los 17,181 titulares de noticias financieras extraídos de Refinitiv/LSEG en la Fase 1.

**¿Por qué VADER?**
- Diseñado específicamente para textos cortos (titulares, tweets).
- No requiere entrenamiento: usa un lexicón validado por humanos.
- Rápido y determinista: ideal para procesar ~17,000 titulares.
- Ampliamente citado en la literatura de finanzas cuantitativas.

### Proceso
1. Cada titular recibe un compound score entre -1 (muy negativo) y +1 (muy positivo).
2. Los scores se agregan a frecuencia semanal (viernes) por ETF, calculando:
   - **sentiment_mean:** media del compound score de la semana (tono general de las noticias).
   - **sentiment_count:** número de titulares de la semana (proxy de "buzz" o atención mediática).
3. Se crea también un índice agregado (todos los ETFs juntos).

### Resultado
- 66 semanas de sentimiento de noticias (dic 2024 → mar 2026)
- 22 columnas nuevas (10 ETFs × 2 métricas + 2 agregadas)
- Distribución de sentimiento: 23.6% positivo, 61.7% neutro, 14.7% negativo
- Score medio: +0.056 (ligeramente positivo)

### Limitación
La API de noticias de Refinitiv solo permite acceder a ~15 meses de histórico. Por tanto, las features de sentimiento de noticias solo están disponibles para las últimas 66 semanas del dataset. Las semanas anteriores tienen NaN en estas columnas. XGBoost maneja nulos nativamente, por lo que esto no impide el entrenamiento del modelo.

---

## 6. Consolidación del Dataset Maestro

**Scripts:** `src/transformers/build_master_dataset.py` + `src/transformers/add_refinitiv_to_master.py`

### Estructura del target
Para cada ETF se creó una columna target con el log-return de la **semana siguiente** (shift(-1)). Esto es fundamental para evitar data leakage: las features contienen información del presente y pasado, mientras que el target es el retorno futuro que el modelo debe predecir.

### Normalización
Se aplicó StandardScaler (z-score: media=0, desviación=1) a las features para que variables con escalas muy diferentes (VIX: 9-80, CPI: 200-320, retornos: -0.10 a +0.10) tengan el mismo peso relativo. Los targets NO se normalizan.

**Nota sobre data leakage:** La normalización del dataset se realizó con todos los datos disponibles para la fase exploratoria. En la fase de modelado, el scaler se ajustará exclusivamente con datos de entrenamiento y se aplicará a los datos de test, para evitar fuga de información del futuro al pasado.

### Dataset final

| Métrica | Valor |
|---------|-------|
| Filas (semanas) | 987 |
| Features | 109 |
| Targets | 10 (uno por ETF) |
| Total columnas | 119 |
| Periodo | 2007-03-30 → 2026-02-20 |
| Nulos en features base | 0 |
| Nulos en features Refinitiv | 955 (esperado: solo 32 semanas con datos) |
| Tamaño | 1.86 MB (normalizado), 1.51 MB (raw) |

Archivos generados:
- `data/processed/master_weekly.csv` — features normalizadas + targets
- `data/processed/master_weekly_raw.csv` — sin normalizar (versión definitiva para modelado)

---

## 7. Scripts creados en la Fase 2

| Script | Función | Comando |
|--------|---------|---------|
| `align_weekly.py` | Alineación temporal a W-FRI | `python src/transformers/align_weekly.py` |
| `clean_data.py` | Limpieza nulos, duplicados, rangos | `python src/transformers/clean_data.py` |
| `feature_engineering.py` | 87 features derivadas | `python src/transformers/feature_engineering.py` |
| `nlp_sentiment.py` | VADER sobre titulares Refinitiv | `python src/transformers/nlp_sentiment.py` |
| `build_master_dataset.py` | Dataset maestro + targets + normalización | `python src/transformers/build_master_dataset.py` |
| `add_refinitiv_to_master.py` | Integra sentimiento Refinitiv al master | `python src/transformers/add_refinitiv_to_master.py` |

---

## 8. Pipeline completo de ejecución (Fase 1 + Fase 2)

Para reproducir todo el proceso desde cero:

```bash
# Fase 1 — Extracción
python src/extractors/extract_etfs.py
python src/extractors/extract_macro.py
python src/extractors/extract_risk.py
python src/extractors/extract_liquidity.py
python src/extractors/extract_sentiment.py
python src/extractors/merge_google_trends.py
# (con Workspace abierto y proxy corriendo):
python src/extractors/refinitiv_proxy.py &
python src/extractors/extract_refinitiv_news.py

# Fase 2 — Transformación
python src/transformers/align_weekly.py
python src/transformers/clean_data.py
python src/transformers/feature_engineering.py
python src/transformers/nlp_sentiment.py
python src/transformers/build_master_dataset.py
python src/transformers/add_refinitiv_to_master.py
```

---

## 9. Pendiente para Fase 3 (EDA / Análisis Exploratorio)

- Estadísticos descriptivos de todas las variables.
- Distribuciones e histogramas.
- Matriz de correlaciones entre dimensiones.
- Evolución temporal de las series clave.
- Detección y análisis de outliers.
- Visualizaciones para la memoria del TFG.
