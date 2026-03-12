# FASE 3 — ANÁLISIS EXPLORATORIO DE DATOS (EDA): Notas para la Memoria del TFG

**Proyecto:** Optimización de Carteras de Inversión mediante Predicción de Retornos con XGBoost y Señales Macroeconómicas, de Riesgo, de Sentimiento Aplicado a ETFs.

**Autor:** Mateo Madrigal Arteaga

**Fecha de esta fase:** Marzo 2026

---

## 1. Visión general

El análisis exploratorio de datos (EDA) es un paso crítico previo al modelado que permite comprender exhaustivamente el comportamiento de las variables, verificar supuestos estadísticos y detectar patrones relevantes. En este trabajo, el EDA se organiza en cuatro bloques: descripción y visualización de ETFs, análisis de las dimensiones macro/riesgo/liquidez/sentimiento, estudio de correlaciones y dependencias entre dimensiones, y análisis de series temporales y outliers.

El resultado son 22 figuras profesionales que documentan el dataset y justifican las decisiones de diseño del modelo.

---

## 2. Parte 1 — ETFs: Descripción y visualización

### 2.1 Estadísticos descriptivos

Se generó una tabla con los estadísticos descriptivos (media, mediana, desviación estándar, mínimo, máximo, asimetría y curtosis) de todas las variables representativas del dataset, agrupadas por dimensión. Los hallazgos principales son:

- Los log-returns semanales de los ETFs tienen media cercana a cero (~0.1% semanal) y desviación estándar variable según la clase de activo: renta fija (AGG: 0.76%) es mucho menos volátil que emergentes (EEM: 3.27%) o inmobiliario (VNQ: 3.0%).
- La curtosis es elevada en todos los ETFs (AGG: 53.4, LQD: 38.0, SPY: 9.6), lo que indica colas pesadas y eventos extremos más frecuentes de lo que asumiría una distribución normal. Este hallazgo es fundamental para justificar el uso de modelos de ML frente a modelos que asumen normalidad.
- La asimetría es negativa en la mayoría de ETFs (SPY: -1.04, EFA: -1.05), indicando que las caídas extremas son más frecuentes que las subidas extremas.

**Figura:** tabla_estadisticos.png

### 2.2 Evolución de precios (base 100, escala logarítmica)

Se normalizaron los precios de los 10 ETFs a base 100 desde enero 2007 y se representaron en escala logarítmica para permitir una comparación justa entre activos con rendimientos muy distintos. Se sombrearon tres periodos de crisis: la Gran Crisis Financiera (GFC, sep 2008 — mar 2009), la pandemia de COVID-19 (feb — abr 2020) y el periodo de inflación y subida de tipos (ene — oct 2022).

Principales observaciones:
- QQQ (Nasdaq 100) ha sido el activo con mayor rendimiento acumulado, multiplicando su valor por más de 8 veces.
- EEM (emergentes) prácticamente no ha generado retorno neto en 19 años, reflejando el bajo rendimiento relativo de los mercados emergentes frente a EE.UU.
- AGG y TIP (renta fija) muestran rendimientos muy estables pero modestos, cumpliendo su papel como activos defensivos.
- Las tres crisis se manifiestan de forma distinta en cada clase de activo, lo que justifica la diversificación y el enfoque multi-activo del proyecto.

**Figura:** etfs_evolucion_log.png

### 2.3 Distribución de retornos vs distribución normal

Se generaron histogramas de los log-returns semanales de los 10 ETFs con la curva de distribución normal teórica superpuesta. Cada panel incluye los valores de curtosis y asimetría.

El resultado es visualmente claro: todos los ETFs presentan colas más pesadas que la distribución normal (leptocurtosis). Esto invalida uno de los supuestos fundamentales del modelo de Markowitz (normalidad de retornos) y justifica el uso de modelos de Machine Learning capaces de capturar relaciones no lineales y eventos extremos.

**Figura:** distribucion_retornos.png

### 2.4 Violin plots de retornos

Los violin plots muestran la distribución completa de los retornos semanales por ETF. Se observa claramente la diferencia de perfil riesgo-retorno entre clases de activos: los ETFs de renta variable (SPY, QQQ, IWM, EEM) tienen distribuciones anchas (mayor volatilidad), mientras que los de renta fija (AGG, LQD, TIP) tienen distribuciones muy concentradas alrededor de cero.

**Figura:** etfs_retornos_violin.png

### 2.5 Drawdowns históricos

Se graficaron los drawdowns (caída desde el máximo histórico) de 5 ETFs representativos: SPY (renta variable US), QQQ (tecnología), EEM (emergentes), AGG (renta fija) y GLD (oro).

Los drawdowns revelan el perfil de riesgo real de cada activo:
- EEM sufrió un drawdown del -65% en 2008 y no recuperó máximos hasta años después.
- SPY y QQQ cayeron ~50% en 2008 pero se recuperaron más rápido.
- AGG (renta fija) mantuvo drawdowns contenidos hasta 2022, cuando la subida agresiva de tipos provocó caídas históricas en bonos.
- GLD muestra un patrón diferente: su mayor drawdown fue en 2013-2015, no en las crisis de renta variable.

**Figura:** drawdowns_comparativo.png

---

## 3. Parte 2 — Macro, riesgo, liquidez y sentimiento

### 3.1 Índice de Volatilidad VIX

El VIX mide la volatilidad implícita del S&P 500 y es el indicador de miedo por excelencia del mercado. El gráfico muestra su evolución desde 2007 con líneas de referencia en VIX=20 (nivel "normal") y VIX=30 (nivel de "pánico"), sombreados de crisis y anotaciones de los picos máximos.

Observaciones clave:
- El VIX alcanzó 79.1 durante la GFC (2008) y 66.0 durante COVID (2020).
- La mayor parte del tiempo se mantiene entre 10 y 25, pero los picos son extremadamente abruptos.
- El comportamiento asimétrico del VIX (sube rápido, baja lento) es una de las razones por las que se incluye tanto su nivel como su variación semanal como features del modelo.

**Figura:** vix_historico.png

### 3.2 Spread Treasury 10Y-2Y (indicador de recesión)

El spread entre el rendimiento del Treasury a 10 años y el de 2 años es uno de los indicadores macroeconómicos más seguidos. Cuando se invierte (valor negativo), históricamente ha anticipado recesiones económicas.

El gráfico muestra:
- Una breve inversión en 2007, justo antes de la Gran Crisis Financiera.
- Una inversión prolongada y profunda en 2022-2023 (-1.0 pp), la más pronunciada en décadas, coincidiendo con la subida agresiva de tipos de la Fed.
- La zona roja sombreada hace visualmente evidente cuándo la curva está invertida.

Esta variable se incluye como feature del modelo (spread_10y_2y) por su reconocido poder predictivo sobre el ciclo económico.

**Figura:** spread_10y2y.png

### 3.3 Balance de la Reserva Federal (WALCL)

El gráfico del balance de la Fed muestra la evolución de los activos totales desde 2007, con anotaciones de los programas de expansión cuantitativa (QE1, QE2, QE3, COVID QE) y contracción (QT).

El patrón es espectacular: el balance pasó de ~0.9 billones USD en 2007 a ~9 billones en 2022, con una duplicación vertical durante COVID-2020. La contracción posterior (QT desde 2022) es visible pero gradual.

La variación semanal del balance de la Fed es una de las features de liquidez del modelo, ya que la inyección o retirada de liquidez tiene impacto directo en los precios de los activos.

**Figura:** fed_balance.png

### 3.4 Sentimiento AAII vs retorno acumulado SPY

Se superpuso el spread Bull-Bear de la encuesta AAII (datos originales sin forward-fill, scatter de observaciones semanales con media móvil de 12 observaciones) con el retorno acumulado de SPY.

El gráfico permite observar:
- El sentimiento se hundió durante la crisis de 2008-2009 (spread muy negativo), coincidiendo con la caída de SPY.
- Existe una relación cíclica entre sentimiento y mercado, aunque el sentimiento suele funcionar como indicador contrario en extremos: cuando el pesimismo es máximo, el mercado tiende a recuperarse.
- La media móvil suaviza el ruido semanal y permite ver la tendencia de fondo del sentimiento.

**Figura:** sentimiento_aaii_vs_spy.png

### 3.5 Google Trends "recession" vs periodos de estrés

Se graficó el índice de búsqueda de "recession" en Google con sombreados en los tres grandes periodos de estrés del mercado (GFC, COVID, inflación/tipos).

El resultado es revelador: los picos de búsqueda de "recession" coinciden con los periodos de estrés del mercado, pero a veces los preceden ligeramente (especialmente en 2022, donde las búsquedas aumentaron antes de que el mercado tocara mínimos). Esto sugiere que Google Trends puede contener información predictiva útil para el modelo.

**Figura:** google_trends_recession.png

### 3.6 Distribución del sentimiento NLP (Refinitiv)

El histograma muestra la distribución de los compound scores de VADER sobre los 17,181 titulares de noticias de Refinitiv, coloreado por zona: negativo (rojo, <-0.05), neutro (gris, entre -0.05 y 0.05) y positivo (verde, >0.05).

Resultados:
- 61.7% de los titulares son neutros (la mayoría de las noticias financieras son factuales, sin tono emocional fuerte).
- 23.6% son positivos vs 14.7% negativos, reflejando un sesgo ligeramente positivo en los medios financieros.
- La distribución valida que el pipeline NLP funciona correctamente y produce resultados coherentes.

**Figura:** nlp_sentiment_dist.png

### 3.7 Visualización antes/después: tratamiento de nulos (para rúbrica)

Se generó un heatmap dual mostrando los valores nulos en los datos de sentimiento antes de la limpieza (sentiment_weekly_aligned.csv, con 4,003 nulos / 40%) y después (sentiment_weekly_clean.csv, con 0 nulos).

El contraste es muy visual: el panel izquierdo muestra grandes bloques rojos en AAII (nulos por desalineación jueves-viernes) y en Google Trends (huecos entre datos mensuales), mientras que el panel derecho está completamente limpio tras aplicar forward-fill.

**Figura:** antes_despues_nulos.png

### 3.8 Visualización antes/después: resampleo de frecuencia (para rúbrica)

Se muestra el precio de SPY en frecuencia diaria original (4,826 observaciones) frente al mismo dato tras resamplear a frecuencia semanal (1,000 observaciones). Se aprecia cómo la serie semanal preserva toda la información relevante (tendencias, crisis) pero con menos ruido intradía.

**Figura:** antes_despues_frecuencia.png

---

## 4. Parte 3 — Correlaciones y relaciones entre variables

### 4.1 Heatmap de correlaciones con clustering jerárquico

Se calculó la matriz de correlación de Pearson entre las 87 features del modelo y se representó como un clustermap con agrupación jerárquica. El clustering organiza automáticamente las variables por similitud, revelando la estructura interna del dataset.

Patrones observados:
- Las volatilidades de todos los ETFs forman un bloque altamente correlacionado (rojo intenso), lo cual tiene sentido: cuando sube la volatilidad, tiende a subir para todos los activos simultáneamente.
- Los retornos y momentum de renta variable (SPY, QQQ, IWM, EFA, VNQ) forman otro bloque correlacionado.
- Los retornos de renta fija (AGG, LQD, TIP) y oro (GLD) están menos correlacionados con renta variable, justificando su inclusión como activos diversificadores.
- Las variables de sentimiento y macro están relativamente poco correlacionadas con las de mercado, lo que indica que aportan información complementaria al modelo.

**Figura:** correlacion_heatmap.png

### 4.2 Correlación media entre dimensiones

Se calculó la correlación media absoluta entre cada par de dimensiones (ETF, Macro, Riesgo, Liquidez, Sentimiento) y se representó como un heatmap 5×5.

Hallazgos:
- La mayor correlación es entre ETF y Riesgo (0.25) y dentro de la propia dimensión de Riesgo (0.34), lo cual refleja que variables como el VIX están intrínsecamente ligadas al comportamiento del mercado.
- Liquidez y Sentimiento son las dimensiones más independientes del resto (correlaciones de 0.05-0.09), lo que confirma que aportan información diferenciada y justifica su inclusión en el modelo.

**Figura:** correlacion_entre_dimensiones.png

### 4.3 Rolling correlation: SPY vs AGG (acciones vs bonos)

Este es uno de los gráficos más relevantes del TFG. Se calculó la correlación rolling de 52 semanas (1 año) entre los retornos de SPY y AGG, coloreando las zonas de correlación positiva (rojo, "diversificación rota") y negativa (verde, "diversificación funciona").

El resultado demuestra visualmente el argumento central del anteproyecto: la correlación entre acciones y bonos NO es estable. Oscila entre -0.6 y +0.6 a lo largo de los 19 años:
- Durante 2009-2012 y 2014-2019: correlación negativa (la diversificación clásica funciona, bonos suben cuando acciones bajan).
- Durante 2008, 2013-2014, 2020 y 2022-2024: correlación positiva (la diversificación se rompe, acciones y bonos caen juntos).

Esto invalida el supuesto de Markowitz de correlaciones estables y justifica un enfoque dinámico basado en ML que se adapte al régimen de correlaciones actual.

**Figura:** rolling_corr_spy_agg.png

### 4.4 Rolling correlations: SPY vs múltiples activos

Se extendió el análisis anterior mostrando la correlación rolling de 52 semanas de SPY contra AGG, GLD, EEM y VNQ simultáneamente.

Observaciones:
- EEM y VNQ mantienen correlación alta y estable con SPY (0.6-0.9), lo que indica que ofrecen poca diversificación real.
- AGG y GLD oscilan mucho más, confirmando que son los verdaderos diversificadores de la cartera, pero su efecto varía según el régimen de mercado.
- Las dinámicas cambian radicalmente en cada crisis, reforzando la necesidad de un modelo adaptativo.

**Figura:** rolling_corr_multi.png

### 4.5 Scatter VIX vs retornos SPY

Se graficó la relación entre el nivel del VIX y los retornos semanales de SPY, coloreando por periodo temporal (2007-2009, 2010-2019, 2020+).

La relación negativa es clara: niveles altos de VIX se asocian con retornos negativos (R²=0.083). Aunque el R² es bajo (esperado para datos financieros semanales con mucho ruido), la relación es estadísticamente significativa y la dispersión aumenta enormemente en niveles altos de VIX. Los puntos extremos abajo a la derecha (VIX>60, retornos<-15%) corresponden a las semanas más críticas de 2008 y 2020.

**Figura:** scatter_vix_spy.png

---

## 5. Parte 4 — Series temporales y outliers

### 5.1 Test de estacionariedad Augmented Dickey-Fuller

Se aplicó el test ADF a 19 variables representativas del dataset. Resultados:
- 17 de 19 variables son estacionarias (p < 0.05), incluyendo todos los log-returns de ETFs y todas las variaciones semanales de indicadores macro/riesgo.
- 2 variables no son estacionarias: spread_10y_2y (p=0.44) y aaii_bull_bear_spread (p=0.12). Ambas son variables de nivel con persistencia temporal (mean-reverting lento), no diferencias.

Se decidió mantener estas dos variables como niveles (sin diferenciar) porque:
1. Su valor absoluto tiene significado económico directo: un spread de -0.5 pp es más informativo que un cambio semanal de -0.1 pp.
2. XGBoost no requiere estacionariedad estricta como los modelos ARIMA; los árboles de decisión pueden trabajar con variables no estacionarias sin problemas.
3. El clustering jerárquico las agrupa correctamente con variables de su misma dimensión, lo que indica que aportan información útil en su forma actual.

**Figura:** tabla_estacionariedad.png

### 5.2 Autocorrelación de retornos semanales (SPY)

Se calculó la función de autocorrelación (ACF) de los retornos semanales de SPY hasta lag 20. Todas las barras se mantienen dentro de la banda de confianza, confirmando que no existe autocorrelación lineal significativa en los retornos.

Implicación para el modelo: los retornos pasados por sí solos no predicen retornos futuros de forma lineal. Esto justifica la necesidad de incorporar features externas (macro, sentimiento, riesgo, liquidez) para la predicción, y valida la aproximación del proyecto de usar señales multidimensionales en vez de depender únicamente del histórico de precios.

Nota técnica: aunque no hay autocorrelación en retornos, la volatilidad sí suele presentar autocorrelación (efecto ARCH/GARCH), lo que justifica incluir volatilidad rolling como feature del modelo.

**Figura:** autocorrelacion_spy.png

### 5.3 Análisis de outliers

Se identificaron outliers en los retornos semanales de los 10 ETFs usando el método IQR (valores fuera de Q1 - 1.5×IQR o Q3 + 1.5×IQR).

Resultados:
- El número de outliers oscila entre 26 (GLD, 2.6%) y 58 (VNQ, 5.9%), con una media de ~4% por ETF.
- VNQ tiene el mayor porcentaje de outliers porque el sector inmobiliario fue el epicentro de la crisis de 2008 y es estructuralmente más volátil.
- GLD tiene el menor porcentaje porque el oro actúa como activo refugio y presenta movimientos más contenidos.
- El scatter temporal de outliers de SPY muestra que los outliers negativos se concentran en 2008-2009, 2020 y 2025, correspondiendo a crisis financieras reales.

Decisión: los outliers se MANTIENEN en el dataset porque corresponden a eventos reales del mercado (crisis financieras, pandemias, shocks geopolíticos). Eliminarlos supondría privar al modelo de información sobre los escenarios más extremos, que son precisamente los que más importa predecir en la gestión de carteras.

**Figura:** outliers_analisis.png

---

## 6. Inventario completo de figuras generadas

| # | Archivo | Parte | Descripción |
|---|---------|-------|-------------|
| 1 | tabla_estadisticos.png | P1 | Estadísticos descriptivos por dimensión |
| 2 | etfs_evolucion_log.png | P1 | Evolución precios base 100, escala log |
| 3 | distribucion_retornos.png | P1 | Histogramas retornos vs normal teórica |
| 4 | etfs_retornos_violin.png | P1 | Violin plots de retornos por ETF |
| 5 | drawdowns_comparativo.png | P1 | Drawdowns de 5 ETFs representativos |
| 6 | vix_historico.png | P2 | VIX con bandas de crisis y picos anotados |
| 7 | spread_10y2y.png | P2 | Spread 10Y-2Y indicador de recesión |
| 8 | fed_balance.png | P2 | Balance de la Fed con QE/QT anotados |
| 9 | sentimiento_aaii_vs_spy.png | P2 | AAII Bull-Bear vs retorno acumulado SPY |
| 10 | google_trends_recession.png | P2 | Google Trends "recession" vs estrés |
| 11 | nlp_sentiment_dist.png | P2 | Distribución sentimiento VADER Refinitiv |
| 12 | antes_despues_nulos.png | P2 | Heatmap nulos antes/después (rúbrica) |
| 13 | antes_despues_frecuencia.png | P2 | Resampleo diario vs semanal (rúbrica) |
| 14 | correlacion_heatmap.png | P3 | Clustermap 87×87 correlaciones |
| 15 | correlacion_entre_dimensiones.png | P3 | Heatmap 5×5 correlación entre dimensiones |
| 16 | rolling_corr_spy_agg.png | P3 | Rolling corr SPY-AGG con zonas verde/rojo |
| 17 | rolling_corr_multi.png | P3 | Rolling corr SPY vs 4 activos |
| 18 | scatter_vix_spy.png | P3 | Scatter VIX vs retornos SPY |
| 19 | tabla_estacionariedad.png | P4 | Test ADF 19 variables |
| 20 | autocorrelacion_spy.png | P4 | ACF retornos SPY hasta lag 20 |
| 21 | outliers_analisis.png | P4 | Análisis outliers IQR + scatter temporal |

Adicionalmente existe tabla_estadisticos.csv con los datos completos de estadísticos descriptivos.

---

## 7. Scripts creados en la Fase 3

| Script | Función | Comando |
|--------|---------|---------|
| `eda_etfs.py` | Parte 1: estadísticos y visualización ETFs (versión inicial) | `python src/eda/eda_etfs.py` |
| `eda_completo.py` | Partes 1-2: gráficos profesionales mejorados | `python src/eda/eda_completo.py` |
| `eda_fixes.py` | Correcciones de 4 gráficos | `python src/eda/eda_fixes.py` |
| `eda_correlaciones.py` | Parte 3: correlaciones y relaciones | `python src/eda/eda_correlaciones.py` |
| `eda_series_temporales.py` | Parte 4: ADF, autocorrelación, outliers | `python src/eda/eda_series_temporales.py` |

---

## 8. Conclusiones del EDA para el modelado

El análisis exploratorio ha revelado varios hallazgos que condicionan el diseño del modelo:

1. **No normalidad de retornos:** La curtosis elevada y la asimetría negativa invalidan los supuestos del modelo de Markowitz clásico y justifican el uso de ML.

2. **Correlaciones inestables:** La correlación entre clases de activos cambia de signo a lo largo del tiempo, lo que hace que las matrices de covarianza históricas sean estimadores poco fiables para el futuro. Un modelo dinámico que se adapte al régimen actual es necesario.

3. **Independencia de dimensiones:** Las dimensiones de liquidez y sentimiento están poco correlacionadas con las de mercado, confirmando que aportan información complementaria útil para la predicción.

4. **Ausencia de autocorrelación lineal:** Los retornos pasados no predicen retornos futuros por sí solos, lo que valida el enfoque de usar features externas multidimensionales.

5. **Outliers reales:** Los eventos extremos corresponden a crisis financieras reales que el modelo debe aprender a gestionar, no a errores de datos.

6. **Estacionariedad general:** La gran mayoría de las features son estacionarias, lo que facilita el modelado. Las dos excepciones se mantienen como niveles por su significado económico.
