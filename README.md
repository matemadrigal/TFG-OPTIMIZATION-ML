# Optimización de Carteras de Inversión mediante Predicción de Retornos con XGBoost

**Trabajo Fin de Grado — Grado en Business Analytics, Universidad Francisco de Vitoria**

**Autor:** Mateo Madrigal Arteaga | **Curso:** 2025-2026

---

## Resumen

Sistema de optimización dinámica de carteras que combina predicciones de retornos semanales generadas por modelos de gradient boosting (XGBoost y LightGBM) con optimización de Markowitz de media-varianza. Aplicado a un universo de 10 ETFs representativos de 5 clases de activos, utilizando 109 variables predictivas en 6 dimensiones. Validado out-of-sample con walk-forward expanding window sobre 778 semanas (2011-2026).

**Resultado principal:** Sharpe ratio 1.397 (XGBoost Tuned), superando significativamente al benchmark 60/40 (Sharpe 0.847) y a la optimización clásica de Markowitz (Sharpe 0.832).

## Resultados

| Estrategia | Sharpe | Retorno Anual | Max Drawdown | Retorno Total |
|---|---|---|---|---|
| **XGBoost Tuned** | **1.397** | **11.09%** | **-14.83%** | **425.9%** |
| LightGBM Tuned | 1.313 | 10.23% | -19.60% | 361.8% |
| 60/40 | 0.847 | 8.60% | -21.67% | 262.3% |
| Markowitz | 0.832 | 5.39% | -19.82% | 124.1% |
| Equal Weight | 0.649 | 7.30% | -24.21% | 198.1% |

## Estructura del proyecto

```
TFG-OPTIMIZATION-ML-1/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── models/                  # Pipeline principal de modelado
│   │   ├── data_loader.py       # Carga del dataset maestro
│   │   ├── walk_forward.py      # Generador de splits walk-forward
│   │   ├── benchmarks.py        # 60/40, Equal Weight, Markowitz
│   │   ├── train_base.py        # Entrenamiento con hiperparámetros por defecto
│   │   ├── tune_optuna.py       # Optimización de hiperparámetros (75 trials)
│   │   ├── train_final.py       # ★ Script principal — genera reporte completo
│   │   ├── diagnostics.py       # Diagnósticos visuales (confusion matrix, etc.)
│   │   ├── shap_analysis.py     # Análisis SHAP (8 figuras interpretabilidad)
│   │   └── backtesting_extra.py # IC, subperíodos, turnover, Monte Carlo
│   ├── extractors/              # Scripts de extracción de datos (FRED, Yahoo, etc.)
│   ├── transformers/            # Pipeline ETL (alineación, limpieza, feature eng.)
│   ├── eda/                     # Análisis exploratorio de datos
│   ├── experiments/             # Experimentos documentados (wavelet, +12 features)
│   └── utils/                   # Utilidades (generación Excel, etc.)
├── data/
│   ├── processed/               # Dataset definitivo
│   │   └── master_weekly_raw.csv    # 987 semanas × 119 columnas
│   └── results/                 # Resultados y métricas
│       ├── extra/               # Backtesting adicional (IC, Monte Carlo, etc.)
│       ├── final/               # Resultados definitivos para la memoria
│       ├── *_tuned_*.csv        # Predicciones y pesos del modelo tuned
│       └── optuna_best_params_*.json  # Hiperparámetros óptimos
├── docs/
│   ├── figures/                 # 30+ figuras (300 DPI, estilo Tufte/Okabe-Ito)
│   └── memoria/                 # Documento Word de la memoria
└── info_entregas/               # Documentos de referencia UFV
```

## Datos

- **Dataset:** 987 semanas × 119 columnas (marzo 2007 – febrero 2026)
- **10 ETFs:** SPY, QQQ, IWM, EFA, EEM (renta variable), AGG, LQD, TIP (renta fija), GLD (oro), VNQ (inmobiliario)
- **109 features en 6 dimensiones:**
  - Mercado (60): retornos, volatilidad, momentum, drawdown por ETF
  - Riesgo (4): VIX nivel/cambio, spread High Yield, NFCI
  - Macro (4): spread 10Y-2Y, CPI, desempleo, sentimiento consumidor
  - Liquidez (4): balance Fed, reverse repo, depósitos bancarios, TGA
  - Sentimiento (15): AAII Bull-Bear, 7 Google Trends
  - NLP Noticias (22): sentimiento VADER sobre 17.181 titulares de Refinitiv

## Reproducción de resultados

```bash
# Instalar dependencias
pip install -r requirements.txt

# Generar reporte completo de resultados (carga predicciones existentes, ~4 seg)
python3 src/models/train_final.py

# Análisis SHAP — genera 8 figuras de interpretabilidad (~1 min)
python3 src/models/shap_analysis.py

# Backtesting adicional: IC, subperíodos, turnover, Monte Carlo (~10 seg)
python3 src/models/backtesting_extra.py

# Reentrenar modelos desde cero (walk-forward completo, ~60 min)
python3 src/models/train_base.py

# Optimización de hiperparámetros con Optuna (~5 horas)
python3 src/models/tune_optuna.py
```

## Cómo funciona el proyecto (de principio a fin)

### El problema

Quieres invertir 100€ en 10 ETFs (fondos que replican índices). Cada semana decides qué porcentaje poner en cada uno. El método clásico (Markowitz, 1952) usa la media histórica para decidir — pero falla porque las correlaciones cambian en crisis. Nuestro enfoque: usar ML para **predecir** los retornos de la semana siguiente y dar esas predicciones al optimizador.

### Fase 1 — Extracción (`src/extractors/`)

8 scripts descargan datos de 6 fuentes:
- **Yahoo Finance:** precios diarios de SPY, QQQ, AGG, GLD... (10 ETFs)
- **FRED** (banco central de EEUU): VIX, tipos de interés, desempleo, CPI...
- **Google Trends:** búsquedas de "recession", "inflation", "bear market"...
- **AAII:** encuesta semanal de sentimiento de inversores
- **Refinitiv:** 17.181 titulares de noticias financieras

Todo esto se guarda en `data/raw/` (datos crudos tal cual vienen de las APIs).

### Fase 2 — Transformación (`src/transformers/`)

6 scripts convierten los datos crudos en el dataset final:
1. **align_weekly.py:** Todo tiene frecuencias distintas (diario, semanal, mensual) → lo alineamos a **viernes semanal**
2. **clean_data.py:** Rellena huecos (forward-fill), elimina duplicados entre dimensiones
3. **feature_engineering.py:** De los precios crudos calcula 109 features con significado financiero:
   - Por cada ETF (×10): log-return, volatilidad 4 y 12 semanas, momentum 4 y 12 semanas, drawdown = **60 features**
   - Macro: spread 10Y-2Y, cambio CPI, desempleo, sentimiento consumidor = **4 features**
   - Riesgo: VIX nivel/cambio, spread high yield, NFCI = **4 features**
   - Liquidez: balance Fed, reverse repo, depósitos, TGA = **4 features**
   - Sentimiento: AAII bull-bear + 7 Google Trends = **15 features**
   - NLP: sentimiento VADER por ETF = **22 features**
4. **build_master_dataset.py:** Junta todo + crea los **targets** (retorno de la semana SIGUIENTE, shift -1)

**Resultado:** `master_weekly_raw.csv` — 987 semanas × 119 columnas.

### Fase 3 — EDA (`src/eda/`)

5 scripts generan ~20 figuras exploratorias que van en la sección de ingeniería del dato:
- Evolución de precios, distribución de retornos (colas pesadas → Markowitz falla)
- VIX y crisis, correlación SPY-AGG (se rompe en crisis → justifica ML)
- Estacionariedad (test ADF), outliers, complementariedad entre dimensiones

### Fase 4 — Modelado (`src/models/`) — El corazón

El flujo es por capas:

**Capa 1 — Infraestructura:**
- `data_loader.py`: Carga el CSV y clasifica las 109 features en 6 dimensiones
- `walk_forward.py`: Genera 778 splits de entrenamiento/test. En cada semana t: entrena con semanas [1..t-2], embargo de 1 semana, predice semana t
- `benchmarks.py`: Calcula las 3 carteras de referencia (60/40, equal weight, Markowitz clásico)

**Capa 2 — Entrenamiento base (`train_base.py`):**
- Entrena XGBoost y LightGBM con parámetros por defecto
- Para cada uno de los 10 ETFs: 778 modelos (uno por semana)
- Cada modelo predice el retorno de 1 ETF para 1 semana
- Las 10 predicciones van al optimizador de Markowitz → pesos de cartera
- Resultado: Sharpe 1.154 (XGB) y 1.000 (LGB) — ya superan al 60/40 (0.847)

**Capa 3 — Optuna (`tune_optuna.py`):**
- Busca los mejores hiperparámetros con 75 trials bayesianos
- Encuentra: lr=0.022, depth=7, min_child_weight=20 para XGB
- Resultado: **Sharpe 1.397** (XGB Tuned) — el resultado definitivo

**Capa 4 — Reporte (`train_final.py`):**
- NO reentrena — carga las predicciones ya calculadas
- Reconstruye los retornos de cartera, calcula benchmarks
- Imprime las 10 tablas y diagnósticos del reporte
- Guarda todo en `data/results/final/`

**Capa 5 — Interpretabilidad (`shap_analysis.py`):**
- Entrena 10 modelos XGBoost (80/20 split) y calcula TreeSHAP
- Genera 8 figuras: importancia global, por dimensión, beeswarm por ETF, heatmap, temporal, waterfall
- Hallazgo clave: **NFCI (condiciones financieras) es la variable #1** (16.5% del poder predictivo)

**Capa 6 — Backtesting extra (`backtesting_extra.py`):**
- Information Coefficient: IC = 0.089 (excelente en finanzas)
- Subperíodos: XGB gana en los 5 períodos vs 60/40
- Turnover: 37%/semana, coste estimado 0.97%/año, Sharpe neto 1.276
- Monte Carlo: p-value = 0.0000 (ninguna de 10.000 carteras aleatorias alcanza 1.397)

### Las 35 figuras (`docs/figures/`)

- **EDA (20 figuras):** evolución ETFs, distribuciones, VIX, correlaciones, outliers, estacionariedad, sentimiento, Google Trends, NLP...
- **SHAP (8 figuras):** importancia global top 20, donut por dimensión, beeswarm SPY/AGG/GLD, heatmap ETFs, evolución temporal, waterfall crisis SVB
- **Portfolio (4 figuras):** frontera eficiente con 5.000 carteras aleatorias, equity curves 100€→526€, subperíodos, turnover
- **Backtesting (3 figuras):** histograma Monte Carlo, barras subperíodos, serie temporal turnover

### Experimentos que no funcionaron (`src/experiments/`)

1. **+12 features nuevas** (WEI, MOVE, STLFSI4...): Sharpe bajó a 1.203 → más features = más ruido
2. **Feature selection SHAP** (60 features): Sharpe bajó a 1.271 → las features "inútiles" contribuyen colectivamente
3. **Wavelet denoising:** Sharpe colapsó a 0.392 → eliminó la señal junto con el ruido
4. **Optuna v2** (100 trials más): Confirmó que v1 ya encontró el óptimo

**Conclusión:** 109 features con los params de Optuna v1 es el óptimo. Documentar lo que NO funciona es tan valioso como lo que sí.

## Metodología (resumen)

1. **Ingeniería del dato:** Extracción de 6 fuentes, alineación semanal, limpieza, 109 features en 6 dimensiones
2. **Modelado:** XGBoost y LightGBM con walk-forward expanding window (778 splits), optimizados con Optuna (75 trials)
3. **Optimización de cartera:** Predicciones ML → optimizador de Markowitz (SLSQP, long-only, max 40% por activo)
4. **Interpretabilidad:** SHAP (TreeSHAP) — NFCI es el driver principal (16.5%)
5. **Validación:** IC = 0.089, Dir. Accuracy = 56.8%, Monte Carlo p-value = 0.0000

## Herramientas

Python 3.12, XGBoost 3.2, LightGBM 4.6, Optuna 4.3, SHAP 0.51, pandas, matplotlib, scipy.

**Asistencia IA:** Claude (Anthropic) — planificación, generación de código y apoyo en redacción. Las decisiones metodológicas, interpretación de resultados y conclusiones son responsabilidad exclusiva del autor.

## Licencia

Proyecto académico — Universidad Francisco de Vitoria, 2025-2026.
