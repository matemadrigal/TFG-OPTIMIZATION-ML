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

## Metodología

1. **Ingeniería del dato:** Extracción de 6 fuentes (Yahoo Finance, FRED, Google Trends, AAII, Refinitiv), alineación a frecuencia semanal, limpieza, feature engineering
2. **Modelado:** XGBoost y LightGBM entrenados con walk-forward expanding window (778 splits), optimizados con Optuna (75 trials bayesianos)
3. **Optimización de cartera:** Predicciones ML como retornos esperados en el optimizador de Markowitz (SLSQP, long-only, max 40% por activo)
4. **Interpretabilidad:** SHAP (TreeSHAP) para identificar variables predictivas — NFCI es el driver principal (16.5%)
5. **Validación:** IC = 0.089, Dir. Accuracy = 56.8%, Monte Carlo p-value = 0.0000

## Herramientas

Python 3.12, XGBoost 3.2, LightGBM 4.6, Optuna 4.3, SHAP 0.51, pandas, matplotlib, scipy.

**Asistencia IA:** Claude (Anthropic) — planificación, generación de código y apoyo en redacción. Las decisiones metodológicas, interpretación de resultados y conclusiones son responsabilidad exclusiva del autor.

## Licencia

Proyecto académico — Universidad Francisco de Vitoria, 2025-2026.
