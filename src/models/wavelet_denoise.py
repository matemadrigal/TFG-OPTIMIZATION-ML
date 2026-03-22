"""
Wavelet denoising para features del dataset.
Daubechies-4, nivel 2, soft thresholding universal.
Solo se aplica a features de entrenamiento (sin data leakage).

Autor: Mateo Madrigal Arteaga, UFV
"""

import numpy as np
import pandas as pd
import pywt


def wavelet_denoise_features(df, feature_cols, wavelet="db4", level=2):
    """
    Aplica wavelet denoising a las columnas de features.
    Solo para datos de entrenamiento (sin leakage).

    Args:
        df: DataFrame con features (solo train data)
        feature_cols: lista de columnas de features (NO targets)
        wavelet: tipo de wavelet (default: db4, Daubechies 4)
        level: nivel de descomposición (default: 2)

    Returns:
        DataFrame con features denoised
    """
    df_denoised = df.copy()

    for col in feature_cols:
        series = df[col].values.astype(float)

        # Skip si la serie es constante o tiene NaN
        if np.nanstd(series) == 0:
            continue

        # Rellenar NaN temporalmente para la transformada
        has_nan = np.any(np.isnan(series))
        if has_nan:
            nan_mask = np.isnan(series)
            series_filled = pd.Series(series).ffill().bfill().values
        else:
            series_filled = series
            nan_mask = None

        # Necesitamos mínimo 2^level + 1 puntos
        min_len = 2 ** level + 1
        if len(series_filled) < min_len:
            continue

        # Descomposición wavelet
        coeffs = pywt.wavedec(series_filled, wavelet, level=level)

        # Umbral universal para cada nivel de detalle (no tocar aproximación [0])
        for i in range(1, len(coeffs)):
            detail = coeffs[i]
            # Estimación robusta de sigma con MAD
            sigma = np.median(np.abs(detail)) / 0.6745
            if sigma > 0:
                threshold = sigma * np.sqrt(2 * np.log(len(series_filled)))
                coeffs[i] = pywt.threshold(detail, threshold, mode="soft")

        # Reconstrucción
        denoised = pywt.waverec(coeffs, wavelet)

        # pywt puede devolver 1 elemento extra por padding
        denoised = denoised[: len(series)]

        # Restaurar NaN originales
        if has_nan and nan_mask is not None:
            denoised[nan_mask] = np.nan

        df_denoised[col] = denoised

    return df_denoised
