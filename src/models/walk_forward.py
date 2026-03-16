"""
Validación Walk-Forward con expanding window.
Fase 4 — Modelado | TFG Optimización de Carteras con ML
Autor: Mateo Madrigal Arteaga, UFV
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.data_loader import load_master_dataset


class WalkForwardValidator:
    """
    Genera splits de train/test con expanding window para series temporales.

    Parámetros:
        min_train_weeks: semanas mínimas de entrenamiento (208 = 4 años)
        retrain_every:   cada cuántas semanas se reentrena (1 = semanal)
        embargo_weeks:   semanas de embargo entre train y test para evitar
                         leakage por autocorrelación temporal
    """

    def __init__(self, min_train_weeks=208, retrain_every=1, embargo_weeks=1):
        self.min_train_weeks = min_train_weeks
        self.retrain_every = retrain_every
        self.embargo_weeks = embargo_weeks

    def generate_splits(self, dates_index):
        """
        Genera splits (train_idx, test_idx) con expanding window.

        Cada split:
            - train: índices [0, t]
            - test:  índice  [t + embargo_weeks] (una sola semana)

        Retorna una lista de tuplas (array de índices train, array de índice test).
        """
        n = len(dates_index)
        splits = []

        # El primer test posible es en min_train_weeks + embargo_weeks
        first_test = self.min_train_weeks + self.embargo_weeks

        for test_pos in range(first_test, n, self.retrain_every):
            # Train: desde el inicio hasta embargo_weeks antes del test
            train_end = test_pos - self.embargo_weeks
            train_idx = list(range(0, train_end))
            test_idx = [test_pos]
            splits.append((train_idx, test_idx))

        # ── Resumen ──
        if splits:
            first_train, first_test_idx = splits[0]
            last_train, last_test_idx = splits[-1]

            print("=" * 60)
            print("WALK-FORWARD VALIDATION")
            print("=" * 60)
            print(f"  Configuración:")
            print(f"    Min train:     {self.min_train_weeks} semanas ({self.min_train_weeks // 52} años)")
            print(f"    Retrain every: {self.retrain_every} semana(s)")
            print(f"    Embargo:       {self.embargo_weeks} semana(s)")
            print(f"\n  Splits generados: {len(splits)}")
            print(f"    Primer test: {dates_index[first_test_idx[0]].date()} "
                  f"(train: {len(first_train)} semanas)")
            print(f"    Último test: {dates_index[last_test_idx[0]].date()} "
                  f"(train: {len(last_train)} semanas)")
            print(f"\n  Semanas totales: {n}")
            print(f"    Solo train (sin test): {self.min_train_weeks + self.embargo_weeks}")
            print(f"    Con predicción (test): {len(splits)}")
            print("=" * 60)
        else:
            print("[AVISO] No se generaron splits. Dataset demasiado corto.")

        return splits


# ── Ejecución directa ──────────────────────────────────────────────

if __name__ == "__main__":
    features, targets = load_master_dataset()

    wf = WalkForwardValidator(min_train_weeks=208, retrain_every=1, embargo_weeks=1)
    splits = wf.generate_splits(features.index)

    # Mostrar algunos splits de ejemplo
    print(f"\nEjemplos de splits:")
    for i in [0, len(splits) // 2, len(splits) - 1]:
        train_idx, test_idx = splits[i]
        print(f"  Split {i:>3d}: train [{features.index[train_idx[0]].date()} → "
              f"{features.index[train_idx[-1]].date()}] ({len(train_idx)} sem) | "
              f"test [{features.index[test_idx[0]].date()}]")
