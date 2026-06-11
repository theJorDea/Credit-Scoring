"""Интерпретация модели через SHAP.

В банке интерпретируемость — не опция, а требование: и внутренняя
валидация, и регулятор должны понимать, почему модель отказала клиенту
(reason codes). SHAP-значения дают точное разложение каждого предсказания
на вклады признаков: sum(shap_values) + base_value = предсказание модели.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # рисуем в файлы, без дисплея

import matplotlib.pyplot as plt
import pandas as pd
import shap


def shap_summary(model, X: pd.DataFrame, out_path: Path, max_display: int = 15) -> pd.DataFrame:
    """Строит SHAP summary plot и возвращает таблицу важности признаков.

    Используем TreeExplainer — для деревьев (LightGBM) он точный
    и быстрый, в отличие от сэмплирующих аппроксимаций.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Для бинарной классификации LightGBM возвращает значения класса 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Summary plot: каждая точка — клиент, цвет — значение признака
    plt.figure()
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    # Глобальная важность = средний |SHAP| по выборке
    importance = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "mean_abs_shap": abs(shap_values).mean(axis=0).round(4),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return importance
