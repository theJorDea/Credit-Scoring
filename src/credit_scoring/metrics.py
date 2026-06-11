"""Метрики качества скоринговой модели.

В кредитном скоринге смотрят на две независимые вещи:
1. Ранжирование — умеет ли модель отделять плохих от хороших:
   ROC-AUC, Gini = 2*AUC - 1, KS-статистика;
2. Калибровка — являются ли предсказанные PD честными вероятностями:
   Brier score, calibration curve (reliability diagram).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """KS (Колмогоров–Смирнов): максимальный разрыв между ROC-кривыми
    распределений скоров «плохих» и «хороших».

    Через ROC это просто max(TPR - FPR). В индустрии KS > 0.3 обычно
    считается приемлемой моделью, KS > 0.4 — хорошей.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Сводка основных метрик по предсказанным вероятностям."""
    auc = roc_auc_score(y_true, y_prob)
    return {
        "roc_auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks": round(ks_statistic(y_true, y_prob), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
    }


def evaluate_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """Считает метрики для всех моделей -> таблица для сравнения."""
    rows = []
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        rows.append({"model": name, **evaluate(y_test, y_prob)})

    return pd.DataFrame(rows).set_index("model")


def decile_table(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Дециль-анализ: делим клиентов на 10 групп по убыванию PD
    и смотрим фактическую долю дефолтов в каждой группе.

    У хорошей модели дефолтность монотонно падает от 1-го дециля
    (самые рискованные) к 10-му (самые надёжные). Это стандартный
    отчёт для бизнеса: по нему видно, где резать одобрение.
    """
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    # Дециль 1 = самые высокие PD
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False), 10, labels=range(1, 11))

    out = df.groupby("decile", observed=True).agg(
        clients=("y", "size"),
        mean_pd=("p", "mean"),
        default_rate=("y", "mean"),
    )
    out["mean_pd"] = out["mean_pd"].round(4)
    out["default_rate"] = out["default_rate"].round(4)
    return out
