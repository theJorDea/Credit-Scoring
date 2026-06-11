"""Инженерия признаков + WoE/IV-анализ.

Две независимые части:
1. `add_features` — ручные признаки поверх исходных колонок
   (утилизация лимита, агрегаты по истории платежей и т.п.);
2. WoE (Weight of Evidence) и IV (Information Value) — классический
   банковский инструмент оценки силы признака до обучения модели.
"""

import numpy as np
import pandas as pd


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Добавляет производные признаки. Возвращает копию DataFrame.

    Логика признаков:
    - utilization: средний счёт / кредитный лимит — насколько клиент
      «выбирает» свой лимит. Высокая утилизация — классический ранний
      индикатор проблем;
    - pay_ratio: сколько клиент реально платит относительно счёта;
    - max_delay / mean_delay: худшая и средняя просрочка за 6 месяцев
      (PAY_x > 0 означает задержку платежа на x месяцев);
    - n_delays: в скольких месяцах из шести была просрочка;
    - bill_trend: растёт или падает задолженность (разница последнего
      и самого старого счёта, нормированная на лимит).
    """
    X = X.copy()

    pay_cols = [f"PAY_{i}" for i in range(1, 7)]
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]

    # Средний счёт и средний платёж за полгода
    mean_bill = X[bill_cols].mean(axis=1)
    mean_pay = X[amt_cols].mean(axis=1)

    # Утилизация лимита (лимит всегда > 0, делить безопасно)
    X["utilization"] = mean_bill / X["LIMIT_BAL"]

    # Доля оплаты счёта; +1 в знаменателе защищает от деления на ноль
    X["pay_ratio"] = mean_pay / (mean_bill.clip(lower=0) + 1)

    # Статистики по просрочкам
    X["max_delay"] = X[pay_cols].max(axis=1)
    X["mean_delay"] = X[pay_cols].mean(axis=1)
    X["n_delays"] = (X[pay_cols] > 0).sum(axis=1)

    # Тренд задолженности: >0 — долг растёт
    X["bill_trend"] = (X["BILL_AMT1"] - X["BILL_AMT6"]) / X["LIMIT_BAL"]

    return X


def woe_iv_table(
    x: pd.Series,
    y: pd.Series,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    """Считает WoE по бинам признака и суммарный IV.

    WoE_i = ln( (доля «хороших» в бине i) / (доля «плохих» в бине i) )
    IV    = sum_i (доля хороших_i - доля плохих_i) * WoE_i

    Числовые признаки бьём на квантильные бины (равное наполнение),
    категориальные (мало уникальных значений) — по самим значениям.
    Эпсилон-сглаживание защищает от ln(0) в пустых бинах.

    Интерпретация IV (стандартные пороги в кредитном скоринге):
    < 0.02 — бесполезный; 0.02–0.1 — слабый; 0.1–0.3 — средний;
    0.3–0.5 — сильный; > 0.5 — подозрительно сильный (проверить утечку).
    """
    eps = 1e-6

    # Категориальный признак — группируем по значениям, иначе по квантилям
    if x.nunique() <= n_bins:
        bins = x.astype(str)
    else:
        bins = pd.qcut(x, q=n_bins, duplicates="drop").astype(str)

    df = pd.DataFrame({"bin": bins, "y": y.values})
    grp = df.groupby("bin", observed=True)["y"].agg(["count", "sum"])
    grp.columns = ["total", "bad"]  # bad = дефолты (y == 1)
    grp["good"] = grp["total"] - grp["bad"]

    # Доли хороших/плохих в бине от всех хороших/плохих
    grp["good_rate"] = grp["good"] / max(grp["good"].sum(), 1)
    grp["bad_rate"] = grp["bad"] / max(grp["bad"].sum(), 1)

    grp["woe"] = np.log((grp["good_rate"] + eps) / (grp["bad_rate"] + eps))
    grp["iv"] = (grp["good_rate"] - grp["bad_rate"]) * grp["woe"]

    return grp, float(grp["iv"].sum())


def iv_report(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """IV по всем признакам, отсортированный по убыванию."""
    rows = []
    for col in X.columns:
        _, iv = woe_iv_table(X[col], y)
        rows.append({"feature": col, "iv": round(iv, 4)})

    return (
        pd.DataFrame(rows)
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )
