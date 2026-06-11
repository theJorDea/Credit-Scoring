"""Модели PD (probability of default) и калибровка вероятностей.

Две модели — это осознанный выбор:
- логистическая регрессия — отраслевой стандарт: интерпретируемая,
  стабильная, её легко защитить перед валидацией и регулятором;
- градиентный бустинг (LightGBM) — сильный нелинейный бенчмарк,
  показывает «потолок» качества на этих данных.

Калибровка: скоринговой модели мало правильно ранжировать клиентов —
predict_proba должен быть честной вероятностью дефолта, потому что PD
дальше идёт в ожидаемые потери (EL = PD * LGD * EAD) и в ценообразование.
Поэтому бустинг дополнительно калибруем изотонической регрессией.
"""

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logreg() -> Pipeline:
    """Логистическая регрессия со стандартизацией признаков.

    - StandardScaler обязателен: признаки имеют разные масштабы
      (LIMIT_BAL ~ 10^5, PAY_x ~ единицы), без него L2-штраф
      несправедливо давит крупные по масштабу коэффициенты;
    - class_weight="balanced" компенсирует дисбаланс классов (22/78);
    - C=1.0 — умеренная L2-регуляризация по умолчанию.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def make_lgbm() -> LGBMClassifier:
    """Градиентный бустинг LightGBM.

    Гиперпараметры консервативные, против переобучения:
    - n_estimators=500 при learning_rate=0.03 — много слабых шагов;
    - num_leaves=31 и min_child_samples=50 ограничивают сложность дерева;
    - subsample/colsample вносят случайность (стохастический бустинг).
    """
    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )


def make_calibrated_lgbm() -> CalibratedClassifierCV:
    """LightGBM + изотоническая калибровка вероятностей.

    CalibratedClassifierCV с cv=5: модель обучается на 4/5 данных,
    калибровочная кривая строится на отложенной 1/5, и так 5 раз.
    Изотоническая регрессия выбрана вместо сигмоиды (Платта), потому
    что данных достаточно (24k строк train) и она не навязывает форму
    искажения вероятностей.
    """
    return CalibratedClassifierCV(make_lgbm(), method="isotonic", cv=5)


def fit_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Обучает все модели и возвращает словарь {имя: модель}."""
    models = {
        "logreg": make_logreg(),
        "lgbm": make_lgbm(),
        "lgbm_calibrated": make_calibrated_lgbm(),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)

    return models
