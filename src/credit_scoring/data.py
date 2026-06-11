"""Загрузка и подготовка данных.

Датасет UCI "Default of Credit Card Clients":
- 30 000 клиентов, 23 признака;
- целевая переменная `default payment next month` (1 — дефолт, 0 — нет);
- доля дефолтов ~22% (умеренный дисбаланс классов).
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Путь к xls по умолчанию (см. scripts/download_data.py)
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "default_of_credit_card_clients.xls"

# Имя целевой переменной после переименования
TARGET = "default"

# Категориальные признаки (кодируются целыми числами уже в исходных данных)
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]


def load_raw(path: Path | str = DEFAULT_PATH) -> pd.DataFrame:
    """Читает исходный xls и приводит его к аккуратному виду.

    Что делаем:
    1. читаем со второй строки (первая строка xls — служебный заголовок);
    2. убираем колонку ID — это просто номер строки, не признак;
    3. переименовываем PAY_0 -> PAY_1, чтобы история платежей шла
       единообразно: PAY_1..PAY_6 (1 — последний месяц, 6 — самый старый);
    4. укорачиваем имя целевой переменной до `default`.
    """
    df = pd.read_excel(path, header=1)

    df = df.drop(columns=["ID"])
    df = df.rename(columns={"PAY_0": "PAY_1", "default payment next month": TARGET})

    # Чистка редких/недокументированных категорий:
    # EDUCATION: 0, 5, 6 не описаны в документации -> объединяем в 4 ("другое")
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    # MARRIAGE: 0 не описан -> объединяем в 3 ("другое")
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Делит данные на train/test со стратификацией по таргету.

    Стратификация важна: доля дефолтов (~22%) должна быть одинаковой
    в обеих выборках, иначе метрики будут смещены.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
