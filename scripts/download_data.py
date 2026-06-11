"""Скачивание датасета UCI "Default of Credit Card Clients".

Датасет: 30 000 держателей кредитных карт (Тайвань, 2005 год).
Целевая переменная — дефолт по платежу в следующем месяце (1/0).

Запуск из корня проекта:
    python scripts/download_data.py
"""

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

# Официальная ссылка UCI ML Repository
URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

# Папка data/ в корне проекта (относительно этого файла)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TARGET = DATA_DIR / "default_of_credit_card_clients.xls"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    if TARGET.exists():
        print(f"Файл уже скачан: {TARGET}")
        return

    print("Скачиваю архив с UCI...")
    raw = urlopen(URL).read()

    # Внутри zip лежит один xls-файл — достаём и сохраняем под удобным именем
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = zf.namelist()[0]
        TARGET.write_bytes(zf.read(name))

    print(f"Готово: {TARGET} ({TARGET.stat().st_size / 1e6:.1f} МБ)")


if __name__ == "__main__":
    main()
