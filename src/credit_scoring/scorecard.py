"""Скоркарта: перевод вероятности дефолта в привычные скоринговые баллы.

Банки не показывают клиентам и менеджерам «PD = 0.37» — используется
шкала баллов. Стандартная параметризация через PDO (Points to Double
the Odds):

    score = offset - factor * ln(odds),   где odds = PD / (1 - PD)

- PDO — на сколько баллов нужно изменить скор, чтобы шансы дефолта
  удвоились (классика: PDO = 20);
- базовая точка: score = base_score при odds = base_odds
  (классика: 600 баллов при шансах 1:19, т.е. PD = 5%).
"""

import numpy as np


class Scorecard:
    """Преобразование PD <-> скоринговый балл по шкале PDO."""

    def __init__(self, base_score: float = 600, base_odds: float = 1 / 19, pdo: float = 20):
        # factor: на сколько баллов меняется скор при изменении ln(odds) на 1
        self.factor = pdo / np.log(2)
        # offset подбирается так, чтобы базовая точка попала в base_score
        self.offset = base_score + self.factor * np.log(base_odds)

    def score(self, pd_: np.ndarray) -> np.ndarray:
        """PD -> баллы. Чем выше балл, тем надёжнее клиент."""
        pd_ = np.clip(pd_, 1e-6, 1 - 1e-6)  # защита от ln(0)
        odds = pd_ / (1 - pd_)
        return self.offset - self.factor * np.log(odds)

    def pd_from_score(self, score: np.ndarray) -> np.ndarray:
        """Обратное преобразование: баллы -> PD."""
        odds = np.exp((self.offset - np.asarray(score)) / self.factor)
        return odds / (1 + odds)
