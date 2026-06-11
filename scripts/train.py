"""Полный пайплайн обучения: от сырых данных до отчётов и графиков.

Шаги:
1. загрузка и чистка данных;
2. инженерия признаков;
3. IV-отчёт (сила признаков до обучения);
4. обучение моделей (logreg, LightGBM, калиброванный LightGBM);
5. метрики на тесте (ROC-AUC, Gini, KS, Brier) + дециль-анализ;
6. графики: ROC-кривые, калибровочная кривая, распределение баллов,
   SHAP summary;
7. сохранение результатов в reports/.

Запуск из корня проекта (после scripts/download_data.py):
    python scripts/train.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

# Добавляем src/ в путь, чтобы скрипт работал без установки пакета
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_scoring.data import load_raw, split_data
from credit_scoring.features import add_features, iv_report
from credit_scoring.interpret import shap_summary
from credit_scoring.metrics import decile_table, evaluate_models
from credit_scoring.models import fit_models
from credit_scoring.scorecard import Scorecard

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def plot_roc(models: dict, X_test, y_test) -> None:
    """ROC-кривые всех моделей на одном графике."""
    plt.figure(figsize=(7, 6))
    for name, model in models.items():
        prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="случайная модель")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривые на тестовой выборке")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "roc_curves.png", dpi=150)
    plt.close()


def plot_calibration(models: dict, X_test, y_test) -> None:
    """Калибровочные кривые: предсказанный PD против фактической частоты."""
    plt.figure(figsize=(7, 6))
    for name in ["lgbm", "lgbm_calibrated"]:
        prob = models[name].predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10)
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="идеальная калибровка")
    plt.xlabel("Средний предсказанный PD")
    plt.ylabel("Фактическая доля дефолтов")
    plt.title("Калибровка вероятностей (до и после)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "calibration.png", dpi=150)
    plt.close()


def plot_scores(y_test, scores: np.ndarray) -> None:
    """Распределение скоринговых баллов отдельно для дефолтов и недефолтов."""
    plt.figure(figsize=(7, 5))
    plt.hist(scores[y_test == 0], bins=50, alpha=0.6, density=True, label="без дефолта")
    plt.hist(scores[y_test == 1], bins=50, alpha=0.6, density=True, label="дефолт")
    plt.xlabel("Скоринговый балл")
    plt.ylabel("Плотность")
    plt.title("Распределение баллов скоркарты по классам")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "score_distribution.png", dpi=150)
    plt.close()


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    # 1-2. Данные и признаки
    print("Загружаю данные...")
    df = load_raw()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train = add_features(X_train)
    X_test = add_features(X_test)
    print(f"train: {X_train.shape}, test: {X_test.shape}, доля дефолтов: {y_train.mean():.3f}")

    # 3. IV-отчёт по train (на тесте подглядывать нельзя)
    print("Считаю Information Value...")
    iv = iv_report(X_train, y_train)
    iv.to_csv(REPORTS / "iv_report.csv", index=False)
    print(iv.head(10).to_string(index=False))

    # 4. Модели
    print("\nОбучаю модели (logreg, LightGBM, калиброванный LightGBM)...")
    models = fit_models(X_train, y_train)

    # 5. Метрики и дециль-анализ
    results = evaluate_models(models, X_test, y_test)
    print("\nМетрики на тесте:")
    print(results.to_string())
    results.to_csv(REPORTS / "metrics.csv")
    (REPORTS / "metrics.json").write_text(
        json.dumps(results.to_dict(orient="index"), indent=2, ensure_ascii=False)
    )

    best_prob = models["lgbm_calibrated"].predict_proba(X_test)[:, 1]
    deciles = decile_table(y_test.values, best_prob)
    deciles.to_csv(REPORTS / "decile_table.csv")
    print("\nДециль-анализ (калиброванный LightGBM):")
    print(deciles.to_string())

    # 6. Графики
    print("\nРисую графики...")
    plot_roc(models, X_test, y_test)
    plot_calibration(models, X_test, y_test)

    scorecard = Scorecard()
    scores = scorecard.score(best_prob)
    plot_scores(y_test.values, scores)

    # SHAP — на некалиброванном LightGBM (TreeExplainer работает с деревьями)
    importance = shap_summary(models["lgbm"], X_test, FIGURES / "shap_summary.png")
    importance.to_csv(REPORTS / "shap_importance.csv", index=False)
    print("\nТоп-10 признаков по SHAP:")
    print(importance.head(10).to_string(index=False))

    print(f"\nГотово. Отчёты в {REPORTS}/, графики в {FIGURES}/")


if __name__ == "__main__":
    main()
