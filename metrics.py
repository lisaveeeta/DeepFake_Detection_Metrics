import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    ConfusionMatrixDisplay,
)

def evaluate_metrics(y_true, y_score, threshold=None, save_dir="results"):
    """
        y_true (list[int] | np.array): истинные метки классов (0 = real, 1 = fake);
        y_score (list[float] | np.array): предсказанные вероятности принадлежности к классу 1 (fake);
        threshold (float): порог бинаризации для перевода вероятностей в метки;
        save_dir (str): папка, куда сохранять графики;

        dict: словарь с рассчитанными метриками
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # бинаризация по порогу
    y_pred = (y_score >= threshold).astype(int)

    # метрики
    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) == 2 else float("nan")
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn + 1e-12)  # ошибка I рода
    fnr = fn / (fn + tp + 1e-12)  # ошибка II рода

    # поиск F1-оптимального порога
    fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_score)
    f1s = []
    for t in thresholds:
        yp = (y_score >= t).astype(int)
        f1s.append(f1_score(y_true, yp))
    t_best = thresholds[int(np.argmax(f1s))] if len(thresholds) > 0 else threshold
    f1_best = max(f1s) if len(f1s) > 0 else f1

    # ROC-кривая
    plt.figure()
    plt.plot(fpr_curve, tpr_curve, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(f"{save_dir}/roc_curve.png")
    plt.close()

    # матрица ошибок
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    disp.figure_.suptitle("Confusion Matrix")
    plt.savefig(f"{save_dir}/confusion_matrix.png")
    plt.close()

    metrics = {
        "AUC": auc,
        "F1": f1,
        "FPR": fpr,
        "FNR": fnr,
        "Best_Threshold": t_best,
        "Best_F1": f1_best,
    }

    return metrics
