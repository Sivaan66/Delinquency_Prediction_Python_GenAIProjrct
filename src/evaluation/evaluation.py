import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve
)


def evaluate_model(model, X_test, y_test, y_prob):
    """
    Evaluate classification model performance.
    """

    y_pred = model.predict(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))


    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))


    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    pr_auc = average_precision_score(
        y_test,
        y_prob
    )


    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    print(f"PR-AUC Score: {pr_auc:.4f}")


    return {
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred
        ),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }



def plot_precision_recall(y_test, y_prob, save_path):
    """
    Save Precision-Recall curve.
    """

    precision, recall, _ = precision_recall_curve(
        y_test,
        y_prob
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    plt.grid(True)

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close()



def plot_roc_curve(y_test, y_prob, save_path):
    """
    Save ROC curve.
    """

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    auc_score = roc_auc_score(
        y_test,
        y_prob
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {auc_score:.2f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close()