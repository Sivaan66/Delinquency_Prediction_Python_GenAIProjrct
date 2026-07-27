import os

from data.load_data import load_dataset
from preprocessing.Imputation import impute_dataset

from models.train_model import train_model

from evaluation.evaluation import (
    evaluate_model,
    plot_precision_recall,
    plot_roc_curve
)

from visualization.plots import (
    plot_feature_importance,
    plot_target_distribution,
    plot_correlation_heatmap
)


# Paths
DATA_PATH = r"Data\Delinquency_prediction_dataset.xlsx"

OUTPUT_DIR = "outputs"
IMAGE_DIR = "images"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


TARGET_COLUMN = "Delinquent_Account"



def main():

    # -----------------------------
    # Load Dataset
    # -----------------------------
    df = load_dataset(DATA_PATH)

    if df is None:
        return


    print("\nDataset Preview:")
    print(df.head())


    # -----------------------------
    # Basic EDA plots
    # -----------------------------
    plot_target_distribution(
        df,
        TARGET_COLUMN,
        f"{IMAGE_DIR}/target_distribution.png"
    )

    plot_correlation_heatmap(
        df,
        f"{IMAGE_DIR}/correlation_heatmap.png"
    )


    # -----------------------------
    # Missing Value Handling
    # -----------------------------
    print("\nApplying Imputation...")

    df = impute_dataset(df)


    print("\nMissing Values After Imputation:")
    print(df.isnull().sum())


    # -----------------------------
    # Split Features and Target
    # -----------------------------
    X = df.drop(
        columns=[
            "Customer_ID",
            TARGET_COLUMN
        ]
    )

    y = df[TARGET_COLUMN]


    # -----------------------------
    # Train Model
    # -----------------------------
    (
        model,
        preprocessor,
        X_test,
        y_test,
        y_prob
    ) = train_model(
        X,
        y
    )


    # -----------------------------
    # Evaluation
    # -----------------------------
    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        y_prob
    )


    # -----------------------------
    # Save Evaluation Curves
    # -----------------------------
    plot_precision_recall(
        y_test,
        y_prob,
        f"{IMAGE_DIR}/precision_recall_curve.png"
    )


    plot_roc_curve(
        y_test,
        y_prob,
        f"{IMAGE_DIR}/roc_curve.png"
    )


    # -----------------------------
    # Feature Importance
    # -----------------------------
    feature_importance = plot_feature_importance(
        model,
        preprocessor,
        f"{IMAGE_DIR}/feature_importance.png"
    )

    print("\nTop Features:")
    print(feature_importance)


    # -----------------------------
    # Save Metrics
    # -----------------------------
    with open(
        f"{OUTPUT_DIR}/metrics.txt",
        "w"
    ) as file:

        file.write(str(metrics))


    print("\nTraining Pipeline Completed Successfully.")



if __name__ == "__main__":
    main()