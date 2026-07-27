import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_feature_importance(model, preprocessor, save_path, top_n=10):
    """
    Plot and save feature importance from Random Forest model.
    """

    feature_names = preprocessor.get_feature_names_out()

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).head(top_n)


    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature"
    )

    plt.title(
        f"Top {top_n} Feature Importance"
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.tight_layout()

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close()

    return importance_df



def plot_target_distribution(df, target_column, save_path):
    """
    Plot target class distribution.
    """

    plt.figure(figsize=(6, 4))

    sns.countplot(
        data=df,
        x=target_column
    )

    plt.title(
        "Target Variable Distribution"
    )

    plt.xlabel(
        target_column
    )

    plt.ylabel(
        "Count"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close()



def plot_correlation_heatmap(df, save_path):
    """
    Plot correlation heatmap for numerical features.
    """

    numerical_df = df.select_dtypes(
        include="number"
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        numerical_df.corr(),
        annot=True,
        fmt=".2f"
    )

    plt.title(
        "Feature Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close()