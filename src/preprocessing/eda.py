from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "Delinquency_prediction_dataset.xlsx"

df = pd.read_excel(
    "Data/Delinquency_prediction_dataset.xlsx",
    engine="openpyxl"
)

print(df.head())

def load_data():
    """Load the delinquency dataset."""
    return pd.read_excel(DATA_PATH)


def show_dataset_info(df):
    """Display dataset information and missing values."""
    print("\nDataset Information")
    print("-" * 50)
    print(df.info())

    print("\nMissing Values")
    print("-" * 50)
    print(df.isnull().sum())


def analyze_missing_income(df):
    """Analyze missing Income values."""

    df["Income_missing"] = df["Income"].isnull().astype(int)

    print("\nRows with Missing Income")
    print("-" * 50)
    print(df[df["Income_missing"] == 1])

    print("\nMissing Income by Employment Status")
    print("-" * 50)
    print(df.groupby("Employment_Status")["Income_missing"].mean())

    print("\nMissing Income by Credit Card Type")
    print("-" * 50)
    print(df.groupby("Credit_Card_Type")["Income_missing"].mean())

    remaining = df[
        (df["Employment_Status"] == "EMP")
        & (df["Income"].isnull())
    ].shape[0]

    print(f"\nRemaining missing EMP incomes: {remaining}")


def main():
    df = load_data()
    show_dataset_info(df)
    analyze_missing_income(df)


if __name__ == "__main__":
    main()