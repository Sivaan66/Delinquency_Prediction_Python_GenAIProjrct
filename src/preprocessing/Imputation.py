# src/preprocessing/imputation.py

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


def create_missing_indicators(df):

    columns = [
        "Income",
        "Credit_Score",
        "Loan_Balance"
    ]

    for col in columns:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    return df



def clean_employment_status(df):

    df["Employment_Status"] = (
        df["Employment_Status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "emp": "employed",
        "employed": "employed",
        "self-employed": "self-employed",
        "unemployed": "unemployed",
        "retired": "retired"
    }

    df["Employment_Status"] = (
        df["Employment_Status"]
        .replace(mapping)
    )

    return df



def regression_income_imputation(df):

    features = [
        "Age",
        "Loan_Balance",
        "Credit_Utilization",
        "Debt_to_Income_Ratio",
        "Account_Tenure"
    ]


    if "Income" not in df.columns:
        return df


    # Train model only on available income
    train_df = df[df["Income"].notna()].copy()


    train_df = train_df.dropna(
        subset=features
    )


    X_train = train_df[features]
    y_train = train_df["Income"]


    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )


    # Predict missing income rows
    missing_df = df[
        df["Income"].isna()
    ].copy()


    missing_df = missing_df.dropna(
        subset=features
    )


    if len(missing_df) > 0:

        predictions = model.predict(
            missing_df[features]
        )


        # Income cannot be negative
        predictions = np.maximum(
            predictions,
            0
        )


        df.loc[
            missing_df.index,
            "Income"
        ] = predictions.round(0)


    return df



def median_imputation(df):

    numerical_columns = [
        "Credit_Score",
        "Loan_Balance"
    ]


    for col in numerical_columns:

        if col in df.columns:

            df[col] = (
                df[col]
                .fillna(
                    df[col].median()
                )
            )

    return df



def impute_dataset(df):

    print("Missing values before:")
    print(df.isnull().sum())


    # Step 1
    df = create_missing_indicators(df)


    # Step 2
    df = clean_employment_status(df)


    # Step 3
    df = regression_income_imputation(df)


    # Step 4
    df = median_imputation(df)


    # Step 5
    # Remaining income fallback
    df["Income"] = (
        df["Income"]
        .fillna(
            df["Income"].median()
        )
    )


    print("\nMissing values after:")
    print(df.isnull().sum())


    return df