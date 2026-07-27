import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE


def train_model(X, y):
    """
    Train Random Forest model with preprocessing and SMOTE.

    Parameters:
        X : Features
        y : Target

    Returns:
        model, preprocessor, X_test, y_test, y_prob
    """

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Identify columns
    numerical_features = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include="object"
    ).columns.tolist()


    # Numerical preprocessing
    numerical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )


    # Categorical preprocessing
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )


    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_transformer,
                numerical_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )


    # Transform data
    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )


    # Handle imbalance
    smote = SMOTE(
        random_state=42
    )

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_processed,
        y_train
    )


    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )


    model.fit(
        X_train_resampled,
        y_train_resampled
    )


    # Prediction probability
    y_prob = model.predict_proba(
        X_test_processed
    )[:, 1]


    print("Model training completed.")

    return (
        model,
        preprocessor,
        X_test_processed,
        y_test,
        y_prob
    )