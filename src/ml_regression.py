from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from data_pipeline import prepare_data


MODEL_PATH = Path("models/resolution_time_model.joblib")


FEATURES = [
    "category",
    "priority",
    "sentiment",
    "customer_spend",
    "previous_tickets",
    "ticket_length",
    "is_high_value_customer",
    "is_frequent_customer",
]

TARGET = "resolution_hours"


def train_resolution_model(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3333,
        random_state=42,
    )

    categorical_features = [
        "category",
        "priority",
        "sentiment",
    ]

    numeric_features = [
        "customer_spend",
        "previous_tickets",
        "ticket_length",
        "is_high_value_customer",
        "is_frequent_customer",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_processed, y_train)

    predictions = model.predict(X_test_processed)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    print(f"Mean Absolute Error: {mae:.2f} hours")
    print(f"Root Mean Squared Error: {rmse:.2f} hours")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
        },
        MODEL_PATH,
    )

    print(f"\nModel saved to: {MODEL_PATH}")

    return model, preprocessor


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "tickets.csv"

    df = prepare_data(str(data_path))

    print(f"Training dataset: {len(df)} records")

    train_resolution_model(df)