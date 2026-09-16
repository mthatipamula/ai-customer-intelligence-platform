from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder

from data_pipeline import prepare_data


MODEL_PATH = Path("models/resolution_time_model_optimized.joblib")


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


def optimize_model(df: pd.DataFrame):
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

    baseline_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    baseline_model.fit(X_train_processed, y_train)

    baseline_predictions = baseline_model.predict(X_test_processed)

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions,
    )

    print(f"Baseline MAE: {baseline_mae:.2f} hours")

    parameter_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
        ),
        param_grid=parameter_grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1,
    )

    grid_search.fit(X_train_processed, y_train)

    optimized_model = grid_search.best_estimator_

    optimized_predictions = optimized_model.predict(
        X_test_processed
    )

    optimized_mae = mean_absolute_error(
        y_test,
        optimized_predictions,
    )

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(f"\nOptimized MAE: {optimized_mae:.2f} hours")

    improvement = baseline_mae - optimized_mae

    print(
        f"MAE improvement: {improvement:.2f} hours"
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": optimized_model,
            "preprocessor": preprocessor,
        },
        MODEL_PATH,
    )

    print(f"\nOptimized model saved to: {MODEL_PATH}")

    return optimized_model, preprocessor


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "tickets.csv"

    df = prepare_data(str(data_path))

    print(f"Dataset records: {len(df)}")

    optimize_model(df)