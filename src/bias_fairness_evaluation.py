from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from data_pipeline import prepare_data


MODEL_PATH = Path("models/resolution_time_model.joblib")


def evaluate_group_fairness(df, model, preprocessor, group_column):
    train_df, test_df = train_test_split(
        df,
        test_size=0.3333,
        random_state=42,
    )

    features = [
        "category",
        "priority",
        "sentiment",
        "customer_spend",
        "previous_tickets",
        "ticket_length",
        "is_high_value_customer",
        "is_frequent_customer",
    ]

    X_test = test_df[features]
    y_test = test_df["resolution_hours"]

    X_test_processed = preprocessor.transform(X_test)
    predictions = model.predict(X_test_processed)

    results = test_df.copy()
    results["prediction"] = predictions
    results["absolute_error"] = (
        results["resolution_hours"] - results["prediction"]
    ).abs()

    results["group"] = results[group_column]

    group_mae = (
        results.groupby("group")["absolute_error"]
        .mean()
        .reset_index(name="mae_hours")
    )

    return group_mae


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "tickets.csv"

    df = prepare_data(str(data_path))

    artifacts = joblib.load(MODEL_PATH)
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]

    print("Bias and Fairness Evaluation")
    print("=" * 40)

    print("\n1. High Value vs Standard Customers")
    print("-" * 40)

    df["customer_value_group"] = (
        df["customer_spend"] >= 3000
    ).map({
        True: "High Value",
        False: "Standard",
    })

    value_results = evaluate_group_fairness(
        df,
        model,
        preprocessor,
        "customer_value_group",
    )

    print(value_results.to_string(index=False))

    print("\n2. Frequent vs Non-Frequent Customers")
    print("-" * 40)

    df["support_frequency_group"] = (
        df["previous_tickets"] >= 5
    ).map({
        True: "Frequent Customer",
        False: "Non-Frequent Customer",
    })

    frequency_results = evaluate_group_fairness(
        df,
        model,
        preprocessor,
        "support_frequency_group",
    )

    print(frequency_results.to_string(index=False))

    print("\nFairness Interpretation")
    print("-" * 40)
    print(
        "MAE is compared across customer behavioral groups "
        "to identify differences in prediction error."
    )
    print(
        "This evaluation does not measure protected-attribute fairness "
        "because the dataset does not contain protected attributes "
        "such as race, gender, or age."
    )


if __name__ == "__main__":
    main()