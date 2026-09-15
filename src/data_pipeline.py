from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "ticket_id",
    "customer_id",
    "ticket_text",
    "category",
    "priority",
    "sentiment",
    "resolution_hours",
    "customer_spend",
    "previous_tickets",
]


def load_data(file_path: str) -> pd.DataFrame:
    """Load customer support tickets from a CSV file."""
    return pd.read_csv(file_path)


def validate_data(df: pd.DataFrame) -> None:
    """Validate that all required columns are present."""
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize customer support data."""

    clean_df = df.copy()

    # Remove duplicate tickets
    clean_df = clean_df.drop_duplicates(subset=["ticket_id"])

    # Remove records without ticket text
    clean_df = clean_df.dropna(subset=["ticket_text"])

    # Normalize text
    clean_df["ticket_text"] = (
        clean_df["ticket_text"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Normalize categorical columns
    for column in ["category", "priority", "sentiment"]:
        clean_df[column] = clean_df[column].astype(str).str.strip()

    # Ensure numeric columns have numeric types
    clean_df["resolution_hours"] = pd.to_numeric(
        clean_df["resolution_hours"], errors="coerce"
    )

    clean_df["customer_spend"] = pd.to_numeric(
        clean_df["customer_spend"], errors="coerce"
    )

    clean_df["previous_tickets"] = pd.to_numeric(
        clean_df["previous_tickets"], errors="coerce"
    )

    # Remove rows where required numeric features are invalid
    clean_df = clean_df.dropna(
        subset=[
            "resolution_hours",
            "customer_spend",
            "previous_tickets",
        ]
    )

    # Feature engineering
    clean_df["ticket_length"] = clean_df["ticket_text"].str.len()

    clean_df["is_high_value_customer"] = (
        clean_df["customer_spend"] >= 3000
    ).astype(int)

    clean_df["is_frequent_customer"] = (
        clean_df["previous_tickets"] >= 5
    ).astype(int)

    return clean_df.reset_index(drop=True)


def prepare_data(file_path: str) -> pd.DataFrame:
    """Load, validate, and clean customer support data."""
    df = load_data(file_path)
    validate_data(df)
    return clean_data(df)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "tickets.csv"

    df = prepare_data(str(data_path))

    print("Cleaned dataset:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())