import sqlite3

import pandas as pd


def create_database(
    df: pd.DataFrame,
    database_path: str = "data/customer_intelligence.db",
) -> None:
    """Store the prepared customer ticket data in SQLite."""

    connection = sqlite3.connect(database_path)

    try:
        df.to_sql(
            "customer_tickets",
            connection,
            if_exists="replace",
            index=False,
        )
    finally:
        connection.close()


def query_customer_summary(
    database_path: str = "data/customer_intelligence.db",
) -> pd.DataFrame:
    """Run SQL analytics against customer ticket data."""

    connection = sqlite3.connect(database_path)

    query = """
        SELECT
            category,
            COUNT(*) AS ticket_count,
            AVG(customer_spend) AS average_customer_spend,
            AVG(resolution_hours) AS average_resolution_hours
        FROM customer_tickets
        GROUP BY category
        ORDER BY ticket_count DESC
    """

    try:
        return pd.read_sql_query(query, connection)
    finally:
        connection.close()


if __name__ == "__main__":
    from data_pipeline import prepare_data

    df = prepare_data("data/tickets.csv")

    create_database(df)

    summary = query_customer_summary()

    print("Customer ticket summary:")
    print(summary)