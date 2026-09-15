from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from data_pipeline import prepare_data


def cluster_customers(df):
    """Group similar customer tickets using K-Means clustering."""

    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
    )

    X = vectorizer.fit_transform(df["ticket_text"])

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10,
    )

    clusters = model.fit_predict(X)

    result = df[
        ["ticket_id", "customer_id", "ticket_text", "category"]
    ].copy()

    result["cluster"] = clusters

    return result, model, vectorizer


def interpret_clusters(result):
    """Show the category distribution within each cluster."""

    cluster_summary = (
        result
        .groupby(["cluster", "category"])
        .size()
        .reset_index(name="ticket_count")
        .sort_values(
            ["cluster", "ticket_count"],
            ascending=[True, False],
        )
    )

    print("\nCluster interpretation:")
    print(cluster_summary.to_string(index=False))

    return cluster_summary


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "tickets.csv"

    df = prepare_data(str(data_path))

    result, model, vectorizer = cluster_customers(df)

    print("Customer ticket clusters:")
    print(
        result[
            ["ticket_id", "category", "cluster"]
        ].to_string(index=False)
    )

    print("\nCluster distribution:")
    print(
        result["cluster"]
        .value_counts()
        .sort_index()
    )

    interpret_clusters(result)