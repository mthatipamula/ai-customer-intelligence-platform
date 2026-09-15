from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from data_pipeline import prepare_data


def train_classifier(df):
    """Train a TF-IDF + Logistic Regression ticket classifier."""

    X = df["ticket_text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3333,
        random_state=42,
        stratify=y,
    )

    # Convert text into TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Logistic Regression classifier
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train_tfidf, y_train)

    # Predict categories for unseen test data
    y_pred = model.predict(X_test_tfidf)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model, vectorizer


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "tickets.csv"

    df = prepare_data(str(data_path))

    train_classifier(df)