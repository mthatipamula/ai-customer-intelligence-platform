from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from data_pipeline import prepare_data


MODEL_PATH = Path("models/ticket_classifier.joblib")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.joblib")


def train_classifier(df):
    """Train and persist a TF-IDF + Logistic Regression classifier."""

    X = df["ticket_text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3333,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print(f"Accuracy: {accuracy:.2f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH,
    )

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")

    return model, vectorizer


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    data_path = (
        project_root
        / "data"
        / "tickets.csv"
    )

    df = prepare_data(
        str(data_path)
    )

    train_classifier(df)