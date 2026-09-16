from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from data_pipeline import prepare_data
from neural_network import TicketClassifier


MODEL_NAME = "distilbert-base-uncased"


def evaluate_logistic_regression(df):
    """Evaluate TF-IDF + Logistic Regression."""

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

    return accuracy


def evaluate_neural_network(df):
    """Evaluate the PyTorch neural network."""

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

    X_train_tensor = torch.tensor(
        X_train_tfidf.toarray(),
        dtype=torch.float32,
    )

    X_test_tensor = torch.tensor(
        X_test_tfidf.toarray(),
        dtype=torch.float32,
    )

    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    y_train_tensor = torch.tensor(
        y_train_encoded,
        dtype=torch.long,
    )

    model = TicketClassifier(
        input_size=X_train_tensor.shape[1],
        num_classes=len(label_encoder.classes_),
    )

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    epochs = 50

    for _ in range(epochs):
        model.train()

        optimizer.zero_grad()

        outputs = model(X_train_tensor)

        loss = criterion(
            outputs,
            y_train_tensor,
        )

        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = torch.argmax(
            outputs,
            dim=1,
        )

    accuracy = (
        (predictions == torch.tensor(y_test_encoded))
        .float()
        .mean()
        .item()
    )

    return accuracy


def evaluate_transformer(df):
    """Evaluate the DistilBERT Transformer classifier."""

    texts = df["ticket_text"].tolist()
    labels = df["category"].tolist()

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        encoded_labels,
        test_size=0.3333,
        random_state=42,
        stratify=encoded_labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    train_tokens = tokenizer(
        train_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    test_tokens = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
    )

    batch_size = 16
    epochs = 2

    model.train()

    for _ in range(epochs):

        for start in range(
            0,
            len(train_labels),
            batch_size,
        ):
            end = start + batch_size

            input_ids = train_tokens["input_ids"][start:end]
            attention_mask = train_tokens["attention_mask"][start:end]

            labels_batch = torch.tensor(
                train_labels[start:end],
                dtype=torch.long,
            )

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_batch,
            )

            outputs.loss.backward()
            optimizer.step()

    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids=test_tokens["input_ids"],
            attention_mask=test_tokens["attention_mask"],
        )

        predictions = torch.argmax(
            outputs.logits,
            dim=1,
        ).numpy()

    accuracy = accuracy_score(
        test_labels,
        predictions,
    )

    return accuracy


def evaluate_clustering(df):
    """Evaluate K-Means clustering using silhouette score."""

    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
    )

    X = vectorizer.fit_transform(
        df["ticket_text"]
    )

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10,
    )

    clusters = model.fit_predict(X)

    score = silhouette_score(
        X,
        clusters,
    )

    return score


def create_evaluation_summary(df):
    """Run model evaluations and create a comparison summary."""

    print("Evaluating Logistic Regression...")
    logistic_accuracy = evaluate_logistic_regression(df)

    print("Evaluating PyTorch Neural Network...")
    neural_network_accuracy = evaluate_neural_network(df)

    print("Evaluating DistilBERT Transformer...")
    transformer_accuracy = evaluate_transformer(df)

    print("Evaluating K-Means clustering...")
    clustering_score = evaluate_clustering(df)

    summary = pd.DataFrame(
        [
            {
                "Model": "TF-IDF + Logistic Regression",
                "Task": "Supervised Classification",
                "Metric": "Accuracy",
                "Score": logistic_accuracy,
            },
            {
                "Model": "PyTorch Neural Network",
                "Task": "Supervised Classification",
                "Metric": "Accuracy",
                "Score": neural_network_accuracy,
            },
            {
                "Model": "DistilBERT",
                "Task": "Transformer Classification",
                "Metric": "Accuracy",
                "Score": transformer_accuracy,
            },
            {
                "Model": "K-Means",
                "Task": "Unsupervised Clustering",
                "Metric": "Silhouette Score",
                "Score": clustering_score,
            },
        ]
    )

    print("\nModel Evaluation Summary:")
    print(
        summary.to_string(
            index=False,
            formatters={
                "Score": "{:.4f}".format,
            },
        )
    )

    output_path = (
        Path(__file__).resolve().parent.parent
        / "outputs"
        / "model_evaluation.csv"
    )

    output_path.parent.mkdir(
        exist_ok=True
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nEvaluation report saved to: "
        f"{output_path}"
    )

    return summary


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

    create_evaluation_summary(df)