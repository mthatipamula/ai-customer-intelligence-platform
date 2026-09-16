from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_pipeline import prepare_data


class TicketClassifier(nn.Module):
    """Simple feed-forward neural network for ticket classification."""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def train_neural_network(df):
    """Train a neural network using TF-IDF features."""

    X = df["ticket_text"]
    y = df["category"]

    # Split into training and test sets
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

    # Convert sparse matrices to PyTorch tensors
    X_train_tensor = torch.tensor(
        X_train_tfidf.toarray(),
        dtype=torch.float32,
    )

    X_test_tensor = torch.tensor(
        X_test_tfidf.toarray(),
        dtype=torch.float32,
    )

    # Convert category names into numeric labels
    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    y_train_tensor = torch.tensor(
        y_train_encoded,
        dtype=torch.long,
    )

    y_test_tensor = torch.tensor(
        y_test_encoded,
        dtype=torch.long,
    )

    # Create neural network
    model = TicketClassifier(
        input_size=X_train_tensor.shape[1],
        num_classes=len(label_encoder.classes_),
    )

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    # Training loop
    epochs = 50

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()

        outputs = model(X_train_tensor)

        loss = criterion(
            outputs,
            y_train_tensor,
        )

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- Loss: {loss.item():.4f}"
            )

    # Evaluate model
    model.eval()

    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = torch.argmax(outputs, dim=1)

    accuracy = (
        (predictions == y_test_tensor)
        .float()
        .mean()
        .item()
    )

    print(f"\nTest Accuracy: {accuracy:.2f}")

    print("\nCategories:")
    print(label_encoder.classes_)

    return model, vectorizer, label_encoder


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "tickets.csv"

    df = prepare_data(str(data_path))

    train_neural_network(df)