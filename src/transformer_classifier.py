from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from data_pipeline import prepare_data


MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5


def train_transformer_classifier(df):
    """Fine-tune DistilBERT for customer ticket classification."""

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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

    train_dataset = TensorDataset(
        train_tokens["input_ids"],
        train_tokens["attention_mask"],
        torch.tensor(train_labels, dtype=torch.long),
    )

    test_dataset = TensorDataset(
        test_tokens["input_ids"],
        test_tokens["attention_mask"],
        torch.tensor(test_labels, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for input_ids, attention_mask, labels_batch in train_loader:

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_batch,
            )

            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {average_loss:.4f}"
        )

    model.eval()

    predictions = []

    with torch.no_grad():
        for input_ids, attention_mask, _ in DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
        ):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            batch_predictions = torch.argmax(
                outputs.logits,
                dim=1,
            )

            predictions.extend(
                batch_predictions.tolist()
            )

    accuracy = accuracy_score(
        test_labels,
        predictions,
    )

    print(f"\nTest Accuracy: {accuracy:.2f}")

    print("\nClassification Report:")
    print(
        classification_report(
            test_labels,
            predictions,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    return model, tokenizer, label_encoder


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "tickets.csv"

    df = prepare_data(str(data_path))

    train_transformer_classifier(df)