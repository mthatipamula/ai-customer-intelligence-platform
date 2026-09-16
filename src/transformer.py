from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

from data_pipeline import prepare_data


MODEL_NAME = "distilbert-base-uncased"


def load_transformer():
    """Load a pretrained DistilBERT tokenizer and model."""

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    model.eval()

    return tokenizer, model


def generate_embeddings(texts, tokenizer, model):
    """Generate transformer embeddings for customer ticket text."""

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state[:, 0, :]

    return embeddings


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "tickets.csv"

    df = prepare_data(str(data_path))

    tokenizer, model = load_transformer()

    sample_texts = df["ticket_text"].head(5).tolist()

    embeddings = generate_embeddings(
        sample_texts,
        tokenizer,
        model,
    )

    print("Transformer model:")
    print(MODEL_NAME)

    print("\nInput tickets:")
    for text in sample_texts:
        print(f"- {text}")

    print("\nEmbedding shape:")
    print(embeddings.shape)