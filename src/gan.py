import numpy as np
import torch
import torch.nn as nn

from data_pipeline import prepare_data


LATENT_SIZE = 5
HIDDEN_SIZE = 32
EPOCHS = 1000
LEARNING_RATE = 0.0002


class Generator(nn.Module):
    """Generate synthetic customer-spend values."""

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(LATENT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        return self.network(x)


class Discriminator(nn.Module):
    """Distinguish real customer-spend values from generated values."""

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def train_gan(df):
    """Train a GAN to generate synthetic customer-spend values."""

    real_values = df["customer_spend"].to_numpy(dtype=np.float32)

    mean = real_values.mean()
    std = real_values.std()

    normalized_values = (
        (real_values - mean) / std
    )

    real_data = torch.tensor(
        normalized_values,
        dtype=torch.float32,
    ).view(-1, 1)

    generator = Generator()
    discriminator = Discriminator()

    criterion = nn.BCELoss()

    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
    )

    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
    )

    for epoch in range(EPOCHS):

        # -------------------------
        # Train Discriminator
        # -------------------------

        discriminator_optimizer.zero_grad()

        real_labels = torch.ones(
            real_data.size(0),
            1,
        )

        real_predictions = discriminator(real_data)

        real_loss = criterion(
            real_predictions,
            real_labels,
        )

        noise = torch.randn(
            real_data.size(0),
            LATENT_SIZE,
        )

        fake_data = generator(noise)

        fake_labels = torch.zeros(
            real_data.size(0),
            1,
        )

        fake_predictions = discriminator(
            fake_data.detach()
        )

        fake_loss = criterion(
            fake_predictions,
            fake_labels,
        )

        discriminator_loss = (
            real_loss + fake_loss
        )

        discriminator_loss.backward()
        discriminator_optimizer.step()

        # -------------------------
        # Train Generator
        # -------------------------

        generator_optimizer.zero_grad()

        noise = torch.randn(
            real_data.size(0),
            LATENT_SIZE,
        )

        generated_data = generator(noise)

        predictions = discriminator(
            generated_data
        )

        generator_loss = criterion(
            predictions,
            real_labels,
        )

        generator_loss.backward()
        generator_optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"- Discriminator Loss: "
                f"{discriminator_loss.item():.4f} "
                f"- Generator Loss: "
                f"{generator_loss.item():.4f}"
            )

    return generator, mean, std


def generate_synthetic_data(
    generator,
    mean,
    std,
    count=10,
):
    """Generate synthetic customer-spend values."""

    noise = torch.randn(
        count,
        LATENT_SIZE,
    )

    with torch.no_grad():
        generated_values = generator(noise).numpy().flatten()

    generated_values = (
        generated_values * std
    ) + mean

    generated_values = np.maximum(
        generated_values,
        0,
    )

    return generated_values


if __name__ == "__main__":
    df = prepare_data("data/tickets.csv")

    generator, mean, std = train_gan(df)

    synthetic_values = generate_synthetic_data(
        generator,
        mean,
        std,
        count=10,
    )

    print("\nSynthetic customer-spend values:")

    for value in synthetic_values:
        print(f"${value:.2f}")