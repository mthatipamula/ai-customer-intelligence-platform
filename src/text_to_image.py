from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def generate_image(prompt: str, output_path: str) -> None:
    """Generate an image from a text prompt."""

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Using device: {device}")

    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    pipeline = pipeline.to(device)

    image = pipeline(
        prompt,
        num_inference_steps=20,
        guidance_scale=7.5,
    ).images[0]

    image.save(output_path)

    print(f"\nImage saved to: {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    output_directory = project_root / "outputs"
    output_directory.mkdir(exist_ok=True)

    output_path = output_directory / "hotel_amenities.png"

    prompt = (
        "A luxurious modern hotel lobby with comfortable seating, "
        "warm lighting, indoor plants, elegant architecture, "
        "professional hospitality photography. Show breakfast bar and people of " 
        "different ages and race enjoying their meals."
    )

    generate_image(
        prompt,
        str(output_path),
    )