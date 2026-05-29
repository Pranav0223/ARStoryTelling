import logging
import os
import urllib.parse
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

logger = logging.getLogger(__name__)

SERVICE_ROOT = Path(__file__).resolve().parent


def generate_character_image(prompt: str, output_path: str = None) -> str:
    """
    Generate a character image from a text prompt using Pollinations.ai (Flux model).

    Args:
        prompt:      Text description of the character to render.
        output_path: Optional absolute path where the PNG will be saved.
                     Defaults to SERVICE_ROOT/outputs/character.png.

    Returns:
        Absolute path to the saved PNG file.

    Raises:
        RuntimeError: If the HTTP request fails or returns a non-200 status.
    """
    logger.info("generate_character_image | prompt=%r", prompt[:100])

    encoded_prompt = urllib.parse.quote(prompt)
    url = (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        f"?model=flux&width=1024&height=1792&seed=2&nologo=true"
    )

    logger.info("Fetching image from Pollinations.ai")
    response = requests.get(url, timeout=120)

    if response.status_code != 200:
        logger.error("Image generation failed: HTTP %s", response.status_code)
        raise RuntimeError(f"Image generation failed (HTTP {response.status_code})")

    image = Image.open(BytesIO(response.content))

    if output_path is None:
        out_dir = SERVICE_ROOT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / "character.png")

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)

    logger.info("Image saved: %s", output_path)
    return output_path
