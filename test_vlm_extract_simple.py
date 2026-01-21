import torch
from transformers import AutoModel
from transformers.models.internvl.processing_internvl import InternVLProcessor
from PIL import Image

MODEL_NAME = "OpenGVLab/InternVL2-1B"

# -----------------------------
# Load InternVL Processor (EXPLICIT)
# -----------------------------
processor = InternVLProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

# -----------------------------
# Load InternVL Model
# -----------------------------
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    dtype=torch.float32,      # CPU-safe
    device_map="cpu"
).eval()

PROMPT = """
You are a vision-language document extraction system used for Indian financial documents.

Extract and return ONLY JSON:

{
  "dealer_name": "...",
  "model_name": "...",
  "horse_power": ...,
  "asset_cost": ...
}

Rules:
- Do not guess
- Use null if unclear
"""

def extract_fields(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")

    # Resize for Mac safety
    MAX_WIDTH = 1024
    if image.width > MAX_WIDTH:
        scale = MAX_WIDTH / image.width
        image = image.resize(
            (MAX_WIDTH, int(image.height * scale)),
            Image.BILINEAR
        )

    # ✅ InternVL visual preprocessing (correct)
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"]

    with torch.no_grad():
        response = model.chat(
            processor,
            pixel_values,
            PROMPT,
            generation_config={
                "max_new_tokens": 256,
                "do_sample": False
            }
        )

    return response


if __name__ == "__main__":
    img_path = "/Users/joffinkoshy/Desktop/IDFC/data/train/172655019_3_pg11.png"
    print("Running inference...")
    print(extract_fields(img_path))
