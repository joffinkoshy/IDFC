import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# -----------------------------
# Load processor and model
# -----------------------------
processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,   # GPU-friendly
    device_map="auto"
).eval()

# -----------------------------
# Prompt (your field logic)
# -----------------------------
PROMPT = """
You are a vision-language document extraction system used for Indian financial documents.

The document may be multilingual (English, Hindi, Marathi, Gujarati) and may contain printed and handwritten text.

FIELD CHARACTERISTICS:

1. Dealer Name
- Located near the top of the document
- Larger or bolder than most text
- Issuing business (not bank, not tractor brand)

2. Model Name
- Tractor model being quoted
- Appears in tables or main body
- Alphanumeric (e.g., 575 DI XP Plus)

3. Horse Power
- Numeric value followed by HP
- Appears near model name or under HP/Capacity column
- Return only the number

4. Asset Cost
- Final total amount
- Largest number on the page
- Near bottom or right side
- Ignore taxes unless included in final total
- Return digits only

RULES:
- Do not guess
- If unclear, return null
- Output only JSON

OUTPUT FORMAT:
{
  "dealer_name": "...",
  "model_name": "...",
  "horse_power": ...,
  "asset_cost": ...
}
"""

# -----------------------------
# Extraction function
# -----------------------------
def extract_fields(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        text=PROMPT,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    return processor.decode(output[0], skip_special_tokens=True)

# -----------------------------
# Run example
# -----------------------------
if __name__ == "__main__":
    img_path = "/path/to/invoice.png"
    print("Running Qwen2.5-VL inference...")
    print(extract_fields(img_path))
