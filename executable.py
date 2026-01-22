import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import re
from inference_sdk import InferenceHTTPClient

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,      # GPU friendly
    device_map="auto"
).eval()

INSTRUCTION = """
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
def extract_fields(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")

    # Qwen expects a chat-style message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": INSTRUCTION}
            ]
        }
    ]

    # 🔑 Apply Qwen's chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Processor now correctly aligns image ↔ image tokens
    inputs = processor(
        images=image,
        text=text,
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

def extract_json_text(raw: str) -> str | None:
    if not raw:
        return None

    # Keep only assistant output
    text = raw.split("assistant", 1)[-1]

    # Prefer fenced JSON
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: first {...} block
    match = re.search(r"\{[\s\S]*?\}", text)
    if match:
        return match.group(0).strip()

    return None

def yolo_to_xyxy(pred):
    x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return (x1, y1, x2, y2)


if __name__ == "__main__":
    img_path = "/content/90018777351_OTHERS_v1_pg1.png"
    raw = extract_fields(img_path)
    json_text = extract_json_text(raw)
    print(json_text)

    # initialize the client
    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="Pr1Q5Z4La0EteVXKMsSo"
    )

    # infer on a local image
    result = CLIENT.infer(img_path, model_id="my-first-project-apncv/2")
