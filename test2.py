import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import json
import os

# Use the same model as vlm_extract.py for compatibility
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# Load processor
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Load model (Mac CPU safe)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
).eval()

PROMPT = """
You are a vision-language document extraction system used for Indian financial documents.

The document may be multilingual (English, Hindi, Marathi, Gujarati).

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
    try:
        # Validate image path
        if not os.path.exists(image_path):
            return json.dumps({"error": f"Image file not found: {image_path}"})

        # Load and validate image
        image = Image.open(image_path).convert("RGB")

        # Resize for Mac safety
        MAX_WIDTH = 1024
        if image.width > MAX_WIDTH:
            scale = MAX_WIDTH / image.width
            image = image.resize(
                (MAX_WIDTH, int(image.height * scale)),
                Image.BILINEAR
            )

        # Process image through processor
        inputs = processor(
            images=image,
            text=PROMPT,
            return_tensors="pt"
        ).to(model.device)

        # Generate response using standard vision-language approach
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256)
            response = processor.decode(output[0], skip_special_tokens=True)

        # Validate and format response
        if not response or not isinstance(response, str):
            return json.dumps({"error": "Invalid model response"})

        # Try to parse as JSON, if not already JSON, wrap in JSON
        try:
            parsed = json.loads(response)
            if not all(key in parsed for key in ["dealer_name", "model_name", "horse_power", "asset_cost"]):
                return json.dumps({"error": "Invalid response format", "raw_response": response})
            return response
        except json.JSONDecodeError:
            # If response is not JSON, wrap it in a standard format
            return json.dumps({
                "dealer_name": None,
                "model_name": None,
                "horse_power": None,
                "asset_cost": None,
                "raw_response": response
            })

    except Exception as e:
        return json.dumps({
            "error": f"Extraction failed: {str(e)}",
            "image_path": image_path
        })


if __name__ == "__main__":
    img_path = "/Users/joffinkoshy/Desktop/IDFC/data/train/172655019_3_pg11.png"
    print("Running inference...")
    print(extract_fields(img_path))
