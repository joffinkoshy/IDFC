import torch
import time
import json
import re
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from inference_sdk import InferenceHTTPClient

# =========================
# CONFIG
# =========================
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
ROBOFLOW_MODEL_ID = "my-first-project-apncv/2"
ROBOFLOW_API_KEY = "Pr1Q5Z4La0EteVXKMsSo"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODELS
# =========================
processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
).eval()

rf_client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

# =========================
# PROMPT
# =========================
INSTRUCTION = """
You are a vision-language document extraction system used for Indian financial documents.

The document may be multilingual (English, Hindi, Marathi, Gujarati) and may contain printed and handwritten text.

Extract ONLY the following fields and return ONLY valid JSON.

Dealer Name:
- Issuing business
- Near top of document
- Not bank or tractor brand

Model Name:
- Tractor model
- Alphanumeric

Horse Power:
- Numeric HP value
- Return number only

Asset Cost:
- Final total amount
- Ignore individual taxes
- Return digits only

Output format:
{
  "dealer_name": "...",
  "model_name": "...",
  "horse_power": ...,
  "asset_cost": ...
}
"""

# =========================
# VLM FUNCTIONS
# =========================
def extract_fields(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": INSTRUCTION}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

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
    text = raw.split("assistant", 1)[-1]
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"\{[\s\S]*?\}", text)
    return match.group(0).strip() if match else None


def normalize_vlm_json(json_text: str) -> dict:
    data = json.loads(json_text)

    # HP normalization
    hp = data.get("horse_power")
    if isinstance(hp, str):
        nums = re.findall(r"\d+", hp)
        data["horse_power"] = int(nums[0]) if nums else None

    # Cost normalization
    cost = data.get("asset_cost")
    if isinstance(cost, str):
        cost = re.sub(r"[^\d]", "", cost)
        data["asset_cost"] = int(cost) if cost else None

    return data

# =========================
# YOLO / IOU FUNCTIONS
# =========================
def yolo_to_xyxy(pred):
    x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
    return (
        int(x - w / 2),
        int(y - h / 2),
        int(x + w / 2),
        int(y + h / 2),
    )


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter)


def extract_stamp_signature(result, gt_stamp_box=None, gt_signature_box=None, iou_thresh=0.5):
    stamp = {"present": False, "bbox": None}
    signature = {"present": False, "bbox": None}

    for pred in result.get("predictions", []):
        bbox = yolo_to_xyxy(pred)

        if pred["class"] == "Stamp":
            if gt_stamp_box is None or compute_iou(bbox, gt_stamp_box) > iou_thresh:
                stamp["present"] = True
                stamp["bbox"] = list(bbox)

        elif pred["class"] == "Signature":
            if gt_signature_box is None or compute_iou(bbox, gt_signature_box) > iou_thresh:
                signature["present"] = True
                signature["bbox"] = list(bbox)

    return stamp, signature

# =========================
# MAIN PIPELINE
# =========================
if __name__ == "__main__":
    start_time = time.time()
    img_path = "/content/172679320_3_pg18.png"

    # -------- VLM --------
    raw = extract_fields(img_path)
    json_text = extract_json_text(raw)
    vlm_fields = normalize_vlm_json(json_text)

    # -------- YOLO --------
    rf_result = rf_client.infer(img_path, model_id=ROBOFLOW_MODEL_ID)
    stamp, signature = extract_stamp_signature(rf_result)

    # -------- FINAL OUTPUT --------
    final_output = {
        "doc_id":img_path,
        "fields": {
            "dealer_name": vlm_fields.get("dealer_name"),
            "model_name": vlm_fields.get("model_name"),
            "horse_power": vlm_fields.get("horse_power"),
            "asset_cost": vlm_fields.get("asset_cost"),
            "signature": signature,
            "stamp": stamp
        },
        "confidence": round(
            sum(p["confidence"] for p in rf_result.get("predictions", [])) /
            max(len(rf_result.get("predictions", [])), 1),
            2
        ),
        "processing_time_sec": round(time.time() - start_time, 2),
        "cost_estimate_usd": 0.002
    }

    print(json.dumps(final_output, indent=2, ensure_ascii=False))
