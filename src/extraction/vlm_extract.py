from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

MODEL_NAME = "openbmb/MiniCPM-V-2_6"



processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

SYSTEM_PROMPT = """You are a vision-language document extraction system used for Indian financial documents.
The document image may contain multiple languages (English, Hindi, Marathi, Gujarati).
Do not guess or infer missing values. Return null if unclear.
"""

TASK_PROMPT = """You are a vision-language model extracting structured information from an Indian tractor quotation or invoice.

The document may be multilingual (English, Hindi, Marathi, Gujarati), may mix printed and handwritten text, and may contain tables.

Extract the following fields based on their visual position, surrounding context, and meaning.

--------------------------------------------------
FIELD CHARACTERISTICS AND LOCATION GUIDANCE
--------------------------------------------------

1. Dealer Name
- Usually located in the top section of the invoice
- Often centered or slightly left-aligned near the top
- Appears in a larger or bolder font than most other text
- Represents the business issuing the quotation
- May appear alongside address or contact details
- Do NOT confuse with:
  - Tractor brand names (e.g., Mahindra, Sonalika, Kubota, Eicher)
  - Bank or financier names (e.g., IDFC FIRST BANK), which usually appear mid-page or near financing terms

2. Model Name
- Refers to the tractor model being quoted for purchase
- Commonly found in the main body of the document
- Often appears:
  - Inside a table under columns like “Particulars”, “Items”, or “Model”
  - In the first or selected row of a table
- Usually an alphanumeric string and may include variant details (e.g., DI, XP, Plus, 2WD, 4WD)
- May be handwritten or printed
- If multiple models are listed, choose the one that is priced or clearly selected

3. Horse Power (HP)
- Numeric value associated with the tractor model
- Usually appears close to the model name
- May be on the same line, in brackets, or in a nearby column
- Can appear under a column such as “HP” or “Capacity”
- Typically written as a number followed by “HP”
- If multiple HP values exist, choose the one corresponding to the quoted tractor
- Return only the numeric value

4. Asset Cost
- Refers to the final total price of the tractor
- Typically the largest numeric amount in the document
- Often located:
  - In the rightmost column of a table
  - Near the bottom section of the invoice
- Commonly associated with labels such as:
  - “Total”
  - “Grand Total”
  - “Net Amount”
  - “After Discount”
  - “Final Amount”
- Ignore individual tax components (CGST, SGST) or accessory prices unless included in the final total
- Return digits only, without commas or currency symbols

--------------------------------------------------
EXTRACTION RULES
--------------------------------------------------
- Extract values only if they are explicitly visible in the image
- Do not infer or guess missing information
- If a field is unclear or ambiguous, return null
- Do not include explanations or additional text

--------------------------------------------------
OUTPUT FORMAT (STRICT)
--------------------------------------------------
Return the result strictly as JSON:

{
  "dealer_name": "...",
  "model_name": "...",
  "horse_power": ...,
  "asset_cost": ...
}
"""

def extract_fields(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        text=SYSTEM_PROMPT + "\n" + TASK_PROMPT,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)

    return processor.decode(output[0], skip_special_tokens=True)

# Example


if __name__ == "__main__":
    img_path = "/Users/joffinkoshy/Desktop/IDFC/data/train/172655019_3_pg11.png"  # change this
    print(extract_fields(img_path))
