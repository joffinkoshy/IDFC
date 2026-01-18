import re

def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.upper()
    text = re.sub(r'[^A-Z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # common OCR confusions
    text = text.replace('0', 'O')
    text = text.replace('1', 'I')

    return text
