import easyocr


reader = easyocr.Reader(
    ["ne"],
    gpu=False,
)

NEPALI_DIGITS = {
    "०": "0",
    "१": "1",
    "२": "2",
    "३": "3",
    "४": "4",
    "५": "5",
    "६": "6",
    "७": "7",
    "८": "8",
    "९": "9",
}


def normalize_digits(text):
    return "".join(
        NEPALI_DIGITS.get(char, char)
        for char in text
    )


def recognize_plate(image_path):
    results = reader.readtext(image_path)

    if not results:
        return {
            "raw_text": "",
            "confidence": 0.0,
        }

    # Sort detected text boxes from top to bottom
    results = sorted(
        results,
        key=lambda item: item[0][0][1],
    )

    texts = []
    confidences = []

    for _, text, confidence in results:
        texts.append(text)
        confidences.append(confidence)

    raw_text = " ".join(texts)

    average_confidence = sum(confidences) / len(confidences)

    normalized_text = normalize_digits(raw_text)

    return {
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "confidence": average_confidence,
    }