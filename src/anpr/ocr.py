import cv2
import easyocr


# ---------------------------------------------------------
# OCR MODEL
# ---------------------------------------------------------

reader = easyocr.Reader(
    ["ne"],
    gpu=False,
)


# ---------------------------------------------------------
# CHARACTER RULES
# ---------------------------------------------------------

NEPALI_DIGIT_CHARS = "०१२३४५६७८९"

DEVANAGARI_CHARACTERS = "".join(
    chr(code)
    for code in range(0x0900, 0x0980)
)

# Top line can contain Nepali letters + digits + dot
TOP_LINE_ALLOWLIST = DEVANAGARI_CHARACTERS + "."

# Bottom registration number should contain digits only
BOTTOM_LINE_ALLOWLIST = NEPALI_DIGIT_CHARS


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


# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------

def normalize_digits(text):
    return "".join(
        NEPALI_DIGITS.get(char, char)
        for char in text
    )


# ---------------------------------------------------------
# READ ONE LINE LEFT → RIGHT
# ---------------------------------------------------------

def read_line(image, allowlist):
    results = reader.readtext(
        image,
        allowlist=allowlist,
        decoder="beamsearch",
        beamWidth=5,
    )

    if not results:
        return "", []

    # Sort detected regions from left to right.
    results = sorted(
        results,
        key=lambda item: item[0][0][0],
    )

    texts = []
    confidences = []

    for _, text, confidence in results:
        texts.append(text)
        confidences.append(confidence)

    return "".join(texts), confidences


# ---------------------------------------------------------
# READ TOP LINE FROM THE FULL PLATE
# ---------------------------------------------------------

def read_top_from_full_plate(image):
    results = reader.readtext(
        image,
        allowlist=TOP_LINE_ALLOWLIST,
        decoder="beamsearch",
        beamWidth=5,
    )

    if not results:
        return "", []

    # Sort detected text regions from top to bottom.
    results = sorted(
        results,
        key=lambda item: item[0][0][1],
    )

    # Keep only the top-most detected region.
    _, text, confidence = results[0]

    return text, [confidence]


# ---------------------------------------------------------
# COMPLETE PLATE RECOGNITION
# ---------------------------------------------------------

def recognize_plate(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not load plate: {image_path}"
        )

    height, _, _ = image.shape

    # We use the full plate for the top line.
    top_text, top_scores = read_top_from_full_plate(
        image
    )

    # For the registration number, inspect the lower 60%.
    # The large region prevents us from cutting digits.
    bottom_start = int(height * 0.40)

    bottom_line = image[bottom_start:, :]

    # Save for debugging.
    cv2.imwrite(
        "outputs/ocr_bottom_line.png",
        bottom_line,
    )

    # Bottom line is digits-only.
    bottom_text, bottom_scores = read_line(
        bottom_line,
        BOTTOM_LINE_ALLOWLIST,
    )

    raw_text = f"{top_text} {bottom_text}".strip()

    normalized_text = normalize_digits(
        raw_text
    )

    all_scores = top_scores + bottom_scores

    if all_scores:
        confidence = sum(all_scores) / len(all_scores)
    else:
        confidence = 0.0

    return {
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "confidence": confidence,
        "top_text": top_text,
        "bottom_text": bottom_text,
    }