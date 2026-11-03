import cv2
import easyocr


DEVANAGARI_CHARACTERS = "".join(
    chr(code)
    for code in range(0x0900, 0x0980)
)

ALLOWLIST = DEVANAGARI_CHARACTERS + ".:"

reader = easyocr.Reader(
    ["ne"],
    gpu=False,
)


image_path = "outputs/front_plate_candidate_0.jpg"

image = cv2.imread(image_path)

if image is None:
    raise ValueError("Could not load front plate crop")


# Simple resize only.
# We intentionally avoid CLAHE/sharpening/splitting because those
# experiments made recognition worse on our current sample.
image = cv2.resize(
    image,
    None,
    fx=3,
    fy=3,
    interpolation=cv2.INTER_CUBIC,
)


results = reader.readtext(
    image,
    allowlist=ALLOWLIST,
    decoder="beamsearch",
    beamWidth=5,
)


# Front plate is single-line, so sort OCR regions left → right.
results = sorted(
    results,
    key=lambda item: item[0][0][0],
)


print("Detected front regions:")

texts = []
confidences = []

for index, (_, text, confidence) in enumerate(results):
    print(
        index,
        text,
        round(confidence, 2),
    )

    texts.append(text)
    confidences.append(confidence)


merged_text = " ".join(texts)

print()
print("Merged front plate:")
print(merged_text)


if confidences:
    average_confidence = (
        sum(confidences) / len(confidences)
    )

    print(
        "Average confidence:",
        round(average_confidence, 2),
    )