import easyocr


image_paths = [
    "outputs/plate_crop.jpeg",
    "outputs/plate_resized.png",
    "outputs/plate_gray.png",
    "outputs/plate_enhanced.png",
    "outputs/plate_thresholded.png",
]


reader = easyocr.Reader(
    ["ne"],
    gpu=False,
)


for image_path in image_paths:

    print("\n")
    print("=" * 60)
    print(f"OCR INPUT: {image_path}")
    print("=" * 60)

    results = reader.readtext(image_path)

    if not results:
        print("No text detected")
        continue

    for bbox, text, confidence in results:
        print(f"{text} ({confidence:.2f})")