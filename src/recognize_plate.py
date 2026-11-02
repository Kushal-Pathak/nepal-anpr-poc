from paddleocr import PaddleOCR

image_paths = [
    "outputs/plate_crop.jpeg",
    "outputs/plate_resized.png",
    "outputs/plate_gray.png",
    "outputs/plate_enhanced.png",
    "outputs/plate_thresholded.png",
]

ocr = PaddleOCR(
    lang="ne",
    ocr_version="PP-OCRv5",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

for image_path in image_paths:

    print("\n")
    print("=" * 60)
    print(f"OCR INPUT: {image_path}")
    print("=" * 60)

    results = ocr.predict(image_path)

    for result in results:
        data = result.json["res"]

        texts = data["rec_texts"]
        scores = data["rec_scores"]

        print("Detected text:")

        for text, score in zip(texts, scores):
            print(f"  {text}  ({score:.2f})")