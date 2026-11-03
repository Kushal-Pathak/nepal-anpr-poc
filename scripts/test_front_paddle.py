from paddleocr import PaddleOCR


image_path = "outputs/front_plate_pipeline.png"


ocr = PaddleOCR(
    lang="ne",
    ocr_version="PP-OCRv5",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


results = ocr.predict(
    image_path
)


print()
print("=" * 50)
print("PADDLE OCR — FRONT PLATE")
print("=" * 50)


for result in results:
    data = result.json["res"]

    texts = data["rec_texts"]
    scores = data["rec_scores"]

    for text, score in zip(
        texts,
        scores,
    ):
        print(
            text,
            round(float(score), 2),
        )