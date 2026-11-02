from paddleocr import PaddleOCR

image_path = "outputs/plate_crop.jpeg"

ocr = PaddleOCR(
    lang="ne",
    ocr_version="PP-OCRv5",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

results = ocr.predict(image_path)

for result in results:
    result.print()