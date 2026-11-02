import cv2
import pytesseract


image_paths = [
    "outputs/plate_crop.jpeg",
    "outputs/plate_resized.png",
    "outputs/plate_gray.png",
    "outputs/plate_enhanced.png",
    "outputs/plate_thresholded.png",
]


for image_path in image_paths:

    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read: {image_path}")
        continue

    print("\n")
    print("=" * 60)
    print(f"OCR INPUT: {image_path}")
    print("=" * 60)

    text = pytesseract.image_to_string(
        image,
        lang="nep",
        config="--psm 6",
    )

    print(text.strip())