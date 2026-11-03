import cv2

from .detector import detect_plate
from .ocr import recognize_plate


def recognize_vehicle(image_path):

    # Step 1: detect plate
    detection = detect_plate(image_path)

    if detection is None:
        return {
            "success": False,
            "error": "No license plate detected",
        }

    plate_crop = detection["crop"]

    # Step 2: enlarge crop for OCR
    plate_resized = cv2.resize(
        plate_crop,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC,
    )

    plate_path = "outputs/plate_pipeline.png"

    cv2.imwrite(
        plate_path,
        plate_resized,
    )

    # Step 3: OCR
    ocr_result = recognize_plate(
        plate_path
    )

    if not ocr_result["raw_text"]:
        return {
            "success": False,
            "error": "Plate detected, but OCR could not recognize any text",
            "box": detection["box"],
            "detection_confidence": detection["confidence"],
        }

    return {
        "success": True,
        "box": detection["box"],
        "detection_confidence": detection["confidence"],
        "raw_text": ocr_result["raw_text"],
        "normalized_text": ocr_result["normalized_text"],
        "ocr_confidence": ocr_result["confidence"],
    }