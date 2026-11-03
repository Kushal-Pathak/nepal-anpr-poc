import cv2

from .detector import (
    detect_front_plate,
    detect_rear_plate,
)

from .ocr import (
    recognize_front_plate,
    recognize_rear_plate,
)

from .matcher import compare_plates


# ---------------------------------------------------------
# FRONT VEHICLE PIPELINE
# ---------------------------------------------------------

def recognize_front_vehicle(image_path):
    detection = detect_front_plate(image_path)

    if detection is None:
        return {
            "success": False,
            "side": "front",
            "error": "No front license plate detected",
        }

    plate_crop = detection["crop"]

    plate_path = "outputs/front_plate_pipeline.png"

    cv2.imwrite(
        plate_path,
        plate_crop,
    )

    ocr_result = recognize_front_plate(
        plate_path
    )

    if not ocr_result["raw_text"]:
        return {
            "success": False,
            "side": "front",
            "error": "Front plate detected, but OCR could not recognize text",
            "box": detection["box"],
            "detection_confidence": detection["confidence"],
        }

    return {
        "success": True,
        "side": "front",
        "box": detection["box"],
        "detection_confidence": detection["confidence"],
        "raw_text": ocr_result["raw_text"],
        "normalized_text": ocr_result["normalized_text"],
        "ocr_confidence": ocr_result["confidence"],
    }


# ---------------------------------------------------------
# REAR VEHICLE PIPELINE
# ---------------------------------------------------------

def recognize_rear_vehicle(image_path):
    detection = detect_rear_plate(image_path)

    if detection is None:
        return {
            "success": False,
            "side": "rear",
            "error": "No rear license plate detected",
        }

    plate_crop = detection["crop"]

    # Rear OCR benefits from enlargement.
    plate_resized = cv2.resize(
        plate_crop,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC,
    )

    plate_path = "outputs/rear_plate_pipeline.png"

    cv2.imwrite(
        plate_path,
        plate_resized,
    )

    ocr_result = recognize_rear_plate(
        plate_path
    )

    if not ocr_result["raw_text"]:
        return {
            "success": False,
            "side": "rear",
            "error": "Rear plate detected, but OCR could not recognize text",
            "box": detection["box"],
            "detection_confidence": detection["confidence"],
        }

    return {
        "success": True,
        "side": "rear",
        "box": detection["box"],
        "detection_confidence": detection["confidence"],
        "raw_text": ocr_result["raw_text"],
        "normalized_text": ocr_result["normalized_text"],
        "ocr_confidence": ocr_result["confidence"],
    }


# ---------------------------------------------------------
# FRONT + REAR PIPELINE
# ---------------------------------------------------------

def recognize_vehicle_pair(
    front_image_path,
    rear_image_path,
):
    front_result = recognize_front_vehicle(
        front_image_path
    )

    rear_result = recognize_rear_vehicle(
        rear_image_path
    )

    front_ok = front_result["success"]
    rear_ok = rear_result["success"]

    # Both sides successfully recognized.
    if front_ok and rear_ok:
        match_result = compare_plates(
            front_result["normalized_text"],
            rear_result["normalized_text"],
        )

        return {
            "success": True,
            "strategy": "front_and_rear",
            "front": front_result,
            "rear": rear_result,
            "match": match_result,
        }

    # Only front succeeded.
    if front_ok:
        return {
            "success": True,
            "strategy": "front_only",
            "front": front_result,
            "rear": rear_result,
            "match": None,
        }

    # Only rear succeeded.
    if rear_ok:
        return {
            "success": True,
            "strategy": "rear_only",
            "front": front_result,
            "rear": rear_result,
            "match": None,
        }

    # Neither side succeeded.
    return {
        "success": False,
        "strategy": "failed",
        "front": front_result,
        "rear": rear_result,
        "match": None,
        "error": "Could not recognize a plate from either image",
    }


# ---------------------------------------------------------
# BACKWARD-COMPATIBILITY
# ---------------------------------------------------------

def recognize_vehicle(image_path):
    """
    Existing evaluation scripts treat their images as rear images.
    Keep this alias so those scripts continue working.
    """
    return recognize_rear_vehicle(image_path)