import cv2

from huggingface_hub import hf_hub_download
from ultralytics import YOLO


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

# Generic license-plate detector.
# Works better for our long, thin front plates.
front_weights_path = hf_hub_download(
    repo_id="Koushim/yolov8-license-plate-detection",
    filename="best.pt",
)

front_model = YOLO(front_weights_path)


# Nepal-specific detector.
# Works better for our rear plates.
rear_model = YOLO(
    "models/nepali_plate_detector.pt"
)


# ---------------------------------------------------------
# COMMON DETECTION LOGIC
# ---------------------------------------------------------

def _detect_best_plate(
    image_path,
    model,
    confidence_threshold=0.25,
):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}"
        )

    results = model.predict(
        source=image,
        conf=confidence_threshold,
        imgsz=960,
        verbose=False,
    )

    result = results[0]

    print(f"Candidates: {len(result.boxes)}")

    for box in result.boxes:
        print(
            "Candidate:",
            box.xyxy[0].tolist(),
            "confidence:",
            float(box.conf[0]),
        )

    if len(result.boxes) == 0:
        return None

    # Choose the highest-confidence prediction.
    best_box = max(
        result.boxes,
        key=lambda box: float(box.conf[0]),
    )

    x1, y1, x2, y2 = map(
        int,
        best_box.xyxy[0].tolist(),
    )

    confidence = float(
        best_box.conf[0]
    )

    crop = image[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return None

    return {
        "box": (x1, y1, x2, y2),
        "confidence": confidence,
        "crop": crop,
        "image": image,
    }


# ---------------------------------------------------------
# FRONT DETECTION
# ---------------------------------------------------------

def detect_front_plate(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}"
        )

    results = front_model.predict(
        source=image,
        conf=0.05,
        imgsz=960,
        verbose=False,
    )

    result = results[0]

    if len(result.boxes) == 0:
        return None

    candidates = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist(),
        )

        confidence = float(box.conf[0])

        width = x2 - x1
        height = y2 - y1

        if height <= 0:
            continue

        aspect_ratio = width / height

        # Front plates are usually long and thin.
        if aspect_ratio < 3.0:
            continue

        candidates.append(
            {
                "box": (x1, y1, x2, y2),
                "confidence": confidence,
                "aspect_ratio": aspect_ratio,
            }
        )

    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda item: item["confidence"],
    )

    x1, y1, x2, y2 = best["box"]

    crop = image[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return None

    return {
        "box": best["box"],
        "confidence": best["confidence"],
        "aspect_ratio": best["aspect_ratio"],
        "crop": crop,
        "image": image,
    }

# ---------------------------------------------------------
# REAR DETECTION
# ---------------------------------------------------------

def detect_rear_plate(image_path):
    return _detect_best_plate(
        image_path=image_path,
        model=rear_model,
        confidence_threshold=0.25,
    )