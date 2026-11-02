import cv2
from ultralytics import YOLO

model = YOLO("models/nepali_plate_detector.pt")

def detect_plate(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    results = model.predict(
        source=image,
        conf=0.05,
        imgsz=960,
        verbose=False,
    )

    result = results[0]

    print(f"YOLO candidates: {len(result.boxes)}")

    for box in result.boxes:
        print(
            "Candidate:",
            box.xyxy[0].tolist(),
            "confidence:",
            float(box.conf[0]),
        )

    if len(result.boxes) == 0:
        return None

    # Pick the detection with the highest confidence.
    best_box = max(
        result.boxes,
        key=lambda box: float(box.conf[0]),
    )

    x1, y1, x2, y2 = map(
        int,
        best_box.xyxy[0].tolist(),
    )

    confidence = float(best_box.conf[0])

    plate_crop = image[y1:y2, x1:x2]

    return {
        "box": (x1, y1, x2, y2),
        "confidence": confidence,
        "crop": plate_crop,
        "image": image,
    }