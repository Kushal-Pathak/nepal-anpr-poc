import cv2

from huggingface_hub import hf_hub_download
from ultralytics import YOLO


weights_path = hf_hub_download(
    repo_id="Koushim/yolov8-license-plate-detection",
    filename="best.pt",
)

model = YOLO(weights_path)


image_path = "samples/img_01_front.jpeg"

image = cv2.imread(image_path)

if image is None:
    raise ValueError("Could not load front image")


results = model.predict(
    source=image,
    conf=0.05,
    imgsz=960,
    verbose=False,
)

result = results[0]

print(
    "Candidates:",
    len(result.boxes),
)


for index, box in enumerate(result.boxes):
    confidence = float(box.conf[0])

    coordinates = box.xyxy[0].tolist()

    print(
        "Candidate:",
        coordinates,
        "confidence:",
        confidence,
    )

    x1, y1, x2, y2 = map(
        int,
        coordinates,
    )

    crop = image[y1:y2, x1:x2]

    cv2.imwrite(
        f"outputs/front_plate_candidate_{index}.jpg",
        crop,
    )