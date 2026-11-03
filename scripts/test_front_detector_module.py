import cv2

from src.anpr.detector import detect_front_plate


image_path = "samples/img_02_front.jpeg"

result = detect_front_plate(
    image_path
)


if result is None:
    print("No front plate detected.")
    raise SystemExit()


print("Front plate detected!")

print(
    "Bounding box:",
    result["box"],
)

print(
    "Confidence:",
    round(
        result["confidence"],
        2,
    ),
)


cv2.imwrite(
    "outputs/front_plate_crop.jpg",
    result["crop"],
)


x1, y1, x2, y2 = result["box"]

preview = result["image"].copy()

cv2.rectangle(
    preview,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    3,
)

cv2.imwrite(
    "outputs/front_vehicle_detected.jpg",
    preview,
)


print("Saved:")
print("- outputs/front_plate_crop.jpg")
print("- outputs/front_vehicle_detected.jpg")