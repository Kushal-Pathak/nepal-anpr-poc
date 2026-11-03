import cv2

from src.anpr.detector import detect_plate


image_path = "samples/vehicle.jpg"

result = detect_plate(image_path)

if result is None:
    print("No license plate detected.")
    raise SystemExit()


x1, y1, x2, y2 = result["box"]

confidence = result["confidence"]

print("Plate detected!")
print("Bounding box:", result["box"])
print("Confidence:", round(confidence, 2))


# Save automatically cropped plate
cv2.imwrite(
    "outputs/plate_auto_crop.jpeg",
    result["crop"],
)


# Draw rectangle on original image
image_with_box = result["image"].copy()

cv2.rectangle(
    image_with_box,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    3,
)

cv2.imwrite(
    "outputs/vehicle_detected.jpeg",
    image_with_box,
)

print("Saved:")
print("- outputs/plate_auto_crop.jpeg")
print("- outputs/vehicle_detected.jpeg")