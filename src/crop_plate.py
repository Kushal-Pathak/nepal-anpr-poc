import cv2

image_path = "samples/vehicle.jpeg"

image = cv2.imread(image_path)

if image is None:
    raise ValueError(f"Could not load image: {image_path}")

height, width, _ = image.shape

print(f"Image size: {width} x {height}")

# Temporary manual coordinates
x1 = 480
y1 = 747
x2 = 585
y2 = 818

plate = image[y1:y2, x1:x2]

if plate.size == 0:
    raise ValueError("Crop coordinates produced an empty image")

cv2.imwrite("outputs/plate_crop.jpeg", plate)

print("Plate crop saved to outputs/plate_crop.jpeg")