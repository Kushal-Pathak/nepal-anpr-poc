import cv2

image_path = "outputs/plate_pipeline.png"

image = cv2.imread(image_path)

if image is None:
    raise ValueError(f"Could not load image: {image_path}")

# 1. Enlarge the plate
resized = cv2.resize(
    image,
    None,
    fx=3,
    fy=3,
    interpolation=cv2.INTER_CUBIC,
)

cv2.imwrite("outputs/plate_resized.png", resized)

# 2. Convert to grayscale
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

cv2.imwrite("outputs/plate_gray.png", gray)

# 3. Improve contrast
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8),
)

enhanced = clahe.apply(gray)

cv2.imwrite("outputs/plate_enhanced.png", enhanced)

# 4. Create black/white version
_, thresholded = cv2.threshold(
    enhanced,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
)

cv2.imwrite(
    "outputs/plate_thresholded.png",
    thresholded,
)

print("Created preprocessing variants:")
print("- plate_resized.png")
print("- plate_gray.png")
print("- plate_enhanced.png")
print("- plate_thresholded.png")