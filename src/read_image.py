import cv2

image_path = "samples/vehicle.jpeg"

image = cv2.imread(image_path)

if image is None:
    raise ValueError(f"Could not load image: {image_path}")

print("Image loaded successfully")

print("Shape:", image.shape)

height, width, channels = image.shape

print("Height:", height)

print("Width:", width)

print("Channels:", channels)

output_path = "outputs/vehicle_copy.jpeg"

cv2.imwrite(output_path, image)

print(f"Image saved to: {output_path}")