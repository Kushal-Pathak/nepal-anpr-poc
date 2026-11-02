from ocr import recognize_plate


result = recognize_plate(
    "outputs/plate_resized.png"
)

print("Raw OCR:")
print(result["raw_text"])

print()

print("Normalized:")
print(result["normalized_text"])

print()

print(
    "Confidence:",
    round(result["confidence"], 2),
)