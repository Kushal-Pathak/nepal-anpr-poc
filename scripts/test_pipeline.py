from src.anpr.pipeline import recognize_vehicle


image_path = "samples/test_02.jpg"

result = recognize_vehicle(image_path)


if not result["success"]:
    print("ANPR failed")
    print(result["error"])
    raise SystemExit()


print()
print("=" * 50)
print("NEPAL ANPR RESULT")
print("=" * 50)

print(
    "Detection confidence:",
    round(result["detection_confidence"], 2),
)

print(
    "OCR confidence:",
    round(result["ocr_confidence"], 2),
)

print()

print("Raw plate:")
print(result["raw_text"])

print()

print("Normalized plate:")
print(result["normalized_text"])

print("=" * 50)