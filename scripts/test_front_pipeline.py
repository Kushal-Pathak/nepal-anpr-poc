from src.anpr.pipeline import recognize_front_vehicle


image_path = "samples/img_02_front.jpeg"

result = recognize_front_vehicle(image_path)


print()
print("=" * 50)
print("FRONT PIPELINE RESULT")
print("=" * 50)

if not result["success"]:
    print("FAILED")
    print("Reason:", result["error"])

    if "detection_confidence" in result:
        print(
            "Detection confidence:",
            round(result["detection_confidence"], 2),
        )

else:
    print(
        "Detection confidence:",
        round(result["detection_confidence"], 2),
    )

    print(
        "OCR confidence:",
        round(result["ocr_confidence"], 2),
    )

    print("Raw plate:")
    print(result["raw_text"])

    print("Normalized:")
    print(result["normalized_text"])