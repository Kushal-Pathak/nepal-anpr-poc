from src.anpr.pipeline import recognize_vehicle


image_paths = [
    "samples/test_01.jpg",
    "samples/test_02.jpg",
    "samples/test_03.jpg",
]


for image_path in image_paths:

    print("\n")
    print("=" * 60)
    print(f"IMAGE: {image_path}")
    print("=" * 60)

    result = recognize_vehicle(image_path)

    if not result["success"]:
        print("FAILED")
        print("Reason:", result["error"])

        if "detection_confidence" in result:
            print(
                "Detection confidence:",
                round(result["detection_confidence"], 2),
            )

        continue

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

    print("Normalized plate:")
    print(result["normalized_text"])