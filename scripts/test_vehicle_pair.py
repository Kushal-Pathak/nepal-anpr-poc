from src.anpr.pipeline import recognize_vehicle_pair


front_image = "samples/img_04_front.jpeg"
rear_image = "samples/img_04_back.jpeg"


result = recognize_vehicle_pair(
    front_image,
    rear_image,
)


print()
print("=" * 60)
print("FRONT + REAR ANPR RESULT")
print("=" * 60)


front = result["front"]

print()
print("FRONT")
print("-" * 30)

if front["success"]:
    print("Plate:", front["normalized_text"])
    print(
        "Detection confidence:",
        round(front["detection_confidence"], 2),
    )
    print(
        "OCR confidence:",
        round(front["ocr_confidence"], 2),
    )
else:
    print("Failed:", front["error"])


rear = result["rear"]

print()
print("REAR")
print("-" * 30)

if rear["success"]:
    print("Plate:", rear["normalized_text"])
    print(
        "Detection confidence:",
        round(rear["detection_confidence"], 2),
    )
    print(
        "OCR confidence:",
        round(rear["ocr_confidence"], 2),
    )
else:
    print("Failed:", rear["error"])


if result["match"] is not None:

    match = result["match"]

    print()
    print("COMPARISON")
    print("-" * 30)

    print(
        "Similarity:",
        round(match["similarity"] * 100, 1),
        "%",
    )

    print("Status:", match["status"])