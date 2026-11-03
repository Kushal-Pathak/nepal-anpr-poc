from src.anpr.ocr import recognize_front_plate


result = recognize_front_plate(
    "outputs/front_plate_candidate_0.jpg"
)


print("Raw:")
print(result["raw_text"])

print()

print("Normalized:")
print(result["normalized_text"])

print()

print(
    "Confidence:",
    round(result["confidence"], 2),
)