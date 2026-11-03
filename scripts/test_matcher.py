from src.anpr.matcher import compare_plates


test_cases = [
    (
        "बा.65प 1785",
        "बा.65प 1785",
    ),
    (
        "बा.65प 1785",
        "बा.65प 1782",
    ),
    (
        "बा.65प 1785",
        "बा.86प 8328",
    ),
]


for front, rear in test_cases:

    result = compare_plates(
        front,
        rear,
    )

    print()
    print("=" * 50)

    print("Front:", front)
    print("Rear:", rear)

    print(
        "Similarity:",
        round(result["similarity"] * 100, 1),
        "%",
    )

    print("Status:", result["status"])