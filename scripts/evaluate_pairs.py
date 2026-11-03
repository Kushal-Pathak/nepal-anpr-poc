import csv
from difflib import SequenceMatcher

from src.anpr.pipeline import (
    recognize_vehicle_pair,
)

from src.anpr.matcher import (
    canonicalize_plate,
)


GROUND_TRUTH_FILE = (
    "evaluation/ground_truth.csv"
)


def calculate_similarity(
    prediction,
    ground_truth,
):
    prediction = canonicalize_plate(
        prediction
    )

    ground_truth = canonicalize_plate(
        ground_truth
    )

    if not prediction or not ground_truth:
        return 0.0

    return SequenceMatcher(
        None,
        prediction,
        ground_truth,
    ).ratio()


def main():
    with open(
        GROUND_TRUTH_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    total = 0

    front_detected = 0
    front_ocr = 0
    front_exact = 0
    front_similarity_total = 0.0

    rear_detected = 0
    rear_ocr = 0
    rear_exact = 0
    rear_similarity_total = 0.0

    pair_available = 0
    pair_matches = 0

    for row in rows:
        sample_id = row["id"]

        ground_truth = row[
            "ground_truth"
        ].strip()

        if not ground_truth:
            continue

        total += 1

        result = recognize_vehicle_pair(
            row["front_image"],
            row["rear_image"],
        )

        expected = canonicalize_plate(
            ground_truth
        )

        print()
        print("=" * 60)
        print(sample_id)
        print("=" * 60)

        print(
            "Ground truth:",
            ground_truth,
        )

        # ---------------------------------------------
        # FRONT
        # ---------------------------------------------

        front = result["front"]

        print()
        print("FRONT")

        # Detection can succeed even if OCR fails.
        if "detection_confidence" in front:
            front_detected += 1

            print(
                "Detected: True"
            )

            print(
                "Detection confidence:",
                round(
                    front[
                        "detection_confidence"
                    ],
                    2,
                ),
            )

        else:
            print(
                "Detected: False"
            )

        if front["success"]:
            front_ocr += 1

            prediction = front[
                "normalized_text"
            ]

            canonical_prediction = (
                canonicalize_plate(
                    prediction
                )
            )

            exact = (
                canonical_prediction
                == expected
            )

            similarity = calculate_similarity(
                prediction,
                ground_truth,
            )

            front_similarity_total += (
                similarity
            )

            if exact:
                front_exact += 1

            print(
                "Prediction:",
                prediction,
            )

            print(
                "OCR confidence:",
                round(
                    front[
                        "ocr_confidence"
                    ],
                    2,
                ),
            )

            print(
                "Ground-truth similarity:",
                round(
                    similarity,
                    2,
                ),
            )

            print(
                "Exact:",
                exact,
            )

        else:
            print(
                "OCR failed:",
                front["error"],
            )

        # ---------------------------------------------
        # REAR
        # ---------------------------------------------

        rear = result["rear"]

        print()
        print("REAR")

        if "detection_confidence" in rear:
            rear_detected += 1

            print(
                "Detected: True"
            )

            print(
                "Detection confidence:",
                round(
                    rear[
                        "detection_confidence"
                    ],
                    2,
                ),
            )

        else:
            print(
                "Detected: False"
            )

        if rear["success"]:
            rear_ocr += 1

            prediction = rear[
                "normalized_text"
            ]

            canonical_prediction = (
                canonicalize_plate(
                    prediction
                )
            )

            exact = (
                canonical_prediction
                == expected
            )

            similarity = calculate_similarity(
                prediction,
                ground_truth,
            )

            rear_similarity_total += (
                similarity
            )

            if exact:
                rear_exact += 1

            print(
                "Prediction:",
                prediction,
            )

            print(
                "OCR confidence:",
                round(
                    rear[
                        "ocr_confidence"
                    ],
                    2,
                ),
            )

            print(
                "Ground-truth similarity:",
                round(
                    similarity,
                    2,
                ),
            )

            print(
                "Exact:",
                exact,
            )

        else:
            print(
                "OCR failed:",
                rear["error"],
            )

        # ---------------------------------------------
        # PAIR
        # ---------------------------------------------

        print()
        print("PAIR")

        match = result["match"]

        if match:
            pair_available += 1

            print(
                "Status:",
                match["status"],
            )

            print(
                "Similarity:",
                round(
                    match["similarity"],
                    2,
                ),
            )

            if match["match"]:
                pair_matches += 1

        else:
            print(
                "Comparison unavailable"
            )

    # ---------------------------------------------
    # SUMMARY
    # ---------------------------------------------

    print()
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(
        "Evaluated pairs:",
        total,
    )

    if total == 0:
        return

    print()
    print(
        "Front detection:",
        f"{front_detected}/{total}",
    )

    print(
        "Front OCR returned text:",
        f"{front_ocr}/{total}",
    )

    print(
        "Front exact OCR:",
        f"{front_exact}/{total}",
    )

    if front_ocr:
        print(
            "Front average similarity:",
            round(
                front_similarity_total
                / front_ocr,
                2,
            ),
        )

    print()
    print(
        "Rear detection:",
        f"{rear_detected}/{total}",
    )

    print(
        "Rear OCR returned text:",
        f"{rear_ocr}/{total}",
    )

    print(
        "Rear exact OCR:",
        f"{rear_exact}/{total}",
    )

    if rear_ocr:
        print(
            "Rear average similarity:",
            round(
                rear_similarity_total
                / rear_ocr,
                2,
            ),
        )

    print()
    print(
        "Pair comparison available:",
        f"{pair_available}/{total}",
    )

    print(
        "Pair possible/exact matches:",
        f"{pair_matches}/{total}",
    )


if __name__ == "__main__":
    main()