from difflib import SequenceMatcher


def canonicalize_plate(text):
    """
    Remove spaces and punctuation before comparison.
    """

    characters_to_remove = " .:-"

    result = text

    for char in characters_to_remove:
        result = result.replace(char, "")

    return result


def compare_plates(front_text, rear_text):
    front = canonicalize_plate(front_text)
    rear = canonicalize_plate(rear_text)

    if not front or not rear:
        return {
            "match": False,
            "status": "insufficient_data",
            "similarity": 0.0,
        }

    similarity = SequenceMatcher(
        None,
        front,
        rear,
    ).ratio()

    if similarity == 1.0:
        status = "exact_match"
        match = True

    elif similarity >= 0.80:
        status = "possible_match"
        match = True

    else:
        status = "mismatch"
        match = False

    return {
        "match": match,
        "status": status,
        "similarity": similarity,
        "front": front,
        "rear": rear,
    }