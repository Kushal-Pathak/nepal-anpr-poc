import re
import unicodedata
from difflib import SequenceMatcher


NEPALI_TO_ASCII = str.maketrans(
    {
        "०": "0",
        "१": "1",
        "२": "2",
        "३": "3",
        "४": "4",
        "५": "5",
        "६": "6",
        "७": "7",
        "८": "8",
        "९": "9",
    }
)


def canonicalize_plate(text):
    if not text:
        return ""

    text = unicodedata.normalize(
        "NFC",
        text,
    )

    # Convert Nepali digits to ASCII.
    text = text.translate(
        NEPALI_TO_ASCII
    )

    # Remove separators/punctuation that do not
    # affect the actual plate identity.
    text = re.sub(
        r"[\s\.\-:ः,;/\\|]+",
        "",
        text,
    )

    return text


def compare_plates(
    front_text,
    rear_text,
):
    front = canonicalize_plate(
        front_text
    )

    rear = canonicalize_plate(
        rear_text
    )

    if not front or not rear:
        return {
            "status": "insufficient_data",
            "match": False,
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
        "status": status,
        "match": match,
        "similarity": similarity,
    }