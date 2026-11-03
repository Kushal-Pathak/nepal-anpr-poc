# Nepal ANPR POC

A proof-of-concept **Automatic Number Plate Recognition (ANPR)** system for Nepali vehicles.

The project detects front and rear vehicle number plates, extracts plate text using OCR, normalizes Nepali plate text, and compares front/rear predictions. It also provides a Streamlit web interface for testing the system with vehicle images.

## Live Demo

**Streamlit:** https://nepal-anpr-poc.streamlit.app/

> This is a proof of concept. Plate detection is currently more reliable than OCR, so predictions should be manually verified.

---

## Problem

Automatic Number Plate Recognition is widely used for parking systems, traffic monitoring, access control, toll systems, and vehicle verification.

Nepali vehicle plates create additional challenges because:

- plates contain **Devanagari characters and Nepali digits**;
- front and rear plates can have different layouts;
- front plates are often long and single-line;
- rear plates can be taller and use two lines;
- glare, low resolution, angle, and plate condition affect OCR;
- general-purpose OCR models are not specifically trained for Nepali number plates.

This project explores a practical ANPR pipeline using pretrained computer-vision and OCR models instead of training a full model from scratch.

---

## System Architecture

```text
                    Vehicle Image Pair
                           |
              +------------+------------+
              |                         |
           FRONT                      REAR
              |                         |
              v                         v
     Generic YOLO Detector     Nepal-specific YOLO Detector
              |                         |
              v                         v
       Front Plate Crop          Rear Plate Crop
              |                         |
              v                         v
      Single-line EasyOCR       Structured EasyOCR
              |                         |
              +------------+------------+
                           |
                           v
                 Plate Normalization
                           |
                           v
                 Front/Rear Matching
                           |
                           v
                    Streamlit UI
```

---

## How It Works

### 1. Front Plate Detection

Front Nepali plates in the tested images are generally long and thin.

A generic YOLO license-plate detector is used with a low confidence threshold. Geometry filtering is then applied to prefer wide plate-shaped detections.

```text
Vehicle image
    ↓
YOLO candidates
    ↓
Aspect-ratio filtering
    ↓
Best front plate crop
```

### 2. Rear Plate Detection

Rear plates are detected using a Nepal-specific YOLO model, which performed better than the generic detector on the tested rear images.

```text
Rear vehicle image
    ↓
Nepal-specific YOLO
    ↓
Best rear plate crop
```

### 3. OCR

The project uses **EasyOCR with Nepali language support**.

Front plates are processed as a mostly single-line layout.

Rear plates use a structured OCR strategy because many rear plates contain two lines:

- OCR the top plate region;
- OCR the lower region separately;
- restrict the lower OCR region mainly to Nepali digits;
- merge the recognized regions.

### 4. Plate Normalization

Nepali digits are converted to ASCII digits for easier comparison.

Example:

```text
ना ३४ प ३१६
        ↓
ना34प316
```

Spaces and common punctuation are removed during matching.

### 5. Front/Rear Matching

The front and rear OCR predictions are compared using string similarity.

Results are classified as:

- `exact_match`
- `possible_match`
- `mismatch`
- `insufficient_data`

This matching step is experimental and is mainly useful as an additional verification signal.

---

## Evaluation

The current POC was evaluated on **4 paired front/rear vehicle samples** with manually recorded ground truth.

| Metric | Result |
|---|---:|
| Front plate detection | 4 / 4 |
| Rear plate detection | 4 / 4 |
| Front OCR returned text | 3 / 4 |
| Rear OCR returned text | 4 / 4 |
| Front exact OCR | 0 / 4 |
| Rear exact OCR | 0 / 4 |
| Front average text similarity | 0.83 |
| Rear average text similarity | 0.77 |
| Pair comparison available | 3 / 4 |
| Possible/exact front-rear matches | 1 / 4 |

### Interpretation

The strongest part of the current system is **plate localization**.

Both front and rear detection succeeded on all four evaluation pairs.

OCR is still the main limitation. Several OCR outputs were close to the correct plate but contained character substitutions, missing digits, or additional characters.

Because the evaluation dataset contains only four pairs, these numbers should be treated as **POC results**, not production accuracy estimates.

---

## Example OCR Behavior

Ground truth:

```text
ना ३४ प ३१६
```

Front OCR prediction:

```text
ना.38 प316
```

Rear OCR prediction:

```text
ना3प 316
```

The predictions are close to the correct plate, but small OCR errors prevent exact matching.

---

## Tech Stack

- **Python 3.11**
- **YOLO / Ultralytics**
- **OpenCV**
- **EasyOCR**
- **PyTorch**
- **Hugging Face Hub**
- **Streamlit**

---

## Project Structure

```text
nepal-anpr-poc/
├── app.py
├── requirements.txt
├── models/
│   └── nepali_plate_detector.pt
├── samples/
├── outputs/
├── evaluation/
│   └── ground_truth.csv
├── src/
│   ├── __init__.py
│   └── anpr/
│       ├── __init__.py
│       ├── detector.py
│       ├── ocr.py
│       ├── matcher.py
│       └── pipeline.py
├── scripts/
│   ├── evaluate_pairs.py
│   ├── evaluate_samples.py
│   ├── test_front_detector_module.py
│   ├── test_front_pipeline.py
│   ├── test_matcher.py
│   └── test_vehicle_pair.py
└── experiments/
    └── ...
```

---

## Local Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd nepal-anpr-poc
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Streamlit app

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## Run the Evaluation

The evaluation ground truth is stored in:

```text
evaluation/ground_truth.csv
```

Run:

```bash
python -m scripts.evaluate_pairs
```

The script reports:

- front/rear detection success;
- OCR availability;
- exact OCR matches;
- similarity against ground truth;
- front/rear prediction similarity.

---

## Experiments Performed

During development, multiple approaches were tested.

### Detection

- generic YOLO license-plate detector;
- Nepal-specific YOLO detector;
- different confidence thresholds;
- geometry filtering for long front plates.

### OCR

- EasyOCR;
- PaddleOCR;
- Tesseract;
- resizing;
- character allowlists;
- beam-search decoding;
- separate OCR regions for two-line rear plates;
- Nepali digit normalization.

EasyOCR provided the most practical baseline for the final POC pipeline.

---

## Limitations

The current system is **not production ready**.

Main limitations are:

1. **Small evaluation dataset**  
   Only four paired vehicles were used for the final evaluation.

2. **OCR errors**  
   Similar Devanagari characters and digits are sometimes confused.

3. **Front OCR weakness**  
   Small or low-resolution front plates can be difficult to recognize.

4. **Lighting and glare**  
   Strong sunlight or reflection can hide character strokes.

5. **Plate variation**  
   Different fonts, plate layouts, sizes, and vehicle categories may require additional handling.

6. **Pretrained models**  
   The models were not trained specifically on a large, representative Nepali ANPR dataset.

7. **Pair matching is experimental**  
   Front/rear similarity is useful as a supporting signal but should not be treated as vehicle identity verification.

---

## Future Improvements

Possible next steps include:

- collect a larger labeled Nepali plate dataset;
- fine-tune a detector specifically for Nepali front and rear plates;
- train or fine-tune OCR for Nepali number-plate characters;
- add character-level segmentation;
- improve image enhancement for glare and low-resolution plates;
- calculate Character Error Rate (CER);
- evaluate performance across more vehicle categories;
- add video/frame-based ANPR;
- add database lookup and vehicle-entry workflows.

---

## Project Status

**Proof of Concept**

The current implementation demonstrates the complete ANPR workflow:

```text
Image
→ Plate Detection
→ Plate Crop
→ OCR
→ Text Normalization
→ Front/Rear Comparison
→ Web Demo
```

The project successfully demonstrates reliable plate detection on the small test set while clearly identifying OCR as the primary area for further improvement.
