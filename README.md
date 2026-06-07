# OCR Pipeline -- Image Text Recognition

A Python-based OCR (Optical Character Recognition) pipeline that extracts text from images using Tesseract OCR with OpenCV pre-processing.

## What It Does

Takes a raw image containing text (invoices, documents, signs) and produces:
- A validated text string with only high-confidence detections
- An annotated image with bounding boxes around detected words
- A detailed results file with word-by-word confidence scores

## Tech Stack

| Tool | Role |
|------|------|
| Python 3.11 | Orchestration and pipeline logic |
| OpenCV 4.13 | Image loading, pre-processing, and annotation |
| Tesseract 5.5 | OCR engine (CNN + BiLSTM neural network) |
| pytesseract 0.3 | Python-to-Tesseract bridge |
| Pillow 12.2 | Image format compatibility |
| NumPy 2.4 | Array data format for image representation |

## Pipeline Flow

```
Raw Image --> Grayscale --> Gaussian Blur --> Adaptive Threshold --> Tesseract OCR --> Confidence Filter (>=80%) --> Validated Text + Annotated Image
```

## Project Structure

```
OCR-Pipeline-Image-Text-Recognition/
├── images/
│   ├── raw/              # Original input images (read-only)
│   └── processed/        # Pre-processed images (grayscale, threshold)
├── output/
│   ├── annotated/        # Images with bounding boxes drawn
│   └── results.txt       # Extracted text + confidence scores
├── src/
│   ├── __init__.py       # Package marker
│   ├── preprocess.py     # Image cleaning pipeline
│   ├── ocr_engine.py     # Tesseract integration + confidence filtering
│   └── visualise.py      # Bounding box drawing + results export
├── main.py               # Entry point -- runs the full pipeline
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup

### Prerequisites
- Python 3.8+
- Tesseract OCR binary installed and on PATH
  - Windows: https://github.com/UB-Mannheim/tesseract/wiki

### Installation
```bash
git clone https://github.com/louayakermi2816-gif/OCR-Pipeline-Image-Text-Recognition.git
cd OCR-Pipeline-Image-Text-Recognition
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## Usage

1. Place your image in `images/raw/`
2. Update the `image_path` in `main.py` if needed
3. Run the pipeline:
```bash
python main.py
```
4. Check results in `output/`

## Validation Results

| Requirement | Result |
|---|---|
| Library integration | pytesseract + OpenCV run without errors |
| Pre-processing integrity | Grayscale + threshold images saved to images/processed/ |
| Confidence benchmarking | 91.2% average confidence, 31/38 words passed 80% threshold |
| Visual confirmation | Annotated image with labeled bounding boxes saved to output/annotated/ |
