import pytesseract


def run_ocr(processed_image):
    """
    Runs Tesseract OCR on a pre-processed image.
    Returns the full detection data including confidence scores per word.
    """

    # Configure Tesseract's behavior
    config = '--psm 6 --oem 3'

    # Run OCR and get detailed word-by-word data
    data = pytesseract.image_to_data(
        processed_image,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    # Count how many words were detected (excluding empty entries)
    total_words = sum(1 for text in data['text'] if text.strip() != '')
    print(f"[INFO] Tesseract detected {total_words} words total")

    return data


def filter_by_confidence(data, threshold=80):
    """
    Filters OCR results to keep only words with confidence >= threshold.
    Returns a list of (word, confidence) tuples and the filtered text.
    """

    high_conf_words = []

    for i in range(len(data['text'])):
        conf = int(data['conf'][i])
        word = data['text'][i].strip()

        # Skip empty entries and low-confidence detections
        if conf >= threshold and word != '':
            high_conf_words.append((word, conf))

    # Build the validated text string
    validated_text = ' '.join(word for word, conf in high_conf_words)

    # Print summary statistics
    total = sum(1 for t in data['text'] if t.strip() != '')
    passed = len(high_conf_words)
    avg_conf = sum(c for _, c in high_conf_words) / passed if passed > 0 else 0

    print(f"\n[FILTER] Confidence threshold: {threshold}%")
    print(f"[FILTER] Words detected: {total}")
    print(f"[FILTER] Words passed filter: {passed}")
    print(f"[FILTER] Words rejected: {total - passed}")
    print(f"[FILTER] Average confidence of passed words: {avg_conf:.1f}%")

    return validated_text, high_conf_words
