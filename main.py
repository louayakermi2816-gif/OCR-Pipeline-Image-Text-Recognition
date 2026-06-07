from src.preprocess import preprocess_image
from src.ocr_engine import run_ocr, filter_by_confidence
from src.visualise import draw_bounding_boxes, save_results


def main():
    # Path to our test image
    image_path = "images/raw/test1.png"

    print("=" * 50)
    print("  OCR Pipeline -- Complete Run")
    print("=" * 50)

    # Phase 2: Pre-process the image
    processed = preprocess_image(image_path)

    # Phase 3: Run OCR on the cleaned image
    data = run_ocr(processed)

    # Phase 4: Filter by confidence
    validated_text, high_conf_words = filter_by_confidence(data, threshold=80)

    # Phase 5: Visual confirmation and save results
    draw_bounding_boxes(image_path, data, high_conf_words)
    save_results(validated_text, high_conf_words)

    # Display the final result
    print("\n" + "=" * 50)
    print("  VALIDATED TEXT OUTPUT")
    print("=" * 50)
    print(validated_text)
    print("=" * 50)


if __name__ == "__main__":
    main()
