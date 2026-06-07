import cv2
import os


def draw_bounding_boxes(image_path, data, high_conf_words, save_dir="output/annotated"):
    """
    Draws bounding boxes around high-confidence words on the original image.
    Saves the annotated image to the output folder.
    """

    # Load the original color image (not the processed one)
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Build a set of high-confidence words for quick lookup
    high_conf_set = set(word for word, conf in high_conf_words)

    # Draw boxes around each high-confidence word
    boxes_drawn = 0
    for i in range(len(data['text'])):
        conf = int(data['conf'][i])
        word = data['text'][i].strip()

        if conf >= 80 and word != '':
            # Get the bounding box coordinates
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]

            # Draw a green rectangle around the word
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 2)

            # Put the confidence score above the box
            label = f"{conf}%"
            cv2.putText(img, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 0), 1)

            boxes_drawn += 1

    # Save the annotated image
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(save_dir, f"{base_name}_annotated.png")
    cv2.imwrite(output_path, img)

    print(f"\n[VISUAL] Drew {boxes_drawn} bounding boxes")
    print(f"[VISUAL] Saved annotated image to: {output_path}")

    return output_path


def save_results(validated_text, high_conf_words, save_path="output/results.txt"):
    """
    Saves the validated text and word-by-word breakdown to a text file.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("OCR Pipeline Results\n")
        f.write("=" * 40 + "\n\n")
        f.write("Validated Text:\n")
        f.write(validated_text + "\n\n")
        f.write("Word-by-Word Breakdown:\n")
        f.write("-" * 40 + "\n")
        for word, conf in high_conf_words:
            f.write(f"  {word:<20} {conf}%\n")

    print(f"[SAVE] Results saved to: {save_path}")
