import cv2
import os


def preprocess_image(image_path, save_dir="images/processed"):
    """
    Takes a raw image path, applies pre-processing steps,
    and returns a cleaned image ready for OCR.
    Also saves intermediate results for visual inspection.
    """

    # Step 1: Load the raw image from disk
    img = cv2.imread(image_path)

    # Check if the image was actually loaded
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Print the raw image shape so we can see what we're working with
    print(f"[INFO] Raw image loaded: {image_path}")
    print(f"[INFO] Shape: {img.shape}  (height, width, channels)")

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"[INFO] Grayscale shape: {gray.shape}  (height, width) -- no more color channels")

    # Step 3: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 4: Apply adaptive thresholding to get crisp black-and-white
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # Save intermediate images so we can visually inspect each step
    os.makedirs(save_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_gray.png"), gray)
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_thresh.png"), thresh)
    print(f"[INFO] Saved processed images to: {save_dir}/")

    return thresh
