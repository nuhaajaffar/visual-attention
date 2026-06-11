import argparse
import cv2
import numpy as np
from pathlib import Path

def resize_to_height(image, target_height):
    height, width = image.shape[:2]
    scale = target_height / height
    new_width = int(width * scale)
    return cv2.resize(image, (new_width, target_height))

def add_title(image, title):
    title_bar = np.full((50, image.shape[1], 3), 255, dtype = np.uint8)
    cv2.putText(
        title_bar,
        title,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([title_bar, image])

def create_comparison(original_path, yolo_path, saliency_path, output_path):
    original = cv2.imread(str(original_path))
    yolo = cv2.imread(str(yolo_path))
    saliency = cv2.imread(str(saliency_path))

    if original is None or yolo is None or saliency is None:
        print(f"Skipped {original_path.stem}: missing image file")
        return

    target_height = 500

    original = resize_to_height(original, target_height)
    yolo = resize_to_height(yolo, target_height)
    saliency = resize_to_height(saliency, target_height)

    original = add_title(original, "Original Image")
    yolo = add_title(yolo, "YOLO Detection")
    saliency = add_title(saliency, "Saliency Overlay")

    comparison = np.hstack([original, yolo, saliency])

    output_path.parent.mkdir(parents = True, exist_ok = True)
    cv2.imwrite(str(output_path), comparison)

    print(f"Saved comparison: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--originals", required = True)
    parser.add_argument("--yolo", required = True)
    parser.add_argument("--saliency", required = True)
    parser.add_argument("--output", required = True)
    args = parser.parse_args()

    originals_dir = Path(args.originals)
    yolo_dir = Path(args.yolo)
    saliency_dir = Path(args.saliency)
    output_dir = Path(args.output)

    expected_files = {
        "cat": "cat_yolo_output.jpg",
        "dog": "dog_yolo_output.jpg",
        "street": "street_yolo_output.jpg",
    }

    for stem, yolo_filename in expected_files.items():
        original_path = originals_dir / f"{stem}.jpg"
        yolo_path = yolo_dir / yolo_filename
        saliency_path = saliency_dir / f"{stem}_saliency_overlay.jpg"
        output_path = output_dir / f"{stem}_comp.jpg"

        create_comparison(original_path, yolo_path, saliency_path, output_path)

if __name__ == "__main__":
    main()