import argparse
import cv2
import numpy as np
from pathlib import Path

def create_masked_image(
    image_path: Path,
    saliency_map_path: Path,
    output_path: Path,
    background_strength: float,
) -> None:
    image = cv2.imread(str(image_path))
    saliency_map = cv2.imread(str(saliency_map_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Skipped unreadable image: {image_path}")
        return

    if saliency_map is None:
        print(f"Missing saliency map for: {image_path.name}")
        return

    saliency_map = cv2.resize(saliency_map, (image.shape[1], image.shape[0]))
    saliency_map = saliency_map.astype(np.float32) / 255.0
    saliency_map = cv2.GaussianBlur(saliency_map, (21, 21), 0)

    mask = background_strength + (1.0 - background_strength) * saliency_map
    mask = np.expand_dims(mask, axis = 2)

    masked_image = image.astype(np.float32) * mask
    masked_image = np.clip(masked_image, 0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents = True, exist_ok = True)
    cv2.imwrite(str(output_path), masked_image)

    print(f"Saved masked image: {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required = True)
    parser.add_argument("--maps", required = True)
    parser.add_argument("--output", required = True)
    parser.add_argument("--background", type = float, default = 0.3)
    args = parser.parse_args()

    if not 0.0 <= args.background <= 1.0:
        raise ValueError("Background strength must be between 0.0 and 1.0")

    images_dir = Path(args.images)
    maps_dir = Path(args.maps)
    output_dir = Path(args.output)

    valid_extensions = {".jpg", ".jpeg", ".png"}

    for image_path in images_dir.iterdir():
        if image_path.suffix.lower() not in valid_extensions:
            continue

        saliency_map_path = maps_dir / f"{image_path.stem}_saliency_map.jpg"
        output_path = output_dir / f"{image_path.stem}_masked.jpg"

        create_masked_image(
            image_path,
            saliency_map_path,
            output_path,
            args.background,
        )

if __name__ == "__main__":
    main()