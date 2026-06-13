import argparse
import shutil
import cv2
import numpy as np
from pathlib import Path

VOC_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

def write_yaml(output_path: Path, dataset_root: Path) -> None:
    names = "\n".join([f"  {index}: {name}" for index, name in enumerate(VOC_NAMES)])

    content = f"""path: {dataset_root.resolve().as_posix()}
train: images/val
val: images/val

names:
{names}
"""

    output_path.write_text(content, encoding = "utf-8")

def create_masked_image(image_path: Path, saliency_map_path: Path, output_path: Path, background: float) -> None:
    image = cv2.imread(str(image_path))
    saliency_map = cv2.imread(str(saliency_map_path), cv2.IMREAD_GRAYSCALE)

    if image is None or saliency_map is None:
        print(f"Skipped: {image_path.name}")
        return

    saliency_map = cv2.resize(saliency_map, (image.shape[1], image.shape[0]))
    saliency_map = saliency_map.astype(np.float32) / 255.0
    saliency_map = cv2.GaussianBlur(saliency_map, (21, 21), 0)

    mask = background + (1.0 - background) * saliency_map
    mask = np.expand_dims(mask, axis = 2)

    masked_image = image.astype(np.float32) * mask
    masked_image = np.clip(masked_image, 0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents = True, exist_ok = True)
    cv2.imwrite(str(output_path), masked_image)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", required = True)
    parser.add_argument("--maps", required = True)
    parser.add_argument("--output-root", required = True)
    parser.add_argument("--background", type = float, default = 0.7)
    args = parser.parse_args()

    original_root = Path(args.original_root)
    maps_dir = Path(args.maps)
    output_root = Path(args.output_root)

    source_images = original_root / "images" / "val"
    source_labels = original_root / "labels" / "val"

    target_images = output_root / "images" / "val"
    target_labels = output_root / "labels" / "val"

    target_images.mkdir(parents = True, exist_ok = True)
    target_labels.mkdir(parents = True, exist_ok = True)

    processed = 0

    for image_path in sorted(source_images.glob("*.jpg")):
        saliency_map_path = maps_dir / f"{image_path.stem}_saliency_map.jpg"
        label_path = source_labels / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        create_masked_image(image_path, saliency_map_path, target_images / image_path.name, args.background)
        shutil.copy2(label_path, target_labels / label_path.name)

        processed += 1

    write_yaml(output_root.parent / "masked.yaml", output_root)

    print(f"Created {processed} masked images")
    print(f"Created YAML: {output_root.parent / 'masked.yaml'}")

if __name__ == "__main__":
    main()