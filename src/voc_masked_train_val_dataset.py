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
train: images/train
val: images/val

names:
{names}
"""

    output_path.write_text(content, encoding="utf-8")

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

def process_split(original_root: Path, maps_root: Path, output_root: Path, split: str, background: float) -> int:
    source_images = original_root / "images" / split
    source_labels = original_root / "labels" / split
    maps_dir = maps_root / split

    target_images = output_root / "images" / split
    target_labels = output_root / "labels" / split

    target_images.mkdir(parents = True, exist_ok = True)
    target_labels.mkdir(parents = True, exist_ok = True)

    processed = 0

    for image_path in sorted(source_images.glob("*.jpg")):
        saliency_map_path = maps_dir / f"{image_path.stem}_saliency_map.jpg"
        label_path = source_labels / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        create_masked_image(image_path, saliency_map_path, target_images / image_path.name, background)
        shutil.copy2(label_path, target_labels / label_path.name)

        processed += 1

    return processed

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", required = True)
    parser.add_argument("--maps-root", required = True)
    parser.add_argument("--output-root", required = True)
    parser.add_argument("--yaml-output", required = True)
    parser.add_argument("--background", type = float, default = 0.5)
    args = parser.parse_args()

    original_root = Path(args.original_root)
    maps_root = Path(args.maps_root)
    output_root = Path(args.output_root)

    train_count = process_split(original_root, maps_root, output_root, "train", args.background)
    val_count = process_split(original_root, maps_root, output_root, "val", args.background)

    write_yaml(Path(args.yaml_output), output_root)

    print(f"Created {train_count} masked training images")
    print(f"Created {val_count} masked validation images")
    print(f"Created YAML: {args.yaml_output}")

if __name__ == "__main__":
    main()