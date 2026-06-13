import argparse
import shutil
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

    output_path.write_text(content, encoding = "utf-8")

def copy_split(voc_root: Path, output_root: Path, source_split: str, target_split: str, limit: int) -> int:
    source_images = voc_root / "images" / source_split
    source_labels = voc_root / "labels" / source_split

    target_images = output_root / "images" / target_split
    target_labels = output_root / "labels" / target_split

    target_images.mkdir(parents = True, exist_ok = True)
    target_labels.mkdir(parents = True, exist_ok = True)

    copied = 0

    for image_path in sorted(source_images.glob("*.jpg")):
        label_path = source_labels / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        shutil.copy2(image_path, target_images / image_path.name)
        shutil.copy2(label_path, target_labels / label_path.name)

        copied += 1

        if copied >= limit:
            break

    return copied

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc-root", required = True)
    parser.add_argument("--output", required = True)
    parser.add_argument("--train-limit", type = int, default = 300)
    parser.add_argument("--val-limit", type = int, default = 100)
    args = parser.parse_args()

    voc_root = Path(args.voc_root)
    experiment_root = Path(args.output)
    original_root = experiment_root / "original"

    train_count = copy_split(voc_root, original_root, "train2012", "train", args.train_limit)
    val_count = copy_split(voc_root, original_root, "val2012", "val", args.val_limit)

    write_yaml(experiment_root / "original.yaml", original_root)

    print(f"Copied {train_count} training image-label pairs")
    print(f"Copied {val_count} validation image-label pairs")
    print(f"Created YAML: {experiment_root / 'original.yaml'}")

if __name__ == "__main__":
    main()