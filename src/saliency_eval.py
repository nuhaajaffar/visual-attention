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
train: images/val
val: images/val

names:
{names}
"""
    
    output_path.write_text(content, encoding = "utf-8")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc-root", required = True)
    parser.add_argument("--output", required = True)
    parser.add_argument("--split", default = "val2012")
    parser.add_argument("--limit", type = int, default = 100)
    args = parser.parse_args()

    voc_root = Path(args.voc_root)
    output_root = Path(args.output)

    source_images = voc_root / "images" / args.split
    source_labels = voc_root / "labels" / args.split

    target_images = output_root / "original" / "images" / "val"
    target_labels = output_root / "original" / "labels" / "val"

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

        if copied >= args.limit:
            break

    write_yaml(output_root / "original.yaml", output_root / "original")

    print(f"Copied {copied} image-label pairs")
    print(f"Created YAML: {output_root / 'original.yaml'}")

if __name__ == "__main__":
    main()