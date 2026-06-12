import argparse
import csv
from pathlib import Path

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

def read_prediction_file(label_path: Path) -> dict:
    detections = {}

    if not label_path.exists():
        return detections

    with label_path.open("r", encoding = "utf-8") as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) < 6:
                continue

            class_id = int(float(parts[0]))
            confidence = float(parts[-1])
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"

            if class_name not in detections:
                detections[class_name] = []

            detections[class_name].append(confidence)

    return detections

def summarise(detections: dict, class_name: str) -> tuple[int, float | None]:
    values = detections.get(class_name, [])

    if not values:
        return 0, None

    return len(values), max(values)

def format_confidence(value: float | None) -> str:
    if value is None:
        return ""

    return f"{value:.4f}"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required = True)
    parser.add_argument("--original-labels", required = True)
    parser.add_argument("--masked-labels", required = True)
    parser.add_argument("--output", required = True)
    args = parser.parse_args()

    images_dir = Path(args.images)
    original_labels_dir = Path(args.original_labels)
    masked_labels_dir = Path(args.masked_labels)
    output_path = Path(args.output)

    rows = []

    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        stem = image_path.stem

        original_detections = read_prediction_file(original_labels_dir / f"{stem}.txt")
        masked_detections = read_prediction_file(masked_labels_dir / f"{stem}_masked.txt")

        class_names = sorted(set(original_detections.keys()) | set(masked_detections.keys()))

        for class_name in class_names:
            original_count, original_conf = summarise(original_detections, class_name)
            masked_count, masked_conf = summarise(masked_detections, class_name)

            confidence_change = ""
            if original_conf is not None and masked_conf is not None:
                confidence_change = f"{masked_conf - original_conf:.4f}"

            rows.append({
                "image": stem,
                "class": class_name,
                "original_count": original_count,
                "original_max_confidence": format_confidence(original_conf),
                "masked_count": masked_count,
                "masked_max_confidence": format_confidence(masked_conf),
                "confidence_change": confidence_change,
            })

    output_path.parent.mkdir(parents=True, exist_ok = True)

    with output_path.open("w", newline = "", encoding = "utf-8") as file:
        fieldnames = [
            "image",
            "class",
            "original_count",
            "original_max_confidence",
            "masked_count",
            "masked_max_confidence",
            "confidence_change",
        ]

        writer = csv.DictWriter(file, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved comparison table to: {output_path}")

if __name__ == "__main__":
    main()