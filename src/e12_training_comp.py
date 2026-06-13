import argparse
import csv
from pathlib import Path
from ultralytics import YOLO

def evaluate_model(model_path: str, data_path: str, project: str, name: str, imgsz: int, batch: int) -> dict:
    model = YOLO(model_path)

    results = model.val(
        data = data_path,
        imgsz = imgsz,
        batch = batch,
        workers = 0,
        project = project,
        name = name,
        exist_ok = True,
    )

    return {
        "model_type": name,
        "precision": results.box.mp,
        "recall": results.box.mr,
        "map50": results.box.map50,
        "map50_95": results.box.map,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-model", required = True)
    parser.add_argument("--original-data", required = True)
    parser.add_argument("--masked-model", required = True)
    parser.add_argument("--masked-data", required = True)
    parser.add_argument("--project", required = True)
    parser.add_argument("--output", required = True)
    parser.add_argument("--imgsz", type = int, default = 416)
    parser.add_argument("--batch", type = int, default = 2)
    args = parser.parse_args()

    rows = [
        evaluate_model(
            args.original_model,
            args.original_data,
            args.project,
            "original_model_on_original_val",
            args.imgsz,
            args.batch,
        ),
        evaluate_model(
            args.masked_model,
            args.masked_data,
            args.project,
            "medium_masked_model_on_medium_val",
            args.imgsz,
            args.batch,
        ),
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    with output_path.open("w", newline = "", encoding = "utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames = ["model_type", "precision", "recall", "map50", "map50_95"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved metrics to: {output_path}")

if __name__ == "__main__":
    main()