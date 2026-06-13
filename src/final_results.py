import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_group(df: pd.DataFrame, output_dir: Path, prefix: str, title_prefix: str) -> None:
    metrics = {
        "precision": "Precision",
        "recall": "Recall",
        "map50": "mAP@50",
        "map50_95": "mAP@50-95",
    }

    for metric, label in metrics.items():
        plt.figure(figsize = (9, 5))
        plt.bar(df["method"], df[metric])
        plt.xticks(rotation = 30, ha = "right")
        plt.ylabel(label)
        plt.title(f"{title_prefix}: {label}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_{metric}.png", dpi = 300)
        plt.close()

def main() -> None:
    input_path = Path("experiments/final_results_summary.csv")
    output_dir = Path("experiments/final_figures")
    output_dir.mkdir(parents = True, exist_ok = True)

    df = pd.read_csv(input_path)

    masking_df = df[df["experiment"].isin(["Experiment 10", "Experiment 11"])]
    training_df = df[df["experiment"] == "Experiment 12"]

    plot_group(
        masking_df,
        output_dir,
        "masking_evaluation",
        "Saliency Masking Evaluation",
    )

    plot_group(
        training_df,
        output_dir,
        "training_comparison",
        "Training Comparison",
    )

    print(f"Saved charts to: {output_dir}")

if __name__ == "__main__":
    main()