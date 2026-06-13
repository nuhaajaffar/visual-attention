import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main() -> None:
    input_path = Path("experiments/final_results_summary.csv")
    output_dir = Path("experiments/final_figures")
    output_dir.mkdir(parents = True, exist_ok = True)

    df = pd.read_csv(input_path)

    metrics = ["precision", "recall", "map50", "map50_95"]

    for metric in metrics:
        plt.figure(figsize = (10, 5))
        plt.bar(df["method"], df[metric])
        plt.xticks(rotation = 35, ha = "right")
        plt.ylabel(metric)
        plt.title(f"Final Results Comparison: {metric}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_comparison.png", dpi = 300)
        plt.close()

    print(f"Saved charts to: {output_dir}")

if __name__ == "__main__":
    main()