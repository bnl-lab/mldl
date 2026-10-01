from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAMPLE_COUNT = 100
RANDOM_SEED = 42
OUTPUT_DIR = Path("outputs")


def create_data() -> pd.DataFrame:
    """Create reproducible data with one feature and a linear target."""
    rng = np.random.default_rng(RANDOM_SEED)
    feature = rng.uniform(0, 10, SAMPLE_COUNT)
    noise = rng.normal(0, 3, SAMPLE_COUNT)
    target = 3 * feature + 5 + noise

    return pd.DataFrame({"feature": feature, "target": target})


def visualize_data(data: pd.DataFrame, output_path: Path) -> tuple[float, float]:
    """Fit a straight line, draw it with the data, and save the chart."""
    slope, intercept = np.polyfit(data["feature"], data["target"], 1)
    feature_line = np.linspace(data["feature"].min(), data["feature"].max(), 200)
    target_line = slope * feature_line + intercept

    plt.figure(figsize=(8, 5))
    plt.scatter(
        data["feature"],
        data["target"],
        alpha=0.75,
        label="Generated data",
    )
    plt.plot(
        feature_line,
        target_line,
        color="red",
        linewidth=2,
        label=f"Regression line: y = {slope:.2f}x + {intercept:.2f}",
    )
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Linear Regression Data (100 Samples, 1 Feature)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return slope, intercept


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    data_path = OUTPUT_DIR / "linear_regression_data.csv"
    plot_path = OUTPUT_DIR / "linear_regression_plot.png"

    data = create_data()
    data.to_csv(data_path, index=False)
    slope, intercept = visualize_data(data, plot_path)

    print(f"Created {len(data)} samples with 1 feature.")
    print(f"Fitted line: y = {slope:.2f}x + {intercept:.2f}")
    print(f"Data saved to: {data_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
