from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAMPLE_COUNT = 1_000
RANDOM_SEED = 42
OUTPUT_DIR = Path("outputs")


def create_multicollinearity_data() -> pd.DataFrame:
    """Create five-feature data with intentionally strong linear relationships."""
    rng = np.random.default_rng(RANDOM_SEED)

    feature_1 = rng.normal(0, 1, SAMPLE_COUNT)
    feature_3 = rng.normal(0, 1, SAMPLE_COUNT)

    # Strongly correlated with feature_1.
    feature_2 = 0.95 * feature_1 + rng.normal(0, 0.08, SAMPLE_COUNT)

    # Strongly negatively correlated with feature_3.
    feature_4 = -0.80 * feature_3 + rng.normal(0, 0.10, SAMPLE_COUNT)

    # Almost a linear combination of feature_1 and feature_3.
    feature_5 = (
        0.60 * feature_1
        + 0.70 * feature_3
        + rng.normal(0, 0.08, SAMPLE_COUNT)
    )

    target = (
        4.0 * feature_1
        - 2.0 * feature_3
        + 1.5 * feature_5
        + rng.normal(0, 1.0, SAMPLE_COUNT)
    )

    return pd.DataFrame(
        {
            "feature_1": feature_1,
            "feature_2": feature_2,
            "feature_3": feature_3,
            "feature_4": feature_4,
            "feature_5": feature_5,
            "target": target,
        }
    )


def save_correlation_plot(data: pd.DataFrame, output_path: Path) -> None:
    """Save a heatmap that makes the correlated feature pairs easy to inspect."""
    feature_columns = [f"feature_{number}" for number in range(1, 6)]
    correlation = data[feature_columns].corr()

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(feature_columns)), feature_columns, rotation=45)
    axis.set_yticks(range(len(feature_columns)), feature_columns)

    for row in range(len(feature_columns)):
        for column in range(len(feature_columns)):
            value = correlation.iloc[row, column]
            text_color = "white" if abs(value) > 0.6 else "black"
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
            )

    axis.set_title("Feature Correlation Matrix")
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    data_path = OUTPUT_DIR / "multicollinearity_dataset.csv"
    plot_path = OUTPUT_DIR / "multicollinearity_correlation.png"

    data = create_multicollinearity_data()
    data.to_csv(data_path, index=False)
    save_correlation_plot(data, plot_path)

    feature_columns = [f"feature_{number}" for number in range(1, 6)]
    correlation = data[feature_columns].corr()

    print(f"Created dataset: {data.shape[0]} rows, 5 features, 1 target")
    print("\nFeature correlation matrix:")
    print(correlation.round(3).to_string())
    print(f"\nDataset saved to: {data_path}")
    print(f"Correlation plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
