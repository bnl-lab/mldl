from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NUM_SAMPLES = 100
RANDOM_SEED = 42
TRUE_WEIGHT = 2.5
TRUE_BIAS = 4.0
NOISE_STANDARD_DEVIATION = 2.0


def create_linear_data(
    num_samples: int = NUM_SAMPLES,
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one-feature data following y = 2.5x + 4 + noise."""
    # RandomState preserves the exact sample sequence used by the original file.
    rng = np.random.RandomState(random_seed)
    features = rng.uniform(0, 10, num_samples)
    noise = rng.normal(0, NOISE_STANDARD_DEVIATION, num_samples)
    targets = TRUE_WEIGHT * features + TRUE_BIAS + noise
    return features, targets


def save_data_plot(
    features: np.ndarray,
    targets: np.ndarray,
    output_path: Path,
) -> None:
    """Save a scatter plot of the generated data and its true relationship."""
    line_x = np.linspace(0, 10, 100)
    line_y = TRUE_WEIGHT * line_x + TRUE_BIAS

    plt.figure(figsize=(9, 6))
    plt.scatter(
        features,
        targets,
        color="dodgerblue",
        alpha=0.7,
        edgecolors="black",
        label=f"Generated data (N={len(features)})",
    )
    plt.plot(
        line_x,
        line_y,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f"True relationship: y = {TRUE_WEIGHT}x + {TRUE_BIAS}",
    )
    plt.title("Linear Regression Data: 1 Feature")
    plt.xlabel("Feature (x)")
    plt.ylabel("Target (y)")
    plt.grid(linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    features, targets = create_linear_data()
    plot_path = output_dir / "linear_data.png"
    save_data_plot(features, targets, plot_path)

    print(f"Created {len(features)} samples with 1 feature.")
    print(f"Feature shape: {features.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
