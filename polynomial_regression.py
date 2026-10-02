from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


NUM_SAMPLES = 1_000
POLYNOMIAL_DEGREE = 3
RANDOM_SEED = 42
NOISE_STANDARD_DEVIATION = 8.0
OUTPUT_DIR = Path("outputs")


def create_polynomial_data(
    num_samples: int = NUM_SAMPLES,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create data following y = 0.5x^3 - 2x^2 + 3x + 10 + noise."""
    rng = np.random.default_rng(random_seed)
    feature = rng.uniform(-5, 5, num_samples)
    noise = rng.normal(0, NOISE_STANDARD_DEVIATION, num_samples)
    target = 0.5 * feature**3 - 2 * feature**2 + 3 * feature + 10 + noise

    return pd.DataFrame({"feature": feature, "target": target})


def train_polynomial_regression(
    data: pd.DataFrame,
) -> tuple[object, float, float]:
    """Train a polynomial linear-regression model and return test metrics."""
    features = data[["feature"]]
    targets = data["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )

    model = make_pipeline(
        PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=False),
        LinearRegression(),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))
    return model, rmse, r2


def save_visualization(
    data: pd.DataFrame,
    model: object,
    output_path: Path,
) -> None:
    """Draw the generated samples and fitted polynomial curve."""
    x_curve = np.linspace(data["feature"].min(), data["feature"].max(), 500)
    curve_frame = pd.DataFrame({"feature": x_curve})
    y_curve = model.predict(curve_frame)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        data["feature"],
        data["target"],
        s=18,
        alpha=0.4,
        color="dodgerblue",
        label=f"Generated samples (N={len(data):,})",
    )
    plt.plot(
        x_curve,
        y_curve,
        color="crimson",
        linewidth=3,
        label=f"Polynomial regression (degree={POLYNOMIAL_DEGREE})",
    )
    plt.title("Polynomial Linear Regression")
    plt.xlabel("Feature (x)")
    plt.ylabel("Target (y)")
    plt.grid(linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    data_path = OUTPUT_DIR / "polynomial_regression_data.csv"
    plot_path = OUTPUT_DIR / "polynomial_regression_plot.png"

    data = create_polynomial_data()
    model, rmse, r2 = train_polynomial_regression(data)
    data.to_csv(data_path, index=False)
    save_visualization(data, model, plot_path)

    print(f"Created {len(data):,} polynomial samples.")
    print(f"Polynomial degree: {POLYNOMIAL_DEGREE}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R-squared: {r2:.4f}")
    print(f"Data saved to: {data_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
