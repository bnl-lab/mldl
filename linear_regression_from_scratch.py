from pathlib import Path

import matplotlib.pyplot as plt

from linear_regression_data import create_linear_data


def predict(feature: float, weight: float, bias: float) -> float:
    """Apply the linear model y_hat = weight * x + bias."""
    return weight * feature + bias


def mean_squared_error(
    features: list[float],
    targets: list[float],
    weight: float,
    bias: float,
) -> float:
    """Return the average squared prediction error."""
    squared_errors = [
        (predict(x, weight, bias) - y) ** 2
        for x, y in zip(features, targets, strict=True)
    ]
    return sum(squared_errors) / len(squared_errors)


def solve_with_least_squares(
    features: list[float],
    targets: list[float],
) -> tuple[float, float]:
    """Calculate the exact weight and bias that minimize MSE."""
    mean_x = sum(features) / len(features)
    mean_y = sum(targets) / len(targets)

    covariance = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(features, targets, strict=True)
    )
    variance = sum((x - mean_x) ** 2 for x in features)

    if variance == 0:
        raise ValueError("Linear regression needs at least two different feature values.")

    weight = covariance / variance
    bias = mean_y - weight * mean_x
    return weight, bias


def train_with_gradient_descent(
    features: list[float],
    targets: list[float],
    learning_rate: float = 0.01,
    epochs: int = 5_000,
) -> tuple[float, float, list[tuple[int, float]]]:
    """Learn weight and bias by repeatedly moving in the downhill direction."""
    weight = 0.0
    bias = 0.0
    sample_count = len(features)
    history: list[tuple[int, float]] = []

    for epoch in range(epochs + 1):
        if epoch % 50 == 0:
            loss = mean_squared_error(features, targets, weight, bias)
            history.append((epoch, loss))

        if epoch == epochs:
            break

        errors = [
            predict(x, weight, bias) - y
            for x, y in zip(features, targets, strict=True)
        ]
        weight_gradient = (2 / sample_count) * sum(
            error * x for error, x in zip(errors, features, strict=True)
        )
        bias_gradient = (2 / sample_count) * sum(errors)

        weight -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

    return weight, bias, history


def save_learning_plot(
    features: list[float],
    targets: list[float],
    optimal_weight: float,
    optimal_bias: float,
    learned_weight: float,
    learned_bias: float,
    history: list[tuple[int, float]],
    output_path: Path,
) -> None:
    """Visualize the fitted lines and the loss during learning."""
    line_x = [min(features), max(features)]
    optimal_y = [predict(x, optimal_weight, optimal_bias) for x in line_x]
    learned_y = [predict(x, learned_weight, learned_bias) for x in line_x]

    figure, (model_axis, loss_axis) = plt.subplots(1, 2, figsize=(13, 5))
    model_axis.scatter(features, targets, alpha=0.7, label="Data")
    model_axis.plot(
        line_x,
        optimal_y,
        color="red",
        linewidth=3,
        label="Exact least-squares solution",
    )
    model_axis.plot(
        line_x,
        learned_y,
        color="green",
        linestyle="--",
        linewidth=2,
        label="Gradient-descent solution",
    )
    model_axis.set_title("Fitted Linear Regression Model")
    model_axis.set_xlabel("Feature (x)")
    model_axis.set_ylabel("Target (y)")
    model_axis.grid(alpha=0.25)
    model_axis.legend()

    epochs = [epoch for epoch, _ in history]
    losses = [loss for _, loss in history]
    loss_axis.plot(epochs, losses, color="purple")
    loss_axis.set_title("MSE During Gradient Descent")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Mean Squared Error")
    loss_axis.set_yscale("log")
    loss_axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    # The data module returns NumPy arrays. The fitting code below converts
    # them to lists and uses ordinary Python equations, loops, and sums.
    feature_array, target_array = create_linear_data()
    features = feature_array.tolist()
    targets = target_array.tolist()

    optimal_weight, optimal_bias = solve_with_least_squares(features, targets)
    learned_weight, learned_bias, history = train_with_gradient_descent(
        features,
        targets,
    )

    optimal_loss = mean_squared_error(
        features, targets, optimal_weight, optimal_bias
    )
    learned_loss = mean_squared_error(
        features, targets, learned_weight, learned_bias
    )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "linear_regression_learning.png"
    save_learning_plot(
        features,
        targets,
        optimal_weight,
        optimal_bias,
        learned_weight,
        learned_bias,
        history,
        plot_path,
    )

    print("Exact least-squares solution (minimum possible training MSE)")
    print(f"  y = {optimal_weight:.6f}x + {optimal_bias:.6f}")
    print(f"  MSE = {optimal_loss:.6f}")
    print("Gradient-descent solution")
    print(f"  y = {learned_weight:.6f}x + {learned_bias:.6f}")
    print(f"  MSE = {learned_loss:.6f}")
    print(
        "  final parameter gap = "
        f"{abs(optimal_weight - learned_weight):.10f}, "
        f"{abs(optimal_bias - learned_bias):.10f}"
    )
    print(f"Learning plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
