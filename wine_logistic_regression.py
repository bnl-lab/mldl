from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
TEST_SIZE = 0.2
OUTPUT_DIR = Path("outputs/wine_logistic_regression")


def load_wine_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the Wine dataset as pandas objects."""
    wine = load_wine()
    features = pd.DataFrame(wine.data, columns=wine.feature_names)
    targets = pd.Series(wine.target, name="target")
    class_names = [f"Class_{class_id}" for class_id in range(len(wine.target_names))]
    return features, targets, class_names


def make_model() -> Pipeline:
    """Create a scaling and multinomial logistic-regression pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=2_000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def save_result_plot(
    model: Pipeline,
    actual: pd.Series,
    predicted: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    output_path: Path,
) -> None:
    """Save a confusion matrix and standardized coefficient heatmap."""
    matrix = confusion_matrix(actual, predicted)
    coefficients = model.named_steps["classifier"].coef_
    feature_label_overrides = {
        "alcalinity_of_ash": "ash alkalinity",
        "nonflavanoid_phenols": "nonflav. phenols",
        "od280/od315_of_diluted_wines": "OD280/OD315",
    }
    display_feature_names = [
        feature_label_overrides.get(name, name.replace("_", " "))
        for name in feature_names
    ]
    coefficient_frame = pd.DataFrame(
        coefficients,
        index=class_names,
        columns=display_feature_names,
    )

    figure, axes = plt.subplots(1, 2, figsize=(21, 7.5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("Actual class")

    coefficient_limit = float(np.abs(coefficients).max())
    sns.heatmap(
        coefficient_frame,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-coefficient_limit,
        vmax=coefficient_limit,
        cbar_kws={"label": "Coefficient"},
        ax=axes[1],
    )
    axes[1].set_title("Class Coefficients after StandardScaler")
    axes[1].set_xlabel("Wine feature")
    axes[1].set_ylabel("Class")
    axes[1].tick_params(axis="x", labelrotation=40, labelsize=8)
    for label in axes[1].get_xticklabels():
        label.set_horizontalalignment("right")

    figure.suptitle("Wine Multiclass Logistic Regression", fontsize=17)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features, targets, class_names = load_wine_data()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=targets,
    )

    model = make_model()
    model.fit(x_train, y_train)

    predicted = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    accuracy = accuracy_score(y_test, predicted)
    cross_entropy = log_loss(y_test, probabilities, labels=model.classes_)

    prediction_data = pd.DataFrame(
        {
            "Actual": y_test.to_numpy(),
            "Predicted": predicted,
            "ActualName": [class_names[value] for value in y_test],
            "PredictedName": [class_names[value] for value in predicted],
        },
        index=y_test.index,
    )
    for class_id, class_name in enumerate(class_names):
        prediction_data[f"Probability_{class_name}"] = probabilities[:, class_id]
    prediction_data = prediction_data.sort_index()

    predictions_path = OUTPUT_DIR / "wine_test_predictions.csv"
    prediction_data.to_csv(predictions_path, index_label="SampleIndex")

    report = classification_report(
        y_test,
        predicted,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_path = OUTPUT_DIR / "classification_report.csv"
    pd.DataFrame(report).transpose().to_csv(report_path)

    plot_path = OUTPUT_DIR / "wine_logistic_regression_results.png"
    save_result_plot(
        model,
        y_test,
        predicted,
        features.columns.tolist(),
        class_names,
        plot_path,
    )

    print(f"Total samples: {len(features)}")
    print(f"Features: {features.shape[1]}")
    print(f"Train/Test samples: {len(x_train)}/{len(x_test)}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test cross-entropy (log loss): {cross_entropy:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predicted,
            target_names=class_names,
            zero_division=0,
        )
    )
    print(f"Predictions saved to: {predictions_path}")
    print(f"Classification report saved to: {report_path}")
    print(f"Visualization saved to: {plot_path}")


if __name__ == "__main__":
    main()
