from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path("data/house-prices-advanced-regression-techniques")
TRAIN_PATH = DATA_DIR / "house_train.csv"
TEST_PATH = DATA_DIR / "house_test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "house_sample_submission.csv"
OUTPUT_DIR = Path("outputs/house_regularized_regression")

TARGET_COLUMN = "SalePrice"
ID_COLUMN = "Id"
RANDOM_SEED = 42
TEST_SIZE = 0.2
TOP_COEFFICIENTS = 20


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the House Prices train, test, and sample-submission data."""
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    if TARGET_COLUMN not in train_data.columns:
        raise ValueError(f"{TRAIN_PATH} does not contain '{TARGET_COLUMN}'.")
    if list(sample_submission.columns) != [ID_COLUMN, TARGET_COLUMN]:
        raise ValueError(
            f"{SAMPLE_SUBMISSION_PATH} must contain only "
            f"'{ID_COLUMN}' and '{TARGET_COLUMN}'."
        )
    if set(test_data[ID_COLUMN]) != set(sample_submission[ID_COLUMN]):
        raise ValueError("Test and sample-submission IDs do not match.")

    return train_data, test_data, sample_submission


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing for numeric and categorical house attributes."""
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def make_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    """Create three regularized linear-regression pipelines."""
    regressors = {
        "Ridge": Ridge(alpha=20.0),
        "Lasso": Lasso(alpha=0.0005, max_iter=30_000),
        "ElasticNet": ElasticNet(
            alpha=0.0005,
            l1_ratio=0.5,
            max_iter=30_000,
        ),
    }
    return {
        name: Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("regressor", regressor),
            ]
        )
        for name, regressor in regressors.items()
    }


def calculate_metrics(
    actual_prices: pd.Series,
    predicted_log_prices: np.ndarray,
) -> dict[str, float]:
    """Calculate metrics in both log-price and original dollar scales."""
    actual_log_prices = np.log1p(actual_prices)
    predicted_prices = np.maximum(np.expm1(predicted_log_prices), 0)

    return {
        "RMSLE": float(np.sqrt(mean_squared_error(actual_log_prices, predicted_log_prices))),
        "RMSE": float(np.sqrt(mean_squared_error(actual_prices, predicted_prices))),
        "MAE": float(mean_absolute_error(actual_prices, predicted_prices)),
        "R2": float(r2_score(actual_prices, predicted_prices)),
    }


def evaluate_models(
    models: dict[str, Pipeline],
    features: pd.DataFrame,
    targets: pd.Series,
) -> pd.DataFrame:
    """Evaluate every model on the same validation split."""
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        targets,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )
    y_train_log = np.log1p(y_train)
    evaluation_rows = []

    for name, model in models.items():
        model.fit(x_train, y_train_log)
        predicted_log_prices = model.predict(x_valid)
        evaluation_rows.append(
            {"Model": name, **calculate_metrics(y_valid, predicted_log_prices)}
        )

    return (
        pd.DataFrame(evaluation_rows)
        .sort_values("RMSLE")
        .reset_index(drop=True)
    )


def train_final_models_and_predict(
    models: dict[str, Pipeline],
    train_features: pd.DataFrame,
    targets: pd.Series,
    test_features: pd.DataFrame,
    test_ids: pd.Series,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Refit models on all training rows and predict the unlabeled test rows."""
    predictions = pd.DataFrame({ID_COLUMN: test_ids.to_numpy()})
    fitted_models = {}
    targets_log = np.log1p(targets)

    for name, model in models.items():
        final_model = clone(model)
        final_model.fit(train_features, targets_log)
        predicted_prices = np.maximum(
            np.expm1(final_model.predict(test_features)),
            0,
        )
        predictions[f"{name}_SalePrice"] = predicted_prices
        fitted_models[name] = final_model

    predictions["Average_SalePrice"] = predictions[
        ["Ridge_SalePrice", "Lasso_SalePrice", "ElasticNet_SalePrice"]
    ].mean(axis=1)
    return fitted_models, predictions


def compare_with_sample_submission(
    predictions: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> pd.DataFrame:
    """Compare model predictions with the sample submission as a reference.

    The sample submission is not ground truth, so these values are diagnostic
    similarity measures and not true test-set performance metrics.
    """
    comparison = sample_submission.merge(
        predictions,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )
    reference_prices = comparison[TARGET_COLUMN]
    rows = []

    for name in ("Ridge", "Lasso", "ElasticNet"):
        predicted_prices = comparison[f"{name}_SalePrice"].to_numpy()
        metrics = calculate_metrics(
            reference_prices,
            np.log1p(predicted_prices),
        )
        rows.append(
            {
                "Model": name,
                "Reference_RMSLE": metrics["RMSLE"],
                "Reference_RMSE": metrics["RMSE"],
                "Reference_MAE": metrics["MAE"],
                "Reference_R2": metrics["R2"],
                "Correlation": float(
                    np.corrcoef(reference_prices, predicted_prices)[0, 1]
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("Reference_RMSLE")
        .reset_index(drop=True)
    )


def save_coefficient_plot(
    fitted_models: dict[str, Pipeline],
    output_path: Path,
) -> None:
    """Plot the largest positive and negative coefficients for each model."""
    figure, axes = plt.subplots(1, 3, figsize=(21, 9))

    for axis, (name, model) in zip(axes, fitted_models.items()):
        preprocessor = model.named_steps["preprocessor"]
        regressor = model.named_steps["regressor"]
        feature_names = preprocessor.get_feature_names_out()
        coefficients = np.asarray(regressor.coef_)

        coefficient_data = pd.DataFrame(
            {"Feature": feature_names, "Coefficient": coefficients}
        )
        coefficient_data["AbsoluteCoefficient"] = coefficient_data[
            "Coefficient"
        ].abs()
        top_coefficients = (
            coefficient_data.nlargest(TOP_COEFFICIENTS, "AbsoluteCoefficient")
            .sort_values("Coefficient")
        )
        colors = np.where(
            top_coefficients["Coefficient"] >= 0,
            "royalblue",
            "tomato",
        )

        axis.barh(
            top_coefficients["Feature"],
            top_coefficients["Coefficient"],
            color=colors,
        )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(f"{name}: Top {TOP_COEFFICIENTS} Coefficients")
        axis.set_xlabel("Coefficient (log SalePrice scale)")
        axis.grid(axis="x", linestyle=":", alpha=0.5)

    figure.suptitle("House Price Model Coefficient Comparison", fontsize=18)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_data, test_data, sample_submission = load_data()
    train_ids = train_data[ID_COLUMN]
    test_ids = test_data[ID_COLUMN]
    train_features = train_data.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    test_features = test_data.drop(columns=[ID_COLUMN])
    targets = train_data[TARGET_COLUMN]

    preprocessor = make_preprocessor(train_features)
    models = make_models(preprocessor)

    evaluation = evaluate_models(models, train_features, targets)
    evaluation_path = OUTPUT_DIR / "validation_metrics.csv"
    evaluation.to_csv(evaluation_path, index=False)

    fitted_models, predictions = train_final_models_and_predict(
        models,
        train_features,
        targets,
        test_features,
        test_ids,
    )
    predictions_path = OUTPUT_DIR / "house_test_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    sample_comparison = compare_with_sample_submission(
        predictions,
        sample_submission,
    )
    sample_comparison_path = OUTPUT_DIR / "sample_submission_comparison.csv"
    sample_comparison.to_csv(sample_comparison_path, index=False)

    best_model_name = evaluation.iloc[0]["Model"]
    kaggle_submission = predictions[
        [ID_COLUMN, f"{best_model_name}_SalePrice"]
    ].rename(columns={f"{best_model_name}_SalePrice": TARGET_COLUMN})
    kaggle_submission_path = OUTPUT_DIR / "best_model_submission.csv"
    kaggle_submission.to_csv(kaggle_submission_path, index=False)

    coefficients_path = OUTPUT_DIR / "regularized_model_coefficients.png"
    save_coefficient_plot(fitted_models, coefficients_path)

    print(f"Training rows: {len(train_ids):,}")
    print(f"Test rows predicted: {len(test_ids):,}")
    print("\nValidation metrics (lower RMSLE/RMSE/MAE is better):")
    print(evaluation.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print(f"\nMetrics saved to: {evaluation_path}")
    print(f"Test predictions saved to: {predictions_path}")
    print(f"Best-model Kaggle submission saved to: {kaggle_submission_path}")
    print(f"Coefficient plot saved to: {coefficients_path}")
    print("\nComparison with house_sample_submission.csv (reference only):")
    print(
        sample_comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:,.4f}",
        )
    )
    print(f"Sample comparison saved to: {sample_comparison_path}")
    print(
        "Note: house_test.csv has no true SalePrice column. True model "
        "evaluation therefore uses a 20% holdout from house_train.csv; "
        "sample-submission metrics only measure similarity to its example "
        "predictions."
    )


if __name__ == "__main__":
    main()
