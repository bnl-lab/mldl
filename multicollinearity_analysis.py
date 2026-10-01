from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


DEFAULT_DATA_PATH = Path("data/multicollinearity_data.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/multicollinearity_analysis")
CORRELATION_THRESHOLD = 0.8
SCALE_RATIO_THRESHOLD = 10.0
VIF_BORDERLINE_LOW = 5.0
VIF_BORDERLINE_HIGH = 10.0
CONDITION_INDEX_THRESHOLD = 15.0
VARIANCE_PROPORTION_THRESHOLD = 0.5


def load_features(data_path: Path, target_column: str) -> pd.DataFrame:
    """Load complete numeric predictors while excluding the target column."""
    data = pd.read_csv(data_path)
    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {list(data.columns)}"
        )

    features = data.drop(columns=[target_column]).select_dtypes(include="number")
    if features.empty:
        raise ValueError("No numeric feature columns were found.")
    if features.isna().any().any():
        missing = features.isna().sum()
        raise ValueError(f"Missing feature values must be handled first:\n{missing[missing > 0]}")

    standard_deviations = features.std(ddof=0)
    constant_columns = standard_deviations[standard_deviations == 0].index.tolist()
    if constant_columns:
        raise ValueError(f"Constant feature columns cannot be analyzed: {constant_columns}")

    return features


def prepare_feature_scale(
    features: pd.DataFrame,
    scale_ratio_threshold: float,
) -> tuple[pd.DataFrame, float, bool]:
    """Standardize when the largest feature scale is much larger than the smallest."""
    standard_deviations = features.std(ddof=0)
    scale_ratio = float(standard_deviations.max() / standard_deviations.min())
    standardized = scale_ratio >= scale_ratio_threshold

    if standardized:
        prepared = (features - features.mean()) / standard_deviations
    else:
        prepared = features.copy()

    return prepared, scale_ratio, standardized


def find_high_correlation_pairs(
    correlation_matrix: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Return each unique feature pair whose absolute correlation meets the threshold."""
    records: list[dict[str, object]] = []
    columns = correlation_matrix.columns

    for left_index, left_feature in enumerate(columns):
        for right_feature in columns[left_index + 1 :]:
            correlation = float(correlation_matrix.loc[left_feature, right_feature])
            if abs(correlation) >= threshold:
                records.append(
                    {
                        "feature_1": left_feature,
                        "feature_2": right_feature,
                        "correlation": correlation,
                        "absolute_correlation": abs(correlation),
                    }
                )

    result = pd.DataFrame(
        records,
        columns=[
            "feature_1",
            "feature_2",
            "correlation",
            "absolute_correlation",
        ],
    )
    if not result.empty:
        result = result.sort_values("absolute_correlation", ascending=False)
    return result


def calculate_vif(features: pd.DataFrame) -> pd.DataFrame:
    """Calculate VIF with an explicit intercept as required by statsmodels."""
    design_matrix = sm.add_constant(features, has_constant="add")
    values = design_matrix.to_numpy(dtype=float)

    records = []
    for index, column in enumerate(design_matrix.columns):
        vif = float(variance_inflation_factor(values, index))
        if column == "const":
            status = "intercept"
        elif vif >= VIF_BORDERLINE_HIGH:
            status = "high"
        elif vif >= VIF_BORDERLINE_LOW:
            status = "borderline"
        else:
            status = "low"
        records.append({"variable": column, "vif": vif, "status": status})

    return pd.DataFrame(records)


def calculate_condition_diagnostics(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate condition indices and Belsley variance-decomposition proportions."""
    design_matrix = sm.add_constant(features, has_constant="add")
    values = design_matrix.to_numpy(dtype=float)

    # Unit-length column scaling prevents measurement units from dominating the SVD.
    column_norms = np.linalg.norm(values, axis=0)
    scaled_design = values / column_norms
    _, singular_values, right_singular_vectors_t = np.linalg.svd(
        scaled_design,
        full_matrices=False,
    )

    eigenvalues = singular_values**2
    epsilon = np.finfo(float).eps
    safe_eigenvalues = np.maximum(eigenvalues, epsilon)
    condition_indices = singular_values.max() / np.maximum(singular_values, epsilon)

    right_singular_vectors = right_singular_vectors_t.T
    variance_components = (
        right_singular_vectors**2 / safe_eigenvalues[np.newaxis, :]
    )
    variance_proportions = variance_components / variance_components.sum(
        axis=1,
        keepdims=True,
    )

    diagnostics = pd.DataFrame(
        {
            "dimension": np.arange(1, len(eigenvalues) + 1),
            "eigenvalue": eigenvalues,
            "condition_index": condition_indices,
        }
    )
    for variable_index, variable in enumerate(design_matrix.columns):
        diagnostics[f"variance_proportion_{variable}"] = variance_proportions[
            variable_index
        ]

    return diagnostics


def find_variance_clusters(
    diagnostics: pd.DataFrame,
    condition_threshold: float,
    proportion_threshold: float,
) -> pd.DataFrame:
    """Find dimensions where at least two features share large variance proportions."""
    proportion_columns = [
        column
        for column in diagnostics.columns
        if column.startswith("variance_proportion_")
        and column != "variance_proportion_const"
    ]
    records: list[dict[str, object]] = []

    for _, row in diagnostics.iterrows():
        if row["condition_index"] < condition_threshold:
            continue

        variables = [
            column.removeprefix("variance_proportion_")
            for column in proportion_columns
            if row[column] >= proportion_threshold
        ]
        if len(variables) >= 2:
            records.append(
                {
                    "dimension": int(row["dimension"]),
                    "condition_index": float(row["condition_index"]),
                    "variables_with_proportion_at_least_0.5": ", ".join(variables),
                    "variable_count": len(variables),
                }
            )

    return pd.DataFrame(
        records,
        columns=[
            "dimension",
            "condition_index",
            "variables_with_proportion_at_least_0.5",
            "variable_count",
        ],
    )


def save_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a labeled heatmap of the correlation matrix."""
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    labels = correlation_matrix.columns.tolist()
    axis.set_xticks(range(len(labels)), labels, rotation=45)
    axis.set_yticks(range(len(labels)), labels)

    for row in range(len(labels)):
        for column in range(len(labels)):
            value = correlation_matrix.iloc[row, column]
            color = "white" if abs(value) >= 0.6 else "black"
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", color=color)

    axis.set_title("Feature Correlation Matrix")
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def analyze(
    data_path: Path,
    target_column: str,
    output_dir: Path,
) -> None:
    """Run all three multicollinearity diagnostics and save their results."""
    features = load_features(data_path, target_column)
    prepared_features, scale_ratio, standardized = prepare_feature_scale(
        features,
        SCALE_RATIO_THRESHOLD,
    )

    correlation_matrix = prepared_features.corr()
    high_correlation_pairs = find_high_correlation_pairs(
        correlation_matrix,
        CORRELATION_THRESHOLD,
    )
    vif_results = calculate_vif(prepared_features)
    condition_diagnostics = calculate_condition_diagnostics(prepared_features)
    variance_clusters = find_variance_clusters(
        condition_diagnostics,
        CONDITION_INDEX_THRESHOLD,
        VARIANCE_PROPORTION_THRESHOLD,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    correlation_matrix.to_csv(output_dir / "correlation_matrix.csv")
    high_correlation_pairs.to_csv(
        output_dir / "high_correlation_pairs.csv",
        index=False,
    )
    vif_results.to_csv(output_dir / "vif_results.csv", index=False)
    condition_diagnostics.to_csv(
        output_dir / "condition_diagnostics.csv",
        index=False,
    )
    variance_clusters.to_csv(output_dir / "variance_clusters.csv", index=False)
    save_correlation_heatmap(
        correlation_matrix,
        output_dir / "correlation_heatmap.png",
    )

    borderline_variables = vif_results.loc[
        vif_results["status"] == "borderline",
        "variable",
    ].tolist()

    print(f"Data: {data_path}")
    print(f"Rows: {len(features)}, numeric features: {features.shape[1]}")
    print(f"Feature scale ratio (largest std / smallest std): {scale_ratio:.3f}")
    print(f"Standardization applied: {standardized}")
    print(f"\nHigh-correlation pairs (|r| >= {CORRELATION_THRESHOLD}):")
    print(
        high_correlation_pairs.to_string(index=False)
        if not high_correlation_pairs.empty
        else "None"
    )
    print("\nVIF results (intercept included):")
    print(vif_results.to_string(index=False))
    print(f"\nBorderline VIF variables (5 <= VIF < 10): {borderline_variables or 'None'}")
    print(
        f"\nVariance clusters (condition index >= {CONDITION_INDEX_THRESHOLD}, "
        f"shared proportion >= {VARIANCE_PROPORTION_THRESHOLD}):"
    )
    print(variance_clusters.to_string(index=False) if not variance_clusters.empty else "None")
    print(f"\nAnalysis files saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze multicollinearity using correlation, VIF, and SVD diagnostics."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--target", default="y")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    analyze(arguments.data, arguments.target, arguments.output_dir)
