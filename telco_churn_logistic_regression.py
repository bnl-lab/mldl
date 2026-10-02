from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("output/telco_churn_featured.csv")
OUTPUT_DIR = Path("output/telco_churn_logistic_regression")
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
RANDOM_SEED = 42

# EDA에서 TotalCharges의 VIF가 9.51로 확인되었다. 동일 정보를 비율형 파생 변수
# Avg_Monthly_Ratio로 일부 보존하면서 원 변수는 선형 모델에서 제외한다.
EDA_EXCLUDED_FEATURES = ["TotalCharges"]


def configure_visualization() -> None:
    """Configure plotting, including Korean font support."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    sns.set_theme(style="whitegrid", font="Malgun Gothic")


def print_section(title: str) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")


def load_modeling_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the EDA-produced modeling dataset."""
    data = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"'{TARGET_COLUMN}' 타깃 컬럼이 없습니다.")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("타깃에 결측치가 있습니다.")

    unexpected_targets = set(data[TARGET_COLUMN].unique()).difference({"Yes", "No"})
    if unexpected_targets:
        raise ValueError(f"알 수 없는 타깃 값입니다: {unexpected_targets}")

    identifiers = data[ID_COLUMN].copy() if ID_COLUMN in data.columns else None
    features = data.drop(
        columns=[TARGET_COLUMN, ID_COLUMN, *EDA_EXCLUDED_FEATURES],
        errors="ignore",
    )
    targets = data[TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype(int)

    if identifiers is not None:
        print("customerID는 추론 결과 연결용으로만 보관하고 모델 입력에서는 제외합니다.")
    if features.isna().sum().sum() > 0:
        print("입력 결측치는 Pipeline 내부 Imputer가 학습 세트 통계로만 처리합니다.")

    return features, targets


def split_data(
    features: pd.DataFrame,
    targets: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Create stratified 60/20/20 train, validation, and test partitions."""
    x_development, x_test, y_development, y_test = train_test_split(
        features,
        targets,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=targets,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_development,
        y_development,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y_development,
    )
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def make_pipeline(features: pd.DataFrame) -> Pipeline:
    """Create a leakage-safe preprocessing and logistic-regression pipeline."""
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
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2_000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def choose_f1_threshold(
    actual: pd.Series,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """Choose a decision threshold using validation data only."""
    precision, recall, thresholds = precision_recall_curve(actual, probabilities)
    f1_values = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    eligible = (thresholds >= 0.10) & (thresholds <= 0.90)
    if not eligible.any():
        best_index = int(np.argmax(f1_values))
    else:
        eligible_indices = np.flatnonzero(eligible)
        best_index = int(eligible_indices[np.argmax(f1_values[eligible])])
    return float(thresholds[best_index]), float(f1_values[best_index])


def calculate_metrics(
    actual: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predicted = (probabilities >= threshold).astype(int)
    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(actual, predicted),
        "Balanced_Accuracy": balanced_accuracy_score(actual, predicted),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "F1": f1_score(actual, predicted, zero_division=0),
        "ROC_AUC": roc_auc_score(actual, probabilities),
        "PR_AUC": average_precision_score(actual, probabilities),
        "Log_Loss": log_loss(actual, probabilities),
        "Predicted_Churn_Rate": predicted.mean(),
    }


def extract_coefficients(model: Pipeline) -> pd.DataFrame:
    """Return model coefficients and odds ratios for interpretation."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    clean_names = [
        name.replace("numeric__", "").replace("categorical__", "")
        for name in feature_names
    ]
    coefficients = classifier.coef_[0]
    coefficient_data = pd.DataFrame(
        {
            "Feature": clean_names,
            "Coefficient": coefficients,
            "Odds_Ratio": np.exp(coefficients),
            "Absolute_Coefficient": np.abs(coefficients),
        }
    )
    return coefficient_data.sort_values("Absolute_Coefficient", ascending=False)


def save_test_predictions(
    actual: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    predictions = (probabilities >= threshold).astype(int)
    result = pd.DataFrame(
        {
            "Actual": actual,
            "Actual_Label": actual.map({0: "No", 1: "Yes"}),
            "Churn_Probability": probabilities,
            "Predicted": predictions,
            "Predicted_Label": np.where(predictions == 1, "Yes", "No"),
            "Threshold": threshold,
        },
        index=actual.index,
    ).sort_index()
    result.index.name = "Original_Row_Index"
    result.to_csv(output_path)


def save_evaluation_plot(
    actual: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    coefficients: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save confusion matrix, ROC/PR curves, and top coefficients in a grid."""
    predicted = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(actual, predicted)
    false_positive_rate, true_positive_rate, _ = roc_curve(actual, probabilities)
    precision, recall, _ = precision_recall_curve(actual, probabilities)
    roc_auc = roc_auc_score(actual, probabilities)
    pr_auc = average_precision_score(actual, probabilities)

    top_coefficients = (
        coefficients.head(20)
        .sort_values("Coefficient", ascending=True)
        .copy()
    )
    colors = np.where(top_coefficients["Coefficient"] >= 0, "#C44E52", "#4C72B0")

    figure, axes = plt.subplots(2, 2, figsize=(18, 14))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["유지(No)", "이탈(Yes)"],
        yticklabels=["유지(No)", "이탈(Yes)"],
        ax=axes[0, 0],
    )
    axes[0, 0].set_title(f"테스트 혼동행렬 (임계값={threshold:.3f})")
    axes[0, 0].set_xlabel("예측")
    axes[0, 0].set_ylabel("실제")

    axes[0, 1].plot(false_positive_rate, true_positive_rate, linewidth=2.5)
    axes[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0, 1].set_title(f"ROC Curve (AUC={roc_auc:.3f})")
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")

    baseline = float(actual.mean())
    axes[1, 0].plot(recall, precision, linewidth=2.5)
    axes[1, 0].axhline(
        baseline,
        color="gray",
        linestyle="--",
        label=f"무작위 기준={baseline:.3f}",
    )
    axes[1, 0].set_title(f"Precision-Recall Curve (AP={pr_auc:.3f})")
    axes[1, 0].set_xlabel("Recall")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].legend()

    axes[1, 1].barh(
        top_coefficients["Feature"],
        top_coefficients["Coefficient"],
        color=colors,
    )
    axes[1, 1].axvline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("절댓값 기준 상위 20개 로지스틱 회귀계수")
    axes[1, 1].set_xlabel("표준화 로그 오즈 계수")
    axes[1, 1].set_ylabel("")
    axes[1, 1].grid(axis="x", linestyle=":", alpha=0.5)

    figure.suptitle("Telco Churn 로지스틱 회귀 테스트 평가", fontsize=18, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def print_split_summary(
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
) -> None:
    print_section("1. 데이터 분할 및 누수 방지 확인")
    split_rows = []
    for name, values in [
        ("Train", y_train),
        ("Validation", y_validation),
        ("Test", y_test),
    ]:
        split_rows.append(
            {
                "Split": name,
                "Rows": len(values),
                "Churn_Count": int(values.sum()),
                "Churn_Rate": values.mean(),
            }
        )
    summary = pd.DataFrame(split_rows)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        "전처리 통계(Imputer, StandardScaler, OneHotEncoder)는 Pipeline 내부에서 "
        "각 학습 데이터에만 fit됩니다."
    )
    print(
        "EDA 권고 반영: TotalCharges 제외, Avg_Monthly_Ratio·Tenure_Group·"
        "Service_Count 유지, class_weight='balanced' 적용"
    )


def print_model_insights(
    metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
    actual: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
    positive = coefficients.nlargest(5, "Coefficient")
    negative = coefficients.nsmallest(5, "Coefficient")

    print_section("4. 테스트 평가 및 비즈니스 해석")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\n혼동행렬: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")
    print(
        f"튜닝 임계값 {threshold:.3f}에서 실제 이탈 고객 {tp + fn:,}명 중 "
        f"{tp:,}명을 탐지하고 {fn:,}명을 놓쳤습니다."
    )
    print("\n[이탈 가능성을 높이는 상위 계수]")
    print(
        positive[["Feature", "Coefficient", "Odds_Ratio"]].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print("\n[이탈 가능성을 낮추는 상위 계수]")
    print(
        negative[["Feature", "Coefficient", "Odds_Ratio"]].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print(
        "\n주의: 회귀계수는 다른 변수가 동일하다는 조건의 연관성이며 인과효과가 아닙니다. "
        "수치형 변수 계수는 1표준편차 변화 기준이고, 원-핫 변수는 기준 범주 대비 값입니다."
    )


def main() -> None:
    configure_visualization()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_section("Telco Customer Churn 로지스틱 회귀")
    features, targets = load_modeling_data()
    print(f"입력 데이터: {DATA_PATH}")
    print(f"전체 크기  : {len(features):,}행 × {features.shape[1]:,}개 모델 피처")
    print(f"전체 이탈률: {targets.mean():.2%}")

    x_train, x_validation, x_test, y_train, y_validation, y_test = split_data(
        features,
        targets,
    )
    print_split_summary(y_train, y_validation, y_test)

    print_section("2. 검증 세트를 이용한 분류 임계값 선택")
    validation_model = make_pipeline(x_train)
    validation_model.fit(x_train, y_train)
    validation_probabilities = validation_model.predict_proba(x_validation)[:, 1]
    tuned_threshold, validation_f1 = choose_f1_threshold(
        y_validation,
        validation_probabilities,
    )
    print(f"기본 임계값       : 0.500")
    print(f"F1 최적 검증 임계값: {tuned_threshold:.4f}")
    print(f"검증 F1           : {validation_f1:.4f}")
    print("테스트 데이터는 임계값 선택에 사용하지 않았습니다.")

    print_section("3. 최종 모델 학습")
    x_development = pd.concat([x_train, x_validation], axis=0)
    y_development = pd.concat([y_train, y_validation], axis=0)
    final_model = make_pipeline(x_development)
    final_model.fit(x_development, y_development)
    test_probabilities = final_model.predict_proba(x_test)[:, 1]
    print(f"최종 학습 데이터: {len(x_development):,}행")
    print(f"최종 테스트 데이터: {len(x_test):,}행")

    metric_rows = []
    for name, threshold in [("Default_0.5", 0.5), ("Validation_F1_Tuned", tuned_threshold)]:
        metric_rows.append(
            {
                "Evaluation": name,
                **calculate_metrics(y_test, test_probabilities, threshold),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    coefficients = extract_coefficients(final_model)

    metrics_path = OUTPUT_DIR / "test_metrics.csv"
    predictions_path = OUTPUT_DIR / "test_predictions.csv"
    coefficients_path = OUTPUT_DIR / "logistic_coefficients.csv"
    plot_path = OUTPUT_DIR / "logistic_regression_evaluation.png"
    model_path = OUTPUT_DIR / "telco_churn_logistic_pipeline.joblib"

    metrics.to_csv(metrics_path, index=False)
    coefficients.to_csv(coefficients_path, index=False)
    save_test_predictions(
        y_test,
        test_probabilities,
        tuned_threshold,
        predictions_path,
    )
    save_evaluation_plot(
        y_test,
        test_probabilities,
        tuned_threshold,
        coefficients,
        plot_path,
    )
    joblib.dump(
        {
            "pipeline": final_model,
            "decision_threshold": tuned_threshold,
            "excluded_features": EDA_EXCLUDED_FEATURES,
            "target_mapping": {"No": 0, "Yes": 1},
        },
        model_path,
    )

    print_model_insights(
        metrics,
        coefficients,
        y_test,
        test_probabilities,
        tuned_threshold,
    )

    print_section("5. 저장된 산출물")
    print(f"평가 지표     : {metrics_path}")
    print(f"테스트 예측   : {predictions_path}")
    print(f"전체 회귀계수 : {coefficients_path}")
    print(f"평가 그래프   : {plot_path}")
    print(f"재사용 모델   : {model_path}")


if __name__ == "__main__":
    main()
