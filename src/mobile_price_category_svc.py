from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_TRAIN_PATH = DATA_DIR / "mobile_price_category_train.csv"
DEFAULT_TEST_PATH = DATA_DIR / "mobile_price_category_test.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mobile_price_category_svc"

TARGET = "price_range"
ID_COLUMN = "id"
EXPECTED_CLASSES = [0, 1, 2, 3]
SUSPECT_ZERO_COLUMNS = ["sc_w", "px_height"]

BINARY_COLUMNS = [
    "blue",
    "dual_sim",
    "four_g",
    "three_g",
    "touch_screen",
    "wifi",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mobile price category 데이터의 SVC 전처리, nested CV 평가 및 최종 예측"
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="코드 확인용 축소 탐색을 사용합니다. 최종 비교에는 기본 모드를 사용하세요.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def configure_plotting() -> None:
    available_fonts = {item.name for item in matplotlib.font_manager.fontManager.ttflist}
    font = "Malgun Gothic" if "Malgun Gothic" in available_fonts else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": font,
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 160,
        }
    )
    sns.set_theme(style="whitegrid", font=font)


def load_and_validate(
    train_path: Path,
    test_path: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    missing_files = [path.resolve() for path in (train_path, test_path) if not path.is_file()]
    if missing_files:
        missing_text = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(f"다음 데이터 파일을 찾을 수 없습니다.\n{missing_text}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if TARGET not in train.columns:
        raise ValueError(f"학습 데이터에 타깃 {TARGET!r}가 없습니다.")
    if TARGET in test.columns:
        raise ValueError("테스트 데이터에 타깃이 포함되어 있습니다. 누수 여부를 확인하세요.")
    if ID_COLUMN not in test.columns:
        raise ValueError(f"테스트 데이터에 식별자 {ID_COLUMN!r}가 없습니다.")
    if test[ID_COLUMN].isna().any() or test[ID_COLUMN].duplicated().any():
        raise ValueError("테스트 id에는 결측값이나 중복값이 없어야 합니다.")

    feature_columns = [column for column in train.columns if column != TARGET]
    test_feature_columns = [column for column in test.columns if column != ID_COLUMN]
    if feature_columns != test_feature_columns:
        raise ValueError(
            "학습/테스트 피처 스키마 또는 순서가 다릅니다.\n"
            f"train={feature_columns}\ntest={test_feature_columns}"
        )
    if train[feature_columns].isna().any().any() or test[feature_columns].isna().any().any():
        raise ValueError("원본 데이터에 예상하지 못한 결측값이 있습니다.")
    if not set(BINARY_COLUMNS + SUSPECT_ZERO_COLUMNS).issubset(feature_columns):
        raise ValueError("예상한 이진 피처 또는 0값 점검 피처가 누락되었습니다.")
    if any(not pd.api.types.is_numeric_dtype(train[column]) for column in feature_columns):
        raise TypeError("모든 예측 피처는 수치형이어야 합니다.")
    if not train[TARGET].isin(EXPECTED_CLASSES).all():
        raise ValueError(f"타깃은 {EXPECTED_CLASSES}만 포함해야 합니다.")

    X = train[feature_columns].copy()
    y = train[TARGET].astype(int).copy()
    X_test = test[feature_columns].copy()
    test_ids = test[ID_COLUMN].copy()
    return X, y, X_test, test_ids, feature_columns


def build_pipeline(feature_columns: list[str], random_state: int) -> Pipeline:
    numeric_columns = [
        column
        for column in feature_columns
        if column not in BINARY_COLUMNS + SUSPECT_ZERO_COLUMNS
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    suspect_zero_pipeline = Pipeline(
        steps=[
            # missing_values=np.nan이면 원본 0을 유지한다. GridSearchCV가 0도 비교한다.
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("suspect_numeric", suspect_zero_pipeline, SUSPECT_ZERO_COLUMNS),
            ("binary", "passthrough", BINARY_COLUMNS),
            (
                "zero_indicators",
                FunctionTransformer(np.logical_not, validate=False),
                SUSPECT_ZERO_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "svc",
                SVC(
                    class_weight=None,
                    cache_size=1024,
                    decision_function_shape="ovr",
                    random_state=random_state,
                ),
            ),
        ]
    )


def parameter_grid(quick: bool) -> list[dict[str, list[Any]]]:
    if quick:
        linear_c = [1.0, 10.0]
        rbf_c = [1.0, 10.0]
        rbf_gamma: list[Any] = ["scale", 0.05]
    else:
        linear_c = [0.1, 1.0, 10.0, 100.0]
        rbf_c = [1.0, 10.0, 100.0]
        rbf_gamma = ["scale", 0.01, 0.05, 0.1]

    # np.nan: 원본 0 유지, 0: 0을 결측으로 간주해 fold 내부 중앙값으로 대체
    zero_missing_value_options: list[Any] = [np.nan, 0]
    return [
        {
            "preprocessor__suspect_numeric__imputer__missing_values": zero_missing_value_options,
            "svc__kernel": ["linear"],
            "svc__C": linear_c,
        },
        {
            "preprocessor__suspect_numeric__imputer__missing_values": zero_missing_value_options,
            "svc__kernel": ["rbf"],
            "svc__C": rbf_c,
            "svc__gamma": rbf_gamma,
        },
    ]


def replaces_suspect_zeros(params: dict[str, Any]) -> bool:
    value = params["preprocessor__suspect_numeric__imputer__missing_values"]
    return bool(not pd.isna(value) and value == 0)


def make_cv(random_state: int, quick: bool) -> tuple[StratifiedKFold, StratifiedKFold]:
    outer_splits = 3 if quick else 5
    inner_splits = 3 if quick else 4
    outer_cv = StratifiedKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=random_state,
    )
    inner_cv = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=random_state + 1,
    )
    return outer_cv, inner_cv


def nested_cv_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
    params: list[dict[str, list[Any]]],
    outer_cv: StratifiedKFold,
    inner_cv: StratifiedKFold,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    oof_predictions = np.full(len(y), fill_value=-1, dtype=int)
    fold_rows: list[dict[str, Any]] = []

    for fold, (train_index, valid_index) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        search = GridSearchCV(
            estimator=clone(pipeline),
            param_grid=params,
            scoring="f1_macro",
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        predictions = search.predict(X_valid).astype(int)
        oof_predictions[valid_index] = predictions

        fold_rows.append(
            {
                "fold": fold,
                "train_size": len(train_index),
                "valid_size": len(valid_index),
                "accuracy": accuracy_score(y_valid, predictions),
                "macro_f1": f1_score(y_valid, predictions, average="macro"),
                "ordinal_mae": mean_absolute_error(y_valid, predictions),
                "quadratic_kappa": cohen_kappa_score(
                    y_valid,
                    predictions,
                    weights="quadratic",
                ),
                "inner_best_macro_f1": search.best_score_,
                "best_kernel": search.best_params_["svc__kernel"],
                "best_C": search.best_params_["svc__C"],
                "best_gamma": search.best_params_.get("svc__gamma", "n.a."),
                "replace_suspect_zeros": replaces_suspect_zeros(search.best_params_),
                "best_params_json": json.dumps(search.best_params_, ensure_ascii=False),
            }
        )
        print(
            f"Outer fold {fold}/{outer_cv.n_splits}: "
            f"accuracy={fold_rows[-1]['accuracy']:.4f}, "
            f"macro_f1={fold_rows[-1]['macro_f1']:.4f}, "
            f"MAE={fold_rows[-1]['ordinal_mae']:.4f}, "
            f"params={search.best_params_}"
        )

    if (oof_predictions < 0).any():
        raise RuntimeError("일부 표본의 out-of-fold 예측이 생성되지 않았습니다.")

    fold_results = pd.DataFrame(fold_rows)
    oof_results = pd.DataFrame(
        {
            "row_index": X.index,
            "actual_price_range": y.to_numpy(),
            "oof_predicted_price_range": oof_predictions,
            "absolute_class_error": np.abs(y.to_numpy() - oof_predictions),
        }
    )
    return fold_results, oof_results, oof_predictions


def fit_final_search(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
    params: list[dict[str, list[Any]]],
    random_state: int,
    quick: bool,
    n_jobs: int,
) -> GridSearchCV:
    final_splits = 3 if quick else 5
    final_cv = StratifiedKFold(
        n_splits=final_splits,
        shuffle=True,
        random_state=random_state + 2,
    )
    search = GridSearchCV(
        estimator=clone(pipeline),
        param_grid=params,
        scoring={
            "accuracy": "accuracy",
            "macro_f1": "f1_macro",
        },
        refit="macro_f1",
        cv=final_cv,
        n_jobs=n_jobs,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(X, y)
    return search


def cv_results_table(search: GridSearchCV) -> pd.DataFrame:
    raw = pd.DataFrame(search.cv_results_)
    selected = pd.DataFrame(
        {
            "rank_macro_f1": raw["rank_test_macro_f1"],
            "mean_test_macro_f1": raw["mean_test_macro_f1"],
            "std_test_macro_f1": raw["std_test_macro_f1"],
            "mean_test_accuracy": raw["mean_test_accuracy"],
            "std_test_accuracy": raw["std_test_accuracy"],
            "mean_train_macro_f1": raw["mean_train_macro_f1"],
            "generalization_gap": raw["mean_train_macro_f1"] - raw["mean_test_macro_f1"],
            "kernel": raw["param_svc__kernel"].astype(str),
            "C": raw["param_svc__C"].astype(float),
            "gamma": raw["param_svc__gamma"].map(
                lambda value: "n.a." if pd.isna(value) else str(value)
            ),
            "replace_suspect_zeros": raw[
                "param_preprocessor__suspect_numeric__imputer__missing_values"
            ].map(lambda value: bool(not pd.isna(value) and value == 0)),
            "mean_fit_time_seconds": raw["mean_fit_time"],
        }
    )
    return selected.sort_values("rank_macro_f1").reset_index(drop=True)


def save_confusion_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_dir: Path,
) -> Path:
    path = output_dir / "nested_cv_confusion_matrix.png"
    count_matrix = confusion_matrix(y_true, y_pred, labels=EXPECTED_CLASSES)
    normalized_matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=EXPECTED_CLASSES,
        normalize="true",
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sns.heatmap(
        count_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=EXPECTED_CLASSES,
        yticklabels=EXPECTED_CLASSES,
        ax=axes[0],
    )
    axes[0].set_title("Nested CV out-of-fold confusion matrix")
    axes[0].set_xlabel("Predicted price range")
    axes[0].set_ylabel("Actual price range")

    sns.heatmap(
        normalized_matrix,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=EXPECTED_CLASSES,
        yticklabels=EXPECTED_CLASSES,
        ax=axes[1],
    )
    axes[1].set_title("Row-normalized confusion matrix")
    axes[1].set_xlabel("Predicted price range")
    axes[1].set_ylabel("Actual price range")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_fold_metrics_plot(fold_results: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "nested_cv_fold_metrics.png"
    metrics = fold_results.melt(
        id_vars="fold",
        value_vars=["accuracy", "macro_f1", "quadratic_kappa"],
        var_name="metric",
        value_name="score",
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.barplot(data=metrics, x="fold", y="score", hue="metric", ax=axes[0])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Nested CV metrics by outer fold")
    axes[0].set_ylabel("Score")
    axes[0].legend(title="Metric")

    sns.barplot(
        data=fold_results,
        x="fold",
        y="ordinal_mae",
        color="#F59E0B",
        ax=axes[1],
    )
    axes[1].set_title("Ordinal class MAE by outer fold")
    axes[1].set_ylabel("Mean absolute class error")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.{float_digits}f}")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    fold_results: pd.DataFrame,
    oof_predictions: np.ndarray,
    final_search: GridSearchCV,
    final_results: pd.DataFrame,
    quick: bool,
) -> Path:
    path = output_dir / "svc_evaluation_report.md"
    oof_accuracy = accuracy_score(y, oof_predictions)
    oof_macro_f1 = f1_score(y, oof_predictions, average="macro")
    oof_mae = mean_absolute_error(y, oof_predictions)
    oof_kappa = cohen_kappa_score(y, oof_predictions, weights="quadratic")
    report_dict = classification_report(
        y,
        oof_predictions,
        labels=EXPECTED_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    class_table = pd.DataFrame(
        [
            {
                "class": class_value,
                "precision": report_dict[str(class_value)]["precision"],
                "recall": report_dict[str(class_value)]["recall"],
                "f1": report_dict[str(class_value)]["f1-score"],
                "support": int(report_dict[str(class_value)]["support"]),
            }
            for class_value in EXPECTED_CLASSES
        ]
    )
    best_kernel_counts = Counter(fold_results["best_kernel"])
    zero_choice_counts = Counter(fold_results["replace_suspect_zeros"])
    top_results = final_results.head(10)[
        [
            "rank_macro_f1",
            "mean_test_macro_f1",
            "std_test_macro_f1",
            "mean_test_accuracy",
            "generalization_gap",
            "kernel",
            "C",
            "gamma",
            "replace_suspect_zeros",
        ]
    ]
    mode_text = "quick smoke-test" if quick else "full nested CV"
    best_params_summary = {
        "kernel": final_search.best_params_["svc__kernel"],
        "C": final_search.best_params_["svc__C"],
        "gamma": final_search.best_params_.get("svc__gamma", "n.a."),
        "replace_suspect_zeros": replaces_suspect_zeros(final_search.best_params_),
    }

    report = f"""# Mobile Price Category SVC 모델 평가

## 실행 범위

- 실행 모드: **{mode_text}**
- 학습 데이터: {len(X):,}행 × {X.shape[1]}개 원본 피처
- 테스트 데이터: {len(X_test):,}행 × {X_test.shape[1]}개 원본 피처
- 전처리: 의심 0값 지시 변수 추가, 수치형 중앙값 대체 + StandardScaler, 이진형 통과
- 모델 선택: Linear/RBF SVC와 `C`, `gamma`, 의심 0값 보정 여부를 내부 CV에서 선택
- 평가: 바깥쪽 StratifiedKFold의 out-of-fold 예측. 테스트셋 라벨은 사용하지 않음

## Nested CV 평가

| 지표 | 값 |
| --- | ---: |
| OOF Accuracy | {oof_accuracy:.4f} |
| OOF Macro F1 | {oof_macro_f1:.4f} |
| OOF ordinal MAE | {oof_mae:.4f} |
| OOF quadratic weighted kappa | {oof_kappa:.4f} |

OOF 평가는 각 행이 해당 행을 학습에 사용하지 않은 외부 fold 모델에서 예측되도록 구성했습니다. 내부 fold는 하이퍼파라미터 선택에만 사용했습니다.

### 외부 fold별 결과

{markdown_table(fold_results[["fold", "accuracy", "macro_f1", "ordinal_mae", "quadratic_kappa", "best_kernel", "best_C", "best_gamma", "replace_suspect_zeros"]])}

- 외부 fold에서 선택된 커널: {dict(best_kernel_counts)}
- 의심 0값을 결측으로 보정한 선택 횟수: {dict(zero_choice_counts)}

![Fold metrics](nested_cv_fold_metrics.png)

### 클래스별 성능

{markdown_table(class_table)}

![Confusion matrix](nested_cv_confusion_matrix.png)

## 전체 학습 데이터에서 선택한 최종 모델

- 최적 파라미터: `{json.dumps(best_params_summary, ensure_ascii=False)}`
- 내부 CV 최적 Macro F1: {final_search.best_score_:.4f}

최종 모델은 전체 학습 데이터를 사용해 다시 적합했으며 `svc_model.joblib`로 저장했습니다. 테스트 예측은 `svc_test_predictions.csv`에 저장했습니다.

### 상위 파라미터 조합

{markdown_table(top_results)}

## 결과 해석 시 주의사항

- `price_range`는 순서형이지만 일반 SVC는 명목형 다중분류로 학습합니다. 그래서 Accuracy/Macro F1과 함께 ordinal MAE와 quadratic weighted kappa를 제시했습니다.
- 클래스가 정확히 균형이므로 SMOTE와 자동 class weight를 사용하지 않았습니다.
- `sc_w=0`, `px_height=0` 처리 선택은 각 내부 CV의 학습 fold에서만 결정됩니다. 중앙값도 fold 내부에서 계산됩니다.
- 테스트셋은 최종 예측에만 사용했으며, 테스트 라벨이 없으므로 테스트 성능이라고 보고할 수 없습니다.
- `--quick` 결과는 코드 검증용입니다. 최종 모델 비교에는 기본 모드 결과를 사용해야 합니다.
"""
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    X, y, X_test, test_ids, feature_columns = load_and_validate(
        args.train.resolve(),
        args.test.resolve(),
    )
    pipeline = build_pipeline(feature_columns, args.random_state)
    params = parameter_grid(args.quick)
    outer_cv, inner_cv = make_cv(args.random_state, args.quick)

    print("1/3 Nested cross-validation evaluation")
    fold_results, oof_results, oof_predictions = nested_cv_evaluation(
        X=X,
        y=y,
        pipeline=pipeline,
        params=params,
        outer_cv=outer_cv,
        inner_cv=inner_cv,
        n_jobs=args.n_jobs,
    )

    print("2/3 Final hyperparameter search on all training rows")
    final_search = fit_final_search(
        X=X,
        y=y,
        pipeline=pipeline,
        params=params,
        random_state=args.random_state,
        quick=args.quick,
        n_jobs=args.n_jobs,
    )
    final_results = cv_results_table(final_search)

    print("3/3 Saving evaluation artifacts and test predictions")
    test_predictions = final_search.best_estimator_.predict(X_test).astype(int)
    prediction_table = pd.DataFrame(
        {
            ID_COLUMN: test_ids.to_numpy(),
            TARGET: test_predictions,
        }
    )
    if not prediction_table[TARGET].isin(EXPECTED_CLASSES).all():
        raise RuntimeError("허용되지 않은 테스트 예측 클래스가 생성되었습니다.")

    fold_results.to_csv(output_dir / "nested_cv_fold_results.csv", index=False, encoding="utf-8-sig")
    oof_results.to_csv(output_dir / "nested_cv_oof_predictions.csv", index=False, encoding="utf-8-sig")
    final_results.to_csv(output_dir / "final_grid_search_results.csv", index=False, encoding="utf-8-sig")
    prediction_table.to_csv(output_dir / "svc_test_predictions.csv", index=False, encoding="utf-8-sig")
    joblib.dump(final_search.best_estimator_, output_dir / "svc_model.joblib")

    save_confusion_plot(y, oof_predictions, output_dir)
    save_fold_metrics_plot(fold_results, output_dir)
    report_path = write_report(
        output_dir=output_dir,
        X=X,
        y=y,
        X_test=X_test,
        fold_results=fold_results,
        oof_predictions=oof_predictions,
        final_search=final_search,
        final_results=final_results,
        quick=args.quick,
    )

    print("\nCompleted")
    print(f"Nested CV accuracy : {accuracy_score(y, oof_predictions):.4f}")
    print(f"Nested CV macro F1 : {f1_score(y, oof_predictions, average='macro'):.4f}")
    print(f"Nested CV class MAE: {mean_absolute_error(y, oof_predictions):.4f}")
    print(f"Final parameters   : {final_search.best_params_}")
    print(f"Report             : {report_path}")
    print(f"Predictions        : {output_dir / 'svc_test_predictions.csv'}")


if __name__ == "__main__":
    main()
