from pathlib import Path

import graphviz
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


RANDOM_SEED = 42
TEST_SIZE = 0.2
OUTPUT_DIR = Path("outputs/wine_decision_tree")


def configure_visualization() -> None:
    """Configure consistent plotting with Korean font support."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="Malgun Gothic")


def load_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the scikit-learn Wine dataset as pandas objects."""
    wine = load_wine()
    features = pd.DataFrame(wine.data, columns=wine.feature_names)
    targets = pd.Series(wine.target, name="target")
    class_names = [f"Class_{class_id}" for class_id in sorted(targets.unique())]
    return features, targets, class_names


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """Select tree complexity using stratified cross-validation."""
    model = DecisionTreeClassifier(random_state=RANDOM_SEED)
    parameter_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 3, 4, 5, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=cross_validation,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    search.fit(x_train, y_train)
    return search


def calculate_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Calculate multiclass classification metrics."""
    metrics = {
        "Accuracy": accuracy_score(actual, predicted),
        "Balanced_Accuracy": balanced_accuracy_score(actual, predicted),
        "Macro_Precision": precision_score(
            actual,
            predicted,
            average="macro",
            zero_division=0,
        ),
        "Macro_Recall": recall_score(
            actual,
            predicted,
            average="macro",
            zero_division=0,
        ),
        "Macro_F1": f1_score(
            actual,
            predicted,
            average="macro",
            zero_division=0,
        ),
        "Macro_ROC_AUC_OVR": roc_auc_score(
            actual,
            probabilities,
            multi_class="ovr",
            average="macro",
        ),
        "Cross_Entropy": log_loss(actual, probabilities),
    }
    return pd.DataFrame([metrics])


def make_feature_importance_table(
    model: DecisionTreeClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    """Create a descending impurity-based feature importance table."""
    importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    )
    return importance.sort_values("Importance", ascending=False).reset_index(drop=True)


def save_graphviz_tree(
    model: DecisionTreeClassifier,
    feature_names: list[str],
    class_names: list[str],
) -> tuple[Path, Path]:
    """Export the fitted decision tree to DOT and render it to PNG."""
    dot_text = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        proportion=False,
        precision=3,
    )
    source = graphviz.Source(dot_text)
    dot_path = Path(
        source.save(
            filename="wine_decision_tree.dot",
            directory=str(OUTPUT_DIR),
        )
    )
    png_path = Path(
        source.render(
            filename="wine_decision_tree",
            directory=str(OUTPUT_DIR),
            format="png",
            cleanup=True,
        )
    )
    return dot_path, png_path


def save_evaluation_plot(
    actual: pd.Series,
    predicted: np.ndarray,
    importance: pd.DataFrame,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Save confusion matrix and feature importance plots in a grid."""
    matrix = confusion_matrix(actual, predicted)
    importance_for_plot = importance.sort_values("Importance", ascending=True)

    figure, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_title("테스트 데이터 혼동행렬")
    axes[0].set_xlabel("예측 클래스")
    axes[0].set_ylabel("실제 클래스")

    sns.barplot(
        data=importance_for_plot,
        x="Importance",
        y="Feature",
        color="#4C72B0",
        ax=axes[1],
    )
    axes[1].set_title("Decision Tree 피처 중요도")
    axes[1].set_xlabel("불순도 기반 중요도")
    axes[1].set_ylabel("")
    axes[1].grid(axis="x", linestyle=":", alpha=0.5)

    figure.suptitle("Wine Decision Tree 모델 평가", fontsize=18, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_predictions(
    actual: pd.Series,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Save test labels, predicted labels, and class probabilities."""
    result = pd.DataFrame(
        {
            "Actual": actual.to_numpy(),
            "Predicted": predicted,
            "ActualName": [class_names[value] for value in actual],
            "PredictedName": [class_names[value] for value in predicted],
        },
        index=actual.index,
    )
    for class_id, class_name in enumerate(class_names):
        result[f"Probability_{class_name}"] = probabilities[:, class_id]
    result.sort_index().to_csv(output_path, index_label="SampleIndex")


def main() -> None:
    configure_visualization()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features, targets, class_names = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=targets,
    )

    print("=" * 80)
    print("Wine Decision Tree 학습")
    print("=" * 80)
    print(f"전체 데이터: {len(features)}개 샘플, {features.shape[1]}개 피처")
    print(f"학습/테스트: {len(x_train)}/{len(x_test)}")
    print("피처 스케일링: 적용하지 않음 (Decision Tree는 스케일에 영향을 받지 않음)")

    search = train_model(x_train, y_train)
    best_model = search.best_estimator_
    predicted = best_model.predict(x_test)
    probabilities = best_model.predict_proba(x_test)

    metrics = calculate_metrics(y_test, predicted, probabilities)
    report = classification_report(
        y_test,
        predicted,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    importance = make_feature_importance_table(
        best_model,
        features.columns.tolist(),
    )

    metrics_path = OUTPUT_DIR / "test_metrics.csv"
    report_path = OUTPUT_DIR / "classification_report.csv"
    predictions_path = OUTPUT_DIR / "test_predictions.csv"
    importance_path = OUTPUT_DIR / "feature_importance.csv"
    evaluation_plot_path = OUTPUT_DIR / "evaluation_and_feature_importance.png"
    model_path = OUTPUT_DIR / "wine_decision_tree.joblib"

    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(report).transpose().to_csv(report_path)
    importance.to_csv(importance_path, index=False)
    save_predictions(
        y_test,
        predicted,
        probabilities,
        class_names,
        predictions_path,
    )
    save_evaluation_plot(
        y_test,
        predicted,
        importance,
        class_names,
        evaluation_plot_path,
    )
    dot_path, tree_png_path = save_graphviz_tree(
        best_model,
        features.columns.tolist(),
        class_names,
    )
    joblib.dump(best_model, model_path)

    print("\n[교차검증 결과]")
    print(f"최적 파라미터: {search.best_params_}")
    print(f"최적 CV Macro F1: {search.best_score_:.4f}")
    print(f"최종 트리 깊이: {best_model.get_depth()}")
    print(f"최종 리프 노드 수: {best_model.get_n_leaves()}")

    print("\n[독립 테스트 평가]")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n[클래스별 평가]")
    print(
        classification_report(
            y_test,
            predicted,
            target_names=class_names,
            zero_division=0,
        )
    )
    print("[상위 5개 중요 피처]")
    print(importance.head(5).to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\n[저장된 산출물]")
    print(f"Graphviz DOT       : {dot_path}")
    print(f"Graphviz 트리 PNG  : {tree_png_path}")
    print(f"평가·중요도 그래프: {evaluation_plot_path}")
    print(f"피처 중요도 CSV   : {importance_path}")
    print(f"평가 지표 CSV     : {metrics_path}")
    print(f"클래스 평가 CSV   : {report_path}")
    print(f"테스트 예측 CSV   : {predictions_path}")
    print(f"저장 모델         : {model_path}")


if __name__ == "__main__":
    main()
