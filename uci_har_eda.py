from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import f_classif


DATA_DIR = Path("data/UCI HAR Dataset")
FEATURE_CSV_PATH = Path("data/har_features.csv")
TARGET_CSV_PATH = Path("data/har_targets.csv")
OUTPUT_DIR = Path("output/uci_har_eda_assets")
REPORT_PATH = Path("output/uci_har_eda_report.md")


def configure_plotting() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="Malgun Gothic")


def display_feature_name(name: str) -> str:
    """Remove the legacy fNNN_ prefix only when it is actually present."""
    if len(name) > 5 and name.startswith("f") and name[1:4].isdigit() and name[4] == "_":
        return name[5:]
    return name


def load_dataset() -> dict[str, object]:
    """Load features/targets from generated CSV and attach official split metadata."""
    if not FEATURE_CSV_PATH.exists() or not TARGET_CSV_PATH.exists():
        raise FileNotFoundError(
            "har_features.csv 또는 har_targets.csv가 없습니다. 먼저 "
            "'uv run python download_har_dataset.py'를 실행하세요."
        )

    all_features = pd.read_csv(FEATURE_CSV_PATH)
    all_targets = pd.read_csv(TARGET_CSV_PATH)
    if "Activity" not in all_targets.columns:
        raise ValueError("har_targets.csv에 'Activity' 컬럼이 없습니다.")

    feature_table = pd.read_csv(
        DATA_DIR / "features.txt",
        sep=r"\s+",
        header=None,
        names=["feature_id", "raw_name"],
    )
    activity_table = pd.read_csv(
        DATA_DIR / "activity_labels.txt",
        sep=r"\s+",
        header=None,
        names=["activity_id", "activity"],
    )
    activity_map = dict(zip(activity_table["activity_id"], activity_table["activity"]))

    subject_train = pd.read_csv(
        DATA_DIR / "train/subject_train.txt",
        header=None,
        names=["subject"],
    )["subject"]
    subject_test = pd.read_csv(
        DATA_DIR / "test/subject_test.txt",
        header=None,
        names=["subject"],
    )["subject"]
    train_size = len(subject_train)
    test_size = len(subject_test)

    if len(all_features) != train_size + test_size:
        raise ValueError(
            "CSV 행 수가 공식 Train/Test 행 수의 합과 다릅니다: "
            f"CSV={len(all_features):,}, 공식={train_size + test_size:,}"
        )
    if len(all_targets) != len(all_features):
        raise ValueError("피처 CSV와 타깃 CSV의 행 수가 다릅니다.")
    if all_features.shape[1] != len(feature_table):
        raise ValueError("피처 CSV 컬럼 수가 features.txt 정의와 다릅니다.")

    target_names = all_targets["Activity"].map(activity_map)
    if target_names.isna().any():
        unknown = sorted(all_targets.loc[target_names.isna(), "Activity"].unique())
        raise ValueError(f"알 수 없는 Activity ID가 있습니다: {unknown}")

    # download_har_dataset.py는 공식 Train을 먼저, Test를 나중에 결합한다.
    return {
        "feature_table": feature_table,
        "activity_table": activity_table,
        "x_train": all_features.iloc[:train_size].reset_index(drop=True),
        "x_test": all_features.iloc[train_size:].reset_index(drop=True),
        "y_train": target_names.iloc[:train_size].reset_index(drop=True),
        "y_test": target_names.iloc[train_size:].reset_index(drop=True),
        "subject_train": subject_train.reset_index(drop=True),
        "subject_test": subject_test.reset_index(drop=True),
    }


def class_distribution(y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    activities = sorted(set(y_train) | set(y_test))
    rows = []
    for activity in activities:
        train_count = int((y_train == activity).sum())
        test_count = int((y_test == activity).sum())
        rows.append(
            {
                "Activity": activity,
                "Train_Count": train_count,
                "Train_Rate": train_count / len(y_train),
                "Test_Count": test_count,
                "Test_Rate": test_count / len(y_test),
            }
        )
    return pd.DataFrame(rows)


def analyze_features(data: dict[str, object]) -> dict[str, object]:
    x_train: pd.DataFrame = data["x_train"]  # type: ignore[assignment]
    x_test: pd.DataFrame = data["x_test"]  # type: ignore[assignment]
    y_train: pd.Series = data["y_train"]  # type: ignore[assignment]
    feature_table: pd.DataFrame = data["feature_table"]  # type: ignore[assignment]

    train_values = x_train.to_numpy()
    test_values = x_test.to_numpy()
    finite_train = np.isfinite(train_values)
    finite_test = np.isfinite(test_values)
    variances = x_train.var(ddof=0)

    f_scores, p_values = f_classif(x_train, y_train)
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
    anova = pd.DataFrame(
        {
            "Feature": x_train.columns,
            "Raw_Feature": feature_table["raw_name"],
            "F_Score": f_scores,
            "P_Value": p_values,
        }
    ).sort_values("F_Score", ascending=False)

    correlations = x_train.corr().to_numpy()
    upper_row, upper_col = np.triu_indices_from(correlations, k=1)
    absolute_correlations = np.abs(correlations[upper_row, upper_col])
    strongest_indices = np.argsort(absolute_correlations)[-10:][::-1]
    top_correlations = pd.DataFrame(
        {
            "Feature_1": x_train.columns[upper_row[strongest_indices]],
            "Feature_2": x_train.columns[upper_col[strongest_indices]],
            "Correlation": correlations[upper_row[strongest_indices], upper_col[strongest_indices]],
        }
    )

    train_mean = x_train.mean()
    test_mean = x_test.mean()
    pooled_std = np.sqrt((x_train.var(ddof=1) + x_test.var(ddof=1)) / 2).replace(0, np.nan)
    standardized_mean_difference = ((test_mean - train_mean) / pooled_std).abs().fillna(0)
    drift = pd.DataFrame(
        {
            "Feature": x_train.columns,
            "Train_Mean": train_mean,
            "Test_Mean": test_mean,
            "Abs_SMD": standardized_mean_difference,
        }
    ).sort_values("Abs_SMD", ascending=False)

    centered = train_values - train_values.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    eigenvalues = np.clip(eigenvalues, 0, None)
    cumulative_variance = np.cumsum(eigenvalues) / eigenvalues.sum()

    q1 = x_train.quantile(0.25)
    q3 = x_train.quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (x_train.lt(q1 - 1.5 * iqr)) | (x_train.gt(q3 + 1.5 * iqr))
    outlier_rates = outlier_mask.mean()

    return {
        "missing_train": int(x_train.isna().sum().sum()),
        "missing_test": int(x_test.isna().sum().sum()),
        "infinite_train": int((~finite_train).sum()),
        "infinite_test": int((~finite_test).sum()),
        "global_min": float(min(train_values.min(), test_values.min())),
        "global_max": float(max(train_values.max(), test_values.max())),
        "duplicate_train_rows": int(x_train.duplicated().sum()),
        "duplicate_test_rows": int(x_test.duplicated().sum()),
        "duplicate_value_columns": int(x_train.T.duplicated().sum()),
        "duplicate_raw_names": int(feature_table["raw_name"].duplicated(keep=False).sum()),
        "zero_variance": int((variances == 0).sum()),
        "near_zero_variance": int((variances < 1e-4).sum()),
        "variance": variances,
        "anova": anova,
        "correlation_pair_count_095": int((absolute_correlations >= 0.95).sum()),
        "correlation_pair_count_099": int((absolute_correlations >= 0.99).sum()),
        "top_correlations": top_correlations,
        "drift": drift,
        "drift_count_020": int((standardized_mean_difference >= 0.20).sum()),
        "drift_count_050": int((standardized_mean_difference >= 0.50).sum()),
        "pca_90": int(np.searchsorted(cumulative_variance, 0.90) + 1),
        "pca_95": int(np.searchsorted(cumulative_variance, 0.95) + 1),
        "pca_99": int(np.searchsorted(cumulative_variance, 0.99) + 1),
        "outlier_median_rate": float(outlier_rates.median()),
        "outlier_max_rate": float(outlier_rates.max()),
        "outlier_top_feature": str(outlier_rates.idxmax()),
    }


def plot_overview(data: dict[str, object], analysis: dict[str, object]) -> Path:
    y_train: pd.Series = data["y_train"]  # type: ignore[assignment]
    y_test: pd.Series = data["y_test"]  # type: ignore[assignment]
    subject_train: pd.Series = data["subject_train"]  # type: ignore[assignment]
    subject_test: pd.Series = data["subject_test"]  # type: ignore[assignment]
    distribution = class_distribution(y_train, y_test)

    subject_frame = pd.concat(
        [
            pd.DataFrame({"Subject": subject_train, "Split": "Train"}),
            pd.DataFrame({"Subject": subject_test, "Split": "Test"}),
        ],
        ignore_index=True,
    )
    subject_counts = subject_frame.groupby(["Subject", "Split"]).size().reset_index(name="Windows")

    figure, axes = plt.subplots(2, 2, figsize=(18, 12))
    distribution_long = distribution.melt(
        id_vars="Activity",
        value_vars=["Train_Rate", "Test_Rate"],
        var_name="Split",
        value_name="Rate",
    )
    distribution_long["Split"] = distribution_long["Split"].map(
        {"Train_Rate": "Train", "Test_Rate": "Test"}
    )
    sns.barplot(data=distribution_long, x="Activity", y="Rate", hue="Split", ax=axes[0, 0])
    axes[0, 0].set_title("활동 클래스 분포")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("비율")
    axes[0, 0].tick_params(axis="x", rotation=25)
    axes[0, 0].yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")

    sns.barplot(data=subject_counts, x="Subject", y="Windows", hue="Split", ax=axes[0, 1])
    axes[0, 1].set_title("피험자별 윈도우 수와 공식 분할")
    axes[0, 1].set_xlabel("피험자 ID")
    axes[0, 1].set_ylabel("윈도우 수")

    variances: pd.Series = analysis["variance"]  # type: ignore[assignment]
    sns.histplot(variances, bins=40, ax=axes[1, 0], color="#4C72B0")
    axes[1, 0].axvline(1e-4, color="red", linestyle="--", label="근저분산 기준 1e-4")
    axes[1, 0].set_title("학습 피처 분산 분포")
    axes[1, 0].set_xlabel("분산")
    axes[1, 0].legend()

    drift: pd.DataFrame = analysis["drift"]  # type: ignore[assignment]
    sns.histplot(drift["Abs_SMD"], bins=40, ax=axes[1, 1], color="#DD8452")
    axes[1, 1].axvline(0.2, color="orange", linestyle="--", label="작은 이동 0.2")
    axes[1, 1].axvline(0.5, color="red", linestyle="--", label="중간 이동 0.5")
    axes[1, 1].set_title("Train-Test 표준화 평균 차이(SMD)")
    axes[1, 1].set_xlabel("절댓값 SMD")
    axes[1, 1].legend()

    figure.suptitle("UCI HAR 데이터 구조와 건전성", fontsize=18, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path = OUTPUT_DIR / "01_dataset_overview.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_top_features(data: dict[str, object], analysis: dict[str, object]) -> Path:
    x_train: pd.DataFrame = data["x_train"]  # type: ignore[assignment]
    y_train: pd.Series = data["y_train"]  # type: ignore[assignment]
    anova: pd.DataFrame = analysis["anova"]  # type: ignore[assignment]
    top_features = anova.head(6)["Feature"].tolist()

    figure, axes = plt.subplots(2, 3, figsize=(20, 11))
    for axis, feature in zip(axes.ravel(), top_features):
        plot_data = pd.DataFrame({"Activity": y_train, "Value": x_train[feature]})
        sns.boxplot(
            data=plot_data,
            x="Activity",
            y="Value",
            showfliers=False,
            color="#55A868",
            ax=axis,
        )
        raw_name = display_feature_name(feature)
        axis.set_title(raw_name, fontsize=10)
        axis.set_xlabel("")
        axis.set_ylabel("정규화 값")
        axis.tick_params(axis="x", rotation=28, labelsize=8)

    figure.suptitle("ANOVA F-score 상위 피처의 클래스별 분포", fontsize=18, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path = OUTPUT_DIR / "02_top_feature_class_separation.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_correlation(data: dict[str, object], analysis: dict[str, object]) -> Path:
    x_train: pd.DataFrame = data["x_train"]  # type: ignore[assignment]
    anova: pd.DataFrame = analysis["anova"]  # type: ignore[assignment]
    top_features = anova.head(20)["Feature"].tolist()
    labels = [display_feature_name(feature) for feature in top_features]
    correlation = x_train[top_features].corr()
    correlation.index = labels
    correlation.columns = labels

    figure, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={"width_ratios": [1.4, 1]})
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=True,
        yticklabels=True,
        ax=axes[0],
    )
    axes[0].set_title("상위 20개 분리 피처 상관계수")
    axes[0].tick_params(axis="x", rotation=75, labelsize=7)
    axes[0].tick_params(axis="y", labelsize=7)

    drift: pd.DataFrame = analysis["drift"]  # type: ignore[assignment]
    drift_top = drift.head(15).sort_values("Abs_SMD")
    sns.barplot(data=drift_top, x="Abs_SMD", y="Feature", color="#C44E52", ax=axes[1])
    axes[1].axvline(0.2, color="orange", linestyle="--")
    axes[1].axvline(0.5, color="red", linestyle="--")
    axes[1].set_title("Train-Test 평균 이동 상위 15개 피처")
    axes[1].set_xlabel("절댓값 SMD")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", labelsize=7)

    figure.suptitle("피처 중복성과 분포 이동", fontsize=18, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path = OUTPUT_DIR / "03_correlation_and_drift.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def markdown_table(frame: pd.DataFrame, float_columns: dict[str, str] | None = None) -> str:
    display = frame.copy()
    for column, format_string in (float_columns or {}).items():
        display[column] = display[column].map(lambda value: format(value, format_string))
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def create_report(data: dict[str, object], analysis: dict[str, object]) -> None:
    x_train: pd.DataFrame = data["x_train"]  # type: ignore[assignment]
    x_test: pd.DataFrame = data["x_test"]  # type: ignore[assignment]
    y_train: pd.Series = data["y_train"]  # type: ignore[assignment]
    y_test: pd.Series = data["y_test"]  # type: ignore[assignment]
    subject_train: pd.Series = data["subject_train"]  # type: ignore[assignment]
    subject_test: pd.Series = data["subject_test"]  # type: ignore[assignment]
    distribution = class_distribution(y_train, y_test)
    distribution["Train_Rate"] *= 100
    distribution["Test_Rate"] *= 100

    anova: pd.DataFrame = analysis["anova"]  # type: ignore[assignment]
    top_anova = anova.head(10)[["Raw_Feature", "F_Score", "P_Value"]].copy()
    top_correlations: pd.DataFrame = analysis["top_correlations"]  # type: ignore[assignment]
    top_correlations = top_correlations.copy()
    top_correlations["Feature_1"] = top_correlations["Feature_1"].map(display_feature_name)
    top_correlations["Feature_2"] = top_correlations["Feature_2"].map(display_feature_name)
    drift: pd.DataFrame = analysis["drift"]  # type: ignore[assignment]
    top_drift = drift.head(10).copy()
    top_drift["Feature"] = top_drift["Feature"].map(display_feature_name)

    train_subjects = sorted(subject_train.unique())
    test_subjects = sorted(subject_test.unique())
    subject_activity_min = int(
        pd.DataFrame({"subject": subject_train, "activity": y_train})
        .groupby("subject")["activity"]
        .nunique()
        .min()
    )
    imbalance_ratio = distribution["Train_Count"].max() / distribution["Train_Count"].min()

    report = f"""# UCI HAR 데이터 탐색적 분석 보고서

## 1. 분석 목적과 올바른 EDA 접근법

이 보고서는 스마트폰 센서 기반 Human Activity Recognition 데이터로 **Random Forest 다중 분류 모델**을 만들기 전에 수행해야 할 EDA를 정리한다. EDA는 단순히 그래프를 많이 그리는 과정이 아니라 다음 질문에 증거로 답하는 과정이다.

1. **관측 단위와 타깃은 무엇인가?** 한 행은 한 사람이 아니라 2.56초 길이의 센서 윈도우이며, 타깃은 6개 활동이다.
2. **결측·무한값·중복·범위 오류가 있는가?** 모델이 숫자를 읽을 수 있는지뿐 아니라 동일 관측이 반복되어 평가가 부풀려지는지도 확인한다.
3. **표본이 독립적인가?** 같은 피험자의 인접 윈도우가 50% 겹치므로 행 단위 무작위 분할은 데이터 누수를 일으킬 수 있다.
4. **타깃이 불균형한가?** 전체 정확도가 소수 클래스를 가리지 않는지 확인하고 Macro F1과 클래스별 Recall을 설계한다.
5. **피처가 유용하고 중복되지 않는가?** 분산, 클래스별 분포, 상관관계, 단변량 분리도를 학습 데이터에서만 진단한다.
6. **Train과 Test의 분포가 다른가?** 피험자가 다르기 때문에 실제 일반화 난도를 나타내는 분포 이동을 점검한다.
7. **예정 모델의 특성과 맞는가?** Random Forest는 스케일링이 필요 없지만, 강한 상관 피처는 중요도를 여러 변수에 분산시키므로 해석 방식을 조정해야 한다.

EDA의 원칙은 **테스트셋을 모델 선택이나 피처 선택에 사용하지 않는 것**이다. 아래 클래스 비율과 분포 이동은 테스트셋의 건전성 확인용이며, 실제 하이퍼파라미터 및 피처 선택은 학습셋 내부의 피험자 그룹 교차검증으로 수행해야 한다.

## 2. 데이터 정의와 실험 설계

- 분석 피처 입력: `data/har_features.csv`
- 분석 타깃 입력: `data/har_targets.csv`
- 피험자·공식 분할 보조 정보: `data/UCI HAR Dataset/train|test/subject_*.txt`
- 피험자: 19~48세 자원자 30명
- 센서: 허리 위치 Samsung Galaxy S II의 3축 가속도계와 자이로스코프
- 샘플링: 50 Hz
- 관측 윈도우: 2.56초, 128개 측정값, 인접 윈도우 50% 중첩
- 입력 피처: 시간·주파수 영역에서 계산된 561개 수치 피처
- 타깃: `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`
- 값 범위: 사전 정규화되어 대체로 [-1, 1]
- 공식 분할: 피험자 기준 Train 21명 / Test 9명

| 구분 | 행 수 | 피처 수 | 피험자 수 |
| --- | ---: | ---: | ---: |
| Train | {len(x_train):,} | {x_train.shape[1]} | {subject_train.nunique()} |
| Test | {len(x_test):,} | {x_test.shape[1]} | {subject_test.nunique()} |
| 전체 | {len(x_train) + len(x_test):,} | {x_train.shape[1]} | {pd.concat([subject_train, subject_test]).nunique()} |

- Train 피험자 ID: {', '.join(map(str, train_subjects))}
- Test 피험자 ID: {', '.join(map(str, test_subjects))}
- 두 집합의 피험자 교집합: **{len(set(train_subjects) & set(test_subjects))}명**

![데이터 구조와 건전성](uci_har_eda_assets/01_dataset_overview.png)

## 3. 데이터 건전성 점검

| 점검 항목 | Train | Test | 해석 |
| --- | ---: | ---: | --- |
| 결측치 | {analysis['missing_train']:,} | {analysis['missing_test']:,} | 별도 대체 불필요 |
| 무한값 | {analysis['infinite_train']:,} | {analysis['infinite_test']:,} | 수치 연산 안전 |
| 완전 중복 행 | {analysis['duplicate_train_rows']:,} | {analysis['duplicate_test_rows']:,} | 동일 피처 벡터 중복 없음 |
| 관측 최솟값 | {analysis['global_min']:.6f} | - | 정규화 범위 확인 |
| 관측 최댓값 | {analysis['global_max']:.6f} | - | 정규화 범위 확인 |

추가 점검 결과:

- 분산이 정확히 0인 피처: **{analysis['zero_variance']}개**
- 분산이 `1e-4` 미만인 근저분산 피처: **{analysis['near_zero_variance']}개**
- 값이 완전히 동일한 중복 피처 열: **{analysis['duplicate_value_columns']}개**
- 원본 `features.txt`에서 중복 이름으로 표기된 엔트리: **{analysis['duplicate_raw_names']}개**. 분석 코드에서는 피처 ID를 접두어로 붙여 이름 충돌을 방지했다.
- IQR 기준 피처별 이상치 비율 중앙값: **{analysis['outlier_median_rate']:.2%}**, 최댓값: **{analysis['outlier_max_rate']:.2%}** (`{display_feature_name(str(analysis['outlier_top_feature']))}`). 센서 행동 데이터는 다봉 분포이므로 IQR 이상치를 오류로 간주해 일괄 삭제하면 안 된다.

**판단:** 결측치와 무한값이 없어 Random Forest 입력을 위한 대체 처리는 현재 필요 없다. 향후 운영 데이터까지 고려하면 `SimpleImputer(strategy="median")`를 파이프라인에 안전장치로 둘 수 있다. 이미 정규화된 값이며 트리 모델은 단조 변환에 불변이므로 `StandardScaler`는 필요 없다. 완전히 동일한 21개 중복 열은 계산량과 중요도 분산을 줄이기 위해 제거할 수 있지만, 반드시 Train에서 중복 여부와 유지 열을 결정한 뒤 동일한 열 선택을 Validation/Test에 적용해야 한다.

## 4. 타깃 분포와 피험자 구조

{markdown_table(distribution, {'Train_Rate': '.2f', 'Test_Rate': '.2f'})}

표의 비율 단위는 `%`이다. 학습셋의 최대/최소 클래스 표본비는 **{imbalance_ratio:.2f}:1**로 심각한 불균형은 아니다. 모든 학습 피험자가 최소 **{subject_activity_min}개 활동**을 수행했다.

그러나 한 피험자에서 많은 윈도우가 반복 생성되므로 행은 독립 표본이 아니다. 인접 윈도우는 원시 신호의 절반을 공유한다. 따라서 `train_test_split(..., stratify=y)`로 행을 무작위 분할하면 동일 피험자의 매우 유사한 윈도우가 양쪽에 들어가 과도하게 낙관적인 점수가 나올 수 있다.

**권장 평가 설계:**

- 최종 평가는 제공된 공식 Test를 그대로 사용한다.
- 하이퍼파라미터 선택은 Train에서 `StratifiedGroupKFold`를 사용하고 `groups=subject_train`으로 지정한다.
- 정확도와 함께 `Macro F1`, `Balanced Accuracy`, 클래스별 Recall, 혼동행렬을 보고한다.

## 5. 피처 분포와 클래스 분리도

ANOVA F-score는 각 피처가 활동 클래스를 단변량으로 얼마나 잘 분리하는지 보는 **탐색용 순위**이다. 이 순위는 학습셋에서만 계산했으며, 피처 간 상호작용이나 인과성을 의미하지 않는다.

{markdown_table(top_anova, {'F_Score': '.2f', 'P_Value': '.3e'})}

![상위 피처 클래스 분포](uci_har_eda_assets/02_top_feature_class_separation.png)

상위 피처는 신체 가속도 Jerk의 엔트로피와 X축 중력가속도의 평균·최솟값·최댓값·에너지 계열이다. 정적 활동(`SITTING`, `STANDING`, `LAYING`)과 동적 활동(`WALKING` 계열)의 분리는 비교적 뚜렷하지만, `SITTING`과 `STANDING`은 유사한 센서 패턴을 보여 주요 오분류 후보가 된다. Random Forest의 비선형 분기와 피처 상호작용이 이 문제에 적합하다.

## 6. 상관관계, 중복성, 유효 차원

- 절댓값 상관계수 0.95 이상 피처 쌍: **{analysis['correlation_pair_count_095']:,}쌍**
- 절댓값 상관계수 0.99 이상 피처 쌍: **{analysis['correlation_pair_count_099']:,}쌍**
- 학습 데이터 분산의 90%를 설명하는 주성분 수: **{analysis['pca_90']}개**
- 95% 설명 주성분 수: **{analysis['pca_95']}개**
- 99% 설명 주성분 수: **{analysis['pca_99']}개**

가장 강한 상관 피처 쌍:

{markdown_table(top_correlations, {'Correlation': '.6f'})}

![상관관계와 분포 이동](uci_har_eda_assets/03_correlation_and_drift.png)

561개 피처에 상당한 중복이 있다. Random Forest 예측 자체는 다중공선성에 비교적 강하지만 다음 해석상 문제가 있다.

- 서로 비슷한 피처가 분할 후보를 나눠 가지므로 기본 `feature_importances_`가 희석되거나 불안정해질 수 있다.
- 연속형·고유값이 많은 피처에 불순도 기반 중요도가 편향될 수 있다.
- 최종 해석에는 공식 Test 또는 그룹 기반 검증 fold의 **Permutation Importance**를 함께 사용해야 한다.

상관 피처를 반드시 사전 제거할 필요는 없다. 우선 `max_features="sqrt"` 등으로 각 트리의 후보 피처를 제한하고, 계산량이나 해석성이 문제일 때만 학습 fold 안에서 상관 기반 제거를 검토한다.

## 7. Train-Test 분포 이동

피험자 집합이 완전히 다르므로 Train과 Test에 어느 정도 분포 이동이 존재하는 것이 자연스럽다. 절댓값 표준화 평균 차이(SMD)를 사용한 결과:

- SMD 0.20 이상 피처: **{analysis['drift_count_020']}개**
- SMD 0.50 이상 피처: **{analysis['drift_count_050']}개**

이동이 큰 상위 피처:

{markdown_table(top_drift, {'Train_Mean': '.4f', 'Test_Mean': '.4f', 'Abs_SMD': '.4f'})}

이는 데이터 오류라기보다 새로운 사람에게 일반화해야 하는 실제 난도를 반영한다. Test 분포에 맞춰 전처리하거나 피처를 선택하면 누수이므로, 모든 선택은 Train 내부 그룹 교차검증에서 수행해야 한다.

## 8. Random Forest 모델링 권고안

### 전처리

1. `X_train`, `y_train`, `subject_train`을 함께 로드한다.
2. 결측치가 현재 없으므로 삭제나 대체는 하지 않는다. 운영 안전장치가 필요하면 학습 fold에서만 중앙값을 학습하는 `SimpleImputer`를 사용한다.
3. 완전 중복 피처 제거를 적용한다면 학습 fold에서 유지 열을 결정하고 Validation/Test에는 그 결정을 그대로 적용한다.
4. `StandardScaler`는 적용하지 않는다.
5. 공식 Test는 최종 1회 평가 전까지 격리한다.
6. 추가 피처 선택이 필요하면 반드시 그룹 교차검증 Pipeline 내부에서 수행한다.

### 권장 교차검증과 탐색 범위

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample",
)

param_distributions = {{
    "n_estimators": [300, 500, 800],
    "max_depth": [None, 12, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.3],
    "bootstrap": [True],
}}

search = RandomizedSearchCV(
    model,
    param_distributions=param_distributions,
    n_iter=30,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    random_state=42,
)
search.fit(X_train, y_train, groups=subject_train)
```

### 평가 지표

- 1순위: `Macro F1` — 6개 활동을 동일 비중으로 평가
- 2순위: `Balanced Accuracy`
- 필수 진단: 클래스별 Precision/Recall/F1과 혼동행렬
- 확률 품질이 필요하면 multiclass Log Loss와 One-vs-Rest ROC-AUC 추가
- 일반화 보고: 그룹 교차검증 평균 ± 표준편차와 공식 Test 점수를 함께 제시

### 모델 해석

- 기본 불순도 중요도는 빠른 개요로만 사용한다.
- 공식 Test 또는 그룹 검증 fold에서 Permutation Importance를 계산한다.
- 상위 피처에 대해 PDP/ALE를 사용하되, 강한 상관 피처에서는 PDP를 인과효과처럼 해석하지 않는다.
- 특히 `SITTING`↔`STANDING`, 세 가지 `WALKING` 활동 간 혼동을 별도로 분석한다.

## 9. 결론

데이터는 결측·무한값·완전 중복 행이 없고 이미 정규화되어 있어 Random Forest 학습을 위한 기본 건전성이 높다. 클래스 불균형도 심각하지 않다. 가장 중요한 위험은 **50% 중첩 윈도우와 피험자 반복 측정으로 인한 평가 누수**이며, 공식 subject-disjoint Test와 그룹 교차검증을 사용해야 한다.

또한 561개 피처에는 강한 상관과 차원 중복이 존재한다. 이는 Random Forest의 예측에는 치명적이지 않지만 중요도 해석을 왜곡할 수 있으므로, `max_features` 튜닝과 Permutation Importance를 권장한다. 이 원칙을 지키면 모델 점수는 새로운 사용자의 행동을 얼마나 잘 인식하는지에 가까운 신뢰 가능한 추정치가 된다.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    configure_plotting()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset()
    analysis = analyze_features(data)
    overview_path = plot_overview(data, analysis)
    top_features_path = plot_top_features(data, analysis)
    correlation_path = plot_correlation(data, analysis)
    create_report(data, analysis)

    print(f"보고서: {REPORT_PATH}")
    print(f"차트 1: {overview_path}")
    print(f"차트 2: {top_features_path}")
    print(f"차트 3: {correlation_path}")
    print(f"Train/Test: {len(data['x_train']):,}/{len(data['x_test']):,}")
    print(
        f"고상관 피처 쌍(|r|>=0.95): "
        f"{analysis['correlation_pair_count_095']:,}"
    )
    print(
        f"PCA 설명분산 95% 성분 수: {analysis['pca_95']} / "
        f"{data['x_train'].shape[1]}"
    )


if __name__ == "__main__":
    main()
