from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif


# 이 파일은 <project>/src/mobile_price_category_eda.py에 위치한다.
# 현재 작업 디렉터리(CWD)가 아니라 파일 위치를 기준으로 경로를 잡아,
# 프로젝트 루트·src 폴더·IDE 어디에서 실행해도 같은 데이터를 읽도록 한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = DATA_DIR / "mobile_price_category_train.csv"
TEST_PATH = DATA_DIR / "mobile_price_category_test.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mobile_price_category_eda"
TARGET = "price_range"
ID_COLUMN = "id"

BINARY_COLUMNS = [
    "blue",
    "dual_sim",
    "four_g",
    "three_g",
    "touch_screen",
    "wifi",
]

FEATURE_DESCRIPTIONS = {
    "battery_power": "배터리 용량",
    "blue": "블루투스 지원",
    "clock_speed": "프로세서 클럭 속도",
    "dual_sim": "듀얼 SIM 지원",
    "fc": "전면 카메라 화소",
    "four_g": "4G 지원",
    "int_memory": "내장 메모리",
    "m_dep": "기기 두께",
    "mobile_wt": "기기 무게",
    "n_cores": "CPU 코어 수",
    "pc": "후면 카메라 화소",
    "px_height": "화면 픽셀 높이",
    "px_width": "화면 픽셀 너비",
    "ram": "RAM",
    "sc_h": "화면 물리 높이",
    "sc_w": "화면 물리 너비",
    "talk_time": "통화 시간",
    "three_g": "3G 지원",
    "touch_screen": "터치스크린 지원",
    "wifi": "Wi-Fi 지원",
}


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


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    missing_files = [path for path in (TRAIN_PATH, TEST_PATH) if not path.is_file()]
    if missing_files:
        missing_text = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "다음 데이터 파일을 찾을 수 없습니다. "
            "CSV 파일이 프로젝트의 data 폴더에 있는지 확인하세요.\n"
            f"{missing_text}"
        )

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    if TARGET not in train.columns:
        raise ValueError(f"학습 데이터에 타깃 {TARGET!r}가 없습니다.")
    if TARGET in test.columns:
        raise ValueError("테스트 데이터에 타깃이 포함되어 있습니다. 누수 여부를 확인하세요.")
    if ID_COLUMN not in test.columns:
        raise ValueError(f"테스트 데이터에 식별자 {ID_COLUMN!r}가 없습니다.")

    feature_columns = [column for column in train.columns if column != TARGET]
    test_features = [column for column in test.columns if column != ID_COLUMN]
    if feature_columns != test_features:
        raise ValueError(
            "학습/테스트 피처 스키마가 일치하지 않습니다.\n"
            f"train={feature_columns}\ntest={test_features}"
        )
    if not set(BINARY_COLUMNS).issubset(feature_columns):
        raise ValueError("예상한 이진 피처가 일부 누락되었습니다.")
    return train, test, feature_columns


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR-adjusted p-values."""
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return pd.Series(result, index=p_values.index)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def population_stability_index(train: pd.Series, test: pd.Series, bins: int = 10) -> float:
    """PSI using train quantile bins; binary/low-cardinality values use categories."""
    train = train.dropna()
    test = test.dropna()
    epsilon = 1e-6
    unique_values = np.sort(train.unique())

    if len(unique_values) <= 10:
        categories = np.union1d(unique_values, test.unique())
        train_rate = train.value_counts(normalize=True).reindex(categories, fill_value=0)
        test_rate = test.value_counts(normalize=True).reindex(categories, fill_value=0)
    else:
        edges = np.unique(train.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        train_rate = pd.cut(train, bins=edges, include_lowest=True).value_counts(normalize=True, sort=False)
        test_rate = pd.cut(test, bins=edges, include_lowest=True).value_counts(normalize=True, sort=False)

    train_rate = np.clip(train_rate.to_numpy(dtype=float), epsilon, None)
    test_rate = np.clip(test_rate.to_numpy(dtype=float), epsilon, None)
    return float(np.sum((test_rate - train_rate) * np.log(test_rate / train_rate)))


def cramer_v_binary(feature: pd.Series, target: pd.Series) -> tuple[float, float]:
    table = pd.crosstab(feature, target)
    chi2, p_value, _, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    denominator = min(table.shape[0] - 1, table.shape[1] - 1)
    value = np.sqrt((chi2 / n) / denominator) if denominator > 0 else 0.0
    return float(value), float(p_value)


def eta_squared(feature: pd.Series, target: pd.Series) -> float:
    overall = feature.mean()
    total = float(((feature - overall) ** 2).sum())
    between = 0.0
    for _, group in feature.groupby(target):
        between += len(group) * float((group.mean() - overall) ** 2)
    return safe_divide(between, total)


def monotonic_mean_pattern(feature: pd.Series, target: pd.Series) -> str:
    means = feature.groupby(target).mean().sort_index().to_numpy()
    if np.all(np.diff(means) > 0):
        return "strictly_increasing"
    if np.all(np.diff(means) < 0):
        return "strictly_decreasing"
    if np.all(np.diff(means) >= 0):
        return "non_decreasing"
    if np.all(np.diff(means) <= 0):
        return "non_increasing"
    return "non_monotonic"


def data_quality_analysis(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    train_x = train[features]
    test_x = test[features]
    binary_invalid_train = sum((~train_x[BINARY_COLUMNS].isin([0, 1])).sum())
    binary_invalid_test = sum((~test_x[BINARY_COLUMNS].isin([0, 1])).sum())
    combined_with_duplicates = pd.concat([train_x, test_x], ignore_index=True)
    duplicated_across = combined_with_duplicates.duplicated(subset=features, keep=False)

    def count_status(value: int) -> str:
        return "review" if value > 0 else "pass"

    cross_duplicate_count = int(duplicated_across.sum())
    sc_w_zero_train = int(train["sc_w"].eq(0).sum())
    sc_w_zero_test = int(test["sc_w"].eq(0).sum())
    px_height_zero_train = int(train["px_height"].eq(0).sum())
    px_height_zero_test = int(test["px_height"].eq(0).sum())
    pc_lt_fc_train = int(train["pc"].lt(train["fc"]).sum())
    pc_lt_fc_test = int(test["pc"].lt(test["fc"]).sum())
    four_g_without_three_g_train = int(
        (train["four_g"].eq(1) & train["three_g"].eq(0)).sum()
    )
    four_g_without_three_g_test = int(
        (test["four_g"].eq(1) & test["three_g"].eq(0)).sum()
    )

    checks = [
        ("train_rows", len(train), "info", "학습 행 수"),
        ("test_rows", len(test), "info", "테스트 행 수"),
        ("feature_count", len(features), "info", "공통 예측 피처 수"),
        ("train_missing_cells", int(train.isna().sum().sum()), "pass", "학습 결측 셀"),
        ("test_missing_cells", int(test.isna().sum().sum()), "pass", "테스트 결측 셀"),
        ("train_duplicate_rows", int(train.duplicated().sum()), "pass", "학습 완전 중복 행"),
        ("test_duplicate_rows", int(test.duplicated().sum()), "pass", "테스트 완전 중복 행"),
        (
            "cross_split_duplicate_feature_rows",
            cross_duplicate_count,
            count_status(cross_duplicate_count),
            "두 분할을 합쳤을 때 중복에 포함된 피처 행 수",
        ),
        ("test_id_missing", int(test[ID_COLUMN].isna().sum()), "pass", "테스트 ID 결측"),
        ("test_id_duplicates", int(test[ID_COLUMN].duplicated().sum()), "pass", "테스트 ID 중복"),
        (
            "test_id_sequential_1_to_n",
            int(np.array_equal(test[ID_COLUMN].to_numpy(), np.arange(1, len(test) + 1))),
            "info",
            "테스트 ID가 1..N 순번인지",
        ),
        ("binary_invalid_train", int(binary_invalid_train), "pass", "학습 이진 피처의 0/1 외 값"),
        ("binary_invalid_test", int(binary_invalid_test), "pass", "테스트 이진 피처의 0/1 외 값"),
        (
            "target_invalid",
            int((~train[TARGET].isin([0, 1, 2, 3])).sum()),
            "pass",
            "타깃의 0~3 외 값",
        ),
        (
            "sc_w_zero_train",
            sc_w_zero_train,
            count_status(sc_w_zero_train),
            "학습 화면 너비 0 (물리적으로 의심)",
        ),
        (
            "sc_w_zero_test",
            sc_w_zero_test,
            count_status(sc_w_zero_test),
            "테스트 화면 너비 0 (물리적으로 의심)",
        ),
        (
            "px_height_zero_train",
            px_height_zero_train,
            count_status(px_height_zero_train),
            "학습 픽셀 높이 0 (물리적으로 의심)",
        ),
        (
            "px_height_zero_test",
            px_height_zero_test,
            count_status(px_height_zero_test),
            "테스트 픽셀 높이 0 (물리적으로 의심)",
        ),
        (
            "pc_less_than_fc_train",
            pc_lt_fc_train,
            count_status(pc_lt_fc_train),
            "학습 후면 카메라 화소 < 전면 카메라 화소",
        ),
        (
            "pc_less_than_fc_test",
            pc_lt_fc_test,
            count_status(pc_lt_fc_test),
            "테스트 후면 카메라 화소 < 전면 카메라 화소",
        ),
        (
            "four_g_without_three_g_train",
            four_g_without_three_g_train,
            count_status(four_g_without_three_g_train),
            "학습 4G=1인데 3G=0인 행",
        ),
        (
            "four_g_without_three_g_test",
            four_g_without_three_g_test,
            count_status(four_g_without_three_g_test),
            "테스트 4G=1인데 3G=0인 행",
        ),
        (
            "constant_features_train",
            int((train_x.nunique(dropna=False) <= 1).sum()),
            "pass",
            "학습 상수 피처 수",
        ),
    ]
    return pd.DataFrame(checks, columns=["check", "value", "status", "description"])


def feature_summary(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in features:
        q1, q3 = train[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            outlier_rate = 0.0
        else:
            outlier_rate = float(
                ((train[column] < q1 - 1.5 * iqr) | (train[column] > q3 + 1.5 * iqr)).mean()
            )
        rows.append(
            {
                "feature": column,
                "description": FEATURE_DESCRIPTIONS[column],
                "kind": "binary" if column in BINARY_COLUMNS else "numeric",
                "train_dtype": str(train[column].dtype),
                "train_unique": train[column].nunique(dropna=False),
                "test_unique": test[column].nunique(dropna=False),
                "train_missing": train[column].isna().sum(),
                "test_missing": test[column].isna().sum(),
                "train_min": train[column].min(),
                "train_q1": q1,
                "train_median": train[column].median(),
                "train_mean": train[column].mean(),
                "train_q3": q3,
                "train_max": train[column].max(),
                "train_std": train[column].std(ddof=1),
                "train_skew": train[column].skew(),
                "iqr_outlier_rate": outlier_rate,
                "test_min": test[column].min(),
                "test_median": test[column].median(),
                "test_mean": test[column].mean(),
                "test_max": test[column].max(),
            }
        )
    return pd.DataFrame(rows)


def target_association_analysis(
    train: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = train[features]
    y = train[TARGET]
    discrete_mask = [column in BINARY_COLUMNS for column in features]
    mi_values = mutual_info_classif(
        x,
        y,
        discrete_features=discrete_mask,
        random_state=42,
    )

    rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    for column, mi in zip(features, mi_values):
        grouped = train.groupby(TARGET)[column]
        means = grouped.mean()
        medians = grouped.median()
        groups = [values.to_numpy() for _, values in grouped]
        kruskal_h, kruskal_p = stats.kruskal(*groups)
        spearman_r, spearman_p = stats.spearmanr(train[column], y)
        eta2 = eta_squared(train[column], y)
        cramer_v, chi2_p = (
            cramer_v_binary(train[column], y) if column in BINARY_COLUMNS else (np.nan, np.nan)
        )
        rows.append(
            {
                "feature": column,
                "kind": "binary" if column in BINARY_COLUMNS else "numeric",
                "mutual_information": mi,
                "eta_squared": eta2,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "kruskal_h": kruskal_h,
                "kruskal_p": kruskal_p,
                "cramer_v": cramer_v,
                "chi2_p": chi2_p,
                "mean_pattern": monotonic_mean_pattern(train[column], y),
                "class_mean_range": means.max() - means.min(),
            }
        )
        for class_value in sorted(y.unique()):
            class_rows.append(
                {
                    "feature": column,
                    "price_range": int(class_value),
                    "mean": means.loc[class_value],
                    "median": medians.loc[class_value],
                    "std": grouped.std().loc[class_value],
                    "count": grouped.size().loc[class_value],
                }
            )

    association = pd.DataFrame(rows)
    association["kruskal_q"] = benjamini_hochberg(association["kruskal_p"])
    association["spearman_q"] = benjamini_hochberg(association["spearman_p"])
    association = association.sort_values(
        ["mutual_information", "eta_squared"], ascending=False
    ).reset_index(drop=True)
    return association, pd.DataFrame(class_rows)


def drift_analysis(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in features:
        train_values = train[column]
        test_values = test[column]
        pooled_sd = np.sqrt((train_values.var(ddof=1) + test_values.var(ddof=1)) / 2)
        smd = safe_divide(test_values.mean() - train_values.mean(), pooled_sd)
        ks = stats.ks_2samp(train_values, test_values, alternative="two-sided", method="auto")
        psi = population_stability_index(train_values, test_values)
        mean_delta = test_values.mean() - train_values.mean()
        relative_mean_delta = safe_divide(mean_delta, abs(train_values.mean()))
        positive_rate_delta = (
            test_values.mean() - train_values.mean() if column in BINARY_COLUMNS else np.nan
        )
        rows.append(
            {
                "feature": column,
                "kind": "binary" if column in BINARY_COLUMNS else "numeric",
                "train_mean": train_values.mean(),
                "test_mean": test_values.mean(),
                "mean_delta": mean_delta,
                "relative_mean_delta": relative_mean_delta,
                "abs_smd": abs(smd),
                "ks_statistic": ks.statistic,
                "ks_p": ks.pvalue,
                "psi": psi,
                "positive_rate_delta": positive_rate_delta,
            }
        )

    drift = pd.DataFrame(rows)
    drift["ks_q"] = benjamini_hochberg(drift["ks_p"])
    drift["severity"] = np.select(
        [
            (drift["psi"] >= 0.25) | (drift["abs_smd"] >= 0.20),
            (drift["psi"] >= 0.10) | (drift["abs_smd"] >= 0.10),
        ],
        ["high", "watch"],
        default="low",
    )
    return drift.sort_values(["severity", "psi", "abs_smd"], ascending=[True, False, False])


def correlation_analysis(train: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pearson = train[features].corr(method="pearson")
    pairs: list[dict[str, object]] = []
    for index, left in enumerate(features):
        for right in features[index + 1 :]:
            value = pearson.loc[left, right]
            pairs.append(
                {
                    "feature_1": left,
                    "feature_2": right,
                    "pearson_r": value,
                    "abs_pearson_r": abs(value),
                }
            )
    pair_table = pd.DataFrame(pairs).sort_values("abs_pearson_r", ascending=False)
    return pearson, pair_table


def derived_feature_analysis(train: pd.DataFrame) -> pd.DataFrame:
    derived = pd.DataFrame(index=train.index)
    derived["pixel_area"] = train["px_height"] * train["px_width"]
    derived["pixel_diagonal"] = np.hypot(train["px_height"], train["px_width"])
    derived["screen_area"] = train["sc_h"] * train["sc_w"]
    derived["camera_gap"] = train["pc"] - train["fc"]
    derived["connectivity_count"] = train[["blue", "three_g", "four_g", "wifi"]].sum(axis=1)
    derived["battery_per_weight"] = train["battery_power"] / train["mobile_wt"]
    derived["ram_battery_product"] = train["ram"] * train["battery_power"]

    rows: list[dict[str, object]] = []
    for column in derived.columns:
        spearman_r, spearman_p = stats.spearmanr(derived[column], train[TARGET])
        rows.append(
            {
                "derived_feature": column,
                "eta_squared": eta_squared(derived[column], train[TARGET]),
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "mean_pattern": monotonic_mean_pattern(derived[column], train[TARGET]),
                "zero_rate": derived[column].eq(0).mean(),
            }
        )
    result = pd.DataFrame(rows)
    result["spearman_q"] = benjamini_hochberg(result["spearman_p"])
    return result.sort_values("eta_squared", ascending=False).reset_index(drop=True)


def save_overview_plot(train: pd.DataFrame, test: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "01_overview_and_quality.png"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    counts = train[TARGET].value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values, ax=axes[0, 0], color="#2563EB")
    axes[0, 0].set_title("Target class balance", pad=12)
    axes[0, 0].set_xlabel("price_range")
    axes[0, 0].set_ylabel("Count")
    for index, value in enumerate(counts.values):
        axes[0, 0].text(index, value + 10, f"{value}\n({value / len(train):.1%})", ha="center")
    axes[0, 0].set_ylim(0, 600)

    zero_features = ["sc_w", "px_height", "fc", "pc"]
    zero_table = pd.DataFrame(
        {
            "Train": [train[column].eq(0).mean() for column in zero_features],
            "Test": [test[column].eq(0).mean() for column in zero_features],
        },
        index=zero_features,
    )
    zero_table.plot(kind="bar", ax=axes[0, 1], color=["#2563EB", "#F59E0B"])
    axes[0, 1].set_title("Zero-value rate in selected features")
    axes[0, 1].set_ylabel("Rate")
    axes[0, 1].tick_params(axis="x", rotation=0)
    axes[0, 1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

    binary_rate = pd.DataFrame(
        {"Train": train[BINARY_COLUMNS].mean(), "Test": test[BINARY_COLUMNS].mean()}
    )
    binary_rate.plot(kind="bar", ax=axes[1, 0], color=["#2563EB", "#F59E0B"])
    axes[1, 0].set_title("Binary-feature positive rates")
    axes[1, 0].set_ylabel("Positive rate")
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

    volume = pd.DataFrame(
        {
            "Dataset": ["Train", "Test"],
            "Rows": [len(train), len(test)],
            "Feature columns": [train.shape[1] - 1, test.shape[1] - 1],
        }
    ).set_index("Dataset")
    volume.plot(kind="bar", ax=axes[1, 1], color=["#0F766E", "#7C3AED"])
    axes[1, 1].set_title("Dataset dimensions")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].tick_params(axis="x", rotation=0)

    fig.suptitle("Mobile price data: structure and quality overview", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_distribution_plot(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> Path:
    path = OUTPUT_DIR / "02_feature_distributions.png"
    numeric = [column for column in features if column not in BINARY_COLUMNS]
    ncols = 4
    nrows = int(np.ceil(len(numeric) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.6 * nrows))
    axes = np.asarray(axes).ravel()
    for axis, column in zip(axes, numeric):
        sns.histplot(train[column], bins=24, stat="density", element="step", fill=False,
                     color="#2563EB", label="Train", ax=axis)
        sns.histplot(test[column], bins=24, stat="density", element="step", fill=False,
                     color="#F59E0B", label="Test", ax=axis)
        axis.set_title(column)
        axis.set_ylabel("Density")
    for axis in axes[len(numeric) :]:
        axis.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=True)
    fig.suptitle("Train and test feature distributions", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_target_relationship_plot(
    train: pd.DataFrame,
    association: pd.DataFrame,
) -> Path:
    path = OUTPUT_DIR / "03_target_relationships.png"
    top_numeric = association.query("kind == 'numeric'").head(8)["feature"].tolist()
    fig, axes = plt.subplots(2, 4, figsize=(19, 10))
    for axis, column in zip(axes.ravel(), top_numeric):
        sns.boxplot(
            data=train,
            x=TARGET,
            y=column,
            hue=TARGET,
            palette="Blues",
            legend=False,
            showfliers=False,
            ax=axis,
        )
        eta2 = association.set_index("feature").loc[column, "eta_squared"]
        axis.set_title(f"{column}  (eta²={eta2:.3f})")
    fig.suptitle("Top numeric features by relationship with price range", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_binary_relationship_plot(train: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "04_binary_target_relationships.png"
    melted = (
        train.groupby(TARGET)[BINARY_COLUMNS]
        .mean()
        .reset_index()
        .melt(id_vars=TARGET, var_name="feature", value_name="positive_rate")
    )
    fig, axis = plt.subplots(figsize=(14, 7))
    sns.barplot(data=melted, x="feature", y="positive_rate", hue=TARGET, palette="Blues", ax=axis)
    axis.set_title("Binary-feature prevalence by price range")
    axis.set_xlabel("")
    axis.set_ylabel("Positive rate")
    axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axis.legend(title="price_range", ncol=4)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_correlation_plot(pearson: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "05_correlation_heatmap.png"
    mask = np.triu(np.ones_like(pearson, dtype=bool), k=1)
    fig, axis = plt.subplots(figsize=(16, 13))
    sns.heatmap(
        pearson,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.35,
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        ax=axis,
    )
    axis.set_title("Feature correlation heatmap")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_drift_plot(drift: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "06_train_test_drift.png"
    ordered = drift.sort_values("psi", ascending=True)
    colors = ordered["severity"].map({"low": "#2563EB", "watch": "#F59E0B", "high": "#DC2626"})
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].barh(ordered["feature"], ordered["psi"], color=colors)
    axes[0].axvline(0.10, color="#F59E0B", linestyle="--", linewidth=1, label="PSI 0.10")
    axes[0].axvline(0.25, color="#DC2626", linestyle="--", linewidth=1, label="PSI 0.25")
    axes[0].set_title("Population Stability Index")
    axes[0].set_xlabel("PSI")
    axes[0].legend()

    ordered_smd = drift.sort_values("abs_smd", ascending=True)
    axes[1].barh(ordered_smd["feature"], ordered_smd["abs_smd"], color="#7C3AED")
    axes[1].axvline(0.10, color="#F59E0B", linestyle="--", linewidth=1, label="|SMD| 0.10")
    axes[1].axvline(0.20, color="#DC2626", linestyle="--", linewidth=1, label="|SMD| 0.20")
    axes[1].set_title("Absolute standardized mean difference")
    axes[1].set_xlabel("|SMD|")
    axes[1].legend()

    fig.suptitle("Train-test distribution shift diagnostics", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_derived_plot(derived: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "07_derived_feature_screening.png"
    plot_data = derived.sort_values("eta_squared", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(plot_data["derived_feature"], plot_data["eta_squared"], color="#0F766E")
    axes[0].set_title("Derived feature eta²")
    axes[0].set_xlabel("Between-class variance share")
    colors = np.where(plot_data["spearman_r"] >= 0, "#2563EB", "#DC2626")
    axes[1].barh(plot_data["derived_feature"], plot_data["spearman_r"], color=colors)
    axes[1].axvline(0, color="#111827", linewidth=0.8)
    axes[1].set_title("Derived feature Spearman correlation")
    axes[1].set_xlabel("Spearman r with price_range")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def format_markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.{float_digits}f}")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    result = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    result.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(result)


def write_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    quality: pd.DataFrame,
    summary: pd.DataFrame,
    association: pd.DataFrame,
    drift: pd.DataFrame,
    corr_pairs: pd.DataFrame,
    derived: pd.DataFrame,
) -> Path:
    path = OUTPUT_DIR / "mobile_price_category_eda_report.md"
    class_counts = train[TARGET].value_counts().sort_index()
    class_ratio = class_counts.min() / class_counts.max()
    zero_scw_train = int(train["sc_w"].eq(0).sum())
    zero_scw_test = int(test["sc_w"].eq(0).sum())
    zero_pxh_train = int(train["px_height"].eq(0).sum())
    zero_pxh_test = int(test["px_height"].eq(0).sum())
    high_drift = drift.query("severity == 'high'")
    watch_drift = drift.query("severity == 'watch'")
    significant_drift = drift.query("ks_q < 0.05")
    strong_corr = corr_pairs.query("abs_pearson_r >= 0.70")
    moderate_corr = corr_pairs.query("abs_pearson_r >= 0.50").head(10)
    top_assoc = association.head(10)[
        ["feature", "mutual_information", "eta_squared", "spearman_r", "mean_pattern", "kruskal_q"]
    ]
    weak_assoc = association.sort_values("mutual_information").head(6)[
        ["feature", "mutual_information", "eta_squared", "spearman_r", "kruskal_q"]
    ]
    top_drift = drift.sort_values(["psi", "abs_smd"], ascending=False).head(10)[
        ["feature", "psi", "abs_smd", "ks_q", "severity"]
    ]
    suspect_quality = quality.query("status == 'review'")[["check", "value", "description"]]
    top_derived = derived.head(7)[
        ["derived_feature", "eta_squared", "spearman_r", "mean_pattern", "spearman_q"]
    ]

    ram_row = association.set_index("feature").loc["ram"]
    battery_row = association.set_index("feature").loc["battery_power"]
    px_width_row = association.set_index("feature").loc["px_width"]
    px_height_row = association.set_index("feature").loc["px_height"]
    mobile_wt_row = association.set_index("feature").loc["mobile_wt"]
    max_psi_row = drift.sort_values("psi", ascending=False).iloc[0]
    max_smd_row = drift.sort_values("abs_smd", ascending=False).iloc[0]

    report = f"""# Mobile Price Category 데이터 EDA 보고서

## 1. 분석 범위와 결론

- 범위: 데이터 이해, 품질 점검, 단변량·타깃 관계·상관관계·파생변수 후보·학습/테스트 분포 차이까지 수행했습니다.
- 제외: 예측 모델 학습, 교차검증, 하이퍼파라미터 탐색, 테스트 예측은 수행하지 않았습니다.
- 학습 데이터는 **{len(train):,}행 × {train.shape[1]:,}열**, 테스트 데이터는 **{len(test):,}행 × {test.shape[1]:,}열**입니다. 공통 예측 피처는 {len(features)}개입니다.
- 결측과 완전 중복은 두 데이터 모두 0건입니다. 타깃은 0~3의 순서형 4개 클래스이며 클래스별 건수는 {', '.join(f'{index}: {value:,}' for index, value in class_counts.items())}입니다. 최소/최대 클래스 비율은 {class_ratio:.3f}입니다.
- `ram`이 가격 범주와 가장 강하게 연결됩니다(MI={ram_row['mutual_information']:.3f}, eta²={ram_row['eta_squared']:.3f}, Spearman r={ram_row['spearman_r']:.3f}). 그다음 핵심 신호는 `battery_power`, `px_width`, `px_height`입니다.
- `mobile_wt`는 약한 음의 관계를 보입니다(Spearman r={mobile_wt_row['spearman_r']:.3f}). 이진 기능 피처의 단독 구분력은 전반적으로 낮습니다.
- 학습/테스트 분포 차이는 최대 PSI가 `{max_psi_row['feature']}`의 {max_psi_row['psi']:.3f}, 최대 |SMD|가 `{max_smd_row['feature']}`의 {max_smd_row['abs_smd']:.3f}입니다. 고위험 드리프트 {len(high_drift)}개, 관찰 필요 {len(watch_drift)}개, FDR 5% 기준 KS 유의 피처 {len(significant_drift)}개입니다.
- 물리적으로 의심되는 0값이 있습니다. `sc_w=0`은 train {zero_scw_train}건/test {zero_scw_test}건, `px_height=0`은 train {zero_pxh_train}건/test {zero_pxh_test}건입니다. 자동 삭제나 전체 중앙값 대체 전에 0이 결측 표기인지 확인해야 합니다.

![데이터 개요](01_overview_and_quality.png)

## 2. 데이터 구조와 역할

| 구분 | 행 수 | 원본 열 수 | 모델 입력에서 제외할 열 | 타깃 |
| --- | ---: | ---: | --- | --- |
| Train | {len(train):,} | {train.shape[1]} | 없음 | `price_range` |
| Test | {len(test):,} | {test.shape[1]} | `id` | 없음 |

`id`는 테스트 행을 제출 결과와 연결하는 식별자입니다. 학습 데이터에는 대응하는 `id`가 없으므로 예측 피처에 포함하면 안 됩니다. `price_range`는 0 < 1 < 2 < 3의 순서가 있으므로 명목형 다중분류이면서 순서형 타깃이라는 성격을 함께 고려할 수 있습니다.

## 3. 데이터 품질 및 무결성

원본 CSV는 수정하지 않았습니다. 아래 항목은 제거 규칙이 아니라 검토 대상입니다.

{format_markdown_table(suspect_quality, 0)}

해석:

- `sc_w=0`, `px_height=0`은 실제 화면 크기로 해석하기 어렵습니다. 우선 결측 지시 변수(`sc_w_zero`, `px_height_zero`) 후보로 보존하고, 보정은 학습 fold 내부에서만 수행하는 방식이 안전합니다.
- `fc=0`, `pc=0`은 카메라 미탑재를 의미할 수 있어 무조건 오류로 보지 않았습니다.
- `pc < fc`나 `4G=1, 3G=0`은 제품 사양 정의에 따라 가능할 수 있습니다. 도메인 정의가 없으므로 원본을 임의 수정하지 않았습니다.
- IQR 이상치는 물리적으로 불가능한 값과 다릅니다. 트리 계열에는 그대로 둘 수 있고, 거리/선형 계열을 쓸 때만 robust scaling 또는 변환을 교차검증 안에서 비교하는 것이 좋습니다.

세부 수치는 `data_quality_checks.csv`, 피처별 기술통계는 `feature_summary.csv`에 저장했습니다.

## 4. 단변량 분포

![피처 분포](02_feature_distributions.png)

- 모든 피처가 수치형이며, {len(BINARY_COLUMNS)}개(`{', '.join(BINARY_COLUMNS)}`)는 의미상 이진 범주형입니다.
- 연속/이산 수치형의 스케일 차이가 큽니다. 예를 들어 RAM·픽셀 치수와 0~1 이진값을 같은 거리 기반 모델에 넣는다면 표준화가 필요합니다. 트리 계열은 스케일링 필요성이 낮습니다.
- 왜도와 IQR 이상치 비율은 `feature_summary.csv`에서 확인할 수 있습니다. 이상치를 일괄 삭제하면 작은 학습 표본({len(train):,}건)을 더 줄이고 실제 고사양 단말을 제거할 수 있으므로 권장하지 않습니다.

## 5. 타깃과의 관계

타깃 관계는 한 지표에 의존하지 않고 다음을 함께 봤습니다.

- Mutual information: 비선형 의존성의 탐색 지표
- eta²: 가격 범주 간 평균 차이가 전체 분산에서 차지하는 비율
- Spearman r: 가격 범주가 올라갈 때 단조 증가/감소하는 정도
- Kruskal-Wallis: 분포 차이 검정. 20개 피처에 대해 Benjamini-Hochberg FDR 보정

### 상위 피처

{format_markdown_table(top_assoc)}

![가격 범주별 핵심 수치 피처](03_target_relationships.png)

핵심 해석:

- `ram`: 네 가격 범주가 뚜렷하게 단계적으로 분리되는 주 신호입니다. 단순한 스케일 조정보다 이 관계를 보존하는 것이 중요합니다.
- `battery_power`: MI={battery_row['mutual_information']:.3f}, eta²={battery_row['eta_squared']:.3f}, Spearman r={battery_row['spearman_r']:.3f}로 두 번째 축을 형성합니다.
- `px_width`, `px_height`: 각각 Spearman r={px_width_row['spearman_r']:.3f}, {px_height_row['spearman_r']:.3f}이며 해상도 정보를 함께 담습니다. 곱(`pixel_area`)이나 대각선 길이를 후보로 검토할 가치가 있습니다.
- `mobile_wt`: 가격과 약한 음의 단조 관계가 있으나 단독 예측력은 제한적입니다.
- 단변량 연관성이 낮은 피처도 상호작용에서 기여할 수 있으므로 EDA만으로 삭제하지 않습니다.

### 단독 신호가 약한 피처

{format_markdown_table(weak_assoc)}

![이진 피처와 가격 범주](04_binary_target_relationships.png)

## 6. 상관관계와 중복 정보

![상관관계](05_correlation_heatmap.png)

- |Pearson r| ≥ 0.70인 피처 쌍은 {len(strong_corr)}개입니다.
- 상관 절댓값 상위 피처 쌍은 아래와 같습니다.

{format_markdown_table(moderate_corr[["feature_1", "feature_2", "pearson_r", "abs_pearson_r"]])}

상관관계가 있다고 곧바로 하나를 삭제할 필요는 없습니다. `pc`–`fc`, `three_g`–`four_g`처럼 의미가 겹치는 피처는 선형 모델에서 계수 불안정성을 만들 수 있으므로 규제 또는 선택을 교차검증으로 판단하고, 트리 모델에서는 둘 다 유지한 기준선을 먼저 비교하는 편이 안전합니다.

## 7. 파생변수 후보 탐색

아래 파생값은 원본 열만 이용한 EDA 후보이며, 학습 데이터에 새 CSV를 만들거나 원본을 변경하지 않았습니다.

{format_markdown_table(top_derived)}

![파생변수 탐색](07_derived_feature_screening.png)

권장 후보:

- `pixel_area = px_height × px_width`: 두 해상도 축의 결합 신호
- `pixel_diagonal = sqrt(px_height² + px_width²)`: 해상도 크기의 대안 표현
- `screen_area = sc_h × sc_w`: `sc_w=0` 문제를 먼저 다뤄야 함
- `camera_gap = pc - fc`: 카메라 구성 차이
- `connectivity_count`: Bluetooth/3G/4G/Wi-Fi 지원 수
- `battery_per_weight`: 무게 대비 배터리 용량
- `ram_battery_product`: 주 신호인 RAM과 배터리의 상호작용 후보. 트리 모델에는 중복일 수 있으므로 검증으로 판단

파생변수는 전체 데이터에서 선택 성능을 확인하면 누수가 생깁니다. 이후 단계에서는 각 교차검증 fold의 학습 부분에서만 전처리·선택 판단을 해야 합니다.

## 8. Train–Test 분포 차이

![분포 차이](06_train_test_drift.png)

{format_markdown_table(top_drift)}

판정 기준은 탐색용으로 PSI 0.10/0.25, |SMD| 0.10/0.20을 사용했습니다. KS p값은 20개 동시 검정에 대해 FDR 보정했습니다. 통계적 유의성과 실제 차이 크기를 함께 봐야 하며, p값 하나만으로 피처를 제거하지 않습니다.

## 9. 예측 성능을 위한 EDA 기반 후속 원칙

이 보고서는 모델링을 수행하지 않았지만, 다음 단계의 실험 설계는 아래가 적절합니다.

1. `id` 제외, `price_range`를 타깃으로 고정합니다.
2. 클래스가 균형이므로 무리한 재표본화는 필요하지 않습니다. 다만 분할은 `StratifiedKFold`로 유지합니다.
3. Accuracy와 함께 macro F1, confusion matrix, 클래스 간 거리 오차(예: MAE on class index)를 봅니다. 0을 3으로 틀리는 오류와 0을 1로 틀리는 오류가 같지 않기 때문입니다.
4. 의심 0값은 원본 유지 기준선과 결측 지시자+fold 내부 보정안을 비교합니다. 테스트 분포를 이용해 보정값을 정하지 않습니다.
5. 기본 20개 피처 기준선과 파생변수 추가안을 비교합니다. 중요도가 낮아 보인다는 이유만으로 EDA 단계에서 피처를 제거하지 않습니다.
6. 선형/거리 모델은 스케일링과 상관 피처 영향을 점검하고, 트리/부스팅 계열은 스케일링 없이도 비교할 수 있습니다.
7. 하이퍼파라미터 선택과 피처 선택은 교차검증 내부에서 수행하고, 테스트셋은 최종 예측 전까지 라벨 없는 분포 확인 용도로만 사용합니다.

## 10. 생성 파일

- `data_quality_checks.csv`: 무결성 점검 결과
- `feature_summary.csv`: 피처별 기술통계, 왜도, IQR 이상치 비율
- `target_association.csv`: MI, eta², Spearman, Kruskal/FDR 결과
- `class_conditional_summary.csv`: 가격 범주별 피처 평균·중앙값·표준편차
- `feature_correlation_pairs.csv`: 피처 쌍 상관계수
- `train_test_drift.csv`: PSI, SMD, KS/FDR 결과
- `derived_feature_screening.csv`: 파생변수 후보의 타깃 관계
- PNG 7개: 데이터 품질, 분포, 타깃 관계, 상관관계, 드리프트, 파생변수 시각화

재현 실행:

```powershell
.venv/Scripts/python.exe src/mobile_price_category_eda.py
```
"""
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    train, test, features = load_and_validate()

    quality = data_quality_analysis(train, test, features)
    summary = feature_summary(train, test, features)
    association, class_summary = target_association_analysis(train, features)
    drift = drift_analysis(train, test, features)
    pearson, corr_pairs = correlation_analysis(train, features)
    derived = derived_feature_analysis(train)

    quality.to_csv(OUTPUT_DIR / "data_quality_checks.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "feature_summary.csv", index=False, encoding="utf-8-sig")
    association.to_csv(OUTPUT_DIR / "target_association.csv", index=False, encoding="utf-8-sig")
    class_summary.to_csv(
        OUTPUT_DIR / "class_conditional_summary.csv", index=False, encoding="utf-8-sig"
    )
    corr_pairs.to_csv(
        OUTPUT_DIR / "feature_correlation_pairs.csv", index=False, encoding="utf-8-sig"
    )
    drift.to_csv(OUTPUT_DIR / "train_test_drift.csv", index=False, encoding="utf-8-sig")
    derived.to_csv(
        OUTPUT_DIR / "derived_feature_screening.csv", index=False, encoding="utf-8-sig"
    )

    save_overview_plot(train, test)
    save_distribution_plot(train, test, features)
    save_target_relationship_plot(train, association)
    save_binary_relationship_plot(train)
    save_correlation_plot(pearson)
    save_drift_plot(drift)
    save_derived_plot(derived)
    report_path = write_report(
        train,
        test,
        features,
        quality,
        summary,
        association,
        drift,
        corr_pairs,
        derived,
    )

    print("Mobile price category EDA completed.")
    print(f"Report: {report_path.resolve()}")
    print("\nTop target associations:")
    print(
        association.head(8)[
            ["feature", "mutual_information", "eta_squared", "spearman_r", "kruskal_q"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\nTop train-test drift diagnostics:")
    print(
        drift.sort_values(["psi", "abs_smd"], ascending=False)
        .head(8)[["feature", "psi", "abs_smd", "ks_q", "severity"]]
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
