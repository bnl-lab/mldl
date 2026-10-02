from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from statsmodels.stats.outliers_influence import variance_inflation_factor


DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTPUT_DIR = Path("outputs/telco_churn_eda")
TARGET = "Churn"
ID_COLUMN = "customerID"

DEMOGRAPHIC_COLUMNS = ["gender", "SeniorCitizen", "Partner", "Dependents"]
ADD_ON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
SERVICE_COLUMNS = ["InternetService", *ADD_ON_COLUMNS]
CONTRACT_BILLING_COLUMNS = ["Contract", "PaperlessBilling", "PaymentMethod"]
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def configure_visualization() -> None:
    """Configure consistent plotting, including Korean font support."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    sns.set_theme(style="whitegrid", font="Malgun Gothic")


def print_section(title: str) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")


def load_and_validate_data() -> pd.DataFrame:
    """Load raw data and validate its expected schema and key integrity."""
    data = pd.read_csv(DATA_PATH, dtype={"TotalCharges": "string"})
    required_columns = {
        ID_COLUMN,
        TARGET,
        *DEMOGRAPHIC_COLUMNS,
        *SERVICE_COLUMNS,
        *CONTRACT_BILLING_COLUMNS,
        *NUMERIC_COLUMNS,
    }
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")

    print_section("Phase 1. 데이터 건전성 및 무결성 점검")
    print(f"원본 크기             : {data.shape[0]:,}행 × {data.shape[1]:,}열")
    print(f"customerID 고유 개수 : {data[ID_COLUMN].nunique():,}")
    print(f"customerID 중복 개수 : {data[ID_COLUMN].duplicated().sum():,}")
    print(f"customerID 결측 개수 : {data[ID_COLUMN].isna().sum():,}")
    duplicate_features = data.drop(columns=ID_COLUMN).duplicated().sum()
    print(f"ID 제외 완전 중복 행 : {duplicate_features:,}")

    if data[ID_COLUMN].isna().any() or data[ID_COLUMN].duplicated().any():
        raise ValueError("customerID는 결측 없이 행마다 고유해야 합니다.")
    return data


def clean_total_charges(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert TotalCharges to float and resolve structurally missing charges."""
    cleaned = data.copy()
    raw_total = cleaned["TotalCharges"].astype("string")
    blank_mask = raw_total.str.strip().eq("").fillna(False)
    converted = pd.to_numeric(raw_total.str.strip(), errors="coerce")
    invalid_mask = converted.isna()
    new_customer_invalid = invalid_mask & cleaned["tenure"].eq(0)
    unexpected_invalid = invalid_mask & ~cleaned["tenure"].eq(0)

    print(f"\nTotalCharges 공백 문자열       : {int(blank_mask.sum()):,}건")
    print(f"수치 변환 불가/결측 전체       : {int(invalid_mask.sum()):,}건")
    print(f"그중 tenure == 0 신규 고객    : {int(new_customer_invalid.sum()):,}건")
    print(f"그중 tenure > 0 비정상 결측   : {int(unexpected_invalid.sum()):,}건")

    # tenure가 0이면 아직 누적 청구가 없다는 도메인 의미가 명확하므로 0이 타당하다.
    converted.loc[new_customer_invalid] = 0.0
    if unexpected_invalid.any():
        # 예외 데이터만 중앙값으로 보완한다. 운영 모델에서는 train fold에서만 중앙값을 학습해야 한다.
        median_total = float(converted.loc[~invalid_mask].median())
        converted.loc[unexpected_invalid] = median_total
        print(
            f"tenure > 0인 비정상 결측은 전체 유효값 중앙값 "
            f"{median_total:,.2f}로 대체했습니다."
        )

    cleaned["TotalCharges"] = converted.astype(float)
    print(
        "처리 근거: tenure=0 고객은 서비스 시작 직후라 누적 청구액이 발생하지 않은 "
        "구조적 결측이므로 0으로 대체했습니다."
    )
    return cleaned, {
        "blank_count": int(blank_mask.sum()),
        "new_customer_blank_count": int(new_customer_invalid.sum()),
        "unexpected_invalid_count": int(unexpected_invalid.sum()),
    }


def diagnose_target_balance(data: pd.DataFrame) -> pd.Series:
    counts = data[TARGET].value_counts()
    rates = data[TARGET].value_counts(normalize=True).mul(100)
    target_summary = pd.DataFrame({"Count": counts, "Rate(%)": rates})
    print("\n[타깃 클래스 분포]")
    print(target_summary.to_string(float_format=lambda value: f"{value:,.2f}"))

    minority_majority_ratio = counts.min() / counts.max()
    if minority_majority_ratio >= 0.67:
        diagnosis = "균형에 가까움"
    elif minority_majority_ratio >= 0.30:
        diagnosis = "중간 수준 불균형"
    else:
        diagnosis = "강한 불균형"
    print(f"소수/다수 클래스 비율: {minority_majority_ratio:.3f} → {diagnosis}")
    print(
        "모델링 권고: 층화 분할을 사용하고 Accuracy 단독 대신 ROC-AUC, "
        "PR-AUC, Recall, F1을 함께 평가하십시오."
    )
    return rates


def print_categorical_distributions(
    data: pd.DataFrame,
    group_name: str,
    columns: list[str],
) -> None:
    print(f"\n[{group_name} 범주 분포]")
    for column in columns:
        summary = (
            data[column]
            .value_counts(dropna=False)
            .rename_axis("Category")
            .to_frame("Count")
        )
        summary["Rate(%)"] = summary["Count"].div(len(data)).mul(100)
        print(f"\n- {column}")
        print(summary.to_string(float_format=lambda value: f"{value:,.2f}"))


def plot_categorical_grid(
    data: pd.DataFrame,
    columns: list[str],
    title: str,
    output_path: Path,
    ncols: int = 3,
) -> None:
    nrows = int(np.ceil(len(columns) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.4 * nrows))
    flat_axes = np.atleast_1d(axes).ravel()

    for axis, column in zip(flat_axes, columns):
        order = data[column].value_counts().index
        sns.countplot(data=data, x=column, order=order, color="#4C72B0", ax=axis)
        axis.set_title(column)
        axis.set_xlabel("")
        axis.set_ylabel("고객 수")
        axis.tick_params(axis="x", rotation=25)
        for container in axis.containers:
            axis.bar_label(container, fmt="{:,.0f}", padding=3, fontsize=8)

    for axis in flat_axes[len(columns) :]:
        axis.set_visible(False)

    figure.suptitle(title, fontsize=17, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def describe_numeric_features(data: pd.DataFrame) -> pd.DataFrame:
    description = data[NUMERIC_COLUMNS].describe().T
    description["skewness"] = data[NUMERIC_COLUMNS].skew()
    description["excess_kurtosis"] = data[NUMERIC_COLUMNS].kurt()

    def shape_description(row: pd.Series) -> str:
        if row["skewness"] > 1:
            skew_shape = "강한 우측 왜도"
        elif row["skewness"] > 0.5:
            skew_shape = "중간 우측 왜도"
        elif row["skewness"] < -1:
            skew_shape = "강한 좌측 왜도"
        elif row["skewness"] < -0.5:
            skew_shape = "중간 좌측 왜도"
        else:
            skew_shape = "대칭에 가까움"

        if row["excess_kurtosis"] > 1:
            peak_shape = "뾰족하고 꼬리가 두꺼움"
        elif row["excess_kurtosis"] < -1:
            peak_shape = "평평한 봉우리"
        else:
            peak_shape = "정규분포와 유사한 첨도"
        return f"{skew_shape}; {peak_shape}"

    description["shape_summary"] = description.apply(shape_description, axis=1)
    print("\n[연속형 변수 기술통계·왜도·첨도]")
    print(description.to_string(float_format=lambda value: f"{value:,.3f}"))
    return description


def plot_numeric_grid(data: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(18, 9))
    for index, column in enumerate(NUMERIC_COLUMNS):
        sns.histplot(data=data, x=column, kde=True, color="#4C72B0", ax=axes[0, index])
        axes[0, index].set_title(f"{column} 분포")
        axes[0, index].set_ylabel("고객 수")

        sns.boxplot(data=data, x=column, color="#55A868", ax=axes[1, index])
        axes[1, index].set_title(f"{column} 이상치 점검")
        axes[1, index].set_xlabel(column)

    figure.suptitle("연속형 변수 분포와 이상치", fontsize=17, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def make_rate_table(data: pd.DataFrame, feature: str) -> pd.DataFrame:
    return (
        data.assign(Churn_Flag=data[TARGET].eq("Yes").astype(int))
        .groupby(feature, observed=True)["Churn_Flag"]
        .agg(Churn_Rate="mean", Customer_Count="size")
        .reset_index()
    )


def calculate_monthly_charge_threshold(data: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    lower = np.floor(data["MonthlyCharges"].min() / 10) * 10
    upper = np.ceil(data["MonthlyCharges"].max() / 10) * 10 + 10
    bins = np.arange(lower, upper, 10)
    charge_band = pd.cut(data["MonthlyCharges"], bins=bins, right=False)
    rate_table = (
        data.assign(
            Charge_Band=charge_band,
            Churn_Flag=data[TARGET].eq("Yes").astype(int),
        )
        .groupby("Charge_Band", observed=True)["Churn_Flag"]
        .agg(Churn_Rate="mean", Customer_Count="size")
        .reset_index()
    )
    rate_table["Rate_Change"] = rate_table["Churn_Rate"].diff()
    reliable = rate_table[rate_table["Customer_Count"] >= 100]
    threshold_row = reliable.loc[reliable["Rate_Change"].idxmax()]
    threshold = float(threshold_row["Charge_Band"].left)
    return rate_table, threshold


def plot_churn_rate_bar(
    axis: plt.Axes,
    data: pd.DataFrame,
    feature: str,
    title: str,
) -> None:
    rates = make_rate_table(data, feature)
    sns.barplot(data=rates, x=feature, y="Churn_Rate", color="#C44E52", ax=axis)
    axis.set_title(title)
    axis.set_xlabel("")
    axis.set_ylabel("이탈률")
    axis.set_ylim(0, max(0.6, rates["Churn_Rate"].max() * 1.2))
    axis.tick_params(axis="x", rotation=25)
    axis.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    for container in axis.containers:
        labels = [f"{bar.get_height():.1%}" for bar in container]
        axis.bar_label(container, labels=labels, padding=3, fontsize=8)


def run_bivariate_analysis(data: pd.DataFrame, output_path: Path) -> dict[str, object]:
    print_section("Phase 3. 타깃 연계 이변량 분석")
    charge_rates, charge_threshold = calculate_monthly_charge_threshold(data)

    tenure_rates = make_rate_table(
        data.assign(
            Tenure_Segment=pd.cut(
                data["tenure"],
                bins=[-1, 12, 24, 48, np.inf],
                labels=["0-12", "13-24", "25-48", "49+"],
            )
        ),
        "Tenure_Segment",
    )
    print("[가입 기간 코호트별 이탈률]")
    print(tenure_rates.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\n월 요금 이탈률 최대 상승 변곡점(10달러 구간 기준): 약 ${charge_threshold:,.0f}")
    print(charge_rates.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    figure, axes = plt.subplots(2, 3, figsize=(20, 12))
    sns.kdeplot(
        data=data,
        x="tenure",
        hue=TARGET,
        hue_order=["No", "Yes"],
        palette={"No": "#4C72B0", "Yes": "#DD8452"},
        common_norm=False,
        fill=True,
        alpha=0.25,
        ax=axes[0, 0],
    )
    axes[0, 0].axvspan(0, 12, color="gold", alpha=0.16, label="초기 0-12개월")
    axes[0, 0].axvline(48, color="green", linestyle="--", label="49개월+ 저항선")
    axes[0, 0].set_title("가입 기간별 이탈/유지 KDE")
    axes[0, 0].set_xlabel("가입 기간(개월)")
    axes[0, 0].legend(
        handles=[
            Patch(facecolor="#4C72B0", alpha=0.3, label="유지(No)"),
            Patch(facecolor="#DD8452", alpha=0.3, label="이탈(Yes)"),
            Patch(facecolor="gold", alpha=0.2, label="초기 0-12개월"),
            Line2D([0], [0], color="green", linestyle="--", label="49개월+ 저항선"),
        ]
    )

    sns.kdeplot(
        data=data,
        x="MonthlyCharges",
        hue=TARGET,
        hue_order=["No", "Yes"],
        palette={"No": "#4C72B0", "Yes": "#DD8452"},
        common_norm=False,
        fill=True,
        alpha=0.25,
        ax=axes[0, 1],
    )
    axes[0, 1].axvline(charge_threshold, color="red", linestyle="--")
    axes[0, 1].set_title("월 요금별 이탈/유지 KDE")
    axes[0, 1].set_xlabel("월 요금")
    axes[0, 1].legend(
        handles=[
            Patch(facecolor="#4C72B0", alpha=0.3, label="유지(No)"),
            Patch(facecolor="#DD8452", alpha=0.3, label="이탈(Yes)"),
            Line2D(
                [0],
                [0],
                color="red",
                linestyle="--",
                label=f"변곡점 약 ${charge_threshold:,.0f}",
            ),
        ]
    )

    charge_plot = charge_rates.copy()
    charge_plot["Band"] = charge_plot["Charge_Band"].astype(str)
    sns.barplot(data=charge_plot, x="Band", y="Churn_Rate", color="#DD8452", ax=axes[0, 2])
    axes[0, 2].set_title(f"월 요금 구간별 이탈률 (변곡점 약 ${charge_threshold:,.0f})")
    axes[0, 2].set_xlabel("월 요금 구간")
    axes[0, 2].set_ylabel("이탈률")
    axes[0, 2].tick_params(axis="x", rotation=45)
    axes[0, 2].yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")

    plot_churn_rate_bar(axes[1, 0], data, "Contract", "계약 유형별 이탈률")
    plot_churn_rate_bar(axes[1, 1], data, "PaymentMethod", "결제 방법별 이탈률")
    plot_churn_rate_bar(axes[1, 2], data, "InternetService", "인터넷 서비스별 이탈률")

    figure.suptitle("이탈과 핵심 변수의 관계", fontsize=18, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return {
        "charge_threshold": charge_threshold,
        "charge_rates": charge_rates,
        "tenure_rates": tenure_rates,
    }


def calculate_vif(data: pd.DataFrame) -> pd.DataFrame:
    numeric_data = data[NUMERIC_COLUMNS].astype(float)
    standardized = (numeric_data - numeric_data.mean()) / numeric_data.std(ddof=0)
    return pd.DataFrame(
        {
            "Feature": standardized.columns,
            "VIF": [
                variance_inflation_factor(standardized.to_numpy(), index)
                for index in range(standardized.shape[1])
            ],
        }
    ).sort_values("VIF", ascending=False)


def run_multivariate_analysis(
    data: pd.DataFrame,
    output_path: Path,
) -> dict[str, object]:
    print_section("Phase 4. 다변량 상호작용 및 다중공선성")
    correlations = data[NUMERIC_COLUMNS].corr()
    vif_table = calculate_vif(data)
    interaction = pd.pivot_table(
        data.assign(Churn_Flag=data[TARGET].eq("Yes").astype(int)),
        index="Contract",
        columns="InternetService",
        values="Churn_Flag",
        aggfunc="mean",
        observed=True,
    )
    top_segment = interaction.stack().idxmax()
    top_segment_rate = float(interaction.stack().max())

    print("[수치형 변수 상관계수]")
    print(correlations.to_string(float_format=lambda value: f"{value:.3f}"))
    print("\n[VIF: 5 이상 주의, 10 이상 강한 다중공선성]")
    print(vif_table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(
        f"\n최대 위험 세그먼트: Contract={top_segment[0]}, "
        f"InternetService={top_segment[1]}, 이탈률={top_segment_rate:.1%}"
    )

    figure, axes = plt.subplots(2, 2, figsize=(17, 13))
    sns.heatmap(
        correlations,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("수치형 변수 상관계수")

    sns.barplot(data=vif_table, x="VIF", y="Feature", color="#8172B2", ax=axes[0, 1])
    axes[0, 1].axvline(5, color="orange", linestyle="--", label="주의 VIF=5")
    axes[0, 1].axvline(10, color="red", linestyle="--", label="강함 VIF=10")
    axes[0, 1].set_title("분산 팽창 계수(VIF)")
    axes[0, 1].legend()

    sns.heatmap(
        interaction,
        annot=True,
        fmt=".1%",
        cmap="YlOrRd",
        vmin=0,
        vmax=max(0.6, top_segment_rate),
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Contract × InternetService 이탈률")
    axes[1, 0].set_xlabel("인터넷 서비스")
    axes[1, 0].set_ylabel("계약 유형")

    lockin_data = []
    for feature in ["OnlineSecurity", "TechSupport"]:
        subset = data[data[feature].isin(["No", "Yes"])]
        rates = make_rate_table(subset, feature)
        rates["Feature"] = feature
        rates = rates.rename(columns={feature: "Subscription"})
        lockin_data.append(rates)
    lockin_rates = pd.concat(lockin_data, ignore_index=True)
    sns.barplot(
        data=lockin_rates,
        x="Feature",
        y="Churn_Rate",
        hue="Subscription",
        hue_order=["No", "Yes"],
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("보안·기술지원 가입의 Lock-in 효과")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("이탈률")
    axes[1, 1].yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")

    figure.suptitle("다변량·상호작용 분석", fontsize=18, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return {
        "correlations": correlations,
        "vif": vif_table,
        "interaction": interaction,
        "top_segment": top_segment,
        "top_segment_rate": top_segment_rate,
        "lockin_rates": lockin_rates,
    }


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic, target-independent features for later modeling."""
    engineered = data.drop(columns=ID_COLUMN).copy()
    engineered["Tenure_Group"] = pd.cut(
        engineered["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["0-12개월", "13-24개월", "25-48개월", "49개월 이상"],
        ordered=True,
    )
    engineered["Service_Count"] = engineered[ADD_ON_COLUMNS].eq("Yes").sum(axis=1)
    denominator = engineered["tenure"] * engineered["MonthlyCharges"] + 1e-5
    engineered["Avg_Monthly_Ratio"] = engineered["TotalCharges"] / denominator
    return engineered


def print_business_insights(
    data: pd.DataFrame,
    bivariate: dict[str, object],
    multivariate: dict[str, object],
) -> None:
    churn_rate = data[TARGET].eq("Yes").mean()
    tenure_rates = bivariate["tenure_rates"].set_index("Tenure_Segment")["Churn_Rate"]
    contract_rates = make_rate_table(data, "Contract").set_index("Contract")["Churn_Rate"]
    payment_rates = make_rate_table(data, "PaymentMethod").set_index("PaymentMethod")["Churn_Rate"]
    automatic_rate = payment_rates[
        payment_rates.index.str.contains("automatic", case=False)
    ].mean()
    lockin = multivariate["lockin_rates"].pivot(
        index="Feature",
        columns="Subscription",
        values="Churn_Rate",
    )
    max_vif_row = multivariate["vif"].iloc[0]

    print_section("주요 비즈니스 인사이트 요약")
    insights = [
        f"전체 고객 이탈률은 {churn_rate:.1%}로, 단순 정확도보다 이탈 클래스의 "
        "Recall/PR-AUC를 함께 관리해야 합니다.",
        f"가입 0-12개월 이탈률은 {tenure_rates.loc['0-12']:.1%}, 49개월 이상은 "
        f"{tenure_rates.loc['49+']:.1%}입니다. 초기 온보딩 1년이 핵심 개입 구간입니다.",
        f"월 요금을 10달러 단위로 나눴을 때 이탈률의 가장 큰 상승 지점은 약 "
        f"${bivariate['charge_threshold']:,.0f}입니다. 이 구간 전후의 가격·혜택 반응을 "
        "별도 실험할 가치가 있습니다.",
        f"월 단위 계약 이탈률은 {contract_rates.loc['Month-to-month']:.1%}이며, "
        f"2년 계약은 {contract_rates.loc['Two year']:.1%}입니다. 장기 계약 전환 인센티브가 "
        "높은 우선순위를 가집니다.",
        f"Electronic check 이탈률은 {payment_rates.loc['Electronic check']:.1%}, "
        f"자동이체 평균은 {automatic_rate:.1%}입니다. 결제수단 전환 캠페인을 검토하십시오.",
        f"보안 서비스 미가입/가입 이탈률은 {lockin.loc['OnlineSecurity', 'No']:.1%}/"
        f"{lockin.loc['OnlineSecurity', 'Yes']:.1%}, 기술지원은 "
        f"{lockin.loc['TechSupport', 'No']:.1%}/{lockin.loc['TechSupport', 'Yes']:.1%}입니다. "
        "다만 이는 인과효과가 아닌 관찰 연관성입니다.",
        f"최대 위험 조합은 {multivariate['top_segment'][0]} 계약 × "
        f"{multivariate['top_segment'][1]} 인터넷이며 이탈률은 "
        f"{multivariate['top_segment_rate']:.1%}입니다.",
        f"가장 높은 VIF는 {max_vif_row['Feature']}의 {max_vif_row['VIF']:.2f}입니다. "
        "선형 모델에서는 규제(Ridge/Lasso), 변수 제거 또는 파생 변수 재설계를 검토하십시오.",
    ]
    for number, insight in enumerate(insights, start=1):
        print(fill(f"{number}. {insight}", width=100))


def main() -> None:
    configure_visualization()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = load_and_validate_data()
    data, health_summary = clean_total_charges(raw_data)
    target_rates = diagnose_target_balance(data)

    print_section("Phase 2. 도메인 그룹별 일변량 분석")
    print_categorical_distributions(data, "인구통계학", DEMOGRAPHIC_COLUMNS)
    print_categorical_distributions(data, "서비스 구독", SERVICE_COLUMNS)
    print_categorical_distributions(data, "계약 및 청구", CONTRACT_BILLING_COLUMNS)
    numeric_summary = describe_numeric_features(data)

    plot_categorical_grid(
        data,
        DEMOGRAPHIC_COLUMNS,
        "인구통계학 변수 분포",
        OUTPUT_DIR / "01_demographics.png",
        ncols=2,
    )
    plot_categorical_grid(
        data,
        SERVICE_COLUMNS,
        "인터넷 및 부가 서비스 분포",
        OUTPUT_DIR / "02_services.png",
        ncols=3,
    )
    plot_categorical_grid(
        data,
        CONTRACT_BILLING_COLUMNS,
        "계약 및 청구 변수 분포",
        OUTPUT_DIR / "03_contract_billing.png",
        ncols=3,
    )
    plot_numeric_grid(data, OUTPUT_DIR / "04_numeric_distributions.png")

    bivariate = run_bivariate_analysis(data, OUTPUT_DIR / "05_bivariate_churn.png")
    multivariate = run_multivariate_analysis(data, OUTPUT_DIR / "06_multivariate.png")

    print_section("Phase 5. 머신러닝 연계 파생 변수 생성")
    engineered = engineer_features(data)
    processed_path = OUTPUT_DIR / "telco_churn_modeling_data.csv"
    engineered.to_csv(processed_path, index=False)
    bivariate["charge_rates"].to_csv(
        OUTPUT_DIR / "monthly_charge_churn_rates.csv",
        index=False,
    )
    multivariate["vif"].to_csv(OUTPUT_DIR / "numeric_vif.csv", index=False)
    numeric_summary.to_csv(OUTPUT_DIR / "numeric_summary.csv")

    print("[파생 변수 요약]")
    print(engineered[["Tenure_Group", "Service_Count", "Avg_Monthly_Ratio"]].head())
    print("\n[최종 데이터 info]")
    engineered.info()
    missing_count = int(engineered.isna().sum().sum())
    print(f"\n최종 전체 결측치 수: {missing_count:,}")
    print(f"최종 데이터 크기  : {engineered.shape[0]:,}행 × {engineered.shape[1]:,}열")
    print(f"customerID 제거 여부: {ID_COLUMN not in engineered.columns}")
    print(
        "누수 방지: Churn을 이용한 파생 변수는 만들지 않았으며, 범주 인코딩과 "
        "StandardScaler는 train/test 분할 후 Pipeline 내부에서 fit해야 합니다."
    )

    print_business_insights(data, bivariate, multivariate)

    print_section("산출물")
    print(f"정제·파생 데이터: {processed_path}")
    print(f"시각화 및 통계표 : {OUTPUT_DIR}")
    print(
        f"건전성 요약      : TotalCharges 공백 {health_summary['blank_count']}건, "
        f"신규 고객 구조적 결측 {health_summary['new_customer_blank_count']}건, "
        f"예외 결측 {health_summary['unexpected_invalid_count']}건"
    )
    print(f"타깃 이탈률      : {target_rates.get('Yes', np.nan):.2f}%")


if __name__ == "__main__":
    main()
