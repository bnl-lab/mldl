"""Telco Customer Churn 데이터를 CNN 입력용 수치형 CSV로 전처리한다."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_PATH = ROOT / "data" / "telco_churn_cnn_preprocessed.csv"
FEATURE_INFO_PATH = ROOT / "data" / "telco_churn_cnn_feature_info.json"


def preprocess_telco_data(input_path: Path) -> tuple[pd.DataFrame, dict]:
    if not input_path.exists():
        raise FileNotFoundError(f"원본 CSV를 찾을 수 없습니다: {input_path}")

    dataframe = pd.read_csv(input_path)
    required_columns = {"Churn", "TotalCharges"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(missing_columns)}")

    # customerID는 식별자일 뿐 예측 피처가 아니므로 제거한다.
    features = dataframe.drop(columns=["Churn", "customerID"], errors="ignore").copy()

    # 공백 문자열을 결측치로 바꾼 뒤 TotalCharges를 숫자형으로 변환한다.
    features = features.replace(r"^\s*$", pd.NA, regex=True)
    features["TotalCharges"] = pd.to_numeric(features["TotalCharges"], errors="coerce")

    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=["number"]).columns.tolist()

    # 수치형은 중앙값, 범주형은 최빈값으로 결측치를 처리한다.
    for column in numeric_columns:
        features[column] = features[column].fillna(features[column].median())
    for column in categorical_columns:
        mode = features[column].mode(dropna=True)
        features[column] = features[column].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    # 범주형 컬럼을 원-핫 인코딩한다. drop_first=False로 모든 범주를 보존한다.
    encoded_features = pd.get_dummies(
        features,
        columns=categorical_columns,
        drop_first=False,
        dtype=float,
    )

    # CNN에 넣을 수 있도록 모든 입력 피처를 표준화한다.
    scaler = StandardScaler()
    encoded_features = pd.DataFrame(
        scaler.fit_transform(encoded_features),
        columns=encoded_features.columns,
        index=encoded_features.index,
    )

    label = dataframe["Churn"].map({"No": 0, "Yes": 1})
    if label.isna().any():
        invalid_labels = sorted(dataframe.loc[label.isna(), "Churn"].astype(str).unique())
        raise ValueError(f"지원하지 않는 Churn 값입니다: {invalid_labels}")

    result = encoded_features.astype("float32")
    result["Churn"] = label.astype("int8")

    feature_info = {
        "source_file": str(input_path),
        "target_column": "Churn",
        "dropped_columns": ["customerID"],
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "feature_columns": encoded_features.columns.tolist(),
        "feature_count": len(encoded_features.columns),
        "row_count": len(result),
        "label_mapping": {"No": 0, "Yes": 1},
        "cnn_input_shape_per_sample": [len(encoded_features.columns), 1],
    }
    return result, feature_info


def main() -> None:
    result, feature_info = preprocess_telco_data(INPUT_PATH)
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    FEATURE_INFO_PATH.write_text(
        json.dumps(feature_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("전처리 완료")
    print(f"원본 크기: {feature_info['row_count']}행")
    print(f"전처리 결과 크기: {result.shape[0]}행 × {result.shape[1]}열")
    print(f"CNN 입력 피처 수: {feature_info['feature_count']}")
    print(f"결측치 수: {int(result.isna().sum().sum())}")
    print(f"Churn 분포: {result['Churn'].value_counts().sort_index().to_dict()}")
    print(f"전처리 CSV: {OUTPUT_PATH}")
    print(f"피처 정보: {FEATURE_INFO_PATH}")


if __name__ == "__main__":
    main()
