"""HAR feature CSV를 표준화한 뒤 PCA 피처로 변환한다.

기본 실행:
    .venv/Scripts/python.exe har_pca_feature_extraction.py

주성분 개수를 직접 지정하는 예:
    .venv/Scripts/python.exe har_pca_feature_extraction.py --n-components 100

학습/평가 누수를 막기 위해 앞 7,352개 행에만 PCA를 적합하는 예:
    .venv/Scripts/python.exe har_pca_feature_extraction.py --fit-rows 7352
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "har_features.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "har_features_pca.csv"


def parse_n_components(value: str) -> int | float:
    """CLI 문자열을 PCA의 n_components 값으로 변환한다."""
    try:
        if any(marker in value.lower() for marker in (".", "e")):
            parsed: int | float = float(value)
        else:
            parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "n-components는 양의 정수 또는 0과 1 사이의 실수여야 합니다."
        ) from exc

    if isinstance(parsed, int) and parsed < 1:
        raise argparse.ArgumentTypeError("정수 n-components는 1 이상이어야 합니다.")
    if isinstance(parsed, float) and not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError(
            "실수 n-components는 0과 1 사이여야 합니다(예: 0.95)."
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HAR 수치형 피처를 StandardScaler와 PCA로 차원 축소합니다."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"입력 CSV 경로 (기본값: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"PCA 피처 CSV 경로 (기본값: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--n-components",
        type=parse_n_components,
        default=0.95,
        help="보존할 누적 설명분산(0~1) 또는 주성분 개수 (기본값: 0.95)",
    )
    parser.add_argument(
        "--fit-rows",
        type=int,
        default=None,
        help=(
            "Scaler/PCA 적합에 사용할 입력의 앞쪽 행 수. 지정하지 않으면 전체 "
            "데이터에 적합합니다. 공식 UCI HAR train split에는 7352를 사용합니다."
        ),
    )
    return parser.parse_args()


def validate_features(features: pd.DataFrame) -> None:
    if features.empty:
        raise ValueError("입력 CSV에 데이터가 없습니다.")

    non_numeric = features.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric:
        preview = ", ".join(map(str, non_numeric[:10]))
        raise ValueError(f"수치형이 아닌 열이 있습니다: {preview}")

    missing_count = int(features.isna().sum().sum())
    if missing_count:
        raise ValueError(
            f"결측치가 {missing_count:,}개 있습니다. 결측치를 처리한 뒤 실행하세요."
        )

    values = features.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("입력 데이터에 무한대 또는 유효하지 않은 값이 있습니다.")


def artifact_paths(output_path: Path) -> tuple[Path, Path, Path]:
    stem = output_path.with_suffix("")
    model_path = stem.parent / f"{stem.name}_pipeline.joblib"
    variance_path = stem.parent / f"{stem.name}_explained_variance.csv"
    summary_path = stem.parent / f"{stem.name}_summary.json"
    return model_path, variance_path, summary_path


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"입력 CSV를 찾을 수 없습니다: {input_path}")

    print(f"[1/4] 데이터 로드: {input_path}")
    features = pd.read_csv(input_path)
    validate_features(features)
    print(f"      샘플 {len(features):,}개, 원본 피처 {features.shape[1]:,}개")

    if args.fit_rows is None:
        fit_features = features
        fit_scope = "all_rows"
    else:
        if not 2 <= args.fit_rows <= len(features):
            raise ValueError(
                f"fit-rows는 2 이상 {len(features):,} 이하여야 합니다."
            )
        fit_features = features.iloc[: args.fit_rows]
        fit_scope = f"first_{args.fit_rows}_rows"

    max_components = min(fit_features.shape)
    if isinstance(args.n_components, int) and args.n_components > max_components:
        raise ValueError(
            f"n-components는 적합 데이터에서 최대 {max_components}까지 가능합니다."
        )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=args.n_components, svd_solver="full")),
        ]
    )

    print(f"[2/4] StandardScaler와 PCA 적합: {fit_scope}")
    pipeline.fit(fit_features)

    print("[3/4] 전체 데이터 PCA 변환")
    transformed = pipeline.transform(features)
    pca = pipeline.named_steps["pca"]
    component_names = [f"PC{i:03d}" for i in range(1, pca.n_components_ + 1)]
    pca_features = pd.DataFrame(
        transformed,
        columns=component_names,
        index=features.index,
    )

    explained_variance = pd.DataFrame(
        {
            "component": component_names,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    )

    model_path, variance_path, summary_path = artifact_paths(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pca_features.to_csv(output_path, index=False)
    explained_variance.to_csv(variance_path, index=False)
    joblib.dump(pipeline, model_path)

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "fit_scope": fit_scope,
        "n_samples": int(features.shape[0]),
        "original_feature_count": int(features.shape[1]),
        "pca_component_count": int(pca.n_components_),
        "requested_n_components": args.n_components,
        "total_explained_variance_ratio": float(
            pca.explained_variance_ratio_.sum()
        ),
        "model_path": str(model_path),
        "explained_variance_path": str(variance_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[4/4] 저장 완료")
    print(f"      PCA 피처: {output_path}")
    print(f"      변환 모델: {model_path}")
    print(f"      설명분산: {variance_path}")
    print(f"      실행 요약: {summary_path}")
    print(
        f"      {features.shape[1]}개 피처 -> {pca.n_components_}개 주성분, "
        f"누적 설명분산 {pca.explained_variance_ratio_.sum():.2%}"
    )


if __name__ == "__main__":
    main()
