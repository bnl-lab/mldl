"""Mall Customers 계층적 클러스터링 분석.

분석 흐름:
    1) Age, Annual Income, Spending Score 선택
    2) StandardScaler로 표준화
    3) Ward + Euclidean linkage로 계층 구조 생성
    4) 덴드로그램 저장
    5) K=3(상위 마케팅)과 K=5(세부 프로모션) 라벨 저장

실행:
    .venv/Scripts/python.exe mall_customers_hierarchical_clustering.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "Mall_Customers.csv"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "mall_customers_hierarchical"
FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ward 계층적 클러스터링으로 고객 세그먼트를 생성합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--show",
        action="store_true",
        help="분석이 끝난 뒤 생성한 그림을 화면에 표시합니다.",
    )
    return parser.parse_args()


def validate_data(data: pd.DataFrame) -> None:
    missing_columns = [column for column in FEATURES if column not in data.columns]
    if missing_columns:
        raise ValueError(f"필수 피처 열이 없습니다: {missing_columns}")
    if data[FEATURES].isna().any().any():
        raise ValueError("분석 피처에 결측치가 있습니다.")
    if not np.isfinite(data[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("분석 피처에 무한대 또는 유효하지 않은 값이 있습니다.")


def save_dendrogram(linkage_matrix: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(15, 8))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=9,
        show_contracted=True,
        color_threshold=0,
    )
    plt.title("Ward hierarchical clustering dendrogram")
    plt.xlabel("Clustered observations (last 30 merges shown)")
    plt.ylabel("Ward distance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_cluster_scatter(
    data: pd.DataFrame, label_column: str, output_path: Path
) -> None:
    colors = data[label_column]
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        data["Annual Income (k$)"],
        data["Spending Score (1-100)"],
        c=colors,
        cmap="tab10",
        s=45,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.3,
    )
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title(f"Customer segments: {label_column}")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=plt.get_cmap("tab10")(i - 1),
            markeredgecolor="white",
            markersize=8,
            label=f"Cluster {i}",
        )
        for i in sorted(colors.unique())
    ]
    plt.legend(handles=handles, title="Segment")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    data = pd.read_csv(input_path)
    validate_data(data)
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[FEATURES])

    # Ward linkage는 Euclidean 거리와 함께 사용합니다.
    Z = linkage(X_scaled, method="ward", metric="euclidean")
    save_dendrogram(Z, output_dir / "dendrogram_ward.png")

    result = data.copy()
    for k in (3, 5):
        # fcluster의 라벨은 1부터 시작하도록 저장합니다.
        result[f"Cluster_K{k}"] = fcluster(Z, t=k, criterion="maxclust").astype(int)

    # 동일한 Ward 설정을 sklearn API로도 확인합니다.
    sklearn_model = AgglomerativeClustering(
        n_clusters=5,
        metric="euclidean",
        linkage="ward",
    )
    sklearn_labels = sklearn_model.fit_predict(X_scaled) + 1
    scipy_labels = result["Cluster_K5"].to_numpy()
    # 군집 번호 자체는 알고리즘마다 바뀔 수 있으므로, 분할이 같은지만 확인합니다.
    scipy_partition = {frozenset(np.flatnonzero(scipy_labels == label)) for label in np.unique(scipy_labels)}
    sklearn_partition = {frozenset(np.flatnonzero(sklearn_labels == label)) for label in np.unique(sklearn_labels)}
    if scipy_partition != sklearn_partition:
        raise RuntimeError("SciPy와 scikit-learn의 K=5 군집 분할이 일치하지 않습니다.")

    result.to_csv(output_dir / "customer_cluster_labels.csv", index=False)

    summary_rows: list[pd.DataFrame] = []
    for k in (3, 5):
        label_column = f"Cluster_K{k}"
        summary = result.groupby(label_column, as_index=False)[FEATURES].mean()
        summary = summary.rename(columns={label_column: "Cluster"})
        sizes = result[label_column].value_counts().rename("Size").rename_axis("Cluster")
        summary = summary.merge(sizes, on="Cluster", how="left")
        summary.insert(0, "K", k)
        summary_rows.append(summary)
        save_cluster_scatter(result, label_column, output_dir / f"scatter_K{k}.png")

    pd.concat(summary_rows, ignore_index=True).to_csv(
        output_dir / "cluster_profile_summary.csv", index=False
    )

    print(f"입력 데이터: {input_path}")
    print(f"샘플 수: {len(result):,}, 사용 피처: {', '.join(FEATURES)}")
    print("거리/연결: Euclidean / Ward")
    print(f"결과 폴더: {output_dir}")
    print(result[["Cluster_K3", "Cluster_K5"]].value_counts().sort_index())
    print("K=3은 상위 마케팅 세그먼트, K=5는 세부 프로모션 세그먼트로 사용할 수 있습니다.")

    if args.show:
        from PIL import Image

        Image.open(output_dir / "dendrogram_ward.png").show()


if __name__ == "__main__":
    main()
