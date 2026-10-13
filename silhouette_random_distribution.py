"""랜덤 합성 데이터에서 클러스터 수별 실루엣 계수 분포를 분석한다.

각 데이터셋 seed마다 Gaussian blob 데이터를 새로 생성하고,
Ward 계층적 클러스터링의 K를 변화시키면서 다음 결과를 저장한다.

* 샘플별 실루엣 계수(long format)
* K별 평균/중앙값/사분위수/음수 비율 요약
* K별 실루엣 계수 분포(boxplot, violin plot)
* 첫 번째 랜덤 데이터셋의 군집 시각화

실행 예:
    .venv/Scripts/python.exe silhouette_random_distribution.py
    .venv/Scripts/python.exe silhouette_random_distribution.py --k-min 2 --k-max 10 --n-datasets 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "output" / "silhouette_random_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="랜덤 데이터에서 K별 실루엣 계수 분포를 분석합니다."
    )
    parser.add_argument("--n-datasets", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--n-features", type=int, default=2)
    parser.add_argument("--true-centers", type=int, default=4)
    parser.add_argument("--cluster-std", type=float, default=2.0)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_datasets < 1 or args.n_samples < 3 or args.n_features < 1:
        raise ValueError("데이터셋 수, 샘플 수, 피처 수는 양수여야 합니다.")
    if args.true_centers < 2 or args.cluster_std <= 0:
        raise ValueError("true-centers는 2 이상, cluster-std는 양수여야 합니다.")
    if args.k_min < 2 or args.k_max < args.k_min:
        raise ValueError("k-min은 2 이상이고 k-max 이상일 수 없습니다.")
    if args.k_max >= args.n_samples:
        raise ValueError("k-max는 샘플 수보다 작아야 합니다.")


def make_random_dataset(args: argparse.Namespace, dataset_seed: int) -> np.ndarray:
    X, _ = make_blobs(
        n_samples=args.n_samples,
        n_features=args.n_features,
        centers=args.true_centers,
        cluster_std=args.cluster_std,
        random_state=dataset_seed,
    )
    # 피처 단위가 다른 상황에서도 거리 기반 분석이 왜곡되지 않도록 표준화한다.
    return StandardScaler().fit_transform(X)


def run_analysis(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    sample_rows: list[dict[str, float | int]] = []
    summary_rows: list[dict[str, float | int]] = []
    first_X: np.ndarray | None = None
    first_labels: np.ndarray | None = None

    rng = np.random.default_rng(args.seed)
    dataset_seeds = rng.integers(0, 2**31 - 1, size=args.n_datasets)

    for dataset_index, dataset_seed in enumerate(dataset_seeds):
        X = make_random_dataset(args, int(dataset_seed))
        if first_X is None:
            first_X = X

        for k in range(args.k_min, args.k_max + 1):
            model = AgglomerativeClustering(
                n_clusters=k,
                metric="euclidean",
                linkage="ward",
            )
            labels = model.fit_predict(X)
            scores = silhouette_samples(X, labels, metric="euclidean")
            mean_score = float(silhouette_score(X, labels, metric="euclidean"))

            if first_labels is None and k == args.k_min:
                first_labels = labels

            for sample_index, (label, score) in enumerate(zip(labels, scores)):
                sample_rows.append(
                    {
                        "dataset_index": dataset_index,
                        "dataset_seed": int(dataset_seed),
                        "k": k,
                        "sample_index": sample_index,
                        "cluster": int(label) + 1,
                        "silhouette": float(score),
                    }
                )

            quantiles = np.quantile(scores, [0.25, 0.50, 0.75])
            summary_rows.append(
                {
                    "dataset_index": dataset_index,
                    "dataset_seed": int(dataset_seed),
                    "k": k,
                    "mean": mean_score,
                    "std": float(scores.std(ddof=1)),
                    "min": float(scores.min()),
                    "q25": float(quantiles[0]),
                    "median": float(quantiles[1]),
                    "q75": float(quantiles[2]),
                    "max": float(scores.max()),
                    "negative_ratio": float((scores < 0).mean()),
                }
            )

    assert first_X is not None and first_labels is not None
    return pd.DataFrame(sample_rows), pd.DataFrame(summary_rows), first_X, first_labels


def plot_distribution(sample_df: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(13, 7))
    sns.violinplot(
        data=sample_df,
        x="k",
        y="silhouette",
        inner=None,
        color="#9ecae1",
        cut=0,
        linewidth=0.7,
    )
    sns.boxplot(
        data=sample_df,
        x="k",
        y="silhouette",
        width=0.18,
        color="#2171b5",
        showfliers=False,
        boxprops={"zorder": 2},
        medianprops={"color": "white", "linewidth": 1.5},
    )
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.title("Silhouette coefficient distribution by cluster count")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Sample silhouette coefficient")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_mean_trend(summary_df: pd.DataFrame, output_path: Path) -> None:
    aggregate = summary_df.groupby("k", as_index=False).agg(
        mean=("mean", "mean"),
        std=("mean", "std"),
        negative_ratio=("negative_ratio", "mean"),
    )
    aggregate["lower"] = aggregate["mean"] - aggregate["std"]
    aggregate["upper"] = aggregate["mean"] + aggregate["std"]

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(aggregate["k"], aggregate["mean"], marker="o", label="Mean silhouette")
    ax1.fill_between(
        aggregate["k"], aggregate["lower"], aggregate["upper"], alpha=0.2
    )
    ax1.axhline(0, color="black", linewidth=1, linestyle="--")
    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel("Mean silhouette ± 1 SD")
    ax1.set_xticks(aggregate["k"])

    ax2 = ax1.twinx()
    ax2.plot(
        aggregate["k"],
        aggregate["negative_ratio"],
        color="#d62728",
        marker="s",
        label="Negative-score ratio",
    )
    ax2.set_ylabel("Negative silhouette ratio")
    ax2.set_ylim(0, 1)
    fig.suptitle("Silhouette trend across random datasets")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_example(X: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA

        X_plot = PCA(n_components=2, random_state=42).fit_transform(X)
        x_label, y_label = "PC1", "PC2"
    else:
        X_plot = X[:, :2]
        x_label, y_label = "Feature 1", "Feature 2"

    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_plot[:, 0],
        X_plot[:, 1],
        c=labels + 1,
        cmap="tab10",
        s=28,
        alpha=0.8,
    )
    plt.title("Example random dataset and K-min clustering")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_df, summary_df, first_X, first_labels = run_analysis(args)
    sample_df.to_csv(output_dir / "sample_silhouette_scores.csv", index=False)
    summary_df.to_csv(output_dir / "silhouette_summary_by_dataset.csv", index=False)
    plot_distribution(sample_df, output_dir / "silhouette_distribution_by_k.png")
    plot_mean_trend(summary_df, output_dir / "silhouette_mean_trend.png")
    plot_example(first_X, first_labels, output_dir / "random_dataset_example.png")

    aggregate = summary_df.groupby("k")["mean"].mean().sort_values(ascending=False)
    best_k = int(aggregate.index[0])
    print(f"데이터셋 수: {args.n_datasets}, 데이터셋당 샘플 수: {args.n_samples}")
    print(f"평가한 K: {args.k_min}~{args.k_max}")
    print(f"평균 실루엣 계수가 가장 큰 K: {best_k} ({aggregate.iloc[0]:.4f})")
    print(f"결과 폴더: {output_dir}")


if __name__ == "__main__":
    main()
