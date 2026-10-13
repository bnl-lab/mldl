"""Mall_Customers.csv에서 K별 실루엣 계수 분포를 분석한다.

앞서 만든 계층적 클러스터링과 같은 데이터/피처/거리/연결을 사용한다.
각 반복마다 동일한 Mall Customers 데이터에서 고객을 무작위로 부분 추출하고,
Ward 계층적 클러스터링을 수행하여 표본 변동에 따른 실루엣 분포를 계산한다.

사용 피처:
    Age, Annual Income (k$), Spending Score (1-100)
분석 설정:
    StandardScaler + Euclidean distance + Ward linkage

실행:
    .venv/Scripts/python.exe mall_customers_silhouette_distribution.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "Mall_Customers.csv"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "mall_customers_silhouette"
FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mall Customers 고객 표본에서 K별 실루엣 계수 분포를 분석합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.8,
        help="각 반복에서 무작위로 사용할 고객 비율 (기본값: 0.8)",
    )
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def validate(data: pd.DataFrame, args: argparse.Namespace) -> None:
    missing = [column for column in FEATURES if column not in data.columns]
    if missing:
        raise ValueError(f"필수 열이 없습니다: {missing}")
    if data[FEATURES].isna().any().any():
        raise ValueError("분석 피처에 결측치가 있습니다.")
    if not np.isfinite(data[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("분석 피처에 무한대 또는 유효하지 않은 값이 있습니다.")
    if args.n_runs < 1:
        raise ValueError("n-runs는 1 이상이어야 합니다.")
    if not 0 < args.sample_frac <= 1:
        raise ValueError("sample-frac는 0보다 크고 1 이하여야 합니다.")
    if args.k_min < 2 or args.k_max < args.k_min:
        raise ValueError("k-min은 2 이상이고 k-max 이상일 수 없습니다.")


def cluster_once(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    model = AgglomerativeClustering(
        n_clusters=k,
        metric="euclidean",
        linkage="ward",
    )
    labels = model.fit_predict(X)
    scores = silhouette_samples(X, labels, metric="euclidean")
    return labels, scores, float(silhouette_score(X, labels, metric="euclidean"))


def plot_distribution(scores: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(13, 7))
    sns.violinplot(
        data=scores,
        x="k",
        y="silhouette",
        inner=None,
        cut=0,
        color="#9ecae1",
        linewidth=0.7,
    )
    sns.boxplot(
        data=scores,
        x="k",
        y="silhouette",
        width=0.18,
        color="#2171b5",
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.5},
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Mall Customers: silhouette distribution by cluster count")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Sample silhouette coefficient")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_trend(summary: pd.DataFrame, full_scores: pd.DataFrame, output_path: Path) -> None:
    trend = summary.groupby("k", as_index=False).agg(
        mean=("run_mean", "mean"),
        std=("run_mean", "std"),
        negative_ratio=("negative_ratio", "mean"),
    )
    trend["lower"] = trend["mean"] - trend["std"]
    trend["upper"] = trend["mean"] + trend["std"]

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(trend["k"], trend["mean"], marker="o", label="Random-subset mean")
    ax1.fill_between(trend["k"], trend["lower"], trend["upper"], alpha=0.2)
    ax1.plot(
        full_scores["k"],
        full_scores["silhouette_mean"],
        marker="D",
        linestyle="--",
        color="black",
        label="Full dataset",
    )
    ax1.axhline(0, color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel("Mean silhouette ± 1 SD")
    ax1.set_xticks(trend["k"])

    ax2 = ax1.twinx()
    ax2.plot(
        trend["k"],
        trend["negative_ratio"],
        color="#d62728",
        marker="s",
        label="Negative-score ratio",
    )
    ax2.set_ylabel("Negative silhouette ratio")
    ax2.set_ylim(0, 1)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    fig.suptitle("Mall Customers silhouette trend across random subsets")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_k5_example(data: pd.DataFrame, scaled: np.ndarray, output_path: Path) -> None:
    labels, _, _ = cluster_once(scaled, k=5)
    plt.figure(figsize=(8, 6))
    plt.scatter(
        data["Annual Income (k$)"],
        data["Spending Score (1-100)"],
        c=labels + 1,
        cmap="tab10",
        s=42,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.3,
    )
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Mall Customers: Ward clusters (K=5)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_silhouette_analysis(
    scaled: np.ndarray, k: int, output_path: Path, title_suffix: str = ""
) -> float:
    """전형적인 군집별 실루엣 막대 그래프를 저장한다."""
    labels, sample_scores, mean_score = cluster_once(scaled, k)
    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    cmap = plt.get_cmap("tab10")

    for cluster_index in range(k):
        cluster_scores = np.sort(sample_scores[labels == cluster_index])
        size = len(cluster_scores)
        y_upper = y_lower + size
        color = cmap(cluster_index % 10)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_scores,
            facecolor=color,
            edgecolor=color,
            alpha=0.75,
        )
        ax.text(-0.05, y_lower + 0.5 * size, str(cluster_index + 1), fontsize=9)
        y_lower = y_upper + 10

    ax.axvline(mean_score, color="red", linestyle="--", linewidth=1.5)
    ax.text(
        mean_score,
        0.98,
        f"mean = {mean_score:.3f}",
        color="red",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
    )
    ax.set_title(f"Silhouette analysis for K={k}{title_suffix}")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster label")
    ax.set_xlim(-0.2, 1.0)
    ax.set_yticks([])
    ax.set_ylim(0, y_lower)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return mean_score


def plot_silhouette_analysis_grid(
    scaled: np.ndarray, k_values: list[int], output_path: Path
) -> None:
    """여러 K의 전형적인 silhouette plot을 한 장으로 비교한다."""
    n_cols = 3
    n_rows = int(np.ceil(len(k_values) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.5 * n_rows), squeeze=False)

    for ax, k in zip(axes.flat, k_values):
        labels, sample_scores, mean_score = cluster_once(scaled, k)
        y_lower = 10
        cmap = plt.get_cmap("tab10")
        for cluster_index in range(k):
            cluster_scores = np.sort(sample_scores[labels == cluster_index])
            y_upper = y_lower + len(cluster_scores)
            color = cmap(cluster_index % 10)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_scores,
                facecolor=color,
                edgecolor=color,
                alpha=0.75,
            )
            ax.text(-0.05, y_lower + 0.5 * len(cluster_scores), str(cluster_index + 1), fontsize=7)
            y_lower = y_upper + 10
        ax.axvline(mean_score, color="red", linestyle="--", linewidth=1.2)
        ax.set_title(f"K={k}, mean={mean_score:.3f}")
        ax.set_xlim(-0.2, 1.0)
        ax.set_ylim(0, y_lower)
        ax.set_yticks([])
        ax.set_xlabel("silhouette")

    for ax in axes.flat[len(k_values):]:
        ax.axis("off")
    fig.suptitle("Mall Customers silhouette analysis by cluster count", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    data = pd.read_csv(input_path)
    validate(data, args)
    n_sample = max(args.k_max + 1, int(round(len(data) * args.sample_frac)))
    if n_sample > len(data):
        n_sample = len(data)
    if args.k_max >= n_sample:
        raise ValueError("표본 크기는 k-max보다 커야 합니다.")

    # 앞서 계층적 클러스터링과 동일하게 세 피처를 표준화한다.
    scaled = StandardScaler().fit_transform(data[FEATURES])
    rng = np.random.default_rng(args.seed)
    score_rows: list[dict[str, float | int]] = []
    summary_rows: list[dict[str, float | int]] = []

    for run in range(args.n_runs):
        selected = np.sort(rng.choice(len(data), size=n_sample, replace=False))
        for k in range(args.k_min, args.k_max + 1):
            labels, scores, run_mean = cluster_once(scaled[selected], k)
            for local_index, (row_index, label, score) in enumerate(
                zip(selected, labels, scores)
            ):
                customer_id = data.iloc[row_index].get("CustomerID", row_index)
                score_rows.append(
                    {
                        "run": run,
                        "k": k,
                        "customer_id": customer_id,
                        "row_index": int(row_index),
                        "cluster": int(label) + 1,
                        "silhouette": float(score),
                    }
                )
            summary_rows.append(
                {
                    "run": run,
                    "k": k,
                    "run_mean": run_mean,
                    "run_std": float(scores.std(ddof=1)),
                    "run_min": float(scores.min()),
                    "run_median": float(np.median(scores)),
                    "run_max": float(scores.max()),
                    "negative_ratio": float((scores < 0).mean()),
                }
            )

    scores_df = pd.DataFrame(score_rows)
    summary_df = pd.DataFrame(summary_rows)
    full_rows = []
    for k in range(args.k_min, args.k_max + 1):
        _, full_sample_scores, full_mean = cluster_once(scaled, k)
        full_rows.append(
            {
                "k": k,
                "silhouette_mean": full_mean,
                "silhouette_std": float(full_sample_scores.std(ddof=1)),
                "negative_ratio": float((full_sample_scores < 0).mean()),
            }
        )
    full_df = pd.DataFrame(full_rows)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(output_dir / "sample_silhouette_scores.csv", index=False)
    summary_df.to_csv(output_dir / "run_summary.csv", index=False)
    full_df.to_csv(output_dir / "full_dataset_silhouette.csv", index=False)
    plot_distribution(scores_df, output_dir / "silhouette_distribution_by_k.png")
    plot_trend(summary_df, full_df, output_dir / "silhouette_trend_by_k.png")
    plot_k5_example(data, scaled, output_dir / "mall_customers_clusters_k5.png")
    k_values = list(range(args.k_min, args.k_max + 1))
    plot_silhouette_analysis_grid(
        scaled, k_values, output_dir / "silhouette_analysis_all_k.png"
    )
    for k in k_values:
        plot_silhouette_analysis(
            scaled, k, output_dir / f"silhouette_analysis_k{k}.png"
        )

    mean_by_k = summary_df.groupby("k")["run_mean"].mean().sort_values(ascending=False)
    print(f"입력: {input_path}")
    print(f"고객 수: {len(data)}, 반복 수: {args.n_runs}, 반복별 표본 수: {n_sample}")
    print(f"분석 K: {args.k_min}~{args.k_max}")
    print("설정: StandardScaler + Euclidean + Ward")
    print(f"무작위 표본 평균 실루엣이 가장 큰 K: {int(mean_by_k.index[0])} ({mean_by_k.iloc[0]:.4f})")
    print(f"결과 폴더: {output_dir}")


if __name__ == "__main__":
    main()
