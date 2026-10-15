"""Sonar 데이터로 히든 레이어 깊이가 다른 DNN을 학습하고 평가한다.

비교 모델
    - DNN-2:  히든 레이어 2개
    - DNN-6:  히든 레이어 6개
    - DNN-12: 히든 레이어 12개

세 모델은 같은 학습/검증/테스트 데이터와 같은 전처리를 사용한다.
실행 결과(평가 지표, 학습 이력, 모델 가중치, 그래프)는
``output/sonar_dnn_comparison`` 폴더에 저장된다.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sonar.all-data-uci.csv"
OUTPUT_DIR = ROOT / "output" / "sonar_dnn_comparison"

RANDOM_SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20  # 학습용 데이터 중 검증 데이터 비율
BATCH_SIZE = 16
MAX_EPOCHS = 300
PATIENCE = 35
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# 각 숫자가 한 히든 레이어의 노드 수이다.
MODEL_CONFIGS: dict[str, list[int]] = {
    "DNN-2": [64, 32],
    "DNN-6": [64, 64, 48, 32, 24, 16],
    "DNN-12": [64, 64, 48, 48, 32, 32, 24, 24, 16, 16, 8, 8],
}
# 층마다 적용되므로 깊은 모델에는 더 작은 dropout을 사용한다.
MODEL_DROPOUTS = {"DNN-2": 0.20, "DNN-6": 0.10, "DNN-12": 0.10}

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Malgun Gothic")


@dataclass
class DataBundle:
    """동일한 데이터 분할을 모든 모델에 전달하기 위한 묶음."""

    train_loader: DataLoader
    x_validation: torch.Tensor
    y_validation: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    split_sizes: tuple[int, int, int]


class SonarDNN(nn.Module):
    """LayerNorm과 Dropout을 포함한 완전연결 이진 분류 DNN."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(previous_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_size = hidden_size

        # BCEWithLogitsLoss를 사용하므로 출력층에 Sigmoid를 넣지 않는다.
        layers.append(nn.Linear(previous_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """실행할 때마다 가능한 한 같은 결과가 나오도록 난수를 고정한다."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_and_prepare_data(file_path: Path) -> tuple[DataBundle, int]:
    """CSV를 읽고 누수 없이 분할 및 표준화한다."""

    if not file_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")

    dataframe = pd.read_csv(file_path)
    if "Label" not in dataframe.columns:
        raise ValueError("CSV에 'Label' 열이 필요합니다.")
    if dataframe.isna().any().any():
        raise ValueError("데이터에 결측치가 있습니다. 결측치를 처리한 뒤 실행하세요.")

    x = dataframe.drop(columns="Label").to_numpy(dtype=np.float32)
    y = dataframe["Label"].map({"R": 0, "M": 1})
    if y.isna().any():
        invalid_labels = sorted(dataframe.loc[y.isna(), "Label"].astype(str).unique())
        raise ValueError(f"지원하지 않는 Label 값입니다: {invalid_labels}")
    y_array = y.to_numpy(dtype=np.float32)

    # 테스트 데이터는 마지막 평가에만 사용한다.
    x_development, x_test, y_development, y_test = train_test_split(
        x,
        y_array,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_array,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_development,
        y_development,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_development,
    )

    # 데이터 누수를 막기 위해 scaler는 학습 데이터에만 fit한다.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_validation = scaler.transform(x_validation).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train)
    )
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    bundle = DataBundle(
        train_loader=train_loader,
        x_validation=torch.from_numpy(x_validation),
        y_validation=torch.from_numpy(y_validation),
        x_test=torch.from_numpy(x_test),
        y_test=torch.from_numpy(y_test),
        split_sizes=(len(x_train), len(x_validation), len(x_test)),
    )
    return bundle, x.shape[1]


@torch.no_grad()
def calculate_loss(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    logits = model(features.to(device))
    return float(criterion(logits, targets.to(device)).item())


def train_model(
    model: nn.Module,
    data: DataBundle,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame, int]:
    """검증 손실 조기 종료를 적용해 한 모델을 학습한다."""

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in data.train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        train_loss = total_loss / len(data.train_loader.dataset)
        validation_loss = calculate_loss(
            model,
            data.x_validation,
            data.y_validation,
            criterion,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

        if validation_loss < best_validation_loss - 1e-4:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    """고정된 테스트 데이터에서 분류 지표를 계산한다."""

    model.eval()
    logits = model(x_test.to(device))
    probabilities = torch.sigmoid(logits).cpu().numpy()
    predictions = (probabilities >= 0.5).astype(np.int64)
    targets = y_test.numpy().astype(np.int64)

    metrics = {
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall": recall_score(targets, predictions, zero_division=0),
        "f1": f1_score(targets, predictions, zero_division=0),
        "roc_auc": roc_auc_score(targets, probabilities),
    }
    return metrics, confusion_matrix(targets, predictions, labels=[0, 1])


def save_comparison_plots(
    results: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    confusion_matrices: dict[str, np.ndarray],
) -> None:
    """손실 곡선, 지표 비교, 혼동행렬을 파일로 저장한다."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for model_name, history in histories.items():
        axes[0].plot(history["epoch"], history["train_loss"], label=model_name)
        axes[1].plot(history["epoch"], history["validation_loss"], label=model_name)
    axes[0].set(title="학습 손실", xlabel="Epoch", ylabel="BCE Loss")
    axes[1].set(title="검증 손실", xlabel="Epoch", ylabel="BCE Loss")
    axes[0].legend()
    axes[1].legend()

    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    melted = results.melt(
        id_vars="model", value_vars=metric_columns, var_name="metric", value_name="score"
    )
    sns.barplot(data=melted, x="metric", y="score", hue="model", ax=axes[2])
    axes[2].set(title="테스트 성능 비교", xlabel="평가 지표", ylabel="점수", ylim=(0, 1.05))
    axes[2].legend(title="Model", loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_and_metrics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, (model_name, matrix) in zip(axes, confusion_matrices.items()):
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Rock (R)", "Mine (M)"],
            yticklabels=["Rock (R)", "Mine (M)"],
            ax=axis,
        )
        axis.set(title=model_name, xlabel="예측값", ylabel="실제값")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = select_device()
    data, input_size = load_and_prepare_data(DATA_PATH)
    train_count, validation_count, test_count = data.split_sizes

    print("=" * 80)
    print("Sonar DNN 깊이별 비교")
    print("=" * 80)
    print(f"데이터: {DATA_PATH}")
    print(f"분할: train={train_count}, validation={validation_count}, test={test_count}")
    print(f"장치: {device}")

    result_rows: list[dict[str, float | int | str]] = []
    histories: dict[str, pd.DataFrame] = {}
    confusion_matrices: dict[str, np.ndarray] = {}

    for model_index, (model_name, hidden_sizes) in enumerate(MODEL_CONFIGS.items()):
        # 모델 초기화도 매번 고정해 비교 실험의 재현성을 높인다.
        set_seed(RANDOM_SEED + model_index)
        # 세 모델에 동일한 mini-batch 순서를 적용한다.
        data.train_loader.generator.manual_seed(RANDOM_SEED)
        dropout = MODEL_DROPOUTS[model_name]
        model = SonarDNN(input_size, hidden_sizes, dropout=dropout).to(device)
        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n[{model_name}] hidden nodes={hidden_sizes}, dropout={dropout:.2f}")
        print(f"학습 파라미터 수: {parameter_count:,}")
        model, history, best_epoch = train_model(model, data, device)
        metrics, matrix = evaluate_model(
            model, data.x_test, data.y_test, device
        )

        history.insert(0, "model", model_name)
        histories[model_name] = history
        confusion_matrices[model_name] = matrix
        result_rows.append(
            {
                "model": model_name,
                "hidden_layers": len(hidden_sizes),
                "hidden_nodes": "-".join(map(str, hidden_sizes)),
                "dropout": dropout,
                "parameters": parameter_count,
                "best_epoch": best_epoch,
                **metrics,
            }
        )
        torch.save(model.state_dict(), OUTPUT_DIR / f"{model_name.lower()}_weights.pt")
        print(
            f"best epoch={best_epoch} | accuracy={metrics['accuracy']:.4f} | "
            f"f1={metrics['f1']:.4f} | ROC-AUC={metrics['roc_auc']:.4f}"
        )

    results = pd.DataFrame(result_rows).sort_values("f1", ascending=False)
    all_histories = pd.concat(histories.values(), ignore_index=True)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    all_histories.to_csv(OUTPUT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")
    save_comparison_plots(results, histories, confusion_matrices)

    print("\n" + "=" * 80)
    print("테스트 성능 비교 (F1 내림차순)")
    print("=" * 80)
    display_columns = ["model", "hidden_layers", "accuracy", "precision", "recall", "f1", "roc_auc"]
    print(results[display_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n가장 높은 F1 모델: {results.iloc[0]['model']}")
    print(f"결과 저장 폴더: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
