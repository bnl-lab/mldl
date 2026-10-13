"""PyTorch MLP로 Fashion-MNIST를 학습하고 테스트 500번째 샘플을 예측한다."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "fashion_mnist"
OUTPUT_DIR = ROOT / "output" / "fashion_mnist_mlp_pytorch"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
TARGET_INDEX = 499  # 사람이 세는 기준으로 테스트 데이터의 500번째 샘플


class FashionMLP(nn.Module):
    """TensorFlow 원본과 동일한 784 → 128 → 64 → 10 DNN 구조."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # CrossEntropyLoss가 내부적으로 softmax를 처리하므로 logits를 그대로 반환한다.
        return self.network(images)


def set_seed(seed: int) -> None:
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


def load_data() -> tuple[DataLoader, DataLoader, datasets.FashionMNIST]:
    """Fashion-MNIST를 내려받고 train 90% / validation 10%로 분리한다."""
    transform = transforms.ToTensor()  # uint8 0~255를 float32 0~1로 변환
    full_train_dataset = datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    train_size = int(len(full_train_dataset) * 0.9)
    validation_size = len(full_train_dataset) - train_size
    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    return train_loader, validation_loader, test_dataset


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size

    return loss_sum / sample_count, correct / sample_count


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
) -> None:
    """오차 역전파와 Adam optimizer로 DNN을 실제 학습한다."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n모델 학습 시작...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()       # 이전 배치의 gradient 초기화
            logits = model(images)      # 순전파
            loss = criterion(logits, labels)
            loss.backward()             # 역전파로 gradient 계산
            optimizer.step()            # 가중치와 편향 갱신

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_count += batch_size

        train_loss = train_loss_sum / train_count
        train_accuracy = train_correct / train_count
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, criterion, device
        )
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"accuracy: {train_accuracy:.4f} - loss: {train_loss:.4f} - "
            f"val_accuracy: {validation_accuracy:.4f} - "
            f"val_loss: {validation_loss:.4f}"
        )


def predict_sample(
    model: nn.Module,
    test_dataset: datasets.FashionMNIST,
    index: int,
    device: torch.device,
) -> tuple[torch.Tensor, int, int, np.ndarray, float]:
    image, actual_label = test_dataset[index]
    model.eval()
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    predicted_label = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_label] * 100)
    return image, int(actual_label), predicted_label, probabilities, confidence


def save_prediction_plot(
    image: torch.Tensor,
    actual_label: int,
    predicted_label: int,
    probabilities: np.ndarray,
    confidence: float,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "test_500th_prediction.png"
    is_correct = actual_label == predicted_label

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].imshow(image.squeeze(0).numpy(), cmap="gray")
    axes[0].set_title(
        f"Actual: {CLASS_NAMES[actual_label]}\n"
        f"Pred: {CLASS_NAMES[predicted_label]} ({confidence:.1f}%)",
        color="blue" if is_correct else "red",
    )
    axes[0].axis("off")

    bars = axes[1].bar(range(10), probabilities, color="#777777")
    bars[actual_label].set_color("blue")
    if not is_correct:
        bars[predicted_label].set_color("red")
    axes[1].set_xticks(range(10), CLASS_NAMES, rotation=45, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Probability")
    axes[1].set_title("Class probability distribution")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    set_seed(RANDOM_SEED)
    device = select_device()
    print(f"사용 장치: {device}")

    train_loader, validation_loader, test_dataset = load_data()
    print(
        f"학습: {len(train_loader.dataset):,}개, "
        f"검증: {len(validation_loader.dataset):,}개, "
        f"테스트: {len(test_dataset):,}개"
    )

    model = FashionMLP().to(device)
    print("\nDNN 구조:")
    print(model)
    train_model(model, train_loader, validation_loader, device)

    image, actual_label, predicted_label, probabilities, confidence = predict_sample(
        model, test_dataset, TARGET_INDEX, device
    )
    print("\n" + "=" * 60)
    print(f"테스트 데이터 [{TARGET_INDEX + 1}번째] 샘플 예측 결과")
    print(f"실제 정답: [{actual_label}] {CLASS_NAMES[actual_label]}")
    print(
        f"예측 결과: [{predicted_label}] {CLASS_NAMES[predicted_label]} "
        f"(신뢰도: {confidence:.2f}%)"
    )
    print(f"일치 여부: {'정답 (O)' if actual_label == predicted_label else '오답 (X)'}")
    print("=" * 60)

    output_path = save_prediction_plot(
        image, actual_label, predicted_label, probabilities, confidence
    )
    print(f"예측 시각화 저장: {output_path}")


if __name__ == "__main__":
    main()
