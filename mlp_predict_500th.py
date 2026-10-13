"""TensorFlow MLP로 Fashion-MNIST를 학습하고 테스트 500번째 샘플을 예측한다."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "fashion_mnist_mlp_tensorflow"

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

BATCH_SIZE = 64
EPOCHS = 5
RANDOM_SEED = 42
TARGET_INDEX = 499  # 사람이 세는 기준으로 테스트 데이터의 500번째 샘플


def build_model() -> tf.keras.Model:
    """784 → 128 → 64 → 10 구조의 DNN 모델을 생성한다."""
    return models.Sequential(
        [
            layers.Input(shape=(28, 28)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ],
        name="fashion_mnist_mlp",
    )


def save_prediction_plot(
    image: np.ndarray,
    actual_label: int,
    predicted_label: int,
    probabilities: np.ndarray,
    confidence: float,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "test_500th_prediction.png"
    is_correct = actual_label == predicted_label

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].imshow(image, cmap="gray")
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
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # 1. Fashion-MNIST 데이터 로드 및 0~1 범위 정규화
    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # 2. TensorFlow DNN 구조 생성
    model = build_model()
    model.summary()

    # 3. 학습 규칙 설정
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 4. 실제 모델 학습: 기본 verbose 출력으로 Epoch 로그가 표시된다.
    print("\n모델 학습 시작...")
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
    )

    # 5. 테스트 데이터의 500번째 샘플 예측
    sample_image = np.expand_dims(x_test[TARGET_INDEX], axis=0)
    actual_label = int(y_test[TARGET_INDEX])
    probabilities = model.predict(sample_image, verbose=0)[0]
    predicted_label = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_label] * 100)

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
        x_test[TARGET_INDEX],
        actual_label,
        predicted_label,
        probabilities,
        confidence,
    )
    print(f"예측 시각화 저장: {output_path}")


if __name__ == "__main__":
    main()
