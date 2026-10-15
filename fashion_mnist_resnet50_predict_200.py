"""Fashion-MNIST를 ResNet50으로 학습하고 테스트 200번째 샘플을 예측한다."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "fashion_mnist_resnet50_predict_200"
MODEL_PATH = OUTPUT_DIR / "fashion_mnist_resnet50.keras"
FIGURE_PATH = OUTPUT_DIR / "test_200_resnet50_prediction.png"

IMAGE_SIZE = 96
TEST_SAMPLE_NUMBER = 200
TEST_SAMPLE_INDEX = TEST_SAMPLE_NUMBER - 1
RANDOM_SEED = 42

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


def prepare_images(images: np.ndarray) -> np.ndarray:
    """28×28 회색조 이미지를 ResNet50용 96×96 RGB 입력으로 변환한다."""

    images = images.astype("float32")
    images = images[..., np.newaxis]
    images = np.repeat(images, 3, axis=-1)
    images = tf.image.resize(images, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
    return preprocess_input(images)


def build_resnet50() -> tf.keras.Model:
    """ImageNet 사전학습 가중치 없이 Fashion-MNIST에 맞춰 ResNet50을 학습한다."""

    backbone = ResNet50(
        include_top=False,
        weights=None,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    # Keras automatically uses training/inference behavior during fit/evaluate/predict.
    features = backbone(inputs)
    features = layers.GlobalAveragePooling2D()(features)
    features = layers.Dropout(0.30)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = models.Model(inputs, outputs, name="fashion_mnist_resnet50")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_prediction_figure(
    image: np.ndarray,
    true_label: int,
    probabilities: np.ndarray,
) -> None:
    predicted_label = int(np.argmax(probabilities))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(
        f"Fashion-MNIST Test #{TEST_SAMPLE_NUMBER}\n"
        f"True: {CLASS_NAMES[true_label]}\n"
        f"Predicted: {CLASS_NAMES[predicted_label]}"
    )
    axes[0].axis("off")

    colors = ["steelblue"] * len(CLASS_NAMES)
    colors[true_label] = "seagreen"
    colors[predicted_label] = "crimson"
    bars = axes[1].barh(CLASS_NAMES, probabilities, color=colors)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Probability")
    axes[1].set_title("ResNet50 prediction probabilities")
    for bar, value in zip(bars, probabilities):
        axes[1].text(min(float(value) + 0.01, 0.98), bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")

    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (x_train_raw, y_train), (x_test_raw, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = prepare_images(x_train_raw)
    x_test = prepare_images(x_test_raw)

    model = build_resnet50()
    model.summary()
    checkpoint = callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=1,
        restore_best_weights=True,
        verbose=1,
    )

    model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=2,
        batch_size=128,
        callbacks=[checkpoint, early_stopping],
        verbose=2,
    )

    # 저장된 최적 ResNet50을 다시 로드해 테스트한다.
    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    test_loss, test_accuracy = loaded_model.evaluate(x_test, y_test, verbose=0)
    probabilities = loaded_model.predict(
        x_test[TEST_SAMPLE_INDEX : TEST_SAMPLE_INDEX + 1],
        verbose=0,
    )[0]
    true_label = int(y_test[TEST_SAMPLE_INDEX])
    predicted_label = int(np.argmax(probabilities))
    save_prediction_figure(x_test_raw[TEST_SAMPLE_INDEX], true_label, probabilities)

    print("\n" + "=" * 70)
    print(f"ResNet50 Fashion-MNIST 테스트 {TEST_SAMPLE_NUMBER}번째 샘플")
    print("=" * 70)
    print(f"정답: {true_label} ({CLASS_NAMES[true_label]})")
    print(f"예측: {predicted_label} ({CLASS_NAMES[predicted_label]})")
    print(f"예측 확률: {probabilities[predicted_label]:.4f}")
    print(f"전체 테스트 정확도: {test_accuracy:.4f}")
    print(f"저장 모델: {MODEL_PATH}")
    print(f"시각화: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
