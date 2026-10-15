"""Fashion-MNIST CNN 학습 후 테스트 데이터 200번째 샘플을 예측·시각화한다."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "fashion_mnist_cnn_predict_200"
MODEL_PATH = OUTPUT_DIR / "fashion_mnist_cnn.keras"
FIGURE_PATH = OUTPUT_DIR / "test_200_prediction.png"
HISTORY_PATH = OUTPUT_DIR / "training_history.png"

RANDOM_SEED = 42
TEST_SAMPLE_NUMBER = 200
TEST_SAMPLE_INDEX = TEST_SAMPLE_NUMBER - 1

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


def build_model() -> tf.keras.Model:
    """28×28 흑백 Fashion-MNIST용 CNN을 생성한다."""

    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.30),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ],
        name="fashion_mnist_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_history(history: tf.keras.callbacks.History) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history.history["loss"]) + 1)

    axes[0].plot(epochs, history.history["loss"], label="Train")
    axes[0].plot(epochs, history.history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history.history["accuracy"], label="Train")
    axes[1].plot(epochs, history.history["val_accuracy"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(HISTORY_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_prediction_figure(
    image: np.ndarray,
    true_label: int,
    probabilities: np.ndarray,
) -> None:
    predicted_label = int(np.argmax(probabilities))
    colors = [
        "tab:green" if index == true_label else "tab:blue"
        for index in range(len(CLASS_NAMES))
    ]
    colors[predicted_label] = "tab:red"

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        gridspec_kw={"width_ratios": [1, 1.6]},
    )
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(
        f"Test #{TEST_SAMPLE_NUMBER}\n"
        f"True: {CLASS_NAMES[true_label]}\n"
        f"Predicted: {CLASS_NAMES[predicted_label]}"
    )
    axes[0].axis("off")

    bars = axes[1].barh(CLASS_NAMES, probabilities, color=colors)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1.0)
    axes[1].set_xlabel("Prediction probability")
    axes[1].set_title("CNN prediction probabilities")
    for bar, probability in zip(bars, probabilities):
        axes[1].text(
            min(probability + 0.01, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{probability:.3f}",
            va="center",
        )

    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    model = build_model()
    model.summary()
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=5,
        batch_size=512,
        callbacks=[early_stopping],
        verbose=2,
    )
    save_training_history(history)

    # 학습이 끝난 모델을 저장하고, 저장된 모델을 다시 로드해 테스트한다.
    model.save(MODEL_PATH)
    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    test_loss, test_accuracy = loaded_model.evaluate(x_test, y_test, verbose=0)

    sample_image = x_test[TEST_SAMPLE_INDEX, :, :, 0]
    probabilities = loaded_model.predict(
        x_test[TEST_SAMPLE_INDEX : TEST_SAMPLE_INDEX + 1],
        verbose=0,
    )[0]
    predicted_label = int(np.argmax(probabilities))
    true_label = int(y_test[TEST_SAMPLE_INDEX])
    save_prediction_figure(sample_image, true_label, probabilities)

    print("\n" + "=" * 70)
    print(f"테스트 데이터 {TEST_SAMPLE_NUMBER}번째 샘플 예측")
    print("=" * 70)
    print(f"정답: {true_label} ({CLASS_NAMES[true_label]})")
    print(f"예측: {predicted_label} ({CLASS_NAMES[predicted_label]})")
    print(f"예측 확률: {probabilities[predicted_label]:.4f}")
    print(f"전체 테스트 정확도: {test_accuracy:.4f}")
    print(f"저장된 모델: {MODEL_PATH}")
    print(f"예측 시각화: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
