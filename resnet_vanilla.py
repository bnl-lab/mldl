"""ResNet의 핵심 구조를 이해하기 위한 작은 바닐라 구현.

핵심 아이디어는 ResidualBlock의 다음 한 줄이다.

    output = ReLU(F(x) + shortcut(x))

입력과 F(x)의 채널 수 또는 해상도가 다를 때만 1×1 projection shortcut을 사용한다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "resnet_vanilla"
MODEL_PATH = OUTPUT_DIR / "vanilla_resnet_fashion_mnist.keras"
HISTORY_PATH = OUTPUT_DIR / "training_history.png"

RANDOM_SEED = 42


class ResidualBlock(layers.Layer):
    """두 개의 3×3 convolution과 skip connection으로 구성된 기본 블록."""

    def __init__(self, filters: int, stride: int = 1, name: str | None = None):
        super().__init__(name=name)
        self.filters = filters
        self.stride = stride

        self.conv1 = layers.Conv2D(
            filters, 3, strides=stride, padding="same", use_bias=False
        )
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(
            filters, 3, strides=1, padding="same", use_bias=False
        )
        self.bn2 = layers.BatchNormalization()

        # 입력과 출력 shape이 다를 때만 shortcut을 projection으로 변환한다.
        self.projection = None

    def build(self, input_shape: tf.TensorShape) -> None:
        input_channels = int(input_shape[-1])
        if self.stride != 1 or input_channels != self.filters:
            self.projection = tf.keras.Sequential(
                [
                    layers.Conv2D(
                        self.filters,
                        1,
                        strides=self.stride,
                        padding="same",
                        use_bias=False,
                    ),
                    layers.BatchNormalization(),
                ],
                name="projection_shortcut",
            )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        shortcut = inputs

        residual = self.conv1(inputs)
        residual = self.bn1(residual, training=training)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual, training=training)

        if self.projection is not None:
            shortcut = self.projection(inputs, training=training)

        # ResNet의 핵심: 변환된 입력과 원래 입력을 더한다.
        return self.relu(residual + shortcut)


class VanillaResNet(tf.keras.Model):
    """Fashion-MNIST에 맞춘 작고 명시적인 ResNet."""

    def __init__(self, number_of_classes: int = 10):
        super().__init__(name="vanilla_resnet")
        self.stem = tf.keras.Sequential(
            [
                layers.Conv2D(32, 3, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
            ],
            name="stem",
        )
        self.stage1 = [
            ResidualBlock(32, name="stage1_block1"),
            ResidualBlock(32, name="stage1_block2"),
        ]
        self.stage2 = [
            ResidualBlock(64, stride=2, name="stage2_block1"),
            ResidualBlock(64, name="stage2_block2"),
        ]
        self.global_average_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(number_of_classes, activation="softmax")

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.stem(inputs, training=training)
        for block in self.stage1:
            x = block(x, training=training)
        for block in self.stage2:
            x = block(x, training=training)
        x = self.global_average_pool(x)
        return self.classifier(x)


def save_history_plot(history: tf.keras.callbacks.History) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history.history["loss"]) + 1)
    axes[0].plot(epochs, history.history["loss"], label="train")
    axes[0].plot(epochs, history.history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, history.history["accuracy"], label="train")
    axes[1].plot(epochs, history.history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(HISTORY_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., tf.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., tf.newaxis]

    model = VanillaResNet(number_of_classes=10)
    # subclassed model은 더미 입력을 한 번 통과시켜 내부 레이어를 build한다.
    model(tf.zeros((1, 28, 28, 1)), training=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=3,
        batch_size=512,
        verbose=2,
    )
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    model.save(MODEL_PATH)
    save_history_plot(history)

    print(f"테스트 손실: {test_loss:.4f}")
    print(f"테스트 정확도: {test_accuracy:.4f}")
    print(f"저장 모델: {MODEL_PATH}")
    print(f"학습 그래프: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
