"""Fashion-MNIST 학습 데이터의 10번째 이미지를 픽셀 단위로 시각화한다."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "fashion_mnist_10th_pixel"
OUTPUT_PATH = OUTPUT_DIR / "fashion_mnist_10th_pixel.png"

SAMPLE_NUMBER = 10
SAMPLE_INDEX = SAMPLE_NUMBER - 1  # 사람 기준 10번째 데이터는 Python 인덱스 9

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


def main() -> None:
    (x_train, y_train), _ = fashion_mnist.load_data()

    image = x_train[SAMPLE_INDEX]
    label = int(y_train[SAMPLE_INDEX])
    class_name = CLASS_NAMES[label]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(13, 6))

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(
        f"Fashion-MNIST #{SAMPLE_NUMBER}\nLabel {label}: {class_name}"
    )
    axes[0].axis("off")

    pixel_image = axes[1].imshow(image, cmap="viridis", vmin=0, vmax=255)
    axes[1].set_title("28 × 28 Pixel Intensity")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")
    axes[1].set_xticks(np.arange(-0.5, 28, 1), minor=True)
    axes[1].set_yticks(np.arange(-0.5, 28, 1), minor=True)
    axes[1].grid(which="minor", color="white", linewidth=0.25, alpha=0.5)
    axes[1].tick_params(which="minor", bottom=False, left=False)
    figure.colorbar(pixel_image, ax=axes[1], label="Pixel value (0–255)")

    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")

    print(f"10번째 데이터 배열 크기: {image.shape}")
    print(f"레이블: {label} ({class_name})")
    print(f"픽셀 최솟값/최댓값: {image.min()} / {image.max()}")
    print(f"이미지 저장 위치: {OUTPUT_PATH}")
    print("\n[28 x 28 픽셀값]")
    print(image)

    plt.show()


if __name__ == "__main__":
    main()
