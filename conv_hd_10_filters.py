"""hd.png의 중앙 그림에 10개 필터를 가진 Conv2D를 적용해 시각화한다."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow.keras import layers, models


ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "hd.png"
FALLBACK_CENTER_PATH = (
    ROOT / "output" / "hd_pixel_visualization" / "hd_center_cropped.png"
)
OUTPUT_DIR = ROOT / "output" / "conv_hd_10_filters"
OUTPUT_PATH = OUTPUT_DIR / "conv_hd_10_maxpooled_feature_maps.png"
CROPPED_PATH = OUTPUT_DIR / "cropped_center.png"

NUM_FILTERS = 10
SATURATION_THRESHOLD = 25
RANDOM_SEED = 42

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def crop_colored_center(image: Image.Image) -> Image.Image:
    """채도를 이용하여 바깥 회색 배경을 제외한 중앙 그림만 자른다."""

    hsv_array = np.asarray(image.convert("HSV"))
    rows, columns = np.where(hsv_array[:, :, 1] > SATURATION_THRESHOLD)

    if columns.size == 0 or rows.size == 0:
        raise ValueError("중앙 컬러 영역을 찾지 못했습니다.")

    crop_box = (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )
    return image.crop(crop_box)


def load_center_image() -> tuple[Image.Image, Path]:
    """정상 hd.png를 우선 사용하고, 읽을 수 없으면 기존 중앙 그림을 사용한다."""

    try:
        original = Image.open(IMAGE_PATH).convert("RGB")
        return crop_colored_center(original), IMAGE_PATH
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        if not FALLBACK_CENTER_PATH.exists():
            raise FileNotFoundError(
                f"사용 가능한 이미지가 없습니다: {IMAGE_PATH}, {FALLBACK_CENTER_PATH}"
            )
        print(f"주의: {IMAGE_PATH}를 읽을 수 없어 기존 중앙 그림을 사용합니다.")
        return Image.open(FALLBACK_CENTER_PATH).convert("RGB"), FALLBACK_CENTER_PATH


def build_convolution_model(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    """학습 없이 특징을 추출하는 10필터 Conv2D 모델을 만든다."""

    return models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(
                filters=NUM_FILTERS,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(seed=RANDOM_SEED),
                bias_initializer="zeros",
                name="conv_10_filters",
            ),
            layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding="valid",
                name="max_pooling_2x2",
            ),
        ],
        name="hd_conv_feature_extractor",
    )


def normalize_feature_map(feature_map: np.ndarray) -> np.ndarray:
    """특성맵을 시각화하기 좋은 0~1 범위로 정규화한다."""

    minimum = float(feature_map.min())
    maximum = float(feature_map.max())
    if np.isclose(minimum, maximum):
        return np.zeros_like(feature_map)
    return (feature_map - minimum) / (maximum - minimum)


def main() -> None:
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cropped_image, source_path = load_center_image()
    cropped_image.save(CROPPED_PATH)

    image_array = np.asarray(cropped_image, dtype=np.float32) / 255.0
    input_tensor = image_array[np.newaxis, ...]

    convolution_model = build_convolution_model(image_array.shape)
    print("[모델 정보]")
    convolution_model.summary()

    # 출력 형태: (batch, height, width, 10)
    feature_maps = convolution_model.predict(input_tensor, verbose=0)[0]

    figure = plt.figure(figsize=(15, 9))
    original_axis = plt.subplot2grid((3, 5), (0, 2))
    original_axis.imshow(cropped_image)
    original_axis.set_title("중앙 원본 이미지", fontweight="bold")
    original_axis.axis("off")

    for index in range(NUM_FILTERS):
        row = 1 + index // 5
        column = index % 5
        axis = plt.subplot2grid((3, 5), (row, column))
        normalized_map = normalize_feature_map(feature_maps[:, :, index])
        axis.imshow(normalized_map, cmap="viridis", vmin=0, vmax=1)
        axis.set_title(f"Filter #{index + 1}")
        axis.axis("off")

        individual_path = OUTPUT_DIR / f"maxpooled_feature_map_{index + 1:02d}.png"
        plt.imsave(individual_path, normalized_map, cmap="viridis", vmin=0, vmax=1)

    figure.suptitle(
        "중앙 그림의 10개 Conv2D Feature Maps",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(OUTPUT_PATH, dpi=170, bbox_inches="tight")

    print(f"\n사용한 이미지: {source_path}")
    print(f"중앙 그림 크기: {cropped_image.width} x {cropped_image.height}")
    print(f"입력 텐서 형태: {input_tensor.shape}")
    print(
        f"MaxPooling 이전 특성맵 형태: "
        f"({input_tensor.shape[1]}, {input_tensor.shape[2]}, {NUM_FILTERS})"
    )
    print(f"MaxPooling 이후 특성맵 형태: {feature_maps.shape}")
    print(f"결과 저장 위치: {OUTPUT_PATH}")

    plt.show()


if __name__ == "__main__":
    main()
