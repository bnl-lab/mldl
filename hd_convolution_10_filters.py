"""hd.png의 중앙 그림에 10개의 3×3 컨볼루션 필터를 적용한다."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "hd.png"
FALLBACK_CENTER_PATH = (
    ROOT / "output" / "hd_pixel_visualization" / "hd_center_cropped.png"
)
OUTPUT_DIR = ROOT / "output" / "hd_convolution_10_filters"
SUMMARY_PATH = OUTPUT_DIR / "convolution_10_filters.png"
CROPPED_PATH = OUTPUT_DIR / "cropped_center.png"

# 회색 배경보다 채도가 높은 중앙 그림을 찾기 위한 기준값
SATURATION_THRESHOLD = 25

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def crop_colored_center(image: Image.Image) -> Image.Image:
    """회색 배경을 제외하고 중앙의 컬러 그림 영역만 반환한다."""

    hsv_array = np.asarray(image.convert("HSV"))
    saturation = hsv_array[:, :, 1]
    rows, columns = np.where(saturation > SATURATION_THRESHOLD)

    if columns.size == 0 or rows.size == 0:
        raise ValueError("중앙 컬러 그림 영역을 찾지 못했습니다.")

    left = int(columns.min())
    top = int(rows.min())
    right = int(columns.max()) + 1
    bottom = int(rows.max()) + 1
    return image.crop((left, top, right, bottom))


def load_center_image() -> tuple[Image.Image, Path]:
    """원본을 우선 사용하고, 손상 시 기존 중앙 크롭 이미지를 사용한다."""

    try:
        original = Image.open(IMAGE_PATH).convert("RGB")
        return crop_colored_center(original), IMAGE_PATH
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        if not FALLBACK_CENTER_PATH.exists():
            raise FileNotFoundError(
                f"정상 이미지가 없습니다: {IMAGE_PATH}, {FALLBACK_CENTER_PATH}"
            )
        print(f"주의: {IMAGE_PATH}를 읽을 수 없어 기존 중앙 이미지를 사용합니다.")
        return Image.open(FALLBACK_CENTER_PATH).convert("RGB"), FALLBACK_CENTER_PATH


def create_filters() -> tuple[list[str], np.ndarray]:
    """서로 다른 특징을 추출하는 3×3 필터 10개를 생성한다."""

    filters = {
        "Identity": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        "Box blur": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        "Sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        "Laplacian": [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
        "Sobel X": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        "Sobel Y": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        "Diagonal NE": [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
        "Diagonal NW": [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
        "Emboss": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
        "Edge enhance": [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
    }

    names = list(filters)
    kernel_array = np.asarray(list(filters.values()), dtype=np.float32)
    kernel_array[1] /= 9.0  # Box blur의 전체 가중치 합을 1로 정규화

    # TensorFlow 필터 형식: [height, width, input_channels, output_channels]
    kernels = np.transpose(kernel_array, (1, 2, 0))[:, :, np.newaxis, :]
    return names, kernels


def apply_convolution(gray_image: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    """한 번의 conv2d 연산으로 10개 특성맵을 생성한다."""

    image_tensor = tf.convert_to_tensor(
        gray_image[np.newaxis, :, :, np.newaxis],
        dtype=tf.float32,
    )
    kernel_tensor = tf.convert_to_tensor(kernels, dtype=tf.float32)
    feature_maps = tf.nn.conv2d(
        image_tensor,
        filters=kernel_tensor,
        strides=1,
        padding="SAME",
    )
    return feature_maps.numpy()[0]


def normalize_for_display(feature_map: np.ndarray) -> np.ndarray:
    """각 컨볼루션 결과를 0~255 범위 이미지로 변환한다."""

    low, high = np.percentile(feature_map, [1, 99])
    if np.isclose(low, high):
        return np.zeros_like(feature_map, dtype=np.uint8)

    clipped = np.clip(feature_map, low, high)
    normalized = (clipped - low) / (high - low) * 255
    return normalized.astype(np.uint8)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    center_image, source_path = load_center_image()
    center_image.save(CROPPED_PATH)

    gray_image = np.asarray(center_image.convert("L"), dtype=np.float32)
    filter_names, kernels = create_filters()
    feature_maps = apply_convolution(gray_image, kernels)

    figure, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()

    axes[0].imshow(center_image)
    axes[0].set_title("Cropped center image")
    axes[0].axis("off")

    for index, filter_name in enumerate(filter_names):
        display_image = normalize_for_display(feature_maps[:, :, index])

        output_path = OUTPUT_DIR / f"filter_{index + 1:02d}_{filter_name.lower().replace(' ', '_')}.png"
        Image.fromarray(display_image, mode="L").save(output_path)

        axis = axes[index + 1]
        axis.imshow(display_image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(f"{index + 1}. {filter_name}")
        axis.axis("off")

    axes[-1].axis("off")
    figure.suptitle("중앙 그림에 적용한 10개 컨볼루션 필터", fontsize=18)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(SUMMARY_PATH, dpi=170, bbox_inches="tight")

    print(f"사용한 이미지: {source_path}")
    print(f"중앙 그림 크기: {center_image.width} x {center_image.height}")
    print(f"컨볼루션 입력 텐서: (1, {gray_image.shape[0]}, {gray_image.shape[1]}, 1)")
    print(f"필터 텐서: {kernels.shape}")
    print(f"출력 특성맵: {feature_maps.shape}")
    print(f"적용한 필터 수: {len(filter_names)}")
    print(f"결과 저장 폴더: {OUTPUT_DIR}")

    plt.show()


if __name__ == "__main__":
    main()
