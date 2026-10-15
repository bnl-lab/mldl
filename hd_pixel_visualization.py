"""hd.png 중앙 그림을 28×28 회색조 픽셀값으로 시각화한다."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "hd.png"
OUTPUT_DIR = ROOT / "output" / "hd_pixel_visualization"
OUTPUT_PATH = OUTPUT_DIR / "hd_center_pixel_values.png"
CROPPED_PATH = OUTPUT_DIR / "hd_center_cropped.png"

PIXEL_SIZE = 28
SATURATION_THRESHOLD = 25

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def crop_colored_center(image: Image.Image) -> Image.Image:
    """저채도 회색 배경을 제외하고 중앙의 컬러 그림 영역만 자른다."""

    hsv_array = np.asarray(image.convert("HSV"))
    saturation = hsv_array[:, :, 1]
    colored_y, colored_x = np.where(saturation > SATURATION_THRESHOLD)

    if colored_x.size == 0 or colored_y.size == 0:
        raise ValueError("중앙 컬러 영역을 찾지 못했습니다.")

    left = int(colored_x.min())
    top = int(colored_y.min())
    right = int(colored_x.max()) + 1
    bottom = int(colored_y.max()) + 1
    return image.crop((left, top, right, bottom))


def convert_to_pixel_array(image: Image.Image) -> np.ndarray:
    """중앙 그림을 28×28 회색조 픽셀 배열로 변환한다."""

    grayscale = image.convert("L")
    resized = grayscale.resize(
        (PIXEL_SIZE, PIXEL_SIZE),
        Image.Resampling.BOX,
    )
    return np.asarray(resized)


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {IMAGE_PATH}")

    original = Image.open(IMAGE_PATH).convert("RGB")
    center_image = crop_colored_center(original)
    pixel_array = convert_to_pixel_array(center_image)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    center_image.save(CROPPED_PATH)

    # Fashion-MNIST 픽셀 시각화와 같은 형태로 구성한다.
    figure, axis = plt.subplots(figsize=(14, 14))
    pixel_plot = axis.imshow(pixel_array, cmap="gray", vmin=0, vmax=255)

    axis.set_xticks(range(PIXEL_SIZE))
    axis.set_yticks(range(PIXEL_SIZE))
    axis.set_xticklabels(range(PIXEL_SIZE), fontsize=8)
    axis.set_yticklabels(range(PIXEL_SIZE), fontsize=8)
    axis.set_xlabel("Pixel column")
    axis.set_ylabel("Pixel row")

    # 각 칸에 실제 회색조 픽셀값(0~255)을 표시한다.
    threshold = pixel_array.max() / 2.0
    for row in range(PIXEL_SIZE):
        for column in range(PIXEL_SIZE):
            pixel_value = int(pixel_array[row, column])
            text_color = "black" if pixel_value > threshold else "white"
            axis.text(
                column,
                row,
                str(pixel_value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=6.5,
            )

    axis.set_title(
        "hd.png 중앙 그림의 28×28 픽셀 상세 시각화\n"
        "[Grayscale intensity: 0~255]",
        fontsize=14,
        pad=15,
    )
    figure.colorbar(
        pixel_plot,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label="Pixel Value (0~255)",
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")

    print(f"원본 이미지: {IMAGE_PATH}")
    print(f"원본 크기: {original.width} x {original.height}")
    print(f"중앙 그림 크기: {center_image.width} x {center_image.height}")
    print(f"픽셀 시각화 크기: {PIXEL_SIZE} x {PIXEL_SIZE}")
    print(f"픽셀값 범위: {pixel_array.min()} ~ {pixel_array.max()}")
    print(f"결과 저장 위치: {OUTPUT_PATH}")

    plt.show()


if __name__ == "__main__":
    main()
