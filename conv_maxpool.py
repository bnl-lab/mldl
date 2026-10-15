import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from PIL import UnidentifiedImageError
from torchvision import transforms

# 1. 이미지 불러오기
ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "hd.png"
FALLBACK_PATH = ROOT / "output" / "hd_pixel_visualization" / "hd_center_cropped.png"
OUTPUT_DIR = ROOT / "output" / "conv_maxpool"
OUTPUT_PATH = OUTPUT_DIR / "conv_maxpool_result.png"

try:
    img = Image.open(IMAGE_PATH).convert("RGB")
    image_source = IMAGE_PATH
except (FileNotFoundError, UnidentifiedImageError, OSError):
    if not FALLBACK_PATH.exists():
        raise FileNotFoundError(
            f"사용 가능한 이미지가 없습니다: {IMAGE_PATH}, {FALLBACK_PATH}"
        )
    print(f"주의: {IMAGE_PATH}를 읽을 수 없어 중앙 크롭 이미지를 사용합니다.")
    img = Image.open(FALLBACK_PATH).convert("RGB")
    image_source = FALLBACK_PATH

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # (C, H, W), 0~1 정규화
])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
img_tensor = transform(img).unsqueeze(0)  # (1, 3, 128, 128)

# 2. 필터 10개짜리 convolution 레이어 정의
# in_channels=3 (RGB), out_channels=10 (필터 10개), kernel_size=3
conv = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)

# 3. 맥스풀링 레이어 정의 (2x2 영역에서 가장 큰 값만 추출, 가로세로 절반으로 축소)
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

# 4. convolution -> maxpooling 순서로 연산 수행
with torch.no_grad():
    conv_output = conv(img_tensor)        # (1, 10, 128, 128)
    pool_output = maxpool(conv_output)     # (1, 10, 64, 64)

print("입력 이미지 shape:", img_tensor.shape)
print("convolution 결과 shape:", conv_output.shape)
print("maxpooling 결과 shape:", pool_output.shape)  # 가로세로가 절반으로 줄어듦

# 5. conv 결과(위) vs maxpooling 결과(아래) 비교 시각화
fig, axes = plt.subplots(2, 10, figsize=(20, 5))

for i in range(10):
    axes[0, i].imshow(conv_output[0, i].numpy(), cmap="viridis")
    axes[0, i].set_title(f"Conv {i + 1}")
    axes[0, i].axis("off")

    axes[1, i].imshow(pool_output[0, i].numpy(), cmap="viridis")
    axes[1, i].set_title(f"MaxPool {i + 1}")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
plt.close()

print(f"\n사용한 이미지: {image_source}")
print(f"결과 이미지 저장: {OUTPUT_PATH}")
