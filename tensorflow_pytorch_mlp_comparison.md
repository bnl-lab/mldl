# Fashion-MNIST MLP: TensorFlow와 PyTorch 구현 비교

## 비교 대상 파일

| 프레임워크 | 파일 |
|---|---|
| TensorFlow | [`mlp_predict_500th.py`](./mlp_predict_500th.py) |
| PyTorch | [`mlp_predict_500th_pytorch.py`](./mlp_predict_500th_pytorch.py) |

두 코드는 다음과 같은 동일한 MLP 구조를 사용한다.

```text
28×28 이미지
  → Flatten(784)
  → Dense/Linear 128 + ReLU
  → Dense/Linear 64 + ReLU
  → 10개 클래스 출력
```

## 1. TensorFlow의 Epoch 로그는 어디에서 출력되는가?

다음과 같은 로그는 개발자가 직접 문자열로 작성한 것이 아니라 TensorFlow의 `model.fit()`이 자동으로 출력한다.

```text
Epoch 1/5
844/844 ━━━━━━━━━━━━━━━━━━━━ 5s 5ms/step
- accuracy: 0.8164
- loss: 0.5208
- val_accuracy: 0.8440
- val_loss: 0.4194
```

해당 코드는 `mlp_predict_500th.py`의 108~114번째 줄이다.

```python
model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
)
```

관련 설정은 다음 위치에 있다.

| 설정 | 코드 위치 | 값 |
|---|---:|---:|
| 배치 크기 | `mlp_predict_500th.py:29` | `64` |
| Epoch 수 | `mlp_predict_500th.py:30` | `5` |
| 검증 데이터 비율 | `mlp_predict_500th.py:113` | `0.1` |

Fashion-MNIST 학습 데이터 60,000개 중 10%인 6,000개는 검증 데이터로 사용되고, 나머지 54,000개가 실제 가중치 학습에 사용된다.

```text
학습 데이터 수 = 60,000 × 0.9 = 54,000
배치 수 = ceil(54,000 ÷ 64) = 844
```

따라서 진행 로그에 `844/844`가 표시된다.

로그 항목의 의미는 다음과 같다.

| 로그 | 의미 |
|---|---|
| `Epoch 1/5` | 총 5번의 전체 데이터 반복 중 첫 번째 반복 |
| `844/844` | 현재 Epoch에서 처리한 미니배치 수 |
| `accuracy` | 학습 데이터 정확도 |
| `loss` | 학습 데이터 손실값 |
| `val_accuracy` | 검증 데이터 정확도 |
| `val_loss` | 검증 데이터 손실값 |

## 2. TensorFlow가 실제로 학습하는 부분

### 학습 규칙 설정

`mlp_predict_500th.py`의 100~104번째 줄에서 학습 방법을 설정한다.

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

- `optimizer="adam"`: 손실이 감소하도록 가중치와 편향을 갱신한다.
- `loss="sparse_categorical_crossentropy"`: 정수 클래스 정답과 예측 확률 사이의 손실을 계산한다.
- `metrics=["accuracy"]`: 학습 및 검증 정확도를 계산한다.

### 실제 학습 실행

실제 순전파, 손실 계산, 역전파, 파라미터 갱신은 `mlp_predict_500th.py:108`의 `model.fit()`에서 수행된다.

TensorFlow/Keras는 다음 과정을 `fit()` 내부에서 자동으로 실행한다.

```text
미니배치 입력
  → 순전파
  → 손실 계산
  → 자동 미분과 역전파
  → Adam으로 가중치 갱신
  → accuracy/loss 집계
  → 검증 데이터 평가
```

## 3. TensorFlow의 DNN 구조를 정의하는 부분

`mlp_predict_500th.py`의 35~46번째 줄이다.

```python
def build_model() -> tf.keras.Model:
    return models.Sequential(
        [
            layers.Input(shape=(28, 28)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
```

`mlp_predict_500th.py:96`에서 이 함수를 호출하여 모델을 생성하고, 97번째 줄의 `model.summary()`가 모델 구조와 파라미터 수를 출력한다.

## 4. PyTorch에서는 Epoch 로그를 어떻게 구현하는가?

PyTorch 버전은 TensorFlow의 `fit()`처럼 전체 학습 과정을 자동 처리하지 않는다. `mlp_predict_500th_pytorch.py`의 `train_model()` 함수에서 학습 루프와 로그 출력을 직접 구현한다.

### Epoch 반복

`mlp_predict_500th_pytorch.py:130`

```python
for epoch in range(EPOCHS):
    model.train()
```

### 미니배치 반복

`mlp_predict_500th_pytorch.py:136`

```python
for images, labels in train_loader:
```

학습 데이터가 54,000개이고 배치 크기가 64이므로 PyTorch의 `train_loader`도 한 Epoch에 844개 배치를 만든다.

### Epoch 결과 출력

`mlp_predict_500th_pytorch.py`의 156~161번째 줄이다.

```python
print(
    f"Epoch {epoch + 1}/{EPOCHS} - "
    f"accuracy: {train_accuracy:.4f} - loss: {train_loss:.4f} - "
    f"val_accuracy: {validation_accuracy:.4f} - "
    f"val_loss: {validation_loss:.4f}"
)
```

TensorFlow는 진행 막대와 배치 수를 자동 출력하지만, 현재 PyTorch 코드는 Epoch가 끝났을 때 요약 결과를 한 줄로 출력한다.

## 5. PyTorch가 실제로 학습하는 부분

손실 함수와 optimizer는 `mlp_predict_500th_pytorch.py`의 126~127번째 줄에서 정의한다.

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

실제 학습은 140~144번째 줄에서 수행한다.

```python
optimizer.zero_grad()       # 이전 미니배치의 gradient 초기화
logits = model(images)      # 순전파
loss = criterion(logits, labels)  # 손실 계산
loss.backward()             # 역전파로 gradient 계산
optimizer.step()            # 가중치와 편향 갱신
```

각 코드의 역할은 다음과 같다.

| PyTorch 코드 | 역할 | TensorFlow 대응 동작 |
|---|---|---|
| `optimizer.zero_grad()` | 이전 gradient 초기화 | `fit()` 내부 자동 처리 |
| `logits = model(images)` | 순전파 | `fit()` 내부 자동 처리 |
| `criterion(logits, labels)` | 손실 계산 | `compile(loss=...)` 설정 후 자동 계산 |
| `loss.backward()` | 자동 미분 및 역전파 | `fit()` 내부 자동 처리 |
| `optimizer.step()` | Adam으로 파라미터 갱신 | `compile(optimizer=...)` 설정 후 자동 처리 |

즉, TensorFlow의 한 줄짜리 `model.fit()` 안에 숨겨진 핵심 학습 과정을 PyTorch에서는 직접 작성했다.

## 6. PyTorch의 DNN 구조를 정의하는 부분

`mlp_predict_500th_pytorch.py`의 32~48번째 줄이다.

```python
class FashionMLP(nn.Module):
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
        return self.network(images)
```

모델 객체는 `mlp_predict_500th_pytorch.py:228`에서 생성한다.

```python
model = FashionMLP().to(device)
```

`.to(device)`는 모델을 CPU, CUDA GPU 또는 Apple MPS 장치로 이동시킨다.

## 7. TensorFlow와 PyTorch DNN 층 대응

| 역할 | TensorFlow | PyTorch |
|---|---|---|
| 입력 | `Input(shape=(28, 28))` | 입력 텐서 자체로 처리 |
| 평탄화 | `Flatten()` | `nn.Flatten()` |
| 첫 번째 은닉층 | `Dense(128, relu)` | `nn.Linear(784, 128)` + `nn.ReLU()` |
| 두 번째 은닉층 | `Dense(64, relu)` | `nn.Linear(128, 64)` + `nn.ReLU()` |
| 출력층 | `Dense(10, softmax)` | `nn.Linear(64, 10)` |
| 손실 함수 | `sparse_categorical_crossentropy` | `nn.CrossEntropyLoss()` |
| 최적화 알고리즘 | `adam` | `torch.optim.Adam` |

## 8. 출력층의 Softmax가 서로 다른 이유

TensorFlow 모델은 출력층에 Softmax를 명시한다.

```python
layers.Dense(10, activation="softmax")
```

PyTorch 모델은 출력층에서 Softmax를 적용하지 않고 `logits`를 반환한다.

```python
nn.Linear(64, 10)
```

그 이유는 PyTorch의 `nn.CrossEntropyLoss()`가 내부적으로 `LogSoftmax`와 분류 손실 계산을 함께 수행하기 때문이다. 학습 모델에 Softmax를 추가로 적용하면 수치 안정성이 나빠질 수 있다.

예측할 때만 `mlp_predict_500th_pytorch.py:174`에서 Softmax를 적용한다.

```python
probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
```

## 9. 검증 데이터 평가 비교

### TensorFlow

```python
model.fit(..., validation_split=0.1)
```

TensorFlow가 학습 데이터의 마지막 10%를 검증 데이터로 분리하고 매 Epoch마다 자동 평가한다.

### PyTorch

PyTorch는 `random_split()`으로 54,000개 학습 데이터와 6,000개 검증 데이터를 먼저 만든다. 이후 `mlp_predict_500th_pytorch.py:153`에서 직접 `evaluate()`를 호출한다.

```python
validation_loss, validation_accuracy = evaluate(
    model, validation_loader, criterion, device
)
```

검증 중에는 다음 두 설정을 사용한다.

```python
model.eval()
with torch.no_grad():
```

- `model.eval()`: 모델을 평가 모드로 전환한다.
- `torch.no_grad()`: gradient 계산을 중지하여 메모리 사용량과 계산량을 줄인다.

## 10. 500번째 이미지 예측 비교

### TensorFlow

`mlp_predict_500th.py:119`

```python
probabilities = model.predict(sample_image, verbose=0)[0]
```

TensorFlow 모델의 출력층에 Softmax가 있으므로 바로 클래스별 확률이 반환된다.

### PyTorch

`mlp_predict_500th_pytorch.py`의 171~174번째 줄이다.

```python
model.eval()
with torch.no_grad():
    logits = model(image.unsqueeze(0).to(device))
    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
```

PyTorch에서는 배치 차원을 추가하고, 모델과 데이터를 같은 장치로 이동한 다음, logits를 Softmax 확률로 변환한다.

## 핵심 요약

| 질문 | TensorFlow | PyTorch |
|---|---|---|
| DNN 구조 정의 | `build_model()` | `FashionMLP` 클래스 |
| 학습 설정 | `model.compile()` | `CrossEntropyLoss`와 `Adam` 생성 |
| 실제 학습 | `model.fit()` | Epoch/배치 반복문과 `backward()`, `step()` |
| Epoch 로그 | `fit()`이 자동 출력 | `print()`로 직접 출력 |
| 검증 | `validation_split=0.1` | `random_split()` 후 `evaluate()` 호출 |
| 예측 | `model.predict()` | `model()` + `torch.softmax()` |

TensorFlow/Keras는 학습 과정을 고수준 API로 간결하게 감싸고 있으며, PyTorch는 같은 과정을 명시적인 Python 학습 루프로 보여준다는 점이 가장 큰 차이이다.
