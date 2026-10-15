"""전처리된 Telco Churn CSV로 Conv1D CNN을 학습하고 테스트 샘플 하나를 예측한다."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "telco_churn_cnn_preprocessed.csv"
OUTPUT_DIR = ROOT / "output" / "telco_churn_cnn_predict_one"
MODEL_PATH = OUTPUT_DIR / "telco_churn_cnn.keras"
FIGURE_PATH = OUTPUT_DIR / "one_sample_prediction.png"

RANDOM_SEED = 42
TEST_SIZE = 0.20
SAMPLE_POSITION_IN_TEST = 0


def build_cnn(feature_count: int) -> tf.keras.Model:
    """표 형식 피처를 1차원 시퀀스로 처리하는 Conv1D CNN."""

    model = models.Sequential(
        [
            layers.Input(shape=(feature_count, 1)),
            layers.Conv1D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, padding="same", activation="relu"),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.25),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="telco_churn_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def save_prediction_figure(
    feature_values: np.ndarray,
    feature_names: list[str],
    true_label: int,
    probability: float,
) -> None:
    predicted_label = int(probability >= 0.5)
    figure, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].plot(np.arange(len(feature_values)), feature_values, marker=".")
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].set_title("Selected test sample features")
    axes[0].set_xlabel("Feature index")
    axes[0].set_ylabel("Preprocessed value")
    axes[0].grid(alpha=0.3)

    axes[1].bar(
        ["No churn (0)", "Churn (1)"],
        [1 - probability, probability],
        color=["steelblue", "tomato"],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Probability")
    axes[1].set_title(
        f"True: {true_label} | Predicted: {predicted_label}\n"
        f"Churn probability: {probability:.4f}"
    )
    for x_position, value in enumerate([1 - probability, probability]):
        axes[1].text(x_position, value + 0.02, f"{value:.3f}", ha="center")

    # 피처명이 길기 때문에 별도 CSV에 이름과 값을 저장하고, 그림은 인덱스로 간결하게 표시한다.
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=170, bbox_inches="tight")
    plt.close(figure)

    pd.DataFrame(
        {"feature": feature_names, "preprocessed_value": feature_values}
    ).to_csv(OUTPUT_DIR / "selected_sample_features.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"전처리 CSV를 찾을 수 없습니다: {DATA_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(DATA_PATH)
    if "Churn" not in dataframe.columns:
        raise ValueError("CSV에 Churn 컬럼이 필요합니다.")
    if dataframe.isna().any().any():
        raise ValueError("전처리 CSV에 결측치가 있습니다.")

    feature_names = [column for column in dataframe.columns if column != "Churn"]
    x = dataframe[feature_names].to_numpy(dtype=np.float32)
    y = dataframe["Churn"].to_numpy(dtype=np.float32)
    row_indices = np.arange(len(dataframe))

    x_train, x_test, y_train, y_test, _, test_indices = train_test_split(
        x,
        y,
        row_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    x_train_cnn = x_train[..., np.newaxis]
    x_test_cnn = x_test[..., np.newaxis]

    model = build_cnn(len(feature_names))
    model.summary()
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True, verbose=1
    )
    model.fit(
        x_train_cnn,
        y_train,
        validation_split=0.1,
        epochs=8,
        batch_size=512,
        callbacks=[early_stopping],
        verbose=2,
    )

    model.save(MODEL_PATH)
    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    test_loss, test_accuracy, test_auc = loaded_model.evaluate(
        x_test_cnn, y_test, verbose=0
    )

    sample = min(SAMPLE_POSITION_IN_TEST, len(x_test) - 1)
    probability = float(loaded_model.predict(x_test_cnn[sample : sample + 1], verbose=0)[0, 0])
    true_label = int(y_test[sample])
    save_prediction_figure(
        x_test[sample], feature_names, true_label, probability
    )

    print("\n" + "=" * 70)
    print("Telco Churn CNN 테스트 샘플 1개 예측")
    print("=" * 70)
    print(f"입력 CSV: {DATA_PATH}")
    print(f"전체 피처 수: {len(feature_names)}")
    print(f"선택한 원본 행 번호(0-based): {int(test_indices[sample])}")
    print(f"정답: {true_label} ({'Churn' if true_label else 'No churn'})")
    print(f"예측: {int(probability >= 0.5)} ({'Churn' if probability >= 0.5 else 'No churn'})")
    print(f"Churn 확률: {probability:.4f}")
    print(f"테스트 정확도: {test_accuracy:.4f}")
    print(f"테스트 AUC: {test_auc:.4f}")
    print(f"저장 모델: {MODEL_PATH}")
    print(f"예측 시각화: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
