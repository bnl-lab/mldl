"""scikit-learn Wine 데이터로 DNN을 학습하고 최적 모델을 다시 로드해 평가한다."""

from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks, layers, models


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "wine_dnn_checkpoint"
MODEL_PATH = OUTPUT_DIR / "best_wine_dnn.keras"
SCALER_PATH = OUTPUT_DIR / "standard_scaler.joblib"
CLASS_INFO_PATH = OUTPUT_DIR / "class_names.json"

RANDOM_SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20  # train+validation 데이터 중 검증 비율
BATCH_SIZE = 16
MAX_EPOCHS = 300
PATIENCE = 30

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Malgun Gothic")


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Python, NumPy, TensorFlow 난수를 고정한다."""

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        # TensorFlow가 이미 초기화된 환경에서는 설정이 제한될 수 있다.
        pass


def prepare_data():
    """Wine 데이터를 계층화 분할하고 데이터 누수 없이 표준화한다."""

    wine = load_wine(as_frame=True)
    x = wine.data.to_numpy(dtype=np.float32)
    y = wine.target.to_numpy(dtype=np.int64)

    # 테스트 데이터는 모델 선택에 사용하지 않고 마지막 평가에만 사용한다.
    x_development, x_test, y_development, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_development,
        y_development,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_development,
    )

    # scaler는 반드시 학습 데이터에만 fit하여 정보 누수를 방지한다.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_validation = scaler.transform(x_validation).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    return (
        x_train,
        x_validation,
        x_test,
        y_train,
        y_validation,
        y_test,
        scaler,
        wine.target_names.tolist(),
    )


def build_model(input_size: int, number_of_classes: int) -> tf.keras.Model:
    """Wine 데이터 크기에 맞춘 완전연결 다중 클래스 DNN을 생성한다."""

    model = models.Sequential(
        [
            layers.Input(shape=(input_size,), name="wine_features"),
            layers.Dense(64, activation="relu", name="hidden_1"),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            layers.Dense(32, activation="relu", name="hidden_2"),
            layers.BatchNormalization(),
            layers.Dropout(0.20),
            layers.Dense(16, activation="relu", name="hidden_3"),
            layers.Dense(number_of_classes, activation="softmax", name="class_output"),
        ],
        name="wine_dnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_callbacks() -> list[callbacks.Callback]:
    """조기 종료와 최적 모델 전체 저장 콜백을 생성한다."""

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=PATIENCE,
        min_delta=1e-4,
        restore_best_weights=True,
        verbose=1,
    )
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    return [early_stopping, model_checkpoint]


def save_training_plot(history: tf.keras.callbacks.History) -> None:
    """학습 및 검증 손실·정확도 곡선을 저장한다."""

    epochs = range(1, len(history.history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history.history["loss"], label="Train")
    axes[0].plot(epochs, history.history["val_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()

    axes[1].plot(epochs, history.history["accuracy"], label="Train")
    axes[1].plot(epochs, history.history["val_accuracy"], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.05))
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_history.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    class_names: list[str],
) -> None:
    """다시 로드한 모델의 테스트 혼동행렬을 저장한다."""

    matrix = confusion_matrix(y_true, y_predicted)
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set(title="Test Confusion Matrix", xlabel="Predicted", ylabel="Actual")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "test_confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (
        x_train,
        x_validation,
        x_test,
        y_train,
        y_validation,
        y_test,
        scaler,
        class_names,
    ) = prepare_data()

    print("=" * 75)
    print("Wine DNN: EarlyStopping + ModelCheckpoint")
    print("=" * 75)
    print(
        f"데이터 분할: train={len(x_train)}, "
        f"validation={len(x_validation)}, test={len(x_test)}"
    )

    model = build_model(
        input_size=x_train.shape[1],
        number_of_classes=len(class_names),
    )
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=create_callbacks(),
        verbose=0,
    )

    # 모델 이외의 추론 필수 객체도 함께 보관한다.
    joblib.dump(scaler, SCALER_PATH)
    CLASS_INFO_PATH.write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(history.history).to_csv(
        OUTPUT_DIR / "training_history.csv",
        index_label="epoch",
        encoding="utf-8-sig",
    )
    save_training_plot(history)

    # 메모리에 남은 model을 평가하지 않고 체크포인트 파일을 새로 로드한다.
    print(f"\n최적 모델 다시 로드: {MODEL_PATH}")
    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    test_loss, test_accuracy = loaded_model.evaluate(x_test, y_test, verbose=0)
    probabilities = loaded_model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    report = classification_report(
        y_test,
        predictions,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "test_classification_report.csv",
        encoding="utf-8-sig",
    )
    save_confusion_matrix(y_test, predictions, class_names)

    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    summary = pd.DataFrame(
        [
            {
                "best_epoch": best_epoch,
                "trained_epochs": len(history.history["loss"]),
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        ]
    )
    summary.to_csv(OUTPUT_DIR / "test_summary.csv", index=False, encoding="utf-8-sig")

    print(f"최적 epoch: {best_epoch}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("\n[Classification Report]")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )
    print(f"모델 및 결과 저장 폴더: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
