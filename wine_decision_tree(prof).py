import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Windows 환경 Graphviz 바이너리 경로 자동 추가
GRAPHVIZ_BIN_DIR = r"C:\Program Files\Graphviz\bin"
if os.path.exists(GRAPHVIZ_BIN_DIR) and GRAPHVIZ_BIN_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + GRAPHVIZ_BIN_DIR

import graphviz

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 0. 한글 폰트 및 스타일 설정
# ============================================================
plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Malgun Gothic")

OUTPUT_DIR = r"C:\apps\mldl\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. Wine 데이터셋 로드
# ============================================================
wine = load_wine()

X = wine.data
y = wine.target

feature_names = wine.feature_names
target_names = wine.target_names

print("=" * 70)
print("[1] Wine 데이터셋 정보")
print("=" * 70)
print(f"샘플 수: {X.shape[0]}")
print(f"피처 수: {X.shape[1]}")
print(f"클래스: {list(target_names)}")
print(f"피처: {list(feature_names)}")


# ============================================================
# 2. 계층적 데이터 분할 (Train 80% / Test 20%)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 70)
print("[2] 데이터 분할")
print("=" * 70)
print(f"Train shape: {X_train.shape}")
print(f"Test shape : {X_test.shape}")


# ============================================================
# 3. 의사결정나무 모델 학습
#    - criterion='gini'
#    - 과적합 방지 및 해석력을 위해 max_depth=3
# ============================================================
dt_clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

dt_clf.fit(X_train, y_train)


# ============================================================
# 4. 5-Fold 교차검증 및 테스트셋 성능 평가
# ============================================================
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    dt_clf,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

y_pred = dt_clf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 70)
print("[3] 모델 성능 평가")
print("=" * 70)
print(f"5-Fold CV 평균 정확도: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print(f"테스트셋 정확도: {test_acc:.4f}\n")

print("[Classification Report]")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_names
    )
)


# ============================================================
# 5. Graphviz 의사결정나무 시각화
# ============================================================
dot_data = export_graphviz(
    dt_clf,
    out_file=None,
    feature_names=feature_names,
    class_names=target_names,
    filled=True,
    rounded=True,
    special_characters=True,
    precision=2
)

graph = graphviz.Source(dot_data)

tree_output_path = os.path.join(
    OUTPUT_DIR,
    "wine_tree_graphviz"
)

graph.render(
    tree_output_path,
    format="png",
    cleanup=True
)

print(f"Graphviz 트리 저장 완료: {tree_output_path}.png")


# ============================================================
# 6. 피처 중요도 DataFrame 생성
# ============================================================
fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": dt_clf.feature_importances_
}).sort_values(
    by="Importance",
    ascending=True
)

print("\n" + "=" * 70)
print("[4] Feature Importance")
print("=" * 70)

fi_print = fi_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

print(fi_print.to_string(index=False))


# ============================================================
# 7. 혼동 행렬 및 피처 중요도 시각화
# ============================================================
fig, axes = plt.subplots(
    1,
    2,
    figsize=(18, 7)
)

# -------------------------
# 7-1. 혼동 행렬
# -------------------------
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names,
    ax=axes[0],
    cbar=False
)

axes[0].set_title(
    f"1. 혼동 행렬 (정확도: {test_acc * 100:.1f}%)",
    fontsize=14,
    fontweight="bold",
    pad=12
)
axes[0].set_xlabel("예측 클래스", fontsize=12)
axes[0].set_ylabel("실제 클래스", fontsize=12)


# -------------------------
# 7-2. 피처 중요도 막대그래프
# -------------------------
colors = [
    "#1f77b4" if imp > 0 else "#cccccc"
    for imp in fi_df["Importance"]
]

axes[1].barh(
    np.arange(len(fi_df)),
    fi_df["Importance"],
    color=colors,
    alpha=0.85
)

axes[1].set_yticks(
    np.arange(len(fi_df))
)
axes[1].set_yticklabels(
    fi_df["Feature"],
    fontsize=11
)

axes[1].set_title(
    "2. 와인 분류 피처 중요도 (Feature Importance)",
    fontsize=14,
    fontweight="bold",
    pad=12
)

axes[1].set_xlabel(
    "Gini 불순도 감소 기여도",
    fontsize=12
)

for i, imp in enumerate(fi_df["Importance"]):
    if imp > 0:
        axes[1].annotate(
            f" {imp:.3f} ({imp * 100:.1f}%)",
            (imp, i),
            va="center",
            fontsize=10,
            fontweight="bold"
        )

plt.tight_layout()

feature_output_path = os.path.join(
    OUTPUT_DIR,
    "wine_feature_importance.png"
)

plt.savefig(
    feature_output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(f"혼동 행렬 및 피처 중요도 저장 완료: {feature_output_path}")


# ============================================================
# 8. 핵심 피처 중요도 요약
# ============================================================
top3 = fi_print.head(3)
top3_sum = top3["Importance"].sum()

print("\n" + "=" * 70)
print("[5] 핵심 피처 중요도 요약")
print("=" * 70)

for idx, row in top3.iterrows():
    print(
        f"{idx + 1}위: {row['Feature']} "
        f"- Importance={row['Importance']:.4f} "
        f"({row['Importance'] * 100:.1f}%)"
    )

print(
    f"\n상위 3개 피처의 총 중요도 비중: "
    f"{top3_sum * 100:.1f}%"
)

print("\n분석 완료.")
print(f"출력 폴더: {OUTPUT_DIR}")
