"""
[사이킷런 없이 순수 파이썬/넘파이로 구현하는 선형회귀 학습]
linear_regression_data.py에서 생성한 100개의 데이터를 활용하여
1) 경사하강법(Gradient Descent)
2) 정규방정식(Normal Equation)
두 가지 방식으로 최적의 가중치(w)와 편향(b)을 구하고 학습 과정을 시각화합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from linear_regression_data import generate_data

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# ==========================================================
# 1. 경사하강법 (Gradient Descent) 구현 클래스
# ==========================================================
class LinearRegressionGD:
    """순수 파이썬/넘파이로 만든 경사하강법 기반 선형회귀 모델"""
    
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = 0.0  # 가중치 (기울기)
        self.b = 0.0  # 편향 (절편)
        self.loss_history = []  # 에폭별 손실값 저장용
        
    def fit(self, X, y):
        n_samples = len(X)
        # 가중치 초기화 (0 또는 작은 무작위 값)
        self.w = 0.0
        self.b = 0.0
        
        print("\n--- 경사하강법(Gradient Descent) 학습 시작 ---")
        for epoch in range(1, self.epochs + 1):
            # 1) 가설(예측값) 계산: y_pred = w * X + b
            y_pred = self.w * X + self.b
            
            # 2) 오차(Error) 계산
            errors = y_pred - y
            
            # 3) 손실 함수 (MSE: 평균 제곱 오차) 계산: L = (1/n) * sum(error^2)
            mse_loss = np.mean(errors ** 2)
            self.loss_history.append(mse_loss)
            
            # 4) 기울기(Gradient) 계산 (손실함수를 w와 b로 각각 편미분)
            # dL/dw = (2 / n) * sum(errors * X)
            # dL/db = (2 / n) * sum(errors)
            dw = (2 / n_samples) * np.sum(errors * X)
            db = (2 / n_samples) * np.sum(errors)
            
            # 5) 파라미터 업데이트 (경사의 반대 방향으로 이동)
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            # 200 에폭마다 진행 상황 출력
            if epoch % 200 == 0 or epoch == 1:
                print(f"Epoch {epoch:4d}/{self.epochs} | Loss(MSE): {mse_loss:8.4f} | w: {self.w:6.4f}, b: {self.b:6.4f}")
                
        print(f"학습 완료! 최종 모델: y = {self.w:.4f}x + {self.b:.4f}")
        return self

    def predict(self, X):
        return self.w * X + self.b


# ==========================================================
# 2. 정규 방정식 (Normal Equation - 해석적 최적해)
# ==========================================================
def solve_normal_equation(X, y):
    """
    정규방정식 공식: theta = (X_b^T * X_b)^(-1) * X_b^T * y
    미분값이 0이 되는 지점을 한 번의 역행렬 계산으로 구함
    """
    n_samples = len(X)
    # 편향(b)을 위해 X 앞에 1로 이루어진 열을 추가: X_b = [1, X]
    X_b = np.c_[np.ones((n_samples, 1)), X]
    
    # (X_b^T * X_b)^(-1) * X_b^T * y 계산
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    
    best_b = theta_best[0][0]
    best_w = theta_best[1][0]
    
    return best_w, best_b


# ==========================================================
# 3. 메인 실행 및 시각화 비교
# ==========================================================
def main():
    # 1) linear_regression_data.py에서 데이터 불러오기
    X, y, true_w, true_b = generate_data(num_samples=100, seed=42)
    
    print("=" * 60)
    print(f" [실제 데이터 관계식 (참값)] : y = {true_w:.4f}x + {true_b:.4f}")
    print("=" * 60)
    
    # 2) 방법 1: 경사하강법으로 학습
    # (학습률 lr=0.01, 에폭 1500)
    model_gd = LinearRegressionGD(learning_rate=0.01, epochs=1500)
    model_gd.fit(X, y)
    
    # 3) 방법 2: 정규방정식으로 이론적 최적해 구하기
    norm_w, norm_b = solve_normal_equation(X, y)
    print("\n--- 정규방정식 (Normal Equation) 계산 결과 ---")
    print(f"이론적 최적해: y = {norm_w:.4f}x + {norm_b:.4f}")
    
    # 4) 성능 지표 (MSE) 비교
    y_pred_gd = model_gd.predict(X)
    y_pred_norm = norm_w * X + norm_b
    
    mse_gd = np.mean((y - y_pred_gd) ** 2)
    mse_norm = np.mean((y - y_pred_norm) ** 2)
    
    print("\n" + "=" * 60)
    print(" [최종 비교 결과] ")
    print(f" 1. 실제 정답 관계식  : y = {true_w:.4f}x + {true_b:.4f}")
    print(f" 2. 경사하강법 학습   : y = {model_gd.w:.4f}x + {model_gd.b:.4f}  (MSE: {mse_gd:.4f})")
    print(f" 3. 정규방정식 최적해 : y = {norm_w:.4f}x + {norm_b:.4f}  (MSE: {mse_norm:.4f})")
    print("=" * 60)
    
    # 5) 시각화 (회귀선 비교 + 학습 곡선)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # [왼쪽 그래프] 데이터와 학습된 회귀선 비교
    axes[0].scatter(X, y, color='dodgerblue', alpha=0.6, edgecolors='k', label='데이터 포인트 (N=100)')
    
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    y_line_true = true_w * X_test + true_b
    y_line_gd = model_gd.predict(X_test)
    y_line_norm = norm_w * X_test + norm_b
    
    axes[0].plot(X_test, y_line_true, 'g--', linewidth=2, label=f'실제 정답 (w={true_w}, b={true_b})')
    axes[0].plot(X_test, y_line_gd, 'r-', linewidth=2.5, alpha=0.8, label=f'경사하강법 (w={model_gd.w:.2f}, b={model_gd.b:.2f})')
    axes[0].plot(X_test, y_line_norm, 'k:', linewidth=2, label=f'정규방정식 (w={norm_w:.2f}, b={norm_b:.2f})')
    
    axes[0].set_title("선형회귀 모델 적합 결과 비교", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("피처 (X)", fontsize=11)
    axes[0].set_ylabel("타깃 (y)", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # [오른쪽 그래프] 에폭에 따른 Loss(MSE) 감소 과정
    axes[1].plot(model_gd.loss_history, color='darkorange', linewidth=2)
    axes[1].set_title("경사하강법 손실(MSE) 감소 곡선", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("에폭 (Epochs)", fontsize=11)
    axes[1].set_ylabel("손실 (Mean Squared Error)", fontsize=11)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    save_path = "regression_result.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n학습 및 비교 시각화 그래프가 '{save_path}'로 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    main()
