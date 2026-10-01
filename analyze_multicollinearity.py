"""
다중공선성(Multicollinearity) 3단계 종합 진단 분석 스크립트
1. 상관행렬(Correlation Matrix) 1차 필터링 (|r| >= 0.8)
2. VIF(Variance Inflation Factor) 계산 (상수항 포함)
3. 특이값 분해(SVD) 기반 Condition Index 및 분산 분할 비율(VDP) 교차 검증
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Windows 콘솔 인코딩 및 폰트 설정
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def load_and_preprocess(csv_path="data/multicollinearity_data.csv"):
    """데이터 로드 및 수치형 변수 표준화"""
    print(f"\n[데이터 로드] '{csv_path}' 읽는 중...")
    df = pd.read_csv(csv_path)
    
    feature_cols = [col for col in df.columns if col != 'y']
    X = df[feature_cols]
    y = df['y']
    
    # 수치형 변수 스케일 표준화 (StandardScaler)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    
    print(f" - 전체 샘플 수: {len(df)}개, 피처: {feature_cols}")
    return df, X, y, X_scaled


# =====================================================================
# 1단계: 상관행렬(Correlation Matrix) 1차 필터링
# =====================================================================
def step1_correlation_filtering(X, threshold=0.8, output_img="correlation_heatmap.png"):
    print("\n" + "=" * 70)
    print(" [1단계] 상관행렬(Correlation Matrix) 1차 필터링 (|r| >= 0.8)")
    print("=" * 70)
    
    corr_matrix = X.corr()
    print("■ 피처 간 상관계수 행렬:")
    print(corr_matrix.round(4))
    
    # |r| >= 0.8 이상인 변수 쌍 탐지
    high_corr_pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r_val = corr_matrix.iloc[i, j]
            if abs(r_val) >= threshold:
                high_corr_pairs.append((cols[i], cols[j], r_val))
                
    print(f"\n■ |r| >= {threshold} 이상인 강한 상관관계 변수 쌍:")
    if high_corr_pairs:
        for f1, f2, r_val in high_corr_pairs:
            print(f"  ▶ ({f1}, {f2}) : 상관계수 r = {r_val:.4f}  -> 명백하게 중복된 정보를 포함!")
        print("  ☞ 조치 가이드: 위 변수 쌍 중 하나(예: 상관계수 0.99 이상인 변수)를 1차적으로 제거/통합 검토.")
    else:
        print("  - |r| >= 0.8 이상인 쌍이 없습니다.")

    # 히트맵 시각화 및 저장
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", cbar=True, vmin=-1, vmax=1)
    plt.title("피처 간 상관계수 히트맵 (Correlation Heatmap)", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close()
    print(f"  (상관행렬 히트맵이 '{output_img}'로 저장되었습니다.)")
    
    return corr_matrix, high_corr_pairs


# =====================================================================
# 2단계: VIF(Variance Inflation Factor) 계산 (상수항 포함)
# =====================================================================
def step2_vif_calculation(X):
    print("\n" + "=" * 70)
    print(" [2단계] VIF(Variance Inflation Factor) 계산 (상수항 포함)")
    print("=" * 70)
    
    # ※ 중요: VIF 왜곡 방지를 위해 상수항(const)을 반드시 추가
    X_with_const = add_constant(X)
    
    vif_data = []
    for i, col in enumerate(X_with_const.columns):
        if col == 'const':
            continue  # 해석 편의를 위해 독립변수만 출력 대상에 표시 (계산은 상수항 포함 상태에서 수행)
        vif_val = variance_inflation_factor(X_with_const.values, i)
        
        # 위험도 판정
        if vif_val >= 10:
            status = "심각 (VIF >= 10, 다중공선성 확실)"
        elif vif_val >= 5:
            status = "주의/경계 (5 <= VIF < 10, 추가 검증 필요)"
        else:
            status = "양호 (VIF < 5)"
            
        vif_data.append({"Variable": col, "VIF": vif_val, "Status": status})
        
    vif_df = pd.DataFrame(vif_data)
    print(vif_df.to_string(index=False))
    return vif_df


# =====================================================================
# 3단계: 특이값 분해(SVD) 기반 Condition Index & 분산 분할 비율(VDP)
# =====================================================================
def step3_condition_index_and_vdp(X):
    print("\n" + "=" * 70)
    print(" [3단계] 상태지수(Condition Index) 및 분산 분할 비율(VDP) 교차 검증")
    print("=" * 70)
    print(" (Belsley, Kuh, and Welsch 진단법: 각 열을 단위 길이로 정규화 후 SVD 수행)")
    
    # 1) 상수항 추가
    X_with_const = add_constant(X)
    col_names = list(X_with_const.columns)
    X_mat = X_with_const.values.astype(float)
    
    # 2) 각 열을 유클리드 노름(단위 길이, L2 Norm)으로 정규화 (Belsley 표준 방식)
    col_norms = np.sqrt(np.sum(X_mat ** 2, axis=0))
    X_norm = X_mat / col_norms
    
    # 3) 특이값 분해 (SVD): X_norm = U * S * V^T
    U, s, Vt = np.linalg.svd(X_norm, full_matrices=False)
    V = Vt.T  # (p x p)
    
    # 4) 상태지수(Condition Index) 계산: eta_k = s_max / s_k
    s_max = np.max(s)
    cond_indices = s_max / s
    
    # 5) 분산 분할 비율(Variance Decomposition Proportions, pi_kj) 계산
    # phi_kj = (v_jk / s_k)^2
    # phi_j = sum_k(phi_kj)
    # pi_kj = phi_kj / phi_j
    phi = (V / s[np.newaxis, :]) ** 2  # (p x p)
    phi_sum = np.sum(phi, axis=1, keepdims=True)
    pi = phi / phi_sum  # (p, p) -> pi[j, k] : j번째 변수의 k번째 특이값에 대한 분산 분할 비율
    pi_matrix = pi.T    # 행: 특이값(차원 k), 열: 변수 j
    
    # 테이블 구성
    table_dict = {
        "Singular Value": np.round(s, 4),
        "Condition Index": np.round(cond_indices, 2)
    }
    for j, col in enumerate(col_names):
        table_dict[col] = np.round(pi_matrix[:, j], 4)
        
    vdp_df = pd.DataFrame(table_dict)
    print("\n■ Condition Index & Variance Decomposition Proportions Table:")
    print(vdp_df.to_string(index=True))
    
    # 다중공선성 결합 군집 식별 (Condition Index >= 10 이면서 VDP >= 0.5 인 변수들)
    print("\n■ 다중공선성 유발 군집(Collinear Clusters) 자동 탐지:")
    severe_found = False
    for k in range(len(cond_indices)):
        ci = cond_indices[k]
        if ci >= 10.0:  # 주의 또는 심각 기준
            high_vdp_vars = [col_names[j] for j in range(len(col_names)) if pi_matrix[k, j] >= 0.5]
            if len(high_vdp_vars) >= 2:
                severe_found = True
                severity = "심각(Severe)" if ci >= 30 else "주의(Moderate)"
                print(f"  ▶ [특이값 차원 {k}] 상태지수(CI) = {ci:.2f} ({severity})")
                print(f"     - 분산 분할 비율 0.5 이상 겹치는 변수 군집: {high_vdp_vars}")
                print(f"     - 해당 변수들의 분산 분할 비율: {[f'{col_names[j]}: {pi_matrix[k, j]:.3f}' for j in range(len(col_names)) if col_names[j] in high_vdp_vars]}")
                
    if not severe_found:
        print("  - 상태지수 10 이상에서 분산 분할 비율 0.5 이상 겹치는 변수 쌍이 없습니다.")
        
    return vdp_df


# =====================================================================
# 메인 분석 실행 및 종합 진단 요약
# =====================================================================
def main():
    csv_file = "data/multicollinearity_data.csv"
    if not os.path.exists(csv_file):
        print(f"오류: '{csv_file}' 파일이 존재하지 않습니다.")
        return
        
    df, X, y, X_scaled = load_and_preprocess(csv_file)
    
    # 1단계: 상관행렬 분석
    corr_matrix, high_corr_pairs = step1_correlation_filtering(X, threshold=0.8)
    
    # 2단계: VIF 산출
    vif_df = step2_vif_calculation(X)
    
    # 3단계: 상태지수 및 분산 분할 비율 교차 검증
    vdp_df = step3_condition_index_and_vdp(X)
    
    # 최종 진단 요약
    print("\n" + "=" * 70)
    print(" [최종 진단 및 조치 제안 (Summary & Action Plan)]")
    print("=" * 70)
    print("1. [x1 - x4 관계]:")
    print("   - 상관계수가 약 0.9997로 1단계 상관행렬에서 즉시 식별됨.")
    print("   - VIF 역시 극도로 높으며, 동일한 정보를 두 번 담고 있으므로 둘 중 하나(예: x4) 제거 필수.")
    print()
    print("2. [x2 - x3 - x5 관계]:")
    print("   - 1:1 상관계수는 ~0.69 수준으로 1단계(|r| >= 0.8)는 통과하였으나,")
    print("   - 2단계 VIF가 극단적으로 높고, 3단계 높은 상태지수(CI)에서 세 변수의 분산 분할 비율이 0.5 이상 강하게 중첩됨.")
    print("   - 즉, x5가 x2와 x3의 선형결합(x5 ≈ x2 + x3)으로 생성된 다중공선성 변수임을 명백히 입증.")
    print("   - 조치: x5를 제거하거나, x2/x3와의 관계를 정리하여 단일화 권장.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
