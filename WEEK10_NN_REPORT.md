# Week 10 · 뉴럴 네트워크 개별 실습 리포트 (11주차 통합분석용)

> 생성: `backend/week10_report.py` (결과 파일: `backend/results/week10_nn_results.json`, 시드 [0, 1, 2]) · 모든 수치는 왕복 비용 0.10% 차감 후, Top-5 동일비중, 현금일(0%) 포함

## 0. 한눈에 보기

- **비ML 최고** (Walk-Forward OOS 샤프): 비ML 룰베이스 최고 **3.74**
- **전통 ML 최고** (Walk-Forward OOS 샤프): Ridge (alpha=100) **4.24**
- **NN 최고** (Walk-Forward OOS 샤프): GRU (3D, L=20) **4.23**
- 신경망은 같은 입력으로 학습한 Ridge 를 **뚜렷하게 넘지 못했다** (GRU 는 동등, MLP 는 오히려 낮음). 비규제 대조군도 붕괴하지 않았다.

## 1. 사용한 신경망 아키텍처 · 입력 · 규제

| 모델 | 구조 | 입력 텐서 | 총 파라미터 | 규제 |
|---|---|---|---:|---|
| MLP (2D) | MLP 17-32(LN,GELU,Drop.3)-16(GELU,Drop.2)-1 | `(N,17)` | 1,185 | Dropout .3/.2 · LayerNorm · AdamW wd 1e-3 · Val 손실 Early Stopping(patience 5, 최적 에폭 복원) |
| GRU (3D, L=20) | GRU(hidden 32, 1층)-last step-LayerNorm-Drop.3-Linear(32,1) | `(N,20,17)` | 4,993 | Dropout .3 · LayerNorm · AdamW wd 1e-3 · Val 손실 Early Stopping(patience 5, 최적 에폭 복원) |
| Overfit_MLP (대조군) | 비규제 대형 MLP 128x3층, Dropout 0, wd 0, Early Stopping 없음 | `(N,17)` | 35,457 | **없음** (Dropout 0 · weight decay 0 · Early Stopping 없음, 고정 60 에폭) |

- 옵티마이저 AdamW (lr 0.001) · 배치 1024 · 손실 **HuberLoss(delta=1, y/σ_train)** (타깃은 Train 구간 표준편차로만 스케일) · 시드 [0, 1, 2] 고정

**입력·타깃·검증 규격**

- 입력 X: 종목별 팩터 **17개**(close_pos, upper_shadow, body_ratio, vol_ratio, vol_surge_up, turnover_rank, ma20_gap, ret_5d, ret_20d, range_pct, r_day, gap_today, rsi_14, night_win_20, night_mean_20, night_sharpe_20, night_sharpe_60)를 **날짜별 100종목 횡단면 백분위 순위(-0.5~+0.5)**로 변환. 스케일러를 fit 하지 않으므로 미래 평균·분산 누수가 없다. **시장 공통 팩터(`mkt_*`)는 입력에서 제외**(레짐은 앞단 게이트로만 사용).
- 타깃 y: 시장 중립 초과수익률 `R_night(i,t+1) − mean_i R_night(t+1)` · 2D `(N, 17)` / 3D `(N, 20, 17)` (최근 20거래일 궤적, 마지막 행 = 당일 t).
- **방식 A 3단 분할**: Train 2020-01-01~2023-12-31 (IS) / Val 2024-01-01~2024-12-31 / OOS Test 2025-01-01~2026-10-21.
- **방식 B 연도별 확장형 Walk-Forward**: 2022~2026 각 해를 그 해 이전 데이터로만 학습(마지막 6개월 = Val/Early Stopping)해 예측, 이어 붙인 OOS 전체 성과 산출.
- 레짐 게이트 `ma120`(KOSPI100 동일가중 지수가 120일선 아래면 쉼)는 **사전에 고정**했고, 나머지 6종 게이트는 JSON 에만 참고용으로 있다.

## 2. 성과 비교표

### 2-A. 3단 분할 (IS = Train / Val / OOS = Test) — 게이트 없음

신경망은 시드 3개 평균. 샤프·누적수익률·MDD·승률(보유일 기준).

| 모델 | 세대 | IS 샤프 | IS 누적 | IS MDD | IS 승률 | Val 샤프 | **OOS 샤프** | OOS 누적 | OOS MDD | OOS 승률 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 비ML 룰베이스 최고 | 비ML | 4.18 | +1,264% | -17.5% | 65.4% | 4.06 | **4.75** | +896% | -17.1% | 64.0% |
| Ridge (alpha=100) | 전통 ML | 4.41 | +1,829% | -15.7% | 68.1% | 4.26 | **4.94** | +967% | -16.7% | 65.7% |
| HistGradientBoosting | 전통 ML | 4.72 | +2,482% | -16.5% | 67.6% | 4.43 | **4.69** | +960% | -19.7% | 64.3% |
| MLP (2D) | NN | 4.81 | +2,576% | -15.8% | 67.7% | 5.00 | **4.90** | +982% | -18.6% | 66.6% |
| GRU (3D, L=20) | NN | 4.83 | +2,486% | -14.8% | 69.0% | 4.73 | **5.30** | +1,162% | -15.4% | 67.7% |
| Overfit_MLP (대조군) | NN(비규제) | 5.08 | +3,243% | -14.3% | 68.8% | 5.34 | **5.13** | +1,259% | -16.7% | 65.2% |

신경망 시드별 샤프 (IS / Val / OOS):

- MLP (2D): seed0 4.71/5.10/4.95 · seed1 4.83/4.74/4.85 · seed2 4.88/5.15/4.91
- GRU (3D, L=20): seed0 4.78/4.86/5.36 · seed1 4.87/4.53/5.53 · seed2 4.84/4.80/5.01
- Overfit_MLP (대조군): seed0 5.12/4.76/5.09 · seed1 4.96/5.56/5.17 · seed2 5.17/5.71/5.14

레짐 게이트 `ma120` 적용 시 OOS 샤프: 비ML 룰베이스 최고 4.77 · Ridge (alpha=100) 5.02 · HistGradientBoosting 4.77 · MLP (2D) 5.06 · GRU (3D, L=20) 5.39 · Overfit_MLP (대조군) 5.11

### 2-B. 연도별 Walk-Forward OOS (2022-01 ~ 2026-10, 1,172 거래일) — 게이트 없음

| 모델 | 세대 | 샤프 | 누적 | MDD | 승률 | 2022 | 2023 | 2024 | 2025 | 2026 | 시드별 샤프 범위 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 비ML 룰베이스 최고 | 비ML | **3.74** | +3,628% | -17.1% | 61.9% | 1.62 | 3.60 | 4.06 | 4.56 | 5.28 | - |
| Ridge (alpha=100) | 전통 ML | **4.24** | +5,614% | -15.2% | 64.7% | 2.21 | 4.61 | 4.47 | 4.83 | 5.76 | - |
| HistGradientBoosting | 전통 ML | **3.39** | +2,815% | -19.2% | 60.5% | 0.40 | 2.60 | 4.55 | 4.34 | 5.08 | - |
| MLP (2D) | NN | **3.94** | +4,415% | -17.3% | 64.2% | 1.95 | 3.77 | 4.61 | 5.03 | 5.08 | 3.85 ~ 4.06 |
| GRU (3D, L=20) | NN | **4.23** | +5,739% | -17.3% | 65.8% | 1.86 | 4.12 | 4.75 | 5.66 | 5.47 | 4.01 ~ 4.38 |
| Overfit_MLP (대조군) | NN(비규제) | **4.17** | +7,500% | -20.1% | 64.8% | 2.24 | 4.46 | 5.10 | 5.21 | 5.12 | 4.02 ~ 4.30 |

레짐 게이트 `ma120` 적용 시 Walk-Forward 샤프: 비ML 룰베이스 최고 3.53 · Ridge (alpha=100) 4.11 · HistGradientBoosting 3.41 · MLP (2D) 3.85 · GRU (3D, L=20) 4.09 · Overfit_MLP (대조군) 3.95

## 3. 실험 중 관찰한 핵심 현상

### ① IS→OOS 붕괴(X자 역전)가 재현되지 않았다 — 비규제 대조군도 OOS 가 유지됨

- Overfit_MLP(3.5만 파라미터, Dropout 0, weight decay 0, Early Stopping 없음)의 3단 분할 샤프는 IS 5.08 / Val 5.34 / **OOS 5.13**, Walk-Forward 4.17. 9주차의 깊은 LightGBM(IS 20.5 → OOS 2.3)과 같은 붕괴가 없다.
- 에폭을 늘려도 Train 손실이 거의 내려가지 않는다 (시드0 Overfit_MLP: 1에폭 0.2939 → 60에폭 0.2918, Val 0.3007). 즉 이 입력으로는 **외울 거리 자체가 적다**.
- **해석(가설)**: 9주차 붕괴의 주범이었던 시장 공통 변수(`MK20` 등)를 입력에서 빼고 모든 피처를 날짜별 횡단면 순위로 바꿨기 때문에, 모델이 '날짜 바코드'를 암기할 단서가 사라졌을 가능성이 크다. 다만 **이 요소만 바꿔 본 대조 실험(예: 시장 공통 변수를 다시 넣은 Overfit_MLP)은 하지 않았으므로 인과는 검증되지 않았다.** 11주차 토의용 후속 실험 후보로 남긴다.

### ② 시드 간 편차

- Walk-Forward 샤프 시드별 범위: MLP 3.85~4.06 · GRU 4.01~4.38 · Overfit_MLP 4.02~4.30. 시드 하나의 폭(약 0.2~0.4)이 GRU·Overfit_MLP·Ridge 사이의 평균 차이(0.1 안팎)보다 크다.
- 따라서 한 번의 학습 결과로 모델 우열을 말하면 안 된다. 시드 3개는 최소한이며, 더 엄밀하게는 10개 이상이 필요하다.

### ③ GRU(20일 시계열 궤적)의 효과 — 방향은 일관되나 확정할 수 없다

- 3단 분할 OOS: GRU 5.30 vs MLP 4.90 (GRU +0.40) · Walk-Forward: GRU 4.23 vs MLP 3.94 (GRU +0.29). 두 검증 모두 GRU 가 앞선다.
- 그러나 (a) 3단 분할 **Val 에서는 반대**(GRU 4.73 < MLP 5.00), (b) GRU 시드 범위가 MLP 와 일부 겹치고, (c) GRU(4,993)는 MLP(1,185)보다 파라미터가 4배라 '20일 궤적'과 '용량' 효과가 섞여 있으며, (d) Walk-Forward 에서 GRU(4.23)는 Ridge(4.24)·비규제 MLP(4.17)와 사실상 같다.
- 결론: 최근 20일 흐름이 **약간의 추가 정보를 줄 가능성**은 있으나, 이 실험만으로 '시계열 궤적 효과'를 주장하기는 어렵다.

### ④ 전통 ML 대비 위치

- Walk-Forward 샤프: Ridge 4.24 ≈ GRU 4.23 ≈ Overfit_MLP 4.17 > MLP 3.94 > 룰베이스 3.74 > HistGB 3.39. 선형 모델이 비선형 신경망과 대등하다 — 이 팩터들의 신호가 대부분 **선형·단조**라는 뜻이다.
- HistGB 는 학습 데이터가 적은 초기 폴드(2022)에서 특히 약했다 (연도별 표 참고). 신경망·Ridge 는 같은 폴드에서 더 안정적이었다.

### ⑤ 기타

- 레짐 게이트 `ma120` 는 Walk-Forward 에서 HistGB 를 뺀 5개 모델의 샤프를 낮췄다 (위 2-B). 3단 분할에서는 MLP·GRU 의 Val/OOS 가 소폭 오르고 IS 는 내려갔지만 Overfit_MLP 는 모두 내려가, 일관된 이득이 아니다.
- Walk-Forward 연도별 샤프가 2022(약 2) → 2026(약 5)로 매년 오르는 것은 학습 데이터 증가 효과와 시장 환경 변화가 섞인 결과로, 이 실험만으로 구분되지 않는다.

## 4. 한계 · 주의

- 유니버스가 **현재** KOSPI 100 구성종목이라 생존편향이 있다. 슬리피지·거래세는 반영하지 않았다. 절대 샤프(4~5)는 이 조건의 산물이므로 **모델 간 상대 비교**로만 읽는다.
- 비ML 룰베이스는 8주차에 2020~2026 **전 구간을 보고 고른** 설정이라 IS/OOS 구분이 의미 없다(사후 최적화). 표의 값은 같은 점수를 게이트 없이 적용한 것이다.
- 누적곡선(대시보드)의 신경망은 시드 0 한 개라 표의 시드평균 샤프와 약간 다르다.
- 체크포인트(`backend/models/week10_best_nn.pt`)는 테스트가 아닌 **Val 샤프**로 골랐다 (선택 편향 방지).

## 5. 재현 · 파일

```bash
# 컨테이너(kis-backend) 안에서
python week10_nn.py --seeds 3        # 학습·평가 -> results/week10_nn_results.json, models/week10_best_nn.pt (약 33분, CPU)
python week10_nn.py --curves         # 시드0 일별 수익률(누적곡선용) -> results/week10_curves.json
python week10_db.py                  # DB(w10_*) 저장
python -m pytest tests/test_week10_dataset.py
```

| 파일 | 설명 |
|---|---|
| `backend/week10_dataset.py` | 누수 없는 2D/3D 데이터셋 · 3단 분할 · Walk-Forward 구간 · 백테스트 헬퍼 |
| `backend/week10_nn.py` | MLP/GRU/Overfit_MLP 학습, 기준선(룰베이스·Ridge·HistGB), 평가 |
| `backend/week10_db.py` | DB 저장(`w10_models/metrics/history/daily`) · API 페이로드 |
| `backend/main.py` | `GET /backtest/week10` |
| `frontend/backtest.html` | WEEK 10 탭(`#m10`) |
| `backend/results/week10_dashboard.png` | 대시보드 캡처 |
