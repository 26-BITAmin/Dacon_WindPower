# 제3회 풍력발전량 예측 AI 경진대회 — BARAM 2026

> 대회 페이지: https://dacon.io/competitions/official/236727/overview/description
> 팀 저장소: https://github.com/26-BITAmin/Dacon_WindPower (**wonwoo 브랜치가 내 작업 브랜치**, private repo — `gh auth login` 필요)
> 모델 실험 기록: [MODEL.md](MODEL.md)

## 1. 대회 개요

| 항목 | 내용 |
|---|---|
| 주최 | 한국동서발전, GS E&R, 태백가덕산풍력발전 (운영: 데이콘) |
| 과제 | 기상 예보 데이터 기반으로 3개 KPX 그룹의 **시간별 풍력발전량(kWh)** 예측 |
| 예측 기간 | 2025-01-01 01:00 ~ 2026-01-01 00:00 (8,760시간) |
| 대회 기간 | 2026-07-06 ~ 2026-08-14 (2차 발표평가: 8-17 ~ 9-04) |
| 제출 제한 | 1일 최대 5회, 최종 제출물 1개 선택 |
| 리더보드 | Public 40% / Private 60% (사전 샘플링), 최종 순위는 Private 100% |
| 외부 데이터 | 공개·합법 데이터만 허용. 평가 기간의 실측/SCADA/재분석자료 사용 금지 |

### 예측 대상

| 컬럼 | 설비용량 | 1시간 환산 | 라벨 제공 기간 |
|---|---:|---:|---|
| `kpx_group_1` | 21.6 MW | 21,600 kWh | 2022~2024 |
| `kpx_group_2` | 21.6 MW | 21,600 kWh | 2022~2024 |
| `kpx_group_3` | 21.0 MW | 21,000 kWh | **2023~2024 (1년 적음)** |

## 2. 평가 지표 (정확한 공식 — `ensemble_q4fold.ipynb`의 `metric()`으로 구현·검증됨)

**총점 = 0.5 × (1-NMAE) + 0.5 × FiCR**

공통 전처리: **실제 발전량이 설비용량의 10% 이상인 시간대만 평가 대상.**
`error_rate = abs(pred − actual) / 설비용량` (그룹별 용량으로 정규화, 실측값이 아님!)

### 1-NMAE (평균 예측오차율)
- 그룹별 `NMAE = mean(error_rate)` → `1-NMAE = 1 − (3개 그룹 NMAE 평균)`

### FiCR (정산금획득률) — **계단 함수 + 발전량 가중**
- 시간별 단가: `error_rate ≤ 6%` → 4원/kWh, `6% < error_rate ≤ 8%` → 3원/kWh, `> 8%` → **0원**
- 그룹별 `FiCR = Σ(actual × 단가) / Σ(actual × 4원)` → 3개 그룹 평균
- 실제 KPX 재생에너지 발전량 예측제도의 정산금 구조를 그대로 반영한 지표

```python
# metric() 핵심 (ensemble_q4fold.ipynb cell-7)
valid = actual >= capacity * 0.10
error_rate = np.abs(forecast - actual)[valid] / capacity
nmae = np.mean(error_rate)
unit_price = np.select([error_rate <= 0.06, error_rate <= 0.08], [4.0, 3.0], default=0.0)
ficr = np.sum(actual[valid] * unit_price) / np.sum(actual[valid] * 4.0)
score = 0.5 * (1 - nmae) + 0.5 * ficr
```

### 왜 FiCR이 승부처인가

- 리더보드에서 **1-NMAE는 팀 간 차이가 미미**하지만 **FiCR은 차이가 큼** (2026-07 초 기준 관찰).
- **(2026-07-21 갱신)** 상위권 1-NMAE가 0.88대(최소 0.87대 후반)로 올라옴 (사용자 확인). 우리 LB 1-NMAE는 0.870(exp-009)~0.871(exp-003)로 **1-NMAE에서도 상위권과 ~0.01 격차** → Score −0.005 상당. FiCR 후처리만으로는 못 메꾸므로 **base 예측 정확도 개선(피처·모델)을 병행**해야 함.
- 1-NMAE는 연속 지표라 평균 오차를 조금 줄여도 점수 변화가 작다. FiCR은 **6%/8% 경계를 넘느냐로 시간당 점수가 4→3→0으로 급락**하는 계단 함수라, 오차 분포의 "모양"에 따라 같은 MAE에서도 FiCR이 크게 갈린다.
- FiCR은 **actual(발전량)로 가중**되므로, 발전량이 큰 시간대(강풍 시간대)를 6% 안에 맞추는 것이 저발전 시간대보다 훨씬 가치가 크다.
- 즉 MAE 최소화(중앙값 예측)와 FiCR 최대화(6% 밴드 적중 확률 최대화)는 **다른 최적화 문제**다.

## 3. 데이터

| 파일 | 내용 | 비고 |
|---|---|---|
| `train/ldaps_train.csv` | LDAPS 예보 (1.5km 해상도, 16격자) | 2022-01 ~ 2025-01 |
| `train/gfs_train.csv` | GFS 예보 (0.25°, 9격자) | 상동. 80m/100m 바람, 상층(850/700/500hPa) 포함 |
| `train/train_labels.csv` | KPX 그룹별 실제 발전량 | group_3은 2022년 빈칸 |
| `train/scada_{vestas,unison}_train.csv` | 터빈별 10분 SCADA (power/ws/wd) | VESTAS 12기 + UNISON 5기. **test 기간엔 없음** |
| `test/{ldaps,gfs}_test.csv` | 2025년 예보 | |
| `info.xlsx` | 터빈 좌표·허브높이·용량·KPX 그룹 매핑 | |

주의사항:
- 예보는 매일 09:00 KST 초기화분에서 다음날 01:00~그다음날 00:00 구간만 추출됨. `data_available_kst_dtm`(전날 13:00) 이후 정보만 쓸 수 있음 → **자기회귀(lag) 피처는 리크**. 예보 horizon은 12~35시간.
- CSV는 전부 `utf-8-sig`.
- SCADA는 train 기간만 제공 → 직접 피처로는 못 쓰고, **터빈 파워커브 학습·중간 타깃(터빈별 발전량)** 용도로만 활용 가능.

## 4. 현재 파이프라인 (wonwoo 브랜치, `ensemble_q4fold.ipynb`)

버전을 v5 → v9 → v10으로 누적해온 구조:

1. **격자 매칭**: `info.xlsx` 터빈 좌표 → 그룹 centroid → haversine 최근접 grid 1개 선택 (LDAPS: 그룹별 5/6/12, GFS: 전 그룹 5)
2. **피처 (87개)**: u/v → 풍속·풍향 sin/cos (10m/80m/100m/pbl), 윈드시어(100-10, 80-10), 달력 피처, LDAPS·GFS **예보 불일치도**(v9), 고변동 계절 인터랙션(v10)
3. **데이터 형태**: 그룹을 long-format으로 쌓아 `kpx_group`을 피처로 사용, 그룹별 개별 학습
4. **모델**: LightGBM(`objective=mae`) + XGBoost(`reg:absoluteerror`) **50:50 블렌딩**, 각 5-seed bagging
5. **샘플 가중**: 평가 제외 구간(actual < 10%cap)은 weight 0.05로 다운웨이트
6. **후처리** (FiCR 대응):
   - **풍속 구간별 편향 보정**: held-out 구간에서 ws bin(0-11/11-13/13+ m/s)별 signed error 추정 → 곱셈 보정
   - **rolling smoothing** (window=3, α=0.7)
   - **하한 clip = 10%cap**: 평가 대상 시간엔 actual ≥ 10%cap이 보장되므로, 그 밑으로 예측하면 손해만 봄 → 바닥을 10%cap으로 올림
7. **검증**: 2024년 분기별 4-fold (expanding window: 과거 전체로 학습 → 다음 분기 검증). fold 평균 + pooled 두 방식으로 집계

### 현재 성능 (4-fold CV, 블렌딩 50:50)

| 지표 | Pooled | fold 평균 |
|---|---:|---:|
| Score | 0.6248 | 0.6259 |
| 1-NMAE | 0.8766 | 0.8769 |
| FiCR | 0.3729 | 0.3748 |

그룹별 (fold 평균): 

| 그룹 | 1-NMAE | FiCR |
|---|---:|---:|
| 1 | 0.884 | 0.389 |
| 2 | 0.881 | 0.420 |
| 3 | **0.865** | **0.316** ← 최약체 (라벨 1년 적음) |

참고: `open (1)/train_ensemble.py`는 초기 베이스라인(전 격자 mean/std/min/max 집계 + RF/XGB/Cat/LGBM 역-MAE 가중 앙상블, 2024년 holdout). 현재는 노트북 파이프라인이 우위.

## 5. FiCR 개선 전략 (우선순위)

1. **분위수(quantile) 기반 제출값 최적화**: LGBM quantile regression으로 시간별 예측 분포를 얻고, 제출값 q를 `E[단가(|q−Y|/cap) × Y]`가 최대가 되도록 선택. 분포가 좁으면 median 근처, 넓으면 기대 정산금을 최대화하는 쪽으로 치우친 값이 최적 — MAE 최적해(median)와 다름.
2. **그룹 3 집중**: FiCR 0.316으로 전체 평균을 끌어내리는 주범. 라벨이 1년 적으므로 그룹 1·2 데이터 전이(그룹을 피처로 한 통합 학습 강화, 파워커브 공유), SCADA 기반 사전학습 검토.
3. **발전량 가중 반영**: FiCR은 actual 가중이므로 강풍(고발전) 시간대 정확도가 지표에 곱으로 들어감. 학습 sample weight를 10%cap 컷 이분법이 아니라 발전량 비례로 조정하는 실험.
4. **편향 보정 고도화**: 현재 ws 3-bin 곱셈 보정 → 6%/8% 경계 기준으로 "아깝게 벗어난" 시간대를 밴드 안으로 넣는 목적함수로 재설계.
5. **검증 시 FiCR 분해 모니터링**: 시간대별 error_rate 히스토그램에서 6~10% 구간(경계 근처) 질량을 추적. 이 구간을 6% 안으로 밀어 넣는 것이 FiCR 개선의 직접 경로.

## 6. 로컬 환경 메모

- 데이터: `open (1)/` 폴더 (train/, test/, info.xlsx, sample_submission.csv)
- `ensemble_q4fold.ipynb`는 Colab 경로(`/content/drive/MyDrive/비타민/풍력발전량예측/data`) 기준 → 로컬 실행 시 `DATA_DIR`만 `open (1)/`로 변경
- 베이스라인 스크립트: `python "open (1)/train_ensemble.py" [--quick] [--jobs N] [--ensemble weighted|mean]`
- 이 폴더는 git repo가 아님. 팀 repo는 private이므로 접근하려면 `gh auth login` 선행
