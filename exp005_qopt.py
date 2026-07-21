# ═══════════════════════════════════════════════════════
# exp-005: v10 블렌딩 + 분위수 기반 제출값 최적화 (FiCR 기대점수 최대화)
#
# 베이스: ensemble_q4fold.ipynb (exp-003, pooled Score 0.6248)
# 변경점: 후처리 단계에서 LGBM quantile 모델 10개로 시간별 예측 분포를 얻고,
#         분포 중앙값을 블렌딩 예측값에 정렬한 뒤, 시간별로
#         E[ -error_rate + 단가(error_rate) * y / (4*ȳ) ]  (점수 공식에서 유도)
#         를 최대화하는 제출값 q를 그리드 탐색으로 선택.
#         spread 스케일 {0.7, 1.0, 1.5} 세 변형 + control(exp-003 재현) 동시 비교.
#
# 실행: python -X utf8 exp005_qopt.py [--quick] [--jobs N] [--no-submit]
# ═══════════════════════════════════════════════════════
import argparse
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgbm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = Path(__file__).parent / "open (1)"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPACITY_KWH = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}

QUANTILES = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]


# ───────────────────────────────────────────────
# 데이터 파이프라인 (노트북 cell 1~8 이식)
# ───────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def dms_to_decimal(s):
    pattern = r"(\d+)°(\d+)'([\d.]+)\"([NS])\s+(\d+)°(\d+)'([\d.]+)\"([EW])"
    m = re.match(pattern, s.strip())
    if not m:
        raise ValueError(f"좌표 파싱 실패: {s}")
    d1, m1, s1, hemi1, d2, m2, s2, hemi2 = m.groups()
    lat = float(d1) + float(m1) / 60 + float(s1) / 3600
    if hemi1 == "S":
        lat = -lat
    lon = float(d2) + float(m2) / 60 + float(s2) / 3600
    if hemi2 == "W":
        lon = -lon
    return lat, lon


def find_nearest_grid(weather_df, coord_df, name_col, lat_col="lat", lon_col="lon"):
    grids = weather_df[["grid_id", "latitude", "longitude"]].drop_duplicates()
    mapping = {}
    for _, row in coord_df.iterrows():
        dists = haversine(row[lat_col], row[lon_col], grids["latitude"], grids["longitude"])
        mapping[row[name_col]] = grids.loc[dists.idxmin(), "grid_id"]
    return mapping


def add_wind_features(df, uv_pairs):
    df = df.copy()
    for u_col, v_col, height in uv_pairs:
        df[f"ws_{height}"] = np.sqrt(df[u_col] ** 2 + df[v_col] ** 2)
        wd_rad = np.arctan2(df[v_col], df[u_col])
        df[f"wd_sin_{height}"] = np.sin(wd_rad)
        df[f"wd_cos_{height}"] = np.cos(wd_rad)
    return df


def build_group_weather(weather_df, group_grid_map, prefix):
    weather_df = weather_df.copy()
    weather_df["forecast_kst_dtm"] = pd.to_datetime(weather_df["forecast_kst_dtm"])
    drop_cols = {"data_available_kst_dtm", "grid_id", "latitude", "longitude"}
    value_cols = [c for c in weather_df.columns if c not in {"forecast_kst_dtm", *drop_cols}]
    frames = []
    for group, grid_id in group_grid_map.items():
        sub = weather_df.loc[weather_df["grid_id"] == grid_id, ["forecast_kst_dtm", *value_cols]].copy()
        sub = sub.rename(columns={c: f"{prefix}_{c}" for c in value_cols})
        sub["kpx_group"] = int(group)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def add_forecast_disagreement(df):
    df = df.copy()
    df["ws_disagreement"] = np.abs(df["ldaps_ws_10m"] - df["gfs_ws_10m"])
    df["wd_agreement"] = df["ldaps_wd_cos_10m"] * df["gfs_wd_cos_10m"] + df["ldaps_wd_sin_10m"] * df["gfs_wd_sin_10m"]
    return df


def add_seasonal_interaction(df):
    df = df.copy()
    month = pd.to_datetime(df["forecast_kst_dtm"]).dt.month
    df["high_variance_season"] = month.isin([1, 2, 3, 4, 5, 7]).astype(int)
    df["ws_x_season"] = df["ldaps_ws_10m"] * df["high_variance_season"]
    df["disagreement_x_season"] = df["ws_disagreement"] * df["high_variance_season"]
    return df


def build_datasets():
    train_labels = pd.read_csv(TRAIN_DIR / "train_labels.csv", encoding="utf-8-sig")
    sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding="utf-8-sig")
    ldaps_train = pd.read_csv(TRAIN_DIR / "ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv(TRAIN_DIR / "gfs_train.csv", encoding="utf-8-sig")
    ldaps_test = pd.read_csv(TEST_DIR / "ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv(TEST_DIR / "gfs_test.csv", encoding="utf-8-sig")

    train_labels["kst_dtm"] = pd.to_datetime(train_labels["kst_dtm"])
    sample_submission["forecast_kst_dtm"] = pd.to_datetime(sample_submission["forecast_kst_dtm"])

    raw = pd.read_excel(DATA_DIR / "info.xlsx", sheet_name="info", header=None)
    info = raw.iloc[4:21].copy()
    info.columns = raw.iloc[3]
    info = info.reset_index(drop=True)
    info = info.rename(columns={
        "단계": "stage", "명칭": "site", "제작사": "maker", "모델명": "model", "호기": "turbine_no",
        "좌표(Google)": "coord", "KPX그룹": "kpx_group", "Hub Height(m)": "hub_height",
        "Rotor Diameter(m)": "rotor_diameter", "설비용량(MW)": "capacity_mw", "그룹설비용량(MW)": "group_capacity_mw",
    })
    info["kpx_group"] = info["kpx_group"].ffill().astype(int)
    coords = info["coord"].apply(dms_to_decimal)
    info["lat"] = coords.apply(lambda x: x[0])
    info["lon"] = coords.apply(lambda x: x[1])
    assert info.groupby("kpx_group")["capacity_mw"].sum().tolist() == [21.6, 21.6, 21.0]

    group_coords = info.groupby("kpx_group")[["lat", "lon"]].mean().reset_index()
    group_coords = group_coords.rename(columns={"kpx_group": "KPX그룹"})

    ldaps_group_grid = find_nearest_grid(ldaps_train, group_coords, "KPX그룹")
    gfs_group_grid = find_nearest_grid(gfs_train, group_coords, "KPX그룹")
    print("LDAPS 매칭:", ldaps_group_grid)
    print("GFS 매칭:", gfs_group_grid)

    LDAPS_UV_PAIRS = [("heightAboveGround_10_10u", "heightAboveGround_10_10v", "10m")]
    GFS_UV_PAIRS = [
        ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "10m"),
        ("heightAboveGround_80_u", "heightAboveGround_80_v", "80m"),
        ("heightAboveGround_100_100u", "heightAboveGround_100_100v", "100m"),
        ("planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v", "pbl"),
    ]
    ldaps_train = add_wind_features(ldaps_train, LDAPS_UV_PAIRS)
    ldaps_test = add_wind_features(ldaps_test, LDAPS_UV_PAIRS)
    gfs_train = add_wind_features(gfs_train, GFS_UV_PAIRS)
    gfs_test = add_wind_features(gfs_test, GFS_UV_PAIRS)
    for df in [gfs_train, gfs_test]:
        df["wind_shear_100_10"] = df["ws_100m"] - df["ws_10m"]
        df["wind_shear_80_10"] = df["ws_80m"] - df["ws_10m"]

    ldaps_group_train = build_group_weather(ldaps_train, ldaps_group_grid, "ldaps")
    ldaps_group_test = build_group_weather(ldaps_test, ldaps_group_grid, "ldaps")
    gfs_group_train = build_group_weather(gfs_train, gfs_group_grid, "gfs")
    gfs_group_test = build_group_weather(gfs_test, gfs_group_grid, "gfs")

    train_labels_long = train_labels.melt(
        id_vars=["kst_dtm"], value_vars=TARGET_COLS, var_name="kpx_group_col", value_name="target",
    )
    train_labels_long["kpx_group"] = train_labels_long["kpx_group_col"].str.extract(r"(\d+)").astype(int)
    train_labels_long = train_labels_long.rename(columns={"kst_dtm": "forecast_kst_dtm"}).drop(columns=["kpx_group_col"])

    test_base = sample_submission[["forecast_id", "forecast_kst_dtm"]].copy()
    test_long = pd.concat([test_base.assign(kpx_group=g) for g in [1, 2, 3]], ignore_index=True)

    train_df = train_labels_long.merge(
        ldaps_group_train, on=["forecast_kst_dtm", "kpx_group"], how="left"
    ).merge(gfs_group_train, on=["forecast_kst_dtm", "kpx_group"], how="left")
    test_df = test_long.merge(
        ldaps_group_test, on=["forecast_kst_dtm", "kpx_group"], how="left"
    ).merge(gfs_group_test, on=["forecast_kst_dtm", "kpx_group"], how="left")
    assert train_df.shape[0] == train_labels_long.shape[0]

    train_df = add_seasonal_interaction(add_forecast_disagreement(train_df))
    test_df = add_seasonal_interaction(add_forecast_disagreement(test_df))
    weather_cols = [c for c in train_df.columns if c not in ["forecast_kst_dtm", "kpx_group", "target"]]
    print(f"v10 피처 개수: {len(weather_cols)}")
    return train_df, test_df, weather_cols, sample_submission


# ───────────────────────────────────────────────
# 검증 공통 함수 (노트북 cell 7 이식)
# ───────────────────────────────────────────────
def calendar_features(dt_series):
    dt = pd.to_datetime(dt_series)
    out = pd.DataFrame(index=dt.index)
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def metric(answer_df, pred_df):
    group_nmae, group_ficr = [], []
    for col in TARGET_COLS:
        actual = answer_df[col].to_numpy(dtype=float)
        forecast = pred_df[col].to_numpy(dtype=float)
        capacity = CAPACITY_KWH[col]
        valid = actual >= capacity * 0.10
        actual, forecast = actual[valid], forecast[valid]
        error_rate = np.abs(forecast - actual) / capacity
        group_nmae.append(np.mean(error_rate))
        unit_price = np.select([error_rate <= 0.06, error_rate <= 0.08], [4.0, 3.0], default=0.0)
        group_ficr.append(np.sum(actual * unit_price) / np.sum(actual * 4.0))
    one_minus_nmae = 1 - np.mean(group_nmae)
    ficr = np.mean(group_ficr)
    return 0.5 * one_minus_nmae + 0.5 * ficr, one_minus_nmae, ficr


def metric_single_group(actual, forecast, capacity):
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    valid = actual >= capacity * 0.10
    actual_v, forecast_v = actual[valid], forecast[valid]
    if len(actual_v) == 0:
        return np.nan, np.nan, 0
    error_rate = np.abs(forecast_v - actual_v) / capacity
    nmae = np.mean(error_rate)
    unit_price = np.select([error_rate <= 0.06, error_rate <= 0.08], [4.0, 3.0], default=0.0)
    ficr = np.sum(actual_v * unit_price) / np.sum(actual_v * 4.0)
    return nmae, ficr, len(actual_v)


def make_sample_weight(y, capacity, low_weight=0.05):
    threshold = capacity * 0.10
    return np.where(y >= threshold, 1.0, low_weight)


def apply_rolling_smoothing(pred_series, window=3, alpha=0.5):
    rolling = pred_series.rolling(window, center=True, min_periods=1).mean()
    return alpha * pred_series + (1 - alpha) * rolling


def apply_bias_correction_v2(pred, ws, group, bias_map):
    pred = pred.copy()
    if group not in bias_map:
        return pred
    for interval, bias in bias_map[group].items():
        mask = (ws > interval.left) & (ws <= interval.right)
        pred[mask] = pred[mask] * (1 - bias)
    return pred


def lgbm_params(n_estimators, n_jobs, seed, objective="mae", alpha=None):
    p = dict(
        n_estimators=n_estimators, max_depth=6, learning_rate=0.03,
        num_leaves=31, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        random_state=seed, n_jobs=n_jobs, verbose=-1, objective=objective,
    )
    if alpha is not None:
        p["alpha"] = alpha
    return p


def estimate_bias_correction_adaptive(train_df, weather_cols, tr_end, n_estimators, n_jobs,
                                      low_weight=0.05, holdout_ratio=0.25, min_fit_days=90):
    bias_records = []
    for group in [1, 2, 3]:
        target_col = f"kpx_group_{group}"
        capacity = CAPACITY_KWH[target_col]
        g_data = train_df[(train_df["kpx_group"] == group) & (train_df["forecast_kst_dtm"] <= tr_end)].copy()
        g_data = g_data[g_data["target"].notna()]
        if len(g_data) == 0:
            continue
        first_date = g_data["forecast_kst_dtm"].min()
        available_days = max((tr_end - first_date).days, 1)
        holdout_days = min(int(available_days * holdout_ratio), 180)
        if available_days - holdout_days < min_fit_days:
            holdout_days = max(0, available_days - min_fit_days)
        tr_sub_end = tr_end - pd.Timedelta(days=holdout_days)
        fit_g = g_data[g_data["forecast_kst_dtm"] <= tr_sub_end]
        check_g = g_data[g_data["forecast_kst_dtm"] > tr_sub_end]
        if len(fit_g) == 0 or len(check_g) == 0:
            continue

        fit_feat = pd.concat([calendar_features(fit_g["forecast_kst_dtm"]).reset_index(drop=True),
                              fit_g[weather_cols].reset_index(drop=True)], axis=1)
        check_feat = pd.concat([calendar_features(check_g["forecast_kst_dtm"]).reset_index(drop=True),
                                check_g[weather_cols].reset_index(drop=True)], axis=1)
        imputer = SimpleImputer(strategy="median")
        X_fit = pd.DataFrame(imputer.fit_transform(fit_feat), columns=fit_feat.columns)
        X_check = pd.DataFrame(imputer.transform(check_feat), columns=check_feat.columns)
        y_fit = fit_g["target"]
        w_fit = make_sample_weight(y_fit.to_numpy(), capacity, low_weight=low_weight)

        model = lgb.LGBMRegressor(**lgbm_params(n_estimators, n_jobs, SEED))
        model.fit(X_fit, y_fit, sample_weight=w_fit)
        pred_check = model.predict(X_check)

        check_g = check_g.copy()
        check_g["signed_error_rate"] = (pred_check - check_g["target"]) / capacity
        valid = check_g["target"] >= capacity * 0.10
        check_g = check_g.loc[valid]
        if len(check_g) == 0:
            continue
        check_g["ws_bin"] = pd.cut(check_g["ldaps_ws_10m"], bins=[0, 11, 13, 100])
        bias_by_bin = check_g.groupby("ws_bin", observed=True)["signed_error_rate"].mean()
        bias_records.append({"group": group, "bias_by_bin": bias_by_bin.to_dict()})
    return bias_records


# ───────────────────────────────────────────────
# 핵심 신규 로직: 분위수 분포 기반 제출값 최적화
# ───────────────────────────────────────────────
def optimize_submission(pred_base, quantile_preds, capacity, ybar,
                        spread_scale=1.0, n_grid=33, stability_eps=1e-4):
    """시간별로 기대 점수 기여 E[-err + 단가(err)*y/(4*ȳ)] 를 최대화하는 q 선택.

    pred_base:      (N,)  블렌딩+편향보정+rolling 이후의 점 예측 (분포 중심으로 사용)
    quantile_preds: (N,K) 분위수 예측 (행별 정렬 전제 아님 — 내부에서 정렬)
    ybar:           그룹의 평가대상(≥10%cap) 시간 평균 실측 발전량
    """
    thr = capacity * 0.10
    y = np.sort(quantile_preds, axis=1)                       # 분위수 교차 방지
    K = y.shape[1]
    med = y[:, K // 2 - 1: K // 2 + 1].mean(axis=1)           # K=10 → q0.45/q0.55 평균
    y = y + (pred_base - med)[:, None]                        # 중앙값을 점 예측에 정렬
    if spread_scale != 1.0:
        y = pred_base[:, None] + spread_scale * (y - pred_base[:, None])
    y = np.clip(y, 0.0, capacity)

    lo, hi = y.min(axis=1), y.max(axis=1)
    t = np.linspace(0.0, 1.0, n_grid)
    cand = lo[:, None] + (hi - lo)[:, None] * t[None, :]
    cand = np.concatenate([cand, pred_base[:, None]], axis=1)  # 점 예측도 후보에 포함
    cand = np.clip(cand, thr, capacity)                        # q<thr는 항상 열등

    err = np.abs(cand[:, :, None] - y[:, None, :]) / capacity  # (N, C, K)
    price = np.where(err <= 0.06, 4.0, np.where(err <= 0.08, 3.0, 0.0))
    mask = (y >= thr)[:, None, :]                              # 평가 제외 시나리오는 기여 0
    gain = mask * (-err + price * y[:, None, :] / (4.0 * ybar))
    obj = gain.mean(axis=2)
    obj -= stability_eps * np.abs(cand - pred_base[:, None]) / capacity  # 동점이면 점 예측 근처

    best = np.argmax(obj, axis=1)
    q = cand[np.arange(len(pred_base)), best]
    all_below = (y < thr).all(axis=1)                          # 전 분포가 평가 제외 → 기존 방식
    q[all_below] = np.clip(pred_base[all_below], thr, capacity)
    return q


# ───────────────────────────────────────────────
# 학습 유닛: 한 (학습셋, 예측셋) 쌍에 대해 base 블렌딩 + 분위수 예측 생성
# ───────────────────────────────────────────────
def fit_predict_group(X_tr, y_tr, w_tr, X_va, args):
    lgbm_preds = []
    for seed in range(args.seeds):
        model = lgb.LGBMRegressor(**lgbm_params(args.trees, args.jobs, seed))
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        lgbm_preds.append(model.predict(X_va))
    pred_lgbm = np.mean(lgbm_preds, axis=0)

    xgb_preds = []
    for seed in range(args.seeds):
        model = xgbm.XGBRegressor(
            n_estimators=args.trees, max_depth=6, learning_rate=0.03,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=args.jobs, verbosity=0, objective="reg:absoluteerror",
        )
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        xgb_preds.append(model.predict(X_va))
    pred_xgb = np.mean(xgb_preds, axis=0)
    pred_blend = 0.5 * pred_lgbm + 0.5 * pred_xgb

    q_preds = []
    for q in QUANTILES:
        model = lgb.LGBMRegressor(**lgbm_params(args.trees, args.jobs, SEED, objective="quantile", alpha=q))
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        q_preds.append(model.predict(X_va))
    quantile_preds = np.column_stack(q_preds)
    return pred_blend, quantile_preds


def prepare_matrices(tr_g, va_g, weather_cols):
    tr_feat = pd.concat([calendar_features(tr_g["forecast_kst_dtm"]).reset_index(drop=True),
                         tr_g[weather_cols].reset_index(drop=True)], axis=1)
    va_feat = pd.concat([calendar_features(va_g["forecast_kst_dtm"]).reset_index(drop=True),
                         va_g[weather_cols].reset_index(drop=True)], axis=1)
    imputer = SimpleImputer(strategy="median")
    X_tr = pd.DataFrame(imputer.fit_transform(tr_feat), columns=tr_feat.columns)
    X_va = pd.DataFrame(imputer.transform(va_feat), columns=va_feat.columns)
    return X_tr, X_va


SPREAD_VARIANTS = {"qopt_s07": 0.7, "qopt_s10": 1.0, "qopt_s15": 1.5}
ALL_VARIANTS = ["control", *SPREAD_VARIANTS.keys()]


def postprocess_variants(pred_blend, quantile_preds, va_g, group, bias_map, capacity, ybar, args):
    """control(exp-003 후처리) + 분위수 최적화 변형들의 예측값 dict 반환"""
    pred = apply_bias_correction_v2(pred_blend, va_g["ldaps_ws_10m"].to_numpy(), group, bias_map)
    pred_series = pd.Series(pred, index=va_g.index)
    pred_series = apply_rolling_smoothing(pred_series, window=3, alpha=0.7)
    pred_base = pred_series.to_numpy()

    out = {"control": np.clip(pred_base, capacity * 0.10, capacity)}
    for name, scale in SPREAD_VARIANTS.items():
        out[name] = optimize_submission(pred_base, quantile_preds, capacity, ybar, spread_scale=scale)
    return out, pred_base


def coverage_report(actual, pred_base, quantile_preds, capacity, spread_scale=1.0):
    """평가대상 시간에서, 정렬된 분위수 분포의 중앙 50%/90% 구간 실측 커버리지"""
    thr = capacity * 0.10
    y = np.sort(quantile_preds, axis=1)
    K = y.shape[1]
    med = y[:, K // 2 - 1: K // 2 + 1].mean(axis=1)
    y = y + (pred_base - med)[:, None]
    if spread_scale != 1.0:
        y = pred_base[:, None] + spread_scale * (y - pred_base[:, None])
    valid = actual >= thr
    if valid.sum() == 0:
        return np.nan, np.nan
    a = actual[valid]
    cov50 = np.mean((a >= y[valid, 2]) & (a <= y[valid, 7]))   # q0.25~q0.75
    cov90 = np.mean((a >= y[valid, 0]) & (a <= y[valid, 9]))   # q0.05~q0.95
    return cov50, cov90


# ───────────────────────────────────────────────
# CV 루프
# ───────────────────────────────────────────────
FOLDS_Q = [
    {"name": "fold_2024Q1", "train_end": pd.Timestamp("2024-01-01 00:00:00"),
     "valid_start": pd.Timestamp("2024-01-01 01:00:00"), "valid_end": pd.Timestamp("2024-04-01 00:00:00")},
    {"name": "fold_2024Q2", "train_end": pd.Timestamp("2024-04-01 00:00:00"),
     "valid_start": pd.Timestamp("2024-04-01 01:00:00"), "valid_end": pd.Timestamp("2024-07-01 00:00:00")},
    {"name": "fold_2024Q3", "train_end": pd.Timestamp("2024-07-01 00:00:00"),
     "valid_start": pd.Timestamp("2024-07-01 01:00:00"), "valid_end": pd.Timestamp("2024-10-01 00:00:00")},
    {"name": "fold_2024Q4", "train_end": pd.Timestamp("2024-10-01 00:00:00"),
     "valid_start": pd.Timestamp("2024-10-01 01:00:00"), "valid_end": pd.Timestamp("2025-01-01 00:00:00")},
]


def run_cv(train_df, weather_cols, folds, args):
    fold_scores = {v: [] for v in ALL_VARIANTS}
    group_scores = []
    all_answer = []
    all_pred = {v: [] for v in ALL_VARIANTS}

    for fold in folds:
        t0 = time.time()
        bias_estimates = estimate_bias_correction_adaptive(
            train_df, weather_cols, fold["train_end"], args.trees, args.jobs)
        bias_map = {b["group"]: b["bias_by_bin"] for b in bias_estimates}

        tr_mask = train_df["forecast_kst_dtm"] <= fold["train_end"]
        va_mask = (train_df["forecast_kst_dtm"] >= fold["valid_start"]) & (train_df["forecast_kst_dtm"] <= fold["valid_end"])
        tr_all = train_df.loc[tr_mask].reset_index(drop=True)
        va_all = train_df.loc[va_mask].reset_index(drop=True)

        va_time_index = sorted(va_all["forecast_kst_dtm"].unique())
        answer_df = pd.DataFrame({"forecast_kst_dtm": va_time_index})
        pred_dfs = {v: pd.DataFrame({"forecast_kst_dtm": va_time_index}) for v in ALL_VARIANTS}

        for group in [1, 2, 3]:
            target_col = f"kpx_group_{group}"
            capacity = CAPACITY_KWH[target_col]
            tr_g = tr_all.loc[tr_all["kpx_group"] == group].copy()
            va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()

            va_answer = va_g[["forecast_kst_dtm", "target"]].rename(columns={"target": target_col})
            answer_df = answer_df.merge(va_answer, on="forecast_kst_dtm", how="left")

            target_mask = tr_g["target"].notna().to_numpy()
            if target_mask.sum() == 0:
                for v in ALL_VARIANTS:
                    pred_dfs[v][target_col] = 0.0
                continue
            tr_g = tr_g.loc[target_mask]

            X_tr, X_va = prepare_matrices(tr_g, va_g, weather_cols)
            y_tr = tr_g["target"]
            w_tr = make_sample_weight(y_tr.to_numpy(), capacity, low_weight=0.05)
            ybar = float(y_tr[y_tr >= capacity * 0.10].mean())

            pred_blend, quantile_preds = fit_predict_group(X_tr, y_tr, w_tr, X_va, args)
            variant_preds, pred_base = postprocess_variants(
                pred_blend, quantile_preds, va_g, group, bias_map, capacity, ybar, args)

            actual_va = va_g["target"].to_numpy()
            cov50, cov90 = coverage_report(actual_va, pred_base, quantile_preds, capacity)
            row = {"fold": fold["name"], "group": group, "cov50": cov50, "cov90": cov90}
            for v in ALL_VARIANTS:
                va_pred = va_g[["forecast_kst_dtm"]].copy()
                va_pred[target_col] = variant_preds[v]
                pred_dfs[v] = pred_dfs[v].merge(va_pred, on="forecast_kst_dtm", how="left")
                nmae_g, ficr_g, valid_n = metric_single_group(actual_va, variant_preds[v], capacity)
                row[f"{v}_1-NMAE"] = 1 - nmae_g if not np.isnan(nmae_g) else np.nan
                row[f"{v}_FICR"] = ficr_g
                row["valid_n"] = valid_n
            group_scores.append(row)

        answer_f = answer_df[TARGET_COLS].fillna(0.0)
        all_answer.append(answer_f)
        line = [f"[{fold['name']}] ({time.time() - t0:.0f}s)"]
        for v in ALL_VARIANTS:
            pred_f = pred_dfs[v][TARGET_COLS].fillna(0.0)
            all_pred[v].append(pred_f)
            score, nmae_s, ficr_s = metric(answer_f, pred_f)
            fold_scores[v].append({"fold": fold["name"], "score": score, "1-NMAE": nmae_s, "FICR": ficr_s})
            line.append(f"{v}: S={score:.4f} N={nmae_s:.4f} F={ficr_s:.4f}")
        print("\n  ".join(line), flush=True)

    print("\n================ 결과 요약 ================")
    pooled_answer = pd.concat(all_answer, ignore_index=True)
    pooled = {}
    for v in ALL_VARIANTS:
        cv = pd.DataFrame(fold_scores[v])
        pooled_pred = pd.concat(all_pred[v], ignore_index=True)
        p_score, p_nmae, p_ficr = metric(pooled_answer, pooled_pred)
        pooled[v] = (p_score, p_nmae, p_ficr)
        print(f"{v:10s} | pooled Score={p_score:.4f} (1-NMAE={p_nmae:.4f}, FICR={p_ficr:.4f})"
              f" | fold평균 Score={cv['score'].mean():.4f} (std {cv['score'].std():.4f})")

    gs = pd.DataFrame(group_scores)
    print("\n--- 그룹별 fold 평균 ---")
    agg_cols = [c for c in gs.columns if c.endswith("1-NMAE") or c.endswith("FICR")]
    print(gs.groupby("group")[agg_cols].mean().T.round(4).to_string())
    print("\n--- 분포 캘리브레이션 (spread=1.0 기준, 목표 cov50≈0.50 / cov90≈0.90) ---")
    print(gs.groupby("group")[["cov50", "cov90"]].mean().round(3).to_string())
    return pooled, gs


# ───────────────────────────────────────────────
# 최종 학습 → 제출 파일
# ───────────────────────────────────────────────
def train_final_and_submit(train_df, test_df, weather_cols, sample_submission, variant, args):
    print(f"\n================ 최종 학습 (variant={variant}) ================")
    max_date = train_df["forecast_kst_dtm"].max()
    bias_estimates = estimate_bias_correction_adaptive(train_df, weather_cols, max_date, args.trees, args.jobs)
    bias_map = {b["group"]: b["bias_by_bin"] for b in bias_estimates}
    print("최종 편향 보정 계수:", bias_map)

    predictions_final = pd.DataFrame(index=test_df.index)
    for group in [1, 2, 3]:
        target_col = f"kpx_group_{group}"
        capacity = CAPACITY_KWH[target_col]
        tr_g = train_df.loc[(train_df["kpx_group"] == group) & train_df["target"].notna()].copy()
        te_g = test_df.loc[test_df["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()

        X_tr, X_te = prepare_matrices(tr_g, te_g, weather_cols)
        y_tr = tr_g["target"]
        w_tr = make_sample_weight(y_tr.to_numpy(), capacity, low_weight=0.05)
        ybar = float(y_tr[y_tr >= capacity * 0.10].mean())

        pred_blend, quantile_preds = fit_predict_group(X_tr, y_tr, w_tr, X_te, args)
        variant_preds, _ = postprocess_variants(
            pred_blend, quantile_preds, te_g, group, bias_map, capacity, ybar, args)
        predictions_final.loc[te_g.index, target_col] = variant_preds[variant]
        print(f"{target_col} train rows: {len(tr_g)}")

    test_pred_long = test_df[["forecast_id", "forecast_kst_dtm", "kpx_group"]].copy()
    for group in [1, 2, 3]:
        mask = test_df["kpx_group"] == group
        test_pred_long.loc[mask, "pred"] = predictions_final.loc[mask, f"kpx_group_{group}"]

    submission = test_pred_long.pivot(index=["forecast_id", "forecast_kst_dtm"], columns="kpx_group", values="pred")
    submission.columns = [f"kpx_group_{int(c)}" for c in submission.columns]
    submission = submission.reset_index()
    submission["forecast_kst_dtm"] = pd.to_datetime(submission["forecast_kst_dtm"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    submission = sample_submission[["forecast_id"]].merge(submission, on="forecast_id", how="left")

    assert submission.shape[0] == sample_submission.shape[0]
    assert submission[TARGET_COLS].isna().sum().sum() == 0
    assert (submission["forecast_id"] == sample_submission["forecast_id"]).all()
    for col in TARGET_COLS:
        assert submission[col].max() <= CAPACITY_KWH[col]

    output_path = DATA_DIR / f"v_{variant}_submit.csv"
    submission.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="스모크 테스트 (1 seed, 150 trees, 마지막 1 fold)")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--no-submit", action="store_true")
    args = parser.parse_args()

    folds = FOLDS_Q
    if args.quick:
        args.seeds, args.trees = 1, 150
        folds = FOLDS_Q[-1:]
        print("### QUICK MODE ###")

    t0 = time.time()
    train_df, test_df, weather_cols, sample_submission = build_datasets()
    print(f"데이터 준비 완료 ({time.time() - t0:.0f}s)\n")

    pooled, _ = run_cv(train_df, weather_cols, folds, args)

    best = max(pooled, key=lambda v: pooled[v][0])
    print(f"\n최고 변형: {best} (pooled Score {pooled[best][0]:.4f}, control {pooled['control'][0]:.4f})")
    if args.no_submit or args.quick:
        return
    if best == "control":
        print("control이 최고 → 새 제출 파일 생성 안 함 (기존 v_blend5050_submit.csv 유지)")
        return
    train_final_and_submit(train_df, test_df, weather_cols, sample_submission, best, args)
    print(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
