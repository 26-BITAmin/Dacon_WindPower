# ═══════════════════════════════════════════════════════
# pd00: 별도 트랙 D (MLP 커브 base + GBM 잔차) — 입력 캐시 빌더 (PD-00)
#
# 사용자 설계 (2026-08-11):
#   "DL이 회귀로 파워 커브를 그리고, 미세한 차이를 GBM이 학습" (잔차형 직렬)
#   DL base 입력 후보: AnEn 벡터 K=30 (u/v · ws 형태 둘 다) / GBM 풍속 점예측 1열
#                       × (±환경 피처) = 6셀 (pd01에서 ablation)
#   피처 풀: "최근 pull한 파일(sinyoung simplified.ipynb, c8855bd)에서 쓴 피처만"
#            = 348(+EC4)에서 null importance 하위 83개 제거 → DL 265 / GBM 269
#   환경 피처: "바람 계열 제외 전부" + calendar 9열 (사용자 선택)
#
# 이 스크립트가 만드는 것:
#   --stage pools : WEAK 83개가 우리 프레임과 이름 일치하는지 검증 → pd00_pools.json
#                   (dl_pool 265 · gbm_pool 269 · env_cols 목록)
#   --stage gbmws : GBM 풍속 점예측 캐시 → pd00_gbmws.parquet
#                   타깃 = park_ws(나셀 파크 풍속, an12 build_scada_vec)
#                   피처 = 352 전체(AN-08/09 GBM352 관례) · LGB e56 · 3시드 평균
#                   split cf = 폴드 학습창 내 그룹별 시간중앙 2분할 교차적합
#                              (자기 실측 복사 방지 — av00 AnEn과 동일 관례)
#                   split va = 학습창 전체 obs로 학습 → 게이트 검증행 예측
#
# AnEn 멤버 60열은 av00_anen_members.parquet 재사용 (폴드·split 구조 동일).
#
# 실행(루트에서):
#   C:\miniconda3\python.exe -X utf8 "open (1)/testing/curve_track/pd00_inputs.py" --stage pools
#   C:\miniconda3\python.exe -X utf8 "open (1)/testing/curve_track/pd00_inputs.py" --stage gbmws
# ═══════════════════════════════════════════════════════
import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np
import pandas as pd

from exp035_raw_dl import FOLD_LIST
from exp056_hpo import RESULT_JSON as E56_JSON
from exp065_gbm3 import build_train_frame
from an12_emos import build_scada_vec

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KEY = ["kpx_group", "forecast_kst_dtm"]
POOLS_JSON = HERE.parent / "pd00_pools.json"
GBMWS_PATH = HERE.parent / "pd00_gbmws.parquet"
N_SEEDS = 3

# sinyoung simplified.ipynb (c8855bd) null importance 하위 83개 —
# score = 100*actual_gain/(null_gain q75 + 1) < 100 판정분. 원본 목록 그대로.
WEAK_FEATURES = [
    'disagreement_x_season', 'gfs_wd_cos_100m', 'gfs_wd_cos_pbl', 'high_variance_season',
    'jgsm_dis', 'ldaps_etc_0_VLCDC', 'ldaps_etc_0_lcc', 'ldaps_heightAboveGround_10_10u',
    'ldaps_surface_0_SNOM', 'ldaps_surface_0_lsm', 'ldaps_surface_0_sp', 'ldaps_wd_cos_10m',
    'ldaps_ws_10m', 'phy_gust_over_ws', 'phy_gust_range', 'sdG_gust_g5', 'sdG_t850_g5',
    'sdG_v100_g6', 'sdG_ws100_g3', 'sdG_ws10_g6', 'sdG_ws10_g7', 'sdG_ws10_g8', 'sdG_ws10_g9',
    'sdL_blh_g1', 'sdL_blh_g10', 'sdL_blh_g11', 'sdL_blh_g13', 'sdL_blh_g14', 'sdL_blh_g15',
    'sdL_blh_g16', 'sdL_blh_g2', 'sdL_blh_g3', 'sdL_blh_g4', 'sdL_blh_g5', 'sdL_blh_g7',
    'sdL_blh_g8', 'sdL_blh_g9', 'sdL_t2_g15', 'sdL_t2_g16', 'sdL_t2_g2', 'sdL_u10_g10',
    'sdL_u10_g14', 'sdL_u10_g15', 'sdL_u10_g16', 'sdL_u10_g2', 'sdL_u10_g5', 'sdL_u10_g8',
    'sdL_v10_g10', 'sdL_v10_g11', 'sdL_v10_g12', 'sdL_v10_g13', 'sdL_v10_g14', 'sdL_v10_g16',
    'sdL_v10_g2', 'sdL_v10_g4', 'sdL_v10_g7', 'sdL_v10_g9', 'sdL_ws10_g1', 'sdL_ws10_g15',
    'sdL_ws10_g2', 'sdL_ws10_g5', 'sdL_ws10_g8', 'sdL_ws50x_g1', 'sdL_ws50x_g14',
    'sdL_ws50x_g5', 'sdL_wsbl_g10', 'sdL_wsbl_g11', 'sdL_wsbl_g13', 'sdL_wsbl_g14',
    'sdL_wsbl_g16', 'sdL_wsbl_g7', 'sdL_wsbl_g8', 'spa_l10_range', 'spa_l10_std',
    'tmp_g100_d1', 'tmp_g100_dm1', 'tmp_g100_dm3', 'tmp_g100_r3_std', 'tmp_l10_d1',
    'tmp_l10_d3', 'tmp_l10_dm1', 'tmp_l10_r3_std', 'tmp_l10_r7_std',
]

# 트랙 D의 DL은 calendar 9열을 환경 피처에 포함한다 (사용자 선택 — 챔피언 DL 관례와 다름)
CALENDAR9 = ["cal_month", "cal_day", "cal_hour", "cal_dayofweek", "cal_is_weekend",
             "cal_hour_sin", "cal_hour_cos", "cal_month_sin", "cal_month_cos"]

# 바람 계열 판정 토큰 — 이 중 하나라도 (소문자) 이름에 들어가면 환경에서 제외
_WIND_TOKENS = ("ws", "wd", "10u", "10v", "u10", "v10", "100u", "100v", "u100", "v100",
                "80_u", "80_v", "gust", "shear", "blws", "mumax", "mvmax", "mumin", "mvmin",
                "disagreement", "agreement", "spa_", "tmp_", "x_season",
                "_dis", "vrate")   # 풍속 불일치도·환기율(풍속×blh)도 바람 비례 정보
_WIND_SUFFIX = re.compile(r"_(u|v)$|_(u|v)_")     # isobaric/pbl u,v 원본


def is_wind(col):
    c = col.lower()
    if c == "phy_density":
        return False
    if "swdir" in c or "swdif" in c:               # 단파 복사 — "wd" 토큰 오검 방지
        return False
    if c.startswith("phy_"):                       # density 외 phy는 전부 풍속 파생
        return True
    if any(t in c for t in _WIND_TOKENS):
        return True
    if _WIND_SUFFIX.search(c):
        return True
    return False


FRAME_CACHE = HERE.parent / "pd_frame352.parquet"


def load_frame_cached():
    """build_train_frame() + add_calendar 결과의 parquet 캐시.
    프로세스마다 CSV 병합을 반복하면 피크 RAM ~2.7GB가 겹쳐 OOM(PD-09 실측) —
    캐시 로드는 병합 중간 사본이 없어 피크가 절반 이하."""
    if FRAME_CACHE.exists():
        return pd.read_parquet(FRAME_CACHE)
    df, _ = build_train_frame()
    df = add_calendar(df)
    df.to_parquet(FRAME_CACHE, index=False)
    return df


def load_pools():
    d = json.loads(POOLS_JSON.read_text(encoding="utf-8"))
    return d["dl_pool"], d["gbm_pool"], d["env_cols"]


def add_calendar(df):
    dt = pd.to_datetime(df["forecast_kst_dtm"])
    df = df.copy()
    df["cal_month"], df["cal_day"], df["cal_hour"] = dt.dt.month, dt.dt.day, dt.dt.hour
    df["cal_dayofweek"] = dt.dt.dayofweek
    df["cal_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    df["cal_hour_sin"] = np.sin(2 * np.pi * df["cal_hour"] / 24)
    df["cal_hour_cos"] = np.cos(2 * np.pi * df["cal_hour"] / 24)
    df["cal_month_sin"] = np.sin(2 * np.pi * df["cal_month"] / 12)
    df["cal_month_cos"] = np.cos(2 * np.pi * df["cal_month"] / 12)
    return df


def stage_pools():
    train_df, cols352 = build_train_frame()
    missing = [c for c in WEAK_FEATURES if c not in cols352]
    assert not missing, f"WEAK 이름 불일치 {len(missing)}개: {missing[:10]}"
    ec_cols = ["ec_ws", "ec_wdsin", "ec_wdcos", "ec_dis"]
    base348 = [c for c in cols352 if c not in ec_cols]
    dl_pool = [c for c in base348 if c not in WEAK_FEATURES]
    gbm_pool = [c for c in cols352 if c not in WEAK_FEATURES]
    env = [c for c in dl_pool if not is_wind(c)]
    print(f"프레임 {train_df.shape} | 348 검증 {len(base348)} | "
          f"DL 풀 {len(dl_pool)} · GBM 풀 {len(gbm_pool)}")
    assert len(dl_pool) == 265 and len(gbm_pool) == 269, "풀 크기 불일치"
    print(f"\n환경 피처 {len(env)}개 (+ calendar 9 = DL env 입력 {len(env)+9}):")
    for c in env:
        print(f"  {c}")
    wind_dropped = [c for c in dl_pool if is_wind(c)]
    print(f"\n바람 계열로 제외된 것 {len(wind_dropped)}개 (표본 20): {wind_dropped[:20]}")
    POOLS_JSON.write_text(json.dumps(dict(
        dl_pool=dl_pool, gbm_pool=gbm_pool, env_cols=env, calendar9=CALENDAR9,
        weak_n=len(WEAK_FEATURES)), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장 → {POOLS_JSON.name}")


def _fit_predict_ws(fit_df, query_df, cols, params, n_est):
    import lightgbm as lgb
    preds = []
    for s in range(N_SEEDS):
        m = lgb.LGBMRegressor(n_estimators=n_est, objective="mae", random_state=s,
                              n_jobs=8, verbose=-1, **params)
        m.fit(fit_df[cols], fit_df["park_ws"])
        preds.append(m.predict(query_df[cols]))
    return np.mean(preds, axis=0)


# ── 정보 분할 GBM 풍속 (gws_hub 셀): base가 HUB10만 보고, 잔차 GBM이 나머지 독점 ──
HUB10 = ["gfs_ws_100m", "gfs_wd_sin_100m", "gfs_wd_cos_100m", "ldaps_ws_10m",
         "ec_ws", "jmsm_ws", "jgsm_ws", "gfs_wind_shear_100_10",
         "ldaps_heightAboveGround_2_t", "ldaps_surface_0_sp"]    # an09 관례 그대로
GBMWS_HUB_PATH = HERE.parent / "pd00_gbmws_hub.parquet"


# ── 분위수 GBM 풍속 (PD-02용): 점예측 대신 τ 15분위 벡터를 "바로 예측" ──
N_Q = 15
TAUS = [(k + 0.5) / N_Q for k in range(N_Q)]          # 0.033 ~ 0.967 균등
QWS_COLS = [f"qws{k}" for k in range(N_Q)]
QWS_PATH = HERE.parent / "pd00_qws.parquet"


def _fit_predict_qws(fit_df, query_df, cols, params):
    """τ별 분위수 LGB (1시드 — 분위수 자체가 목적함수 차이라 시드 평균 없음). GPU."""
    import lightgbm as lgb
    out = np.empty((len(query_df), N_Q))
    for k, tau in enumerate(TAUS):
        m = lgb.LGBMRegressor(n_estimators=params["n"], objective="quantile", alpha=tau,
                              random_state=0, n_jobs=4, verbose=-1, device="gpu",
                              **params["lgb"])
        m.fit(fit_df[cols], fit_df["park_ws"])
        out[:, k] = m.predict(query_df[cols])
    return out


def stage_qws(folds):
    t0 = time.time()
    e56 = json.loads(E56_JSON.read_text())
    params = {"lgb": e56["lgb"], "n": e56["lgb_n"]}
    train_df, cols352 = build_train_frame()
    vec = build_scada_vec()
    for g in [1, 2, 3]:
        train_df = train_df.merge(
            vec[g][["forecast_kst_dtm", "park_ws"]].assign(kpx_group=g),
            on=KEY, how="left", suffixes=("", "_d"))
        if "park_ws_d" in train_df:
            train_df["park_ws"] = train_df["park_ws"].fillna(train_df.pop("park_ws_d"))
    print(f"프레임 준비 {train_df.shape} — {time.time()-t0:.0f}s", flush=True)

    suffix = f"_{'-'.join(f.replace('fold_', '') for f in folds)}" if folds else ""
    path = HERE.parent / f"pd00_qws{suffix}.parquet"
    prev = pd.read_parquet(path) if path.exists() else None
    done = set(prev["fold"].unique()) if prev is not None else set()
    rows = [prev] if prev is not None else []
    for fname, fold in FOLD_LIST:
        if folds and fname not in folds:
            continue
        if fname in done:
            print(f"[{fname}] 캐시 재사용", flush=True)
            continue
        tf = time.time()
        win = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
        va = train_df.loc[(train_df["forecast_kst_dtm"] >= fold["valid_start"])
                          & (train_df["forecast_kst_dtm"] <= fold["valid_end"])]
        for g in [1, 2, 3]:
            win_g = win.loc[win["kpx_group"] == g]
            va_g = va.loc[va["kpx_group"] == g]
            obs = win_g.loc[win_g["park_ws"].notna()]
            med = obs["forecast_kst_dtm"].quantile(0.5)
            qA = win_g.loc[win_g["forecast_kst_dtm"] <= med]
            qB = win_g.loc[win_g["forecast_kst_dtm"] > med]
            fitA = obs.loc[obs["forecast_kst_dtm"] <= med]
            fitB = obs.loc[obs["forecast_kst_dtm"] > med]
            for fit_df, q_df in ((fitB, qA), (fitA, qB)):
                arr = _fit_predict_qws(fit_df, q_df, cols352, params)
                rows.append(q_df[KEY].assign(fold=fname, split="cf",
                                             **{c: arr[:, k] for k, c in enumerate(QWS_COLS)}))
            arr = _fit_predict_qws(obs, va_g, cols352, params)
            rows.append(va_g[KEY].assign(fold=fname, split="va",
                                         **{c: arr[:, k] for k, c in enumerate(QWS_COLS)}))
            print(f"  [{fname} g{g}] {time.time()-tf:.0f}s", flush=True)
        out = pd.concat(rows, ignore_index=True)
        out.to_parquet(path, index=False)
        print(f"[{fname}] 완료 {time.time()-tf:.0f}s (누적 {len(out):,}행)", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분 → {path.name}")


def stage_gbmws(folds, hub=False):
    t0 = time.time()
    e56 = json.loads(E56_JSON.read_text())
    params, n_est = e56["lgb"], e56["lgb_n"]
    train_df, cols352 = build_train_frame()
    if hub:
        missing = [c for c in HUB10 if c not in cols352]
        assert not missing, f"HUB10 이름 불일치: {missing}"
        cols352 = HUB10
    out_path = GBMWS_HUB_PATH if hub else GBMWS_PATH
    vec = build_scada_vec()
    for g in [1, 2, 3]:
        train_df = train_df.merge(
            vec[g][["forecast_kst_dtm", "park_ws"]].assign(kpx_group=g),
            on=KEY, how="left", suffixes=("", "_d"))
        if "park_ws_d" in train_df:
            train_df["park_ws"] = train_df["park_ws"].fillna(train_df.pop("park_ws_d"))
    print(f"프레임 준비 {train_df.shape} · park_ws 커버 "
          f"{train_df['park_ws'].notna().mean():.3f} — {time.time()-t0:.0f}s", flush=True)

    prev = pd.read_parquet(out_path) if out_path.exists() else None
    done = set(prev["fold"].unique()) if prev is not None else set()
    rows = [prev] if prev is not None else []
    for fname, fold in FOLD_LIST:
        if folds and fname not in folds:
            continue
        if fname in done:
            print(f"[{fname}] 캐시 재사용", flush=True)
            continue
        tf = time.time()
        win = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
        va = train_df.loc[(train_df["forecast_kst_dtm"] >= fold["valid_start"])
                          & (train_df["forecast_kst_dtm"] <= fold["valid_end"])]
        for g in [1, 2, 3]:
            win_g = win.loc[win["kpx_group"] == g]
            va_g = va.loc[va["kpx_group"] == g]
            obs = win_g.loc[win_g["park_ws"].notna()]
            assert len(obs) > 1000, f"{fname} g{g} obs {len(obs)}행"
            med = obs["forecast_kst_dtm"].quantile(0.5)
            # cf: 그룹별 시간중앙 2분할 교차적합 (질의 = 학습창 전 행)
            qA = win_g.loc[win_g["forecast_kst_dtm"] <= med]
            qB = win_g.loc[win_g["forecast_kst_dtm"] > med]
            fitA = obs.loc[obs["forecast_kst_dtm"] <= med]
            fitB = obs.loc[obs["forecast_kst_dtm"] > med]
            for fit_df, q_df in ((fitB, qA), (fitA, qB)):
                p = _fit_predict_ws(fit_df, q_df, cols352, params, n_est)
                rows.append(q_df[KEY].assign(fold=fname, split="cf", ws_gbm=p))
            # va: 학습창 전체 obs → 검증행
            pv = _fit_predict_ws(obs, va_g, cols352, params, n_est)
            rows.append(va_g[KEY].assign(fold=fname, split="va", ws_gbm=pv))
        out = pd.concat(rows, ignore_index=True)
        out.to_parquet(out_path, index=False)
        print(f"[{fname}] 완료 {time.time()-tf:.0f}s (누적 {len(out):,}행)", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분 → {out_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pools", "gbmws", "gbmws_hub", "qws"], required=True)
    ap.add_argument("--folds", type=str, default="")
    args = ap.parse_args()
    folds = [f for f in args.folds.split(",") if f]
    if args.stage == "pools":
        stage_pools()
    elif args.stage == "gbmws":
        stage_gbmws(folds)
    elif args.stage == "gbmws_hub":
        stage_gbmws(folds, hub=True)
    else:
        stage_qws(folds)


if __name__ == "__main__":
    main()
