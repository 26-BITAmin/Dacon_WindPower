# ═══════════════════════════════════════════════════════
# exp-082 게이트: 기상청 ASOS 지상관측 기반 "최근 예보오차 상태" 피처 → 챔피언 GBM base
#
# 근거:
#   - 지금까지 투입한 외부 소스는 전부 "예보"(LDAPS·GFS·JMA·EC·enfo) → exp-073에서
#     정보 중복(corr(cf,cf_hpo) 0.9996)으로 전이 실패. 관측은 유일하게 성격이 다른 자료형.
#   - 핵심 신호 = obs_gap: 최근 24h 동안 (관측 풍속 − 같은 시각 예보 풍속).
#     "요즘 모델이 과대/과소예보 중"이라는 상태 변수로, 기존 피처에 없는 유형.
#   - exp-078 진단상 이 계열은 레벨(일평균) 성분에 작용 → 오라클 상한 +0.0152. 기대 소액.
#
# 규정: 관측시각 ≤ D-1 12:00 (예측기준시점 13:00보다 1시간 앞)만 사용 —
#   대회 규칙 "예측기준시점 전에 관측·공개·확정된 과거 자료" 조항 준수. 상세는 exp082_fetch_asos.py.
#
# 피처 변형 2개 (GBM에만 — exp-054/073 선례, DL 무변경):
#   cf_obs3: +3 = obs_gap24, obs_ws24, obs_gap_ratio
#   cf_obs1: +1 = obs_gap24                        (exp-073의 "적을수록 낫다" 반영)
#
# 실행: C:\miniconda3\python.exe -X utf8 exp082_gate.py [--check] [--topk 5]
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, DATA_DIR
from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import RESULT_JSON as E56_JSON, train_gate_fold
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import build_train_frame

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ASOS_DIR = DATA_DIR / "external" / "kma_asos"
OBS_CACHE = CACHE_DIR / "exp082_gbm_obs.parquet"
OBS3 = ["obs_gap24", "obs_ws24", "obs_gap_ratio"]
OBS1 = ["obs_gap24"]


def load_asos():
    files = sorted(ASOS_DIR.glob("*.csv"))
    assert files, f"ASOS 수집물 없음: {ASOS_DIR}"
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["tm"] = pd.to_datetime(df["tm"].astype(str), format="%Y%m%d%H%M")
    df = df[["tm", "stn", "ws", "ta"]].dropna(subset=["ws"])
    df["ws"] = pd.to_numeric(df["ws"], errors="coerce")
    df = df.dropna(subset=["ws"]).drop_duplicates(["tm", "stn"])
    print(f"ASOS: {len(df)}행 · 지점 {df['stn'].nunique()} · "
          f"{df['tm'].min():%Y-%m-%d} ~ {df['tm'].max():%Y-%m-%d}", flush=True)
    return df


def pick_stations(asos, train_df, topk):
    """train 기간 라벨과 일평균 풍속 상관이 높은 지점 top-k (선택은 train에서만 수행)"""
    tr = train_df[train_df["target"].notna()].copy()
    tr["date"] = (tr["forecast_kst_dtm"] - pd.Timedelta(hours=1)).dt.date
    cap = tr["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"])
    tr["cf"] = tr["target"] / cap
    day_cf = tr.groupby("date")["cf"].mean()
    a = asos.copy()
    a["date"] = a["tm"].dt.date
    day_ws = a.groupby(["stn", "date"])["ws"].mean().unstack(0)
    common = day_ws.index.intersection(day_cf.index)
    corr = day_ws.loc[common].corrwith(day_cf.loc[common]).dropna().sort_values(ascending=False)
    top = corr.head(topk)
    print(f"지점 선택 (일평균 ws ↔ 일평균 CF 상관, train 기준 상위 {topk}):", flush=True)
    for stn, c in top.items():
        print(f"  stn {int(stn)}: corr {c:.4f}", flush=True)
    return [int(s) for s in top.index]


def build_obs_features(asos, train_df, stns):
    """타깃일 D → 관측창 W=[D-2 12:00, D-1 12:00] 집계 (규정: 관측시각 ≤ D-1 12:00).
    수제 집계(obs3/obs1)와 함께, exp-035 교훈("원시 > 수제 축약")을 반영해
    지점별 원시 수준(obs_s{stn})과 그룹 예보 수준(fc_ws24)도 함께 낸다."""
    a = asos[asos["stn"].isin(stns)]
    piv = a.pivot_table(index="tm", columns="stn", values="ws").sort_index()
    obs_s = piv.mean(axis=1)                                    # 선택 지점 시간별 평균
    oi = obs_s.index.to_numpy()
    per_stn = {s: piv[s].to_numpy() for s in piv.columns}
    ov = obs_s.to_numpy()

    df = train_df[["forecast_kst_dtm", "kpx_group", "ldaps_ws_10m"]].copy()
    df["tday"] = (df["forecast_kst_dtm"] - pd.Timedelta(hours=1)).dt.normalize()
    days = pd.DatetimeIndex(sorted(df["tday"].unique()))
    rows = []
    for g, sub in df.groupby("kpx_group"):
        s = sub.set_index("forecast_kst_dtm")["ldaps_ws_10m"].sort_index()
        si, sv = s.index.to_numpy(), s.to_numpy()
        for d in days:
            d64 = np.datetime64(d)
            w0, w1 = d64 - np.timedelta64(36, "h"), d64 - np.timedelta64(12, "h")
            om_mask = (oi >= w0) & (oi <= w1)
            f = sv[(si >= w0) & (si <= w1)]
            if om_mask.sum() < 6 or len(f) < 6:
                continue
            om, fm = float(np.nanmean(ov[om_mask])), float(f.mean())
            rec = {"tday": d, "kpx_group": g, "obs_gap24": om - fm, "obs_ws24": om,
                   "obs_gap_ratio": om / fm if fm > 0.1 else np.nan, "fc_ws24": fm}
            for stn, arr in per_stn.items():
                v = arr[om_mask]
                rec[f"obs_s{int(stn)}"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
            rows.append(rec)
    out = pd.DataFrame(rows)
    print(f"관측 피처: {len(out)}행 (일×그룹) | gap 평균 {out['obs_gap24'].mean():+.3f} m/s "
          f"· std {out['obs_gap24'].std():.3f}", flush=True)
    return out


def prepare_frame(topk):
    train_df, use_cols = build_train_frame()
    asos = load_asos()
    stns = pick_stations(asos, train_df, topk)
    obs = build_obs_features(asos, train_df, stns)
    train_df["tday"] = (train_df["forecast_kst_dtm"] - pd.Timedelta(hours=1)).dt.normalize()
    train_df = train_df.merge(obs, on=["tday", "kpx_group"], how="left")
    raw_cols = [c for c in obs.columns if c.startswith("obs_s")] + ["fc_ws24"]
    cov = train_df.groupby(train_df["forecast_kst_dtm"].dt.year)["obs_gap24"] \
                  .apply(lambda d: d.notna().mean().round(3))
    print("연도별 관측 피처 커버리지:\n" + cov.to_string(), flush=True)
    # 신호 점검: gap이 그날 실측 CF·예보 잔차와 상관 있나 (라벨 있는 행)
    m = train_df["target"].notna() & train_df["obs_gap24"].notna()
    for g in (1, 2, 3):
        mg = m & (train_df["kpx_group"] == g)
        capg = CAPACITY_KWH[f"kpx_group_{g}"]
        s = train_df.loc[mg]
        cfg = s["target"] / capg
        c1 = np.corrcoef(s["obs_gap24"], cfg)[0, 1]
        # 예보 잔차 프록시: 실측CF − (예보풍속의 CF 스케일 회귀 적합값)
        b = np.polyfit(s["ldaps_ws_10m"], cfg, 1)
        resid = cfg - np.polyval(b, s["ldaps_ws_10m"])
        c2 = np.corrcoef(s["obs_gap24"], resid)[0, 1]
        print(f"  그룹{g}: corr(obs_gap24, CF) {c1:+.4f} | corr(obs_gap24, 예보잔차) {c2:+.4f} "
              f"(n={mg.sum()})", flush=True)
    return train_df, use_cols, raw_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-082 게이트: ASOS 관측 기반 최근 예보오차 상태 피처 ###", flush=True)

    train_df, use_cols, raw_cols = prepare_frame(args.topk)
    print(f"입력 준비 완료 — {time.time()-t0:.0f}s", flush=True)
    if args.check:
        print("[--check 모드 종료 — GBM 학습 없음]")
        return
    e56 = json.loads(E56_JSON.read_text())

    cache, champ_bias = load_cache_full()
    want = {"cf_obs3": use_cols + OBS3, "cf_obs1": use_cols + OBS1,
            "cf_obsraw": use_cols + raw_cols}
    have = merge_cached_cols(cache, OBS_CACHE, list(want))
    for col, cols in want.items():
        if col in have:
            print(f"[{col}] 캐시 재사용", flush=True)
            continue
        preds = []
        for fname, fold in FOLD_LIST:
            tf = time.time()
            p = train_gate_fold(train_df, cols, fold, args.seeds, args.jobs,
                                e56["lgb"], e56["lgb_n"], e56["xgb"], e56["xgb_n"])
            preds.append(p.rename(columns={"cf_hpo": col}))
            print(f"  [{col}:{fname}] {time.time()-tf:.0f}s", flush=True)
        p = pd.concat(preds, ignore_index=True)
        m = cache[KEY].merge(p, on=KEY, how="left")
        assert m[col].notna().all(), "정렬 실패"
        cache[col] = m[col].to_numpy()
        have.append(col)
        save_cols(cache, OBS_CACHE, have)
        print(f"  캐시 저장 → {OBS_CACHE.name}", flush=True)

    mm = cache["target"].notna()
    for col in want:
        c = np.corrcoef(cache.loc[mm, col], cache.loc[mm, "cf_hpo"])[0, 1]
        shift = (cache[col].to_numpy() - cache["cf_hpo"].to_numpy()).mean() * 100
        print(f"  corr({col}, cf_hpo) = {c:.4f} | 평균 이동 {shift:+.3f}%cap", flush=True)

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} "
          f"FiCR {ref['cv'][2]:.4f}) | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)", flush=True)
    res = {"hpo_solo": ev("cf_hpo", None, 1.0, PP_NEW)}
    for tag, col in (("obs3", "cf_obs3"), ("obs1", "cf_obs1"), ("obsraw", "cf_obsraw")):
        res[f"{tag}_solo"] = ev(col, None, 1.0, PP_NEW)
        for w in (0.60, 0.65, 0.70):
            res[f"{tag}_w{int(w*100)}"] = ev(col, "dl_cln03", w, PP_NEW)
    print_table(res, ref)
    for tag in ("obs3", "obs1", "obsraw"):
        d_cv = res[f"{tag}_solo"]["cv"][0] - res["hpo_solo"]["cv"][0]
        d_hd = res[f"{tag}_solo"]["hold"][0] - res["hpo_solo"]["hold"][0]
        print(f"GBM 단독 델타 ({tag}−hpo): ΔCV {d_cv:+.4f} / Δhold {d_hd:+.4f}")
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
