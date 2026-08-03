# ═══════════════════════════════════════════════════════
# exp-110: 학습 가중의 가용성 지표 교체 — `avail_strict` → 유효출력률 `eff`
#
# 발견 (2026-08-03, 사용자 지적 "완전 정지/완전 운행만 있는 게 아니라 저속 모드도 있지 않나"):
#   현행 `avail_strict`는 터빈이 **도느냐 마느냐**만 센다 (dead = ws≥5 & power≤1).
#   SCADA 터빈 슬롯을 자기 자신의 파워커브(터빈×0.5m/s 격자 p85) 대비로 재면:
#     기대의 30~60%만 내는 "저속" 상태 — **UNISON 20.1% vs VESTAS 2.1%**
#   그 결과 `avail_strict=1.0`("완전가동")인 시간의 실제 유효출력률이
#     g1 중앙 0.880 / g2 0.887 / **g3 0.711**, g3는 **66.1%가 eff<0.8**.
#   → **avail_strict는 그룹3에서 오라벨**이고, exp-049의 cln03 가중이 g3 열화 시간의
#     3분의 2를 "정상"으로 학습시키고 있다.
#
# 가설: 가중 기준을 eff로 바꾸면 라벨 품질이 올라간다. exp-049(가중 도입)가
#   게이트 +0.0053 / LB +0.00058을 냈으므로 기준 교체는 근거가 분명한 축이다.
#   ※ 메모리의 "가용성 정제 축 재탐색 금지"는 **같은 avail_strict 위의 λ·τ 스윕**을 가리킨다.
#     여기서는 지표 정의 자체를 교체하므로 다른 축이다 (exp-056이 피처셋 변경을 근거로
#     "하이퍼 축 폐쇄"를 다시 연 것과 같은 유형).
#
# 변형 2개 (사용자 지시 — GBM에만 적용, DL·후처리·블렌드는 챔피언 고정 = 변경점 분리):
#   effw  : eff만       — eff<0.60 → 0.05 / <0.85 → 0.3 / else 1.0  (eff 결측은 1.0 중립)
#   bothw : 둘 다       — cln03(avail_strict) × eff03(eff)
#
# 판정 Δ = 보드 챔피언(hpo×dC_w65). 주기준 CV 4-fold(홀드아웃·corr은 참고 — 사용자 지시).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp110_effweight.py --stage build
#       C:\miniconda3\python.exe -X utf8 exp110_effweight.py --stage gate
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, TRAIN_DIR, make_sample_weight, prepare_matrices
from exp015_scada_pc import GROUP_TURBINES
from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW, avail_factor
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import RESULT_JSON as E56_JSON
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EFF_PARQUET = CACHE_DIR / "exp098_eff.parquet"
WS_LO, WS_HI = 5.0, 14.0          # 참조 곡선이 의미 있는 구간
REF_Q, REF_MIN = 0.85, 50.0


def build_eff():
    """터빈 슬롯을 자기 파워커브(p85) 대비로 정규화 → 시간×그룹 평균 유효출력률."""
    raw = {m: pd.read_csv(TRAIN_DIR / f"scada_{m}_train.csv", encoding="utf-8-sig",
                          parse_dates=["kst_dtm"]) for m in ("vestas", "unison")}
    longs = []
    for g, (maker, turbs, _) in GROUP_TURBINES.items():
        df = raw[maker]
        for t in turbs:
            s = df[["kst_dtm", f"{t}_power_kw10m", f"{t}_ws"]].copy()
            s.columns = ["dtm", "power", "ws"]
            s["turbine"], s["group"] = t, g
            longs.append(s)
    a = pd.concat(longs, ignore_index=True).dropna(subset=["power", "ws"])
    a = a[(a["ws"] >= WS_LO) & (a["ws"] < WS_HI)].copy()
    a["wsb"] = (a["ws"] * 2).round() / 2
    ref = a.groupby(["turbine", "wsb"])["power"].quantile(REF_Q).rename("ref").reset_index()
    a = a.merge(ref, on=["turbine", "wsb"])
    a = a[a["ref"] > REF_MIN]
    a["r"] = (a["power"] / a["ref"]).clip(0, 1.2)
    a["forecast_kst_dtm"] = a["dtm"].dt.ceil("h")        # exp-015 규약 (구간 종료 시각)
    h = (a.groupby(["group", "forecast_kst_dtm"])
           .agg(eff=("r", "mean"), n_slot=("r", "size")).reset_index())
    h = h[h["n_slot"] >= 3]                              # 시간당 슬롯이 너무 적으면 버림
    h.to_parquet(EFF_PARQUET, index=False)
    print(f"eff 저장 → {EFF_PARQUET.name} | {len(h)}행", flush=True)
    print(h.groupby("group")["eff"].describe(percentiles=[.05, .25, .5, .75]).round(3).to_string(),
          flush=True)
    return h


def eff_factor(e):
    a = np.asarray(e, dtype=float)
    return np.where(np.isnan(a), 1.0, np.where(a < 0.60, 0.05, np.where(a < 0.85, 0.3, 1.0)))


def make_weight(tr_g, cap, scheme):
    y = tr_g["target"].to_numpy()
    base = make_sample_weight(y, cap, low_weight=0.05)
    av = avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
    ef = eff_factor(tr_g["eff"].to_numpy())
    if scheme == "cln03":
        return base * av
    if scheme == "effw":
        return base * ef
    if scheme == "bothw":
        return base * av * ef
    raise ValueError(scheme)


def train_fold(train_df, cols, fold, seeds, jobs, e56, scheme, colname):
    import lightgbm as lgb
    import xgboost as xgbm
    tr_all, va_all = slice_fold(train_df, fold)
    rows = []
    for group in [1, 2, 3]:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
        va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
        if len(tr_g) == 0 or len(va_g) == 0:
            continue
        X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
        y_tr = tr_g["target"].to_numpy()
        w_tr = make_weight(tr_g, cap, scheme)
        pl = np.mean([lgb.LGBMRegressor(n_estimators=e56["lgb_n"], objective="mae",
                      random_state=s, n_jobs=jobs, verbose=-1, **e56["lgb"])
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        px = np.mean([xgbm.XGBRegressor(n_estimators=e56["xgb_n"], objective="reg:absoluteerror",
                      tree_method="hist", random_state=s, n_jobs=jobs, verbosity=0, **e56["xgb"])
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        rows.append(pd.DataFrame({"fold": fold["name"], "group": group,
                                  "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                                  colname: (0.5 * pl + 0.5 * px) / cap}))
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "gate"], default="gate")
    ap.add_argument("--variants", default="effw,bothw")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-110 유효출력률(eff) 기반 학습 가중 ###", flush=True)

    if args.stage == "build":
        build_eff()
        print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")
        return

    train_df, cols = build_train_frame()
    eff = pd.read_parquet(EFF_PARQUET)
    train_df = train_df.merge(eff[["forecast_kst_dtm", "group", "eff"]]
                              .rename(columns={"group": "kpx_group"}),
                              on=["forecast_kst_dtm", "kpx_group"], how="left")
    lab = train_df["target"].notna()
    print(f"입력 {len(cols)}피처 | 라벨행 {lab.sum()} | eff 커버리지 "
          f"{train_df.loc[lab, 'eff'].notna().mean():.3f}", flush=True)
    for g in [1, 2, 3]:
        m = lab & (train_df["kpx_group"] == g)
        w = {s: make_weight(train_df[m], CAPACITY_KWH[f"kpx_group_{g}"], s)
             for s in ("cln03", "effw", "bothw")}
        print(f"  그룹{g} 유효표본(가중합/n): " +
              " | ".join(f"{k} {v.sum()/m.sum():.3f}" for k, v in w.items()), flush=True)

    e56 = json.loads(E56_JSON.read_text())
    cache, champ_bias = load_cache_full()
    for tag in [t.strip() for t in args.variants.split(",") if t.strip()]:
        col = f"cf_{tag}"
        cpath = CACHE_DIR / f"exp098_gbm_{tag}.parquet"
        ppath = CACHE_DIR / f"exp098_gbm_{tag}_partial.parquet"
        if col in merge_cached_cols(cache, cpath, [col]):
            continue
        out = np.full(len(cache), np.nan)
        if ppath.exists():
            prev = pd.read_parquet(ppath)
            out = cache[KEY].merge(prev[KEY + [col]], on=KEY, how="left")[col].to_numpy()
        for fname, fold in FOLD_LIST:
            sel = (cache["fold"] == fname).to_numpy()
            if not np.isnan(out[sel]).any():
                continue
            tf = time.time()
            p = train_fold(train_df, cols, fold, args.seeds, args.jobs, e56, tag, col)
            m = cache.loc[sel, KEY].merge(p, on=KEY, how="left")
            assert m[col].notna().all(), f"정렬 실패 {fname}"
            out[sel] = m[col].to_numpy()
            cache[col] = out
            cache[KEY + [col]].to_parquet(ppath, index=False)
            print(f"  [{tag}:{fname}] {time.time()-tf:.0f}s", flush=True)
        cache[col] = out
        save_cols(cache, cpath, [col])

    mm = cache["target"].notna().to_numpy()
    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)",
          flush=True)
    res = {"hpo_solo": ev("cf_hpo", None, 1.0, PP_NEW)}
    for tag in [t.strip() for t in args.variants.split(",") if t.strip()]:
        col = f"cf_{tag}"
        v, h = cache[col].to_numpy(), cache["cf_hpo"].to_numpy()
        print(f"  corr({col}, cf_hpo) = {np.corrcoef(v[mm], h[mm])[0,1]:.4f} (참고) | "
              f"평균 이동 {(v-h).mean()*100:+.3f}%cap", flush=True)
        res[f"{tag}_solo"] = ev(col, None, 1.0, PP_NEW)
        for w in (0.65, 0.70):
            res[f"{tag}_w{int(w*100)}"] = ev(col, "dl_cln03", w, PP_NEW)
    print_table(res, ref)
    for tag in [t.strip() for t in args.variants.split(",") if t.strip()]:
        d = res[f"{tag}_solo"]["cv"][0] - res["hpo_solo"]["cv"][0]
        print(f"GBM 단독 ΔCV ({tag}−hpo): {d:+.4f}")
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
